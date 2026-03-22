package proxy

import (
	"bytes"
	"context"
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/cost"
	"github.com/mostlydev/cllama/internal/feeds"
	"github.com/mostlydev/cllama/internal/identity"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

// ContextLoader resolves per-agent context by ID.
type ContextLoader func(agentID string) (*agentctx.AgentContext, error)

// Handler proxies OpenAI-compatible chat requests to upstream providers.
type Handler struct {
	registry    *provider.Registry
	loadContext ContextLoader
	client      *http.Client
	logger      *logging.Logger
	accumulator *cost.Accumulator
	pricing     *cost.Pricing
	feedFetcher *feeds.Fetcher
}

// HandlerOption configures optional Handler behaviour.
type HandlerOption func(*Handler)

// WithCostTracking enables per-request cost recording.
func WithCostTracking(acc *cost.Accumulator, pricing *cost.Pricing) HandlerOption {
	return func(h *Handler) {
		h.accumulator = acc
		h.pricing = pricing
	}
}

// WithFeeds enables feed injection using the given pod name for identity headers.
func WithFeeds(podName string) HandlerOption {
	return func(h *Handler) {
		h.feedFetcher = feeds.NewFetcher(podName, nil, h.logger)
	}
}

func NewHandler(registry *provider.Registry, contextLoader ContextLoader, logger *logging.Logger, opts ...HandlerOption) *Handler {
	if registry == nil {
		registry = provider.NewRegistry("")
	}
	if contextLoader == nil {
		contextLoader = func(string) (*agentctx.AgentContext, error) {
			return nil, fmt.Errorf("context loader not configured")
		}
	}
	if logger == nil {
		logger = logging.New(io.Discard)
	}
	h := &Handler{
		registry:    registry,
		loadContext: contextLoader,
		client:      &http.Client{},
		logger:      logger,
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}

func (h *Handler) fetchFeeds(reqCtx context.Context, agentID string, agentCtx *agentctx.AgentContext) string {
	if h.feedFetcher == nil || agentCtx == nil || agentCtx.ContextDir == "" {
		return ""
	}

	entries, err := feeds.LoadManifest(agentCtx.ContextDir)
	if err != nil {
		h.logger.LogError(agentID, "", 0, 0, fmt.Errorf("load feed manifest: %w", err))
		return ""
	}
	if len(entries) == 0 {
		return ""
	}

	results := make([]feeds.FeedResult, 0, len(entries))
	for _, entry := range entries {
		result, err := h.feedFetcher.Fetch(reqCtx, agentID, entry)
		if err != nil {
			continue
		}
		results = append(results, result)
	}
	return feeds.FormatAllFeeds(results)
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	if r.Method != http.MethodPost {
		h.fail(w, http.StatusMethodNotAllowed, "method not allowed", "", "", start, nil)
		return
	}

	agentID, secret, err := identity.ParseBearer(r.Header.Get("Authorization"))
	if err != nil {
		h.fail(w, http.StatusUnauthorized, "invalid bearer token", "", "", start, err)
		return
	}

	ctx, err := h.loadContext(agentID)
	if err != nil {
		h.fail(w, http.StatusForbidden, "agent context not found", agentID, "", start, err)
		return
	}
	if err := validateSecret(ctx, agentID, secret); err != nil {
		h.fail(w, http.StatusForbidden, "invalid agent secret", agentID, "", start, err)
		return
	}

	// Route based on path: /v1/messages → Anthropic flow, everything else → OpenAI flow
	if strings.HasPrefix(r.URL.Path, "/v1/messages") {
		h.handleAnthropicMessages(w, r, agentID, ctx, start)
		return
	}

	h.handleOpenAI(w, r, agentID, ctx, start)
}

func (h *Handler) handleOpenAI(w http.ResponseWriter, r *http.Request, agentID string, agentCtx *agentctx.AgentContext, start time.Time) {
	inBody, err := io.ReadAll(r.Body)
	if err != nil {
		h.fail(w, http.StatusBadRequest, "failed to read request body", agentID, "", start, err)
		return
	}
	defer r.Body.Close()

	var payload map[string]any
	if err := json.Unmarshal(inBody, &payload); err != nil {
		h.fail(w, http.StatusBadRequest, "invalid JSON body", agentID, "", start, err)
		return
	}
	if feedBlock := h.fetchFeeds(r.Context(), agentID, agentCtx); feedBlock != "" {
		feeds.InjectOpenAI(payload, feedBlock)
	}

	requestedModel, _ := payload["model"].(string)
	requestedModel = strings.TrimSpace(requestedModel)
	if requestedModel == "" {
		h.fail(w, http.StatusBadRequest, "missing model field", agentID, "", start, fmt.Errorf("missing model"))
		return
	}

	providerName, upstreamModel, err := splitModel(requestedModel)
	if err != nil {
		h.fail(w, http.StatusBadRequest, err.Error(), agentID, requestedModel, start, err)
		return
	}

	prov, err := h.registry.Get(providerName)
	if err != nil {
		// Compatibility: some clients targeting OpenRouter via an OpenAI-compatible
		// provider config send vendor-prefixed model IDs like "anthropic/claude-*"
		// instead of "openrouter/anthropic/claude-*". If no first-party provider is
		// configured for that prefix, route through OpenRouter and preserve the full
		// vendor/model path as the upstream model.
		bridge, bridgeErr := h.registry.Get("openrouter")
		if bridgeErr != nil || strings.EqualFold(providerName, "openrouter") {
			h.fail(w, http.StatusBadGateway, "unknown provider", agentID, requestedModel, start, err)
			return
		}
		upstreamModel = providerName + "/" + upstreamModel
		providerName = bridge.Name
		prov = bridge
	}

	// Format bridge: if the resolved provider uses Anthropic format but
	// the incoming request is OpenAI format (/v1/chat/completions), route
	// through OpenRouter instead (which accepts OpenAI format for all models).
	if strings.EqualFold(prov.APIFormat, "anthropic") {
		bridge, bridgeErr := h.registry.Get("openrouter")
		if bridgeErr != nil {
			h.fail(w, http.StatusBadGateway,
				fmt.Sprintf("provider %q uses anthropic format but request is openai format; configure openrouter for format bridging", providerName),
				agentID, requestedModel, start, bridgeErr)
			return
		}
		// Reconstruct model as provider/model so OpenRouter routes correctly.
		upstreamModel = providerName + "/" + upstreamModel
		providerName = bridge.Name
		prov = bridge
	}

	if h.accumulator != nil && h.pricing != nil && isStreamingChatCompletions(r.URL.Path, payload) {
		ensureStreamUsage(payload)
	}

	payload["model"] = upstreamModel
	outBody, err := json.Marshal(payload)
	if err != nil {
		h.fail(w, http.StatusInternalServerError, "failed to encode upstream body", agentID, requestedModel, start, err)
		return
	}

	targetURL, err := buildUpstreamURL(prov.BaseURL, r.URL.Path, r.URL.RawQuery)
	if err != nil {
		h.fail(w, http.StatusBadGateway, "invalid provider URL", agentID, requestedModel, start, err)
		return
	}

	outReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, targetURL, bytes.NewReader(outBody))
	if err != nil {
		h.fail(w, http.StatusBadGateway, "failed to create upstream request", agentID, requestedModel, start, err)
		return
	}
	copyRequestHeaders(outReq.Header, r.Header)
	outReq.Header.Set("Content-Type", "application/json")

	if err := h.setProviderAuth(outReq, prov, agentID, requestedModel, start, w); err != nil {
		return // error already written
	}

	h.proxyAndLog(w, outReq, agentID, providerName, requestedModel, upstreamModel, start)
}

func (h *Handler) handleAnthropicMessages(w http.ResponseWriter, r *http.Request, agentID string, agentCtx *agentctx.AgentContext, start time.Time) {
	inBody, err := io.ReadAll(r.Body)
	if err != nil {
		h.fail(w, http.StatusBadRequest, "failed to read request body", agentID, "", start, err)
		return
	}
	defer r.Body.Close()

	var payload map[string]any
	if err := json.Unmarshal(inBody, &payload); err != nil {
		h.fail(w, http.StatusBadRequest, "invalid JSON body", agentID, "", start, err)
		return
	}
	if feedBlock := h.fetchFeeds(r.Context(), agentID, agentCtx); feedBlock != "" {
		feeds.InjectAnthropic(payload, feedBlock)
	}

	requestedModel, _ := payload["model"].(string)
	requestedModel = strings.TrimSpace(requestedModel)
	if requestedModel == "" {
		h.fail(w, http.StatusBadRequest, "missing model field", agentID, "", start, fmt.Errorf("missing model"))
		return
	}

	// Anthropic models don't use provider prefix — route directly to "anthropic" provider
	prov, err := h.registry.Get("anthropic")
	if err != nil {
		h.fail(w, http.StatusBadGateway, "anthropic provider not configured", agentID, requestedModel, start, err)
		return
	}

	outBody, err := json.Marshal(payload)
	if err != nil {
		h.fail(w, http.StatusInternalServerError, "failed to encode upstream body", agentID, requestedModel, start, err)
		return
	}

	targetURL, err := buildUpstreamURL(prov.BaseURL, r.URL.Path, r.URL.RawQuery)
	if err != nil {
		h.fail(w, http.StatusBadGateway, "invalid provider URL", agentID, requestedModel, start, err)
		return
	}

	outReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, targetURL, bytes.NewReader(outBody))
	if err != nil {
		h.fail(w, http.StatusBadGateway, "failed to create upstream request", agentID, requestedModel, start, err)
		return
	}
	copyRequestHeaders(outReq.Header, r.Header)
	outReq.Header.Set("Content-Type", "application/json")

	// Forward Anthropic-specific headers
	for _, hdr := range []string{"Anthropic-Version", "Anthropic-Beta"} {
		if v := r.Header.Get(hdr); v != "" {
			outReq.Header.Set(hdr, v)
		}
	}

	if err := h.setProviderAuth(outReq, prov, agentID, requestedModel, start, w); err != nil {
		return // error already written
	}

	h.proxyAndLog(w, outReq, agentID, "anthropic", requestedModel, requestedModel, start)
}

// setProviderAuth applies the provider's auth method to the upstream request.
// Returns an error (and writes the HTTP response) if auth cannot be applied.
func (h *Handler) setProviderAuth(outReq *http.Request, prov *provider.Provider, agentID, requestedModel string, start time.Time, w http.ResponseWriter) error {
	switch strings.ToLower(strings.TrimSpace(prov.Auth)) {
	case "", "bearer":
		if strings.TrimSpace(prov.APIKey) == "" {
			h.fail(w, http.StatusBadGateway, "provider API key not configured", agentID, requestedModel, start, fmt.Errorf("missing API key for %s", prov.Name))
			return fmt.Errorf("missing API key")
		}
		outReq.Header.Set("Authorization", "Bearer "+prov.APIKey)
	case "x-api-key":
		if strings.TrimSpace(prov.APIKey) == "" {
			h.fail(w, http.StatusBadGateway, "provider API key not configured", agentID, requestedModel, start, fmt.Errorf("missing API key for %s", prov.Name))
			return fmt.Errorf("missing API key")
		}
		outReq.Header.Del("Authorization")
		outReq.Header.Set("X-Api-Key", prov.APIKey)
	case "none":
		outReq.Header.Del("Authorization")
	default:
		h.fail(w, http.StatusBadGateway, "unsupported provider auth", agentID, requestedModel, start, fmt.Errorf("unsupported auth mode: %s", prov.Auth))
		return fmt.Errorf("unsupported auth mode")
	}
	return nil
}

// proxyAndLog forwards the request upstream, streams the response, and logs.
func (h *Handler) proxyAndLog(w http.ResponseWriter, outReq *http.Request, agentID, providerName, requestedModel, upstreamModel string, start time.Time) {
	h.logger.LogRequest(agentID, requestedModel)
	resp, err := h.client.Do(outReq)
	if err != nil {
		h.fail(w, http.StatusBadGateway, "upstream request failed", agentID, requestedModel, start, err)
		return
	}
	defer resp.Body.Close()

	copyResponseHeaders(w.Header(), resp.Header)
	w.WriteHeader(resp.StatusCode)

	var responseBuf bytes.Buffer
	tee := io.TeeReader(resp.Body, &responseBuf)
	if err := streamBody(w, tee); err != nil {
		h.logger.LogError(agentID, requestedModel, resp.StatusCode, time.Since(start).Milliseconds(), err)
		return
	}

	var costInfo *logging.CostInfo
	if h.accumulator != nil && h.pricing != nil {
		captured := responseBuf.Bytes()
		var usage cost.Usage
		if isSSE(resp.Header) {
			usage, _ = cost.ExtractUsageFromSSE(captured)
		} else {
			usage, _ = cost.ExtractUsage(captured)
		}
		if usage.PromptTokens > 0 || usage.CompletionTokens > 0 {
			rate, ok := h.pricing.Lookup(providerName, upstreamModel)
			costUSD := 0.0
			if ok {
				costUSD = rate.Compute(usage.PromptTokens, usage.CompletionTokens)
			}
			h.accumulator.Record(agentID, providerName, upstreamModel,
				usage.PromptTokens, usage.CompletionTokens, costUSD)
			costInfo = &logging.CostInfo{
				InputTokens:  usage.PromptTokens,
				OutputTokens: usage.CompletionTokens,
				CostUSD:      costUSD,
			}
		}
	}

	latency := time.Since(start).Milliseconds()
	if costInfo != nil {
		h.logger.LogResponseWithCost(agentID, requestedModel, resp.StatusCode, latency, costInfo)
	} else {
		h.logger.LogResponse(agentID, requestedModel, resp.StatusCode, latency)
	}
}

func (h *Handler) fail(w http.ResponseWriter, status int, msg, clawID, model string, start time.Time, err error) {
	writeJSONError(w, status, msg)
	h.logger.LogError(clawID, model, status, time.Since(start).Milliseconds(), err)
}

func validateSecret(ctx *agentctx.AgentContext, agentID, presentedSecret string) error {
	stored := strings.TrimSpace(ctx.MetadataToken())
	if stored == "" {
		return fmt.Errorf("metadata token missing")
	}

	if strings.HasPrefix(strings.ToLower(stored), "bearer ") {
		stored = strings.TrimSpace(stored[7:])
	}

	storedAgent, storedSecret, hasColon := strings.Cut(stored, ":")
	if hasColon {
		if storedAgent != "" && storedAgent != agentID {
			return fmt.Errorf("token agent mismatch")
		}
		if !constantTimeEqual(storedSecret, presentedSecret) {
			return fmt.Errorf("secret mismatch")
		}
		return nil
	}

	if !constantTimeEqual(stored, presentedSecret) {
		return fmt.Errorf("secret mismatch")
	}
	return nil
}

func splitModel(model string) (providerName, upstreamModel string, err error) {
	providerName, upstreamModel, ok := strings.Cut(strings.TrimSpace(model), "/")
	if !ok || providerName == "" || upstreamModel == "" {
		return "", "", fmt.Errorf("model must be provider-prefixed: <provider>/<model>")
	}
	return strings.ToLower(providerName), upstreamModel, nil
}

func isStreamingChatCompletions(path string, payload map[string]any) bool {
	if !strings.HasPrefix(path, "/v1/chat/completions") {
		return false
	}
	stream, _ := payload["stream"].(bool)
	return stream
}

func ensureStreamUsage(payload map[string]any) {
	streamOptions, _ := payload["stream_options"].(map[string]any)
	if streamOptions == nil {
		streamOptions = make(map[string]any)
	}
	streamOptions["include_usage"] = true
	payload["stream_options"] = streamOptions
}

func buildUpstreamURL(baseURL, incomingPath, rawQuery string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return "", err
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("invalid base URL: %q", baseURL)
	}

	suffix := incomingPath
	if !strings.HasPrefix(suffix, "/") {
		suffix = "/" + suffix
	}
	if strings.HasPrefix(suffix, "/v1/") {
		suffix = strings.TrimPrefix(suffix, "/v1")
	} else if suffix == "/v1" {
		suffix = "/"
	}

	u.Path = strings.TrimRight(u.Path, "/") + suffix
	u.RawQuery = rawQuery
	return u.String(), nil
}

func copyRequestHeaders(dst, src http.Header) {
	for k, vals := range src {
		if isHopByHopHeader(k) || strings.EqualFold(k, "Authorization") || strings.EqualFold(k, "Accept-Encoding") {
			continue
		}
		for _, v := range vals {
			dst.Add(k, v)
		}
	}
}

func copyResponseHeaders(dst, src http.Header) {
	for k, vals := range src {
		if isHopByHopHeader(k) {
			continue
		}
		dst.Del(k)
		for _, v := range vals {
			dst.Add(k, v)
		}
	}
}

func isHopByHopHeader(name string) bool {
	switch strings.ToLower(name) {
	case "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailer", "transfer-encoding", "upgrade":
		return true
	default:
		return false
	}
}

func isSSE(h http.Header) bool {
	return strings.Contains(h.Get("Content-Type"), "text/event-stream")
}

func streamBody(w http.ResponseWriter, body io.Reader) error {
	flusher, _ := w.(http.Flusher)
	if flusher == nil {
		_, err := io.Copy(w, body)
		return err
	}

	buf := make([]byte, 32*1024)
	for {
		n, err := body.Read(buf)
		if n > 0 {
			if _, werr := w.Write(buf[:n]); werr != nil {
				return werr
			}
			flusher.Flush()
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

func writeJSONError(w http.ResponseWriter, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": msg,
		},
	})
}

func constantTimeEqual(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(a), []byte(b)) == 1
}
