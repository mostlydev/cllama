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
	"github.com/mostlydev/cllama/internal/alert"
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
	notifier    *alert.Notifier
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

// WithNotifier attaches an alert notifier for pool transition events.
func WithNotifier(n *alert.Notifier) HandlerOption {
	return func(h *Handler) {
		h.notifier = n
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

	// Resolve provider (with format-bridge fallback for vendor-prefixed models).
	resolvedProvider, resolvedUpstream, err := h.resolveOpenAIProvider(providerName, upstreamModel)
	if err != nil {
		h.fail(w, http.StatusBadGateway, err.Error(), agentID, requestedModel, start, err)
		return
	}
	providerName = resolvedProvider
	upstreamModel = resolvedUpstream

	if h.accumulator != nil && h.pricing != nil && isStreamingChatCompletions(r.URL.Path, payload) {
		ensureStreamUsage(payload)
	}

	payload["model"] = upstreamModel
	outBody, err := json.Marshal(payload)
	if err != nil {
		h.fail(w, http.StatusInternalServerError, "failed to encode upstream body", agentID, requestedModel, start, err)
		return
	}

	h.dispatchWithRetry(w, r, agentID, providerName, requestedModel, upstreamModel, outBody, start)
}

// resolveOpenAIProvider maps a parsed provider name to the actual provider name
// and potentially rewrites the upstream model (format bridge).
// Returns the final (providerName, upstreamModel) to use.
func (h *Handler) resolveOpenAIProvider(providerName, upstreamModel string) (string, string, error) {
	prov, err := h.registry.Get(providerName)
	if err != nil {
		// Vendor-prefix bridge: route unknown provider prefix through openrouter.
		bridge, bridgeErr := h.registry.Get("openrouter")
		if bridgeErr != nil || strings.EqualFold(providerName, "openrouter") {
			return "", "", fmt.Errorf("unknown provider %q", providerName)
		}
		return bridge.Name, providerName + "/" + upstreamModel, nil
	}

	// Format bridge: anthropic provider on OpenAI path → route via openrouter.
	if strings.EqualFold(prov.APIFormat, "anthropic") {
		bridge, bridgeErr := h.registry.Get("openrouter")
		if bridgeErr != nil {
			return "", "", fmt.Errorf("provider %q uses anthropic format but request is openai format; configure openrouter for format bridging", providerName)
		}
		return bridge.Name, providerName + "/" + upstreamModel, nil
	}

	return providerName, upstreamModel, nil
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

	outBody, err := json.Marshal(payload)
	if err != nil {
		h.fail(w, http.StatusInternalServerError, "failed to encode upstream body", agentID, requestedModel, start, err)
		return
	}

	h.dispatchWithRetry(w, r, agentID, "anthropic", requestedModel, requestedModel, outBody, start)
}

// dispatchWithRetry selects a key, dispatches the request, and retries on key-level
// failures (401/403/402/quota-429 → dead, rate-limit-429 → cooldown).
// 5xx and transport errors do NOT cause key state changes.
func (h *Handler) dispatchWithRetry(w http.ResponseWriter, r *http.Request, agentID, providerName, requestedModel, upstreamModel string, outBody []byte, start time.Time) {
	const maxKeyAttempts = 5

	for attempt := 0; attempt < maxKeyAttempts; attempt++ {
		prov, lease, err := h.registry.SelectKey(providerName)
		if err != nil {
			// All keys dead/disabled — give up.
			if _, ok := err.(*provider.CooldownError); ok {
				h.fail(w, http.StatusServiceUnavailable, "all provider keys in cooldown", agentID, requestedModel, start, err)
				return
			}
			h.fail(w, http.StatusBadGateway, err.Error(), agentID, requestedModel, start, err)
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

		// Forward Anthropic-specific headers for the Anthropic path.
		if strings.HasPrefix(r.URL.Path, "/v1/messages") {
			for _, hdr := range []string{"Anthropic-Version", "Anthropic-Beta"} {
				if v := r.Header.Get(hdr); v != "" {
					outReq.Header.Set(hdr, v)
				}
			}
		}

		if err := applyProviderAuth(outReq, prov); err != nil {
			h.fail(w, http.StatusBadGateway, "provider auth not configured", agentID, requestedModel, start, err)
			return
		}

		h.logger.LogRequest(agentID, requestedModel)
		resp, err := h.client.Do(outReq)
		if err != nil {
			// Transport error — no key state change, return 502.
			h.fail(w, http.StatusBadGateway, "upstream request failed", agentID, requestedModel, start, err)
			return
		}

		classification := classifyResponse(resp)

		switch classification {
		case classAuth:
			// 401/403/402 or quota-429: key is permanently dead.
			resp.Body.Close()
			reason := fmt.Sprintf("http_%d", resp.StatusCode)
			_ = h.registry.MarkDead(lease.ProviderName, lease.KeyID, reason, resp.StatusCode)
			h.logger.LogProviderPool(lease.ProviderName, lease.KeyID, "dead", reason, "")
			if h.notifier != nil {
				h.notifier.Notify(alert.PoolEvent{Provider: lease.ProviderName, KeyID: lease.KeyID, Action: "dead", Reason: reason})
			}
			_ = h.registry.SaveToFile()
			continue // try next key

		case classRateLimit:
			// Rate-limit 429: cooldown.
			resp.Body.Close()
			cooldownDur := parseCooldownDuration(resp)
			until := time.Now().UTC().Add(cooldownDur)
			_ = h.registry.MarkCooldown(lease.ProviderName, lease.KeyID, "rate_limit", until)
			cooldownUntil := until.Format(time.RFC3339)
			h.logger.LogProviderPool(lease.ProviderName, lease.KeyID, "cooldown", "rate_limit", cooldownUntil)
			if h.notifier != nil {
				h.notifier.Notify(alert.PoolEvent{
					Provider:      lease.ProviderName,
					KeyID:         lease.KeyID,
					Action:        "cooldown",
					Reason:        "rate_limit",
					CooldownUntil: cooldownUntil,
				})
			}
			_ = h.registry.SaveToFile()
			continue // try next key

		default:
			// Success or 5xx: stream response back, no key state change.
			h.streamResponse(w, resp, agentID, providerName, requestedModel, upstreamModel, start)
			return
		}
	}

	h.fail(w, http.StatusBadGateway, "no usable provider key after retries", agentID, requestedModel, start, fmt.Errorf("exhausted %d key attempts", maxKeyAttempts))
}

type responseClass int

const (
	classOK        responseClass = iota
	classAuth                    // dead key: 401, 403, 402, quota-429
	classRateLimit               // cooldown: rate-limit 429
)

func classifyResponse(resp *http.Response) responseClass {
	switch resp.StatusCode {
	case http.StatusUnauthorized, http.StatusForbidden, http.StatusPaymentRequired:
		return classAuth
	case http.StatusTooManyRequests:
		if isQuotaExhausted(resp) {
			return classAuth
		}
		return classRateLimit
	default:
		return classOK
	}
}

// isQuotaExhausted peeks at the response body to distinguish billing/quota
// exhaustion from ordinary rate limiting. The body is replaced so the caller
// can still read it.
func isQuotaExhausted(resp *http.Response) bool {
	if resp.Body == nil {
		return false
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4096))
	resp.Body.Close()
	resp.Body = io.NopCloser(bytes.NewReader(body))
	if err != nil {
		return false
	}
	lower := strings.ToLower(string(body))
	return strings.Contains(lower, "insufficient_quota") ||
		strings.Contains(lower, "billing") ||
		strings.Contains(lower, "exceeded your current quota")
}

// parseCooldownDuration returns how long to cool down this key.
// It respects Retry-After if present, with a cap of 60s as a sane default.
func parseCooldownDuration(resp *http.Response) time.Duration {
	const defaultCooldown = 10 * time.Second
	const maxCooldown = 60 * time.Second

	ra := resp.Header.Get("Retry-After")
	if ra == "" {
		return defaultCooldown
	}
	// Retry-After may be seconds (integer) or an HTTP-date; only parse integer form.
	var secs int
	if _, err := fmt.Sscanf(ra, "%d", &secs); err == nil && secs > 0 {
		d := time.Duration(secs) * time.Second
		if d > maxCooldown {
			return maxCooldown
		}
		return d
	}
	return defaultCooldown
}

// applyProviderAuth sets the appropriate auth header on the outgoing request.
func applyProviderAuth(outReq *http.Request, prov *provider.Provider) error {
	switch strings.ToLower(strings.TrimSpace(prov.Auth)) {
	case "", "bearer":
		if strings.TrimSpace(prov.APIKey) == "" {
			return fmt.Errorf("missing API key for %s", prov.Name)
		}
		outReq.Header.Set("Authorization", "Bearer "+prov.APIKey)
	case "x-api-key":
		if strings.TrimSpace(prov.APIKey) == "" {
			return fmt.Errorf("missing API key for %s", prov.Name)
		}
		outReq.Header.Del("Authorization")
		outReq.Header.Set("X-Api-Key", prov.APIKey)
	case "none":
		outReq.Header.Del("Authorization")
	default:
		return fmt.Errorf("unsupported auth mode: %s", prov.Auth)
	}
	return nil
}

// streamResponse forwards the upstream response to the client and logs it.
func (h *Handler) streamResponse(w http.ResponseWriter, resp *http.Response, agentID, providerName, requestedModel, upstreamModel string, start time.Time) {
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
