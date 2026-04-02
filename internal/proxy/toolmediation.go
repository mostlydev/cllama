package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/alert"
	"github.com/mostlydev/cllama/internal/cost"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
	"github.com/mostlydev/cllama/internal/sessionhistory"
)

const (
	defaultManagedToolMaxRounds    = 8
	defaultManagedToolTimeoutMS    = 30000
	defaultManagedToolTotalTimeout = 120000
	maxManagedToolResultBytes      = 16 * 1024
)

var (
	managedToolPathParam = regexp.MustCompile(`\{([A-Za-z0-9_]+)\}`)

	errManagedToolBudget = errors.New("managed tool budget exhausted")
)

const managedToolModeMessage = "This request is in mediated mode. Action required: re-emit only managed service tools for this turn, or respond in text."

type capturedResponse struct {
	StatusCode    int
	Header        http.Header
	Body          []byte
	ProviderName  string
	UpstreamModel string
	RequestBody   []byte
}

type dispatchJSONAttemptResult struct {
	Response               *capturedResponse
	AdvanceToNextCandidate bool
	CandidateSawCooldown   bool
	ClientStatus           int
	ClientMessage          string
	Err                    error
}

type openAIToolCall struct {
	ID           string
	Name         string
	Arguments    map[string]any
	ArgumentsRaw json.RawMessage
	ParseErr     error
}

type managedToolOutcome struct {
	RawJSON []byte
	Trace   sessionhistory.ToolCallTrace
}

type managedToolPolicy struct {
	MaxRounds      int
	PerToolTimeout time.Duration
	TotalTimeout   time.Duration
}

type managedUsageAggregate struct {
	PromptTokens     int
	CompletionTokens int
	ReportedCostUSD  float64
	HasReportedCost  bool
	TotalRounds      int
	LoggedCostUSD    float64
	LoggedCostKnown  bool
	SawCostUsage     bool
}

func (h *Handler) handleManagedOpenAI(w http.ResponseWriter, r *http.Request, agentID string, agentCtx *agentctx.AgentContext, requestedModel string, payload map[string]any, candidates []dispatchCandidate, requestOriginal []byte, start time.Time) {
	policy := resolveManagedToolPolicy(agentCtx)
	loopCtx, cancel := context.WithTimeout(r.Context(), policy.TotalTimeout)
	defer cancel()

	usageAgg := managedUsageAggregate{LoggedCostKnown: true}
	var toolTrace []sessionhistory.ToolRoundTrace
	var requestEffective []byte
	var lastProvider string
	var lastUpstreamModel string

	for {
		resp, status, msg, err := h.dispatchCandidatesJSON(loopCtx, r, agentID, requestedModel, payload, candidates)
		if requestEffective == nil && resp != nil && len(resp.RequestBody) > 0 {
			requestEffective = append([]byte(nil), resp.RequestBody...)
		}
		if resp == nil {
			if len(toolTrace) > 0 {
				h.recordManagedFailure(agentID, lastProvider, requestedModel, lastUpstreamModel, r.URL.Path, requestOriginal, requestEffective, status, jsonErrorPayload(msg), usageAgg, toolTrace)
			}
			h.fail(w, status, msg, agentID, requestedModel, start, err)
			return
		}

		lastProvider = resp.ProviderName
		lastUpstreamModel = resp.UpstreamModel

		usage, _ := cost.ExtractUsage(resp.Body)
		usageAgg.AddRound(agentID, usage, resp.ProviderName, resp.UpstreamModel, h)

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			copyResponseHeaders(w.Header(), resp.Header)
			w.WriteHeader(resp.StatusCode)
			if len(resp.Body) > 0 {
				_, _ = w.Write(resp.Body)
			}
			if len(toolTrace) > 0 {
				h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, resp.StatusCode, ensureJSONPayload(resp.Body), usageAgg, toolTrace)
			}
			if costInfo := usageAgg.costInfo(); costInfo != nil {
				h.logger.LogResponseWithCost(agentID, requestedModel, resp.StatusCode, time.Since(start).Milliseconds(), costInfo)
			} else {
				h.logger.LogResponse(agentID, requestedModel, resp.StatusCode, time.Since(start).Milliseconds())
			}
			return
		}

		assistantMessage, toolCalls, parseErr := parseOpenAIToolResponse(resp.Body)
		if parseErr != nil || len(toolCalls) == 0 {
			copyResponseHeaders(w.Header(), resp.Header)
			w.WriteHeader(resp.StatusCode)
			if len(resp.Body) > 0 {
				_, _ = w.Write(resp.Body)
			}
			h.recordManagedSuccess(agentID, agentCtx, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, resp.StatusCode, resp.Body, usageAgg, toolTrace, time.Since(start).Milliseconds())
			return
		}

		if len(toolTrace) >= policy.MaxRounds {
			msg := fmt.Sprintf("managed tool max rounds exceeded (%d)", policy.MaxRounds)
			h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload(msg), usageAgg, toolTrace)
			h.fail(w, http.StatusBadGateway, msg, agentID, requestedModel, start, fmt.Errorf(msg))
			return
		}

		toolMessages := make([]any, 0, len(toolCalls))
		roundTrace := sessionhistory.ToolRoundTrace{
			Round: len(toolTrace) + 1,
			RoundUsage: sessionhistory.Usage{
				PromptTokens:     usage.PromptTokens,
				CompletionTokens: usage.CompletionTokens,
				ReportedCostUSD:  usage.ReportedCostUSD,
			},
		}
		for _, call := range toolCalls {
			outcome, execErr := h.executeManagedOpenAITool(loopCtx, agentID, agentCtx, call, policy)
			if execErr != nil {
				msg := "managed tool mediation timed out"
				h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload(msg), usageAgg, append(toolTrace, roundTrace))
				h.fail(w, http.StatusBadGateway, msg, agentID, requestedModel, start, execErr)
				return
			}
			roundTrace.ToolCalls = append(roundTrace.ToolCalls, outcome.Trace)
			toolMessages = append(toolMessages, map[string]any{
				"role":         "tool",
				"tool_call_id": call.ID,
				"content":      string(outcome.RawJSON),
			})
		}
		toolTrace = append(toolTrace, roundTrace)
		appendOpenAIAssistantAndToolMessages(payload, assistantMessage, toolMessages)
	}
}

func (h *Handler) dispatchCandidatesJSON(ctx context.Context, r *http.Request, agentID string, requestedModel string, payload map[string]any, candidates []dispatchCandidate) (*capturedResponse, int, string, error) {
	sawCooldown := false
	for i, candidate := range candidates {
		payload["model"] = candidate.UpstreamModel
		outBody, err := json.Marshal(payload)
		if err != nil {
			return nil, http.StatusInternalServerError, "failed to encode upstream body", err
		}
		result := h.dispatchJSONWithRetry(ctx, r, agentID, requestedModel, candidate, outBody)
		if result.Response != nil {
			return result.Response, 0, "", nil
		}
		if !result.AdvanceToNextCandidate {
			return nil, result.ClientStatus, result.ClientMessage, result.Err
		}
		sawCooldown = sawCooldown || result.CandidateSawCooldown
		if i+1 < len(candidates) {
			h.logger.LogIntervention(agentID, requestedModel, "provider_exhausted_failover")
		}
	}

	if sawCooldown {
		err := fmt.Errorf("all declared providers cooling down")
		return nil, http.StatusServiceUnavailable, "all declared provider keys in cooldown", err
	}
	err := fmt.Errorf("exhausted declared model candidates")
	return nil, http.StatusBadGateway, "no usable declared provider key after retries", err
}

func (h *Handler) dispatchJSONWithRetry(ctx context.Context, r *http.Request, agentID string, requestedModel string, candidate dispatchCandidate, outBody []byte) dispatchJSONAttemptResult {
	const maxKeyAttempts = 5
	sawCooldown := false

	for attempt := 0; attempt < maxKeyAttempts; attempt++ {
		prov, lease, err := h.registry.SelectKey(candidate.ProviderName)
		if err != nil {
			if _, ok := err.(*provider.CooldownError); ok {
				return dispatchJSONAttemptResult{
					AdvanceToNextCandidate: true,
					CandidateSawCooldown:   true,
				}
			}
			return dispatchJSONAttemptResult{
				AdvanceToNextCandidate: true,
			}
		}

		targetURL, err := buildUpstreamURL(prov.BaseURL, r.URL.Path, r.URL.RawQuery)
		if err != nil {
			return dispatchJSONAttemptResult{
				ClientStatus:  http.StatusBadGateway,
				ClientMessage: "invalid provider URL",
				Err:           err,
			}
		}

		outReq, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(outBody))
		if err != nil {
			return dispatchJSONAttemptResult{
				ClientStatus:  http.StatusBadGateway,
				ClientMessage: "failed to create upstream request",
				Err:           err,
			}
		}
		copyRequestHeaders(outReq.Header, r.Header)
		outReq.Header.Set("Content-Type", "application/json")
		if err := applyProviderAuth(outReq, prov); err != nil {
			return dispatchJSONAttemptResult{
				ClientStatus:  http.StatusBadGateway,
				ClientMessage: "provider auth not configured",
				Err:           err,
			}
		}

		h.logger.LogRequest(agentID, requestedModel)
		resp, err := h.client.Do(outReq)
		if err != nil {
			return dispatchJSONAttemptResult{
				ClientStatus:  http.StatusBadGateway,
				ClientMessage: "upstream request failed",
				Err:           err,
			}
		}

		classification := classifyResponse(resp)
		switch classification {
		case classAuth:
			resp.Body.Close()
			reason := fmt.Sprintf("http_%d", resp.StatusCode)
			_ = h.registry.MarkDead(lease.ProviderName, lease.KeyID, reason, resp.StatusCode)
			h.logger.LogProviderPool(lease.ProviderName, lease.KeyID, "dead", reason, "")
			if h.notifier != nil {
				h.notifier.Notify(alert.PoolEvent{Provider: lease.ProviderName, KeyID: lease.KeyID, Action: "dead", Reason: reason})
			}
			_ = h.registry.SaveToFile()
			continue
		case classRateLimit:
			resp.Body.Close()
			cooldownDur := parseCooldownDuration(resp)
			until := time.Now().UTC().Add(cooldownDur)
			_ = h.registry.MarkCooldown(lease.ProviderName, lease.KeyID, "rate_limit", until)
			cooldownUntil := until.Format(time.RFC3339)
			sawCooldown = true
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
			continue
		default:
			body, readErr := io.ReadAll(resp.Body)
			resp.Body.Close()
			if readErr != nil {
				return dispatchJSONAttemptResult{
					ClientStatus:  http.StatusBadGateway,
					ClientMessage: "failed to read upstream response",
					Err:           readErr,
				}
			}
			return dispatchJSONAttemptResult{
				Response: &capturedResponse{
					StatusCode:    resp.StatusCode,
					Header:        resp.Header.Clone(),
					Body:          body,
					ProviderName:  candidate.ProviderName,
					UpstreamModel: candidate.UpstreamModel,
					RequestBody:   append([]byte(nil), outBody...),
				},
			}
		}
	}

	return dispatchJSONAttemptResult{
		AdvanceToNextCandidate: true,
		CandidateSawCooldown:   sawCooldown,
	}
}

func resolveManagedToolPolicy(agentCtx *agentctx.AgentContext) managedToolPolicy {
	maxRounds := defaultManagedToolMaxRounds
	perTool := time.Duration(defaultManagedToolTimeoutMS) * time.Millisecond
	total := time.Duration(defaultManagedToolTotalTimeout) * time.Millisecond
	if agentCtx != nil && agentCtx.Tools != nil {
		if agentCtx.Tools.Policy.MaxRounds > 0 {
			maxRounds = agentCtx.Tools.Policy.MaxRounds
		}
		if agentCtx.Tools.Policy.TimeoutPerToolMS > 0 {
			perTool = time.Duration(agentCtx.Tools.Policy.TimeoutPerToolMS) * time.Millisecond
		}
		if agentCtx.Tools.Policy.TotalTimeoutMS > 0 {
			total = time.Duration(agentCtx.Tools.Policy.TotalTimeoutMS) * time.Millisecond
		}
	}
	return managedToolPolicy{
		MaxRounds:      maxRounds,
		PerToolTimeout: perTool,
		TotalTimeout:   total,
	}
}

func (h *Handler) executeManagedOpenAITool(ctx context.Context, agentID string, agentCtx *agentctx.AgentContext, call openAIToolCall, policy managedToolPolicy) (managedToolOutcome, error) {
	trace := sessionhistory.ToolCallTrace{
		Name:      call.Name,
		Arguments: call.ArgumentsRaw,
	}
	tool, ok := lookupManagedTool(agentCtx, call.Name)
	if !ok {
		raw := toolErrorPayload("unsupported_tool", managedToolModeMessage, 0, nil)
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}
	trace.Service = tool.Execution.Service
	if call.ParseErr != nil {
		raw := toolErrorPayload("invalid_arguments", fmt.Sprintf("Tool arguments for %s are invalid JSON: %v", call.Name, call.ParseErr), 0, nil)
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}

	start := time.Now()
	childCtx, cancel := context.WithTimeout(ctx, policy.PerToolTimeout)
	defer cancel()
	raw, statusCode, err := h.callManagedHTTPTool(childCtx, agentID, tool, call.Arguments)
	trace.LatencyMS = time.Since(start).Milliseconds()
	trace.StatusCode = statusCode
	if errors.Is(err, errManagedToolBudget) {
		return managedToolOutcome{}, err
	}
	if err != nil {
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}
	trace.Result = raw
	return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
}

func (h *Handler) callManagedHTTPTool(ctx context.Context, agentID string, tool agentctx.ToolManifestEntry, args map[string]any) ([]byte, int, error) {
	if !strings.EqualFold(tool.Execution.Transport, "http") {
		return toolErrorPayload("unsupported_transport", fmt.Sprintf("Managed tool transport %q is unsupported", tool.Execution.Transport), 0, nil), 0, nil
	}

	renderedPath, remaining, err := renderManagedToolPath(tool.Execution.Path, agentID, args)
	if err != nil {
		return toolErrorPayload("invalid_arguments", err.Error(), 0, nil), 0, nil
	}
	targetURL, body, err := buildManagedToolRequest(tool.Execution.BaseURL, strings.ToUpper(tool.Execution.Method), renderedPath, remaining)
	if err != nil {
		return toolErrorPayload("request_build_failed", err.Error(), 0, nil), 0, nil
	}
	req, err := http.NewRequestWithContext(ctx, strings.ToUpper(tool.Execution.Method), targetURL, body)
	if err != nil {
		return toolErrorPayload("request_build_failed", err.Error(), 0, nil), 0, nil
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if err := applyManagedToolAuth(req, tool.Execution.Auth); err != nil {
		return toolErrorPayload("unsupported_auth", err.Error(), 0, nil), 0, nil
	}

	resp, err := h.client.Do(req)
	if err != nil {
		if errors.Is(ctx.Err(), context.DeadlineExceeded) && parentContextTimedOut(ctx) {
			return nil, 0, errManagedToolBudget
		}
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			return toolErrorPayload("timeout", fmt.Sprintf("Service did not respond within %s", policyDurationText(ctx)), http.StatusGatewayTimeout, nil), http.StatusGatewayTimeout, nil
		}
		return toolErrorPayload("request_failed", err.Error(), 0, nil), 0, nil
	}
	defer resp.Body.Close()

	respBody, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return toolErrorPayload("read_failed", readErr.Error(), resp.StatusCode, nil), resp.StatusCode, nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		details := decodeManagedToolBody(respBody, resp.Header.Get("Content-Type"))
		return toolErrorPayload(fmt.Sprintf("http_%d", resp.StatusCode), fmt.Sprintf("Service returned HTTP %d", resp.StatusCode), resp.StatusCode, details), resp.StatusCode, nil
	}
	return toolSuccessPayload(resp.StatusCode, decodeManagedToolBody(respBody, resp.Header.Get("Content-Type"))), resp.StatusCode, nil
}

func renderManagedToolPath(path string, agentID string, args map[string]any) (string, map[string]any, error) {
	remaining := copyArgumentMap(args)
	var replaceErr error
	rendered := managedToolPathParam.ReplaceAllStringFunc(path, func(match string) string {
		if replaceErr != nil {
			return match
		}
		parts := managedToolPathParam.FindStringSubmatch(match)
		if len(parts) != 2 {
			return match
		}
		name := parts[1]
		var value any
		if name == "claw_id" {
			value = agentID
		} else {
			var ok bool
			value, ok = remaining[name]
			if !ok {
				replaceErr = fmt.Errorf("missing required argument %q", name)
				return match
			}
			delete(remaining, name)
		}
		text, err := scalarArgumentString(value)
		if err != nil {
			replaceErr = fmt.Errorf("argument %q cannot be rendered in path: %w", name, err)
			return match
		}
		return url.PathEscape(text)
	})
	if replaceErr != nil {
		return "", nil, replaceErr
	}
	return rendered, remaining, nil
}

func buildManagedToolRequest(baseURL string, method string, path string, args map[string]any) (string, io.Reader, error) {
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return "", nil, err
	}
	if u.Scheme == "" || u.Host == "" {
		return "", nil, fmt.Errorf("invalid tool base URL %q", baseURL)
	}
	u.Path = strings.TrimRight(u.Path, "/") + path
	if method == http.MethodGet || method == http.MethodDelete {
		q := u.Query()
		for key, value := range args {
			q.Set(key, queryArgumentValue(value))
		}
		u.RawQuery = q.Encode()
		return u.String(), nil, nil
	}
	if len(args) == 0 {
		args = map[string]any{}
	}
	body, err := json.Marshal(args)
	if err != nil {
		return "", nil, err
	}
	return u.String(), bytes.NewReader(body), nil
}

func parseOpenAIToolResponse(body []byte) (map[string]any, []openAIToolCall, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, nil, err
	}
	choices, _ := payload["choices"].([]any)
	if len(choices) == 0 {
		return nil, nil, fmt.Errorf("openai response missing choices")
	}
	choice, _ := choices[0].(map[string]any)
	if choice == nil {
		return nil, nil, fmt.Errorf("openai choice is not an object")
	}
	message, _ := choice["message"].(map[string]any)
	if message == nil {
		return nil, nil, fmt.Errorf("openai choice missing message")
	}

	rawCalls, _ := message["tool_calls"].([]any)
	if len(rawCalls) == 0 {
		return message, nil, nil
	}
	calls := make([]openAIToolCall, 0, len(rawCalls))
	for i, raw := range rawCalls {
		callMap, _ := raw.(map[string]any)
		if callMap == nil {
			continue
		}
		callID, _ := callMap["id"].(string)
		if strings.TrimSpace(callID) == "" {
			callID = fmt.Sprintf("call_%d", i+1)
		}
		fn, _ := callMap["function"].(map[string]any)
		name, _ := fn["name"].(string)
		argText, _ := fn["arguments"].(string)
		argText = strings.TrimSpace(argText)
		var (
			args     map[string]any
			parseErr error
			rawArgs  json.RawMessage
		)
		switch {
		case argText == "":
			args = map[string]any{}
			rawArgs = json.RawMessage([]byte(`{}`))
		case json.Valid([]byte(argText)):
			rawArgs = json.RawMessage([]byte(argText))
			if err := json.Unmarshal([]byte(argText), &args); err != nil {
				parseErr = err
			}
			if args == nil {
				parseErr = fmt.Errorf("tool arguments must decode to an object")
			}
		default:
			parseErr = fmt.Errorf("arguments are not valid JSON")
			rawArgs, _ = json.Marshal(argText)
		}
		calls = append(calls, openAIToolCall{
			ID:           callID,
			Name:         strings.TrimSpace(name),
			Arguments:    args,
			ArgumentsRaw: rawArgs,
			ParseErr:     parseErr,
		})
	}
	return message, calls, nil
}

func appendOpenAIAssistantAndToolMessages(payload map[string]any, assistantMessage map[string]any, toolMessages []any) {
	messages, _ := payload["messages"].([]any)
	messages = append(messages, assistantMessage)
	messages = append(messages, toolMessages...)
	payload["messages"] = messages
}

func lookupManagedTool(agentCtx *agentctx.AgentContext, name string) (agentctx.ToolManifestEntry, bool) {
	if agentCtx == nil || agentCtx.Tools == nil {
		return agentctx.ToolManifestEntry{}, false
	}
	for _, tool := range agentCtx.Tools.Tools {
		if tool.Name == name {
			return tool, true
		}
	}
	return agentctx.ToolManifestEntry{}, false
}

func applyManagedToolAuth(req *http.Request, auth *agentctx.AuthEntry) error {
	if auth == nil || strings.TrimSpace(auth.Type) == "" || strings.EqualFold(auth.Type, "none") {
		return nil
	}
	if !strings.EqualFold(auth.Type, "bearer") {
		return fmt.Errorf("unsupported tool auth type %q", auth.Type)
	}
	if strings.TrimSpace(auth.Token) == "" {
		return fmt.Errorf("managed tool bearer token is empty")
	}
	req.Header.Set("Authorization", "Bearer "+auth.Token)
	return nil
}

func copyArgumentMap(args map[string]any) map[string]any {
	if len(args) == 0 {
		return map[string]any{}
	}
	out := make(map[string]any, len(args))
	for k, v := range args {
		out[k] = v
	}
	return out
}

func scalarArgumentString(v any) (string, error) {
	switch typed := v.(type) {
	case string:
		return typed, nil
	case bool:
		if typed {
			return "true", nil
		}
		return "false", nil
	case float64, float32, int, int64, int32, int16, int8, uint, uint64, uint32, uint16, uint8:
		return fmt.Sprint(typed), nil
	case json.Number:
		return typed.String(), nil
	default:
		return "", fmt.Errorf("expected scalar, got %T", v)
	}
}

func queryArgumentValue(v any) string {
	text, err := scalarArgumentString(v)
	if err == nil {
		return text
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprint(v)
	}
	return string(raw)
}

func decodeManagedToolBody(body []byte, contentType string) any {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return nil
	}
	if len(trimmed) > maxManagedToolResultBytes {
		return map[string]any{
			"data":           string(trimmed[:maxManagedToolResultBytes]),
			"truncated":      true,
			"original_bytes": len(trimmed),
		}
	}
	if strings.Contains(strings.ToLower(contentType), "application/json") || json.Valid(trimmed) {
		var parsed any
		if err := json.Unmarshal(trimmed, &parsed); err == nil {
			return parsed
		}
	}
	return string(trimmed)
}

func toolSuccessPayload(statusCode int, data any) []byte {
	payload := map[string]any{
		"ok":          true,
		"status_code": statusCode,
	}
	switch typed := data.(type) {
	case map[string]any:
		if truncated, _ := typed["truncated"].(bool); truncated {
			payload["data"] = typed["data"]
			payload["truncated"] = true
			payload["original_bytes"] = typed["original_bytes"]
		} else if len(typed) > 0 {
			payload["data"] = typed
		}
	case nil:
	default:
		payload["data"] = typed
	}
	raw, _ := json.Marshal(payload)
	return raw
}

func toolErrorPayload(code, message string, statusCode int, details any) []byte {
	errPayload := map[string]any{
		"code":    code,
		"message": message,
	}
	switch typed := details.(type) {
	case map[string]any:
		if truncated, _ := typed["truncated"].(bool); truncated {
			errPayload["details"] = typed["data"]
		} else if len(typed) > 0 {
			errPayload["details"] = typed
		}
	case nil:
	default:
		errPayload["details"] = typed
	}
	payload := map[string]any{
		"ok":    false,
		"error": errPayload,
	}
	if statusCode > 0 {
		payload["status_code"] = statusCode
	}
	if typed, ok := details.(map[string]any); ok {
		if truncated, _ := typed["truncated"].(bool); truncated {
			payload["truncated"] = true
			payload["original_bytes"] = typed["original_bytes"]
		}
	}
	raw, _ := json.Marshal(payload)
	return raw
}

func ensureJSONPayload(body []byte) []byte {
	trimmed := bytes.TrimSpace(body)
	if len(trimmed) == 0 {
		return jsonErrorPayload("empty response body")
	}
	if json.Valid(trimmed) {
		return append([]byte(nil), trimmed...)
	}
	wrapped, _ := json.Marshal(map[string]any{
		"raw": string(trimmed),
	})
	return wrapped
}

func jsonErrorPayload(msg string) []byte {
	raw, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": msg,
		},
	})
	return raw
}

func parentContextTimedOut(ctx context.Context) bool {
	return ctx.Err() != nil && errors.Is(ctx.Err(), context.DeadlineExceeded)
}

func policyDurationText(ctx context.Context) string {
	deadline, ok := ctx.Deadline()
	if !ok {
		return "configured timeout"
	}
	remaining := time.Until(deadline)
	if remaining < time.Second {
		return "configured timeout"
	}
	return remaining.Round(time.Second).String()
}

func (a *managedUsageAggregate) AddRound(agentID string, usage cost.Usage, providerName, upstreamModel string, h *Handler) {
	a.TotalRounds++
	a.PromptTokens += usage.PromptTokens
	a.CompletionTokens += usage.CompletionTokens
	if usage.ReportedCostUSD != nil {
		a.HasReportedCost = true
		a.ReportedCostUSD += *usage.ReportedCostUSD
	}
	if usage.PromptTokens > 0 || usage.CompletionTokens > 0 || usage.ReportedCostUSD != nil {
		a.SawCostUsage = true
		costUSD, costKnown := h.resolveCost(providerName, upstreamModel, usage)
		if h.accumulator != nil {
			h.accumulator.RecordWithStatus(agentID, providerName, upstreamModel, usage.PromptTokens, usage.CompletionTokens, costUSD, costKnown)
		}
		if costKnown {
			a.LoggedCostUSD += costUSD
		} else {
			a.LoggedCostKnown = false
		}
	}
}

func (a managedUsageAggregate) sessionUsage() sessionhistory.Usage {
	usage := sessionhistory.Usage{
		PromptTokens:     a.PromptTokens,
		CompletionTokens: a.CompletionTokens,
		TotalRounds:      a.TotalRounds,
	}
	if a.HasReportedCost {
		reported := a.ReportedCostUSD
		usage.ReportedCostUSD = &reported
	}
	return usage
}

func (a managedUsageAggregate) costInfo() *logging.CostInfo {
	ci := &logging.CostInfo{
		InputTokens:  a.PromptTokens,
		OutputTokens: a.CompletionTokens,
	}
	if a.SawCostUsage && a.LoggedCostKnown {
		total := a.LoggedCostUSD
		ci.CostUSD = &total
	}
	if ci.InputTokens == 0 && ci.OutputTokens == 0 && ci.CostUSD == nil {
		return nil
	}
	return ci
}

func (h *Handler) recordManagedSuccess(agentID string, agentCtx *agentctx.AgentContext, providerName, requestedModel, upstreamModel string, requestPath string, requestOriginal []byte, requestEffective []byte, statusCode int, captured []byte, usage managedUsageAggregate, toolTrace []sessionhistory.ToolRoundTrace, latencyMS int64) {
	if statusCode < 200 || statusCode >= 300 {
		return
	}
	entry := sessionhistory.Entry{
		Version:           1,
		Status:            "ok",
		ClawID:            agentID,
		TS:                time.Now().UTC().Format(time.RFC3339),
		Path:              requestPath,
		RequestedModel:    requestedModel,
		EffectiveProvider: providerName,
		EffectiveModel:    upstreamModel,
		StatusCode:        statusCode,
		Stream:            false,
		RequestOriginal:   json.RawMessage(requestOriginal),
		RequestEffective:  json.RawMessage(requestEffective),
		Response: sessionhistory.Payload{
			Format: "json",
			JSON:   json.RawMessage(captured),
		},
		Usage:     usage.sessionUsage(),
		ToolTrace: toolTrace,
	}
	if err := entry.EnsureID(); err != nil {
		h.logger.LogError(agentID, requestedModel, 0, 0, fmt.Errorf("session history id: %w", err))
		return
	}
	if h.sessionRecorder != nil {
		if err := h.sessionRecorder.Record(agentID, entry); err != nil {
			h.logger.LogError(agentID, requestedModel, 0, 0, fmt.Errorf("session history write: %w", err))
		}
	}
	h.retainMemory(agentID, agentCtx, entry)

	if costInfo := usage.costInfo(); costInfo != nil {
		h.logger.LogResponseWithCost(agentID, requestedModel, statusCode, latencyMS, costInfo)
		return
	}
	h.logger.LogResponse(agentID, requestedModel, statusCode, latencyMS)
}

func (h *Handler) recordManagedFailure(agentID string, providerName, requestedModel, upstreamModel string, requestPath string, requestOriginal []byte, requestEffective []byte, statusCode int, responseBody []byte, usage managedUsageAggregate, toolTrace []sessionhistory.ToolRoundTrace) {
	if h.sessionRecorder == nil {
		return
	}
	entry := sessionhistory.Entry{
		Version:           1,
		Status:            "error",
		ClawID:            agentID,
		TS:                time.Now().UTC().Format(time.RFC3339),
		Path:              requestPath,
		RequestedModel:    requestedModel,
		EffectiveProvider: providerName,
		EffectiveModel:    upstreamModel,
		StatusCode:        statusCode,
		Stream:            false,
		RequestOriginal:   json.RawMessage(requestOriginal),
		RequestEffective:  json.RawMessage(requestEffective),
		Response: sessionhistory.Payload{
			Format: "json",
			JSON:   json.RawMessage(ensureJSONPayload(responseBody)),
		},
		Usage:     usage.sessionUsage(),
		ToolTrace: toolTrace,
	}
	if err := entry.EnsureID(); err != nil {
		h.logger.LogError(agentID, requestedModel, 0, 0, fmt.Errorf("session history id: %w", err))
		return
	}
	if err := h.sessionRecorder.Record(agentID, entry); err != nil {
		h.logger.LogError(agentID, requestedModel, 0, 0, fmt.Errorf("session history write: %w", err))
	}
}
