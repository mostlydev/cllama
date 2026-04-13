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
	managedKeepaliveInterval       = 250 * time.Millisecond
	maxManagedToolResultBytes      = 16 * 1024
	maxManagedLLMResponseBytes     = 4 * 1024 * 1024
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

type anthropicToolUse struct {
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
	TotalTokens      int
	ReportedCostUSD  float64
	HasReportedCost  bool
	TotalRounds      int
	LoggedCostUSD    float64
	LoggedCostKnown  bool
	SawCostUsage     bool
}

type managedDispatchResult struct {
	Response *capturedResponse
	Status   int
	Message  string
	Err      error
}

type managedOpenAIToolExecResult struct {
	Outcome managedToolOutcome
	Err     error
}

type managedAnthropicToolExecResult struct {
	Outcome managedToolOutcome
	Err     error
}

type managedStreamKeepalive struct {
	w       http.ResponseWriter
	started bool
}

type limitedReadResult struct {
	Body          []byte
	Truncated     bool
	ObservedBytes int
}

func (h *Handler) handleManagedOpenAI(w http.ResponseWriter, r *http.Request, agentID string, agentCtx *agentctx.AgentContext, requestedModel string, payload map[string]any, candidates []dispatchCandidate, requestOriginal []byte, downstreamStream bool, downstreamIncludeUsage bool, start time.Time) {
	policy := resolveManagedToolPolicy(agentCtx)
	loopCtx, cancel := context.WithTimeout(r.Context(), policy.TotalTimeout)
	defer cancel()
	streamKeepalive := newManagedStreamKeepalive(w, downstreamStream)

	usageAgg := managedUsageAggregate{LoggedCostKnown: true}
	var toolTrace []sessionhistory.ToolRoundTrace
	var hiddenMessages []json.RawMessage
	var requestEffective []byte
	var lastProvider string
	var lastUpstreamModel string

	for {
		dispatchResult := waitWithManagedKeepalive(streamKeepalive, managedModelWaitComment(len(toolTrace)+1), func() managedDispatchResult {
			resp, status, msg, err := h.dispatchCandidatesJSON(loopCtx, r, agentID, requestedModel, payload, candidates)
			return managedDispatchResult{
				Response: resp,
				Status:   status,
				Message:  msg,
				Err:      err,
			}
		})
		resp, status, msg, err := dispatchResult.Response, dispatchResult.Status, dispatchResult.Message, dispatchResult.Err
		if requestEffective == nil && resp != nil && len(resp.RequestBody) > 0 {
			requestEffective = append([]byte(nil), resp.RequestBody...)
		}
		if resp == nil {
			if len(toolTrace) > 0 {
				h.recordManagedFailure(agentID, lastProvider, requestedModel, lastUpstreamModel, r.URL.Path, requestOriginal, requestEffective, status, jsonErrorPayload(msg), usageAgg, toolTrace)
			}
			if streamKeepalive != nil && streamKeepalive.started {
				streamKeepalive.writeOpenAIError(jsonErrorPayload(msg))
				h.logger.LogError(agentID, requestedModel, status, time.Since(start).Milliseconds(), err)
				return
			}
			h.fail(w, status, msg, agentID, requestedModel, start, err)
			return
		}

		lastProvider = resp.ProviderName
		lastUpstreamModel = resp.UpstreamModel

		usage, _ := cost.ExtractUsage(resp.Body)
		usageAgg.AddRound(agentID, usage, resp.ProviderName, resp.UpstreamModel, h)

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			if streamKeepalive != nil && streamKeepalive.started {
				streamKeepalive.writeOpenAIError(resp.Body)
			} else {
				copyResponseHeaders(w.Header(), resp.Header)
				w.WriteHeader(resp.StatusCode)
				if len(resp.Body) > 0 {
					_, _ = w.Write(resp.Body)
				}
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
			responseBytes := resp.Body
			if downstreamStream {
				sse, synthErr := synthesizeOpenAIStream(resp.Body, resp.UpstreamModel, usageAgg, downstreamIncludeUsage)
				if synthErr != nil {
					h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload("failed to synthesize managed stream"), usageAgg, toolTrace)
					if streamKeepalive != nil && streamKeepalive.started {
						streamKeepalive.writeOpenAIError(jsonErrorPayload("failed to synthesize managed stream"))
						h.logger.LogError(agentID, requestedModel, http.StatusBadGateway, time.Since(start).Milliseconds(), synthErr)
						return
					}
					h.fail(w, http.StatusBadGateway, "failed to synthesize managed stream", agentID, requestedModel, start, synthErr)
					return
				}
				streamKeepalive.writeFinal(sse)
				responseBytes = sse
			} else {
				copyResponseHeaders(w.Header(), resp.Header)
				w.WriteHeader(resp.StatusCode)
				if len(resp.Body) > 0 {
					_, _ = w.Write(resp.Body)
				}
			}
			h.managedTurns.ObserveTerminalAssistant(agentID, assistantMessage, hiddenMessages)
			h.recordManagedSuccess(agentID, agentCtx, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, resp.StatusCode, responseBytes, usageAgg, toolTrace, downstreamStream, time.Since(start).Milliseconds())
			return
		}

		managedCalls, nativeCalls := partitionManagedOpenAIToolCalls(agentCtx, toolCalls)
		if len(managedCalls) == 0 {
			if len(toolTrace) > 0 || len(hiddenMessages) > 0 {
				msg := "runner-native tool calls after managed tool rounds are not supported in one request"
				h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload(msg), usageAgg, toolTrace)
				if streamKeepalive != nil && downstreamStream {
					streamKeepalive.writeOpenAIError(jsonErrorPayload(msg))
					h.logger.LogError(agentID, requestedModel, http.StatusBadGateway, time.Since(start).Milliseconds(), fmt.Errorf(msg))
					return
				}
				h.fail(w, http.StatusBadGateway, msg, agentID, requestedModel, start, fmt.Errorf(msg))
				return
			}

			responseBytes := resp.Body
			responseHeader := resp.Header.Clone()
			if downstreamStream {
				sse, synthErr := synthesizeOpenAIToolCallStream(resp.Body, resp.UpstreamModel, usageAgg, downstreamIncludeUsage)
				if synthErr != nil {
					h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload("failed to synthesize managed stream"), usageAgg, toolTrace)
					if streamKeepalive != nil {
						streamKeepalive.writeOpenAIError(jsonErrorPayload("failed to synthesize managed stream"))
						h.logger.LogError(agentID, requestedModel, http.StatusBadGateway, time.Since(start).Milliseconds(), synthErr)
						return
					}
					h.fail(w, http.StatusBadGateway, "failed to synthesize managed stream", agentID, requestedModel, start, synthErr)
					return
				}
				streamKeepalive.writeFinal(sse)
				responseBytes = sse
				responseHeader = syntheticSSEHeader()
			} else {
				copyResponseHeaders(w.Header(), resp.Header)
				w.WriteHeader(resp.StatusCode)
				if len(resp.Body) > 0 {
					_, _ = w.Write(resp.Body)
				}
			}
			h.recordResponse(agentID, agentCtx, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, resp.StatusCode, responseHeader, responseBytes, start)
			return
		}

		if len(nativeCalls) > 0 {
			msg := "mixed managed and runner-native tool calls are not supported in one model response"
			h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload(msg), usageAgg, toolTrace)
			if streamKeepalive != nil && downstreamStream {
				streamKeepalive.writeOpenAIError(jsonErrorPayload(msg))
				h.logger.LogError(agentID, requestedModel, http.StatusBadGateway, time.Since(start).Milliseconds(), fmt.Errorf(msg))
				return
			}
			h.fail(w, http.StatusBadGateway, msg, agentID, requestedModel, start, fmt.Errorf(msg))
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
		for _, call := range managedCalls {
			execResult := waitWithManagedKeepalive(streamKeepalive, managedToolWaitComment(len(toolTrace)+1, managedToolDisplayName(agentCtx, call.Name)), func() managedOpenAIToolExecResult {
				outcome, execErr := h.executeManagedOpenAITool(loopCtx, agentID, agentCtx, call, policy)
				return managedOpenAIToolExecResult{Outcome: outcome, Err: execErr}
			})
			outcome, execErr := execResult.Outcome, execResult.Err
			if execErr != nil {
				msg := "managed tool mediation timed out"
				h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload(msg), usageAgg, append(toolTrace, roundTrace))
				if streamKeepalive != nil && streamKeepalive.started {
					streamKeepalive.writeOpenAIError(jsonErrorPayload(msg))
					h.logger.LogError(agentID, requestedModel, http.StatusBadGateway, time.Since(start).Milliseconds(), execErr)
					return
				}
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
		hiddenMessages = appendManagedOpenAIContinuityMessages(hiddenMessages, assistantMessage, toolMessages)
	}
}

func (h *Handler) handleManagedAnthropic(w http.ResponseWriter, r *http.Request, agentID string, agentCtx *agentctx.AgentContext, requestedModel string, payload map[string]any, candidates []dispatchCandidate, requestOriginal []byte, downstreamStream bool, start time.Time) {
	policy := resolveManagedToolPolicy(agentCtx)
	loopCtx, cancel := context.WithTimeout(r.Context(), policy.TotalTimeout)
	defer cancel()
	streamKeepalive := newManagedStreamKeepalive(w, downstreamStream)

	usageAgg := managedUsageAggregate{LoggedCostKnown: true}
	var toolTrace []sessionhistory.ToolRoundTrace
	var hiddenMessages []json.RawMessage
	var requestEffective []byte
	var lastProvider string
	var lastUpstreamModel string

	for {
		dispatchResult := waitWithManagedKeepalive(streamKeepalive, managedModelWaitComment(len(toolTrace)+1), func() managedDispatchResult {
			resp, status, msg, err := h.dispatchCandidatesJSON(loopCtx, r, agentID, requestedModel, payload, candidates)
			return managedDispatchResult{
				Response: resp,
				Status:   status,
				Message:  msg,
				Err:      err,
			}
		})
		resp, status, msg, err := dispatchResult.Response, dispatchResult.Status, dispatchResult.Message, dispatchResult.Err
		if requestEffective == nil && resp != nil && len(resp.RequestBody) > 0 {
			requestEffective = append([]byte(nil), resp.RequestBody...)
		}
		if resp == nil {
			if len(toolTrace) > 0 {
				h.recordManagedFailure(agentID, lastProvider, requestedModel, lastUpstreamModel, r.URL.Path, requestOriginal, requestEffective, status, jsonErrorPayload(msg), usageAgg, toolTrace)
			}
			if streamKeepalive != nil && streamKeepalive.started {
				streamKeepalive.writeAnthropicError(msg)
				h.logger.LogError(agentID, requestedModel, status, time.Since(start).Milliseconds(), err)
				return
			}
			h.fail(w, status, msg, agentID, requestedModel, start, err)
			return
		}

		lastProvider = resp.ProviderName
		lastUpstreamModel = resp.UpstreamModel

		usage, _ := cost.ExtractUsage(resp.Body)
		usageAgg.AddRound(agentID, usage, resp.ProviderName, resp.UpstreamModel, h)

		if resp.StatusCode < 200 || resp.StatusCode >= 300 {
			if streamKeepalive != nil && streamKeepalive.started {
				streamKeepalive.writeAnthropicError(extractAnthropicErrorMessage(resp.Body))
			} else {
				copyResponseHeaders(w.Header(), resp.Header)
				w.WriteHeader(resp.StatusCode)
				if len(resp.Body) > 0 {
					_, _ = w.Write(resp.Body)
				}
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

		assistantMessage, toolUses, parseErr := parseAnthropicToolResponse(resp.Body)
		if parseErr != nil || len(toolUses) == 0 {
			responseBytes := resp.Body
			if downstreamStream {
				sse, synthErr := synthesizeAnthropicStream(resp.Body, resp.UpstreamModel, usageAgg)
				if synthErr != nil {
					h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload("failed to synthesize managed stream"), usageAgg, toolTrace)
					if streamKeepalive != nil && streamKeepalive.started {
						streamKeepalive.writeAnthropicError("failed to synthesize managed stream")
						h.logger.LogError(agentID, requestedModel, http.StatusBadGateway, time.Since(start).Milliseconds(), synthErr)
						return
					}
					h.fail(w, http.StatusBadGateway, "failed to synthesize managed stream", agentID, requestedModel, start, synthErr)
					return
				}
				streamKeepalive.writeFinal(sse)
				responseBytes = sse
			} else {
				copyResponseHeaders(w.Header(), resp.Header)
				w.WriteHeader(resp.StatusCode)
				if len(resp.Body) > 0 {
					_, _ = w.Write(resp.Body)
				}
			}
			h.managedAnthropicTurns.ObserveTerminalAssistant(agentID, assistantMessage, hiddenMessages)
			h.recordManagedSuccess(agentID, agentCtx, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, resp.StatusCode, responseBytes, usageAgg, toolTrace, downstreamStream, time.Since(start).Milliseconds())
			return
		}

		managedToolUses, nativeToolUses := partitionManagedAnthropicToolUses(agentCtx, toolUses)
		if len(managedToolUses) == 0 {
			if len(toolTrace) > 0 || len(hiddenMessages) > 0 {
				msg := "runner-native tool calls after managed tool rounds are not supported in one request"
				h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload(msg), usageAgg, toolTrace)
				if streamKeepalive != nil && downstreamStream {
					streamKeepalive.writeAnthropicError(msg)
					h.logger.LogError(agentID, requestedModel, http.StatusBadGateway, time.Since(start).Milliseconds(), fmt.Errorf(msg))
					return
				}
				h.fail(w, http.StatusBadGateway, msg, agentID, requestedModel, start, fmt.Errorf(msg))
				return
			}

			responseBytes := resp.Body
			responseHeader := resp.Header.Clone()
			if downstreamStream {
				sse, synthErr := synthesizeAnthropicToolUseStream(resp.Body, resp.UpstreamModel, usageAgg)
				if synthErr != nil {
					h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload("failed to synthesize managed stream"), usageAgg, toolTrace)
					if streamKeepalive != nil {
						streamKeepalive.writeAnthropicError("failed to synthesize managed stream")
						h.logger.LogError(agentID, requestedModel, http.StatusBadGateway, time.Since(start).Milliseconds(), synthErr)
						return
					}
					h.fail(w, http.StatusBadGateway, "failed to synthesize managed stream", agentID, requestedModel, start, synthErr)
					return
				}
				streamKeepalive.writeFinal(sse)
				responseBytes = sse
				responseHeader = syntheticSSEHeader()
			} else {
				copyResponseHeaders(w.Header(), resp.Header)
				w.WriteHeader(resp.StatusCode)
				if len(resp.Body) > 0 {
					_, _ = w.Write(resp.Body)
				}
			}
			h.recordResponse(agentID, agentCtx, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, resp.StatusCode, responseHeader, responseBytes, start)
			return
		}

		if len(nativeToolUses) > 0 {
			msg := "mixed managed and runner-native tool calls are not supported in one model response"
			h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload(msg), usageAgg, toolTrace)
			if streamKeepalive != nil && downstreamStream {
				streamKeepalive.writeAnthropicError(msg)
				h.logger.LogError(agentID, requestedModel, http.StatusBadGateway, time.Since(start).Milliseconds(), fmt.Errorf(msg))
				return
			}
			h.fail(w, http.StatusBadGateway, msg, agentID, requestedModel, start, fmt.Errorf(msg))
			return
		}

		if len(toolTrace) >= policy.MaxRounds {
			msg := fmt.Sprintf("managed tool max rounds exceeded (%d)", policy.MaxRounds)
			h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload(msg), usageAgg, toolTrace)
			h.fail(w, http.StatusBadGateway, msg, agentID, requestedModel, start, fmt.Errorf(msg))
			return
		}

		toolResults := make([]map[string]any, 0, len(toolUses))
		roundTrace := sessionhistory.ToolRoundTrace{
			Round: len(toolTrace) + 1,
			RoundUsage: sessionhistory.Usage{
				PromptTokens:     usage.PromptTokens,
				CompletionTokens: usage.CompletionTokens,
				ReportedCostUSD:  usage.ReportedCostUSD,
			},
		}
		for _, call := range managedToolUses {
			execResult := waitWithManagedKeepalive(streamKeepalive, managedToolWaitComment(len(toolTrace)+1, managedToolDisplayName(agentCtx, call.Name)), func() managedAnthropicToolExecResult {
				outcome, execErr := h.executeManagedAnthropicTool(loopCtx, agentID, agentCtx, call, policy)
				return managedAnthropicToolExecResult{Outcome: outcome, Err: execErr}
			})
			outcome, execErr := execResult.Outcome, execResult.Err
			if execErr != nil {
				msg := "managed tool mediation timed out"
				h.recordManagedFailure(agentID, resp.ProviderName, requestedModel, resp.UpstreamModel, r.URL.Path, requestOriginal, requestEffective, http.StatusBadGateway, jsonErrorPayload(msg), usageAgg, append(toolTrace, roundTrace))
				if streamKeepalive != nil && streamKeepalive.started {
					streamKeepalive.writeAnthropicError(msg)
					h.logger.LogError(agentID, requestedModel, http.StatusBadGateway, time.Since(start).Milliseconds(), execErr)
					return
				}
				h.fail(w, http.StatusBadGateway, msg, agentID, requestedModel, start, execErr)
				return
			}
			roundTrace.ToolCalls = append(roundTrace.ToolCalls, outcome.Trace)
			toolResults = append(toolResults, anthropicToolResultBlock(call.ID, outcome.RawJSON))
		}
		toolTrace = append(toolTrace, roundTrace)
		toolResultMessage := appendAnthropicAssistantAndToolResultMessages(payload, assistantMessage, toolResults)
		hiddenMessages = appendManagedAnthropicContinuityMessages(hiddenMessages, assistantMessage, toolResultMessage)
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
			limited, readErr := readBodyLimited(resp.Body, maxManagedLLMResponseBytes)
			resp.Body.Close()
			if readErr != nil {
				return dispatchJSONAttemptResult{
					ClientStatus:  http.StatusBadGateway,
					ClientMessage: "failed to read upstream response",
					Err:           readErr,
				}
			}
			if limited.Truncated {
				return dispatchJSONAttemptResult{
					ClientStatus:  http.StatusBadGateway,
					ClientMessage: "managed tool mediation upstream response exceeded size limit",
					Err:           fmt.Errorf("upstream response exceeded %d bytes during managed tool mediation", maxManagedLLMResponseBytes),
				}
			}
			return dispatchJSONAttemptResult{
				Response: &capturedResponse{
					StatusCode:    resp.StatusCode,
					Header:        resp.Header.Clone(),
					Body:          limited.Body,
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
		Name:      managedToolDisplayName(agentCtx, call.Name),
		Arguments: call.ArgumentsRaw,
	}
	resolved, ok := resolveManagedTool(agentCtx, call.Name)
	if !ok {
		raw := toolErrorPayload("unsupported_tool", managedToolModeMessage, 0, nil)
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}
	trace.Name = resolved.CanonicalName
	trace.Service = resolved.Manifest.Execution.Service
	if call.ParseErr != nil {
		raw := toolErrorPayload("invalid_arguments", fmt.Sprintf("Tool arguments for %s are invalid JSON: %v", trace.Name, call.ParseErr), 0, nil)
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}

	start := time.Now()
	childCtx, cancel := context.WithTimeout(ctx, policy.PerToolTimeout)
	defer cancel()
	raw, statusCode, err := h.callManagedHTTPTool(childCtx, agentID, resolved.Manifest, call.Arguments)
	trace.LatencyMS = time.Since(start).Milliseconds()
	if errors.Is(err, errManagedToolBudget) {
		return managedToolOutcome{}, err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		if ctx.Err() != nil {
			return managedToolOutcome{}, errManagedToolBudget
		}
		raw = toolErrorPayload("timeout", fmt.Sprintf("Service did not respond within %s", formatDuration(policy.PerToolTimeout)), http.StatusGatewayTimeout, nil)
		statusCode = http.StatusGatewayTimeout
		trace.StatusCode = statusCode
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}
	if errors.Is(err, context.Canceled) {
		if ctx.Err() != nil {
			return managedToolOutcome{}, errManagedToolBudget
		}
		raw = toolErrorPayload("canceled", "Tool execution was canceled", 0, nil)
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}
	trace.StatusCode = statusCode
	if err != nil {
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}
	trace.Result = raw
	return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
}

func (h *Handler) executeManagedAnthropicTool(ctx context.Context, agentID string, agentCtx *agentctx.AgentContext, call anthropicToolUse, policy managedToolPolicy) (managedToolOutcome, error) {
	trace := sessionhistory.ToolCallTrace{
		Name:      managedToolDisplayName(agentCtx, call.Name),
		Arguments: call.ArgumentsRaw,
	}
	resolved, ok := resolveManagedTool(agentCtx, call.Name)
	if !ok {
		raw := toolErrorPayload("unsupported_tool", managedToolModeMessage, 0, nil)
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}
	trace.Name = resolved.CanonicalName
	trace.Service = resolved.Manifest.Execution.Service
	if call.ParseErr != nil {
		raw := toolErrorPayload("invalid_arguments", fmt.Sprintf("Tool arguments for %s are invalid JSON: %v", trace.Name, call.ParseErr), 0, nil)
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}

	start := time.Now()
	childCtx, cancel := context.WithTimeout(ctx, policy.PerToolTimeout)
	defer cancel()
	raw, statusCode, err := h.callManagedHTTPTool(childCtx, agentID, resolved.Manifest, call.Arguments)
	trace.LatencyMS = time.Since(start).Milliseconds()
	if errors.Is(err, errManagedToolBudget) {
		return managedToolOutcome{}, err
	}
	if errors.Is(err, context.DeadlineExceeded) {
		if ctx.Err() != nil {
			return managedToolOutcome{}, errManagedToolBudget
		}
		raw = toolErrorPayload("timeout", fmt.Sprintf("Service did not respond within %s", formatDuration(policy.PerToolTimeout)), http.StatusGatewayTimeout, nil)
		statusCode = http.StatusGatewayTimeout
		trace.StatusCode = statusCode
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}
	if errors.Is(err, context.Canceled) {
		if ctx.Err() != nil {
			return managedToolOutcome{}, errManagedToolBudget
		}
		raw = toolErrorPayload("canceled", "Tool execution was canceled", 0, nil)
		trace.Result = raw
		return managedToolOutcome{RawJSON: raw, Trace: trace}, nil
	}
	trace.StatusCode = statusCode
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
	targetURL, body, err := buildManagedToolRequest(tool.Execution.BaseURL, strings.ToUpper(tool.Execution.Method), renderedPath, tool.Execution.BodyKey, remaining)
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
		if ctx.Err() != nil {
			return nil, 0, ctx.Err()
		}
		return toolErrorPayload("request_failed", err.Error(), 0, nil), 0, nil
	}
	defer resp.Body.Close()

	limited, readErr := readBodyLimited(resp.Body, maxManagedToolResultBytes)
	if readErr != nil {
		return toolErrorPayload("read_failed", readErr.Error(), resp.StatusCode, nil), resp.StatusCode, nil
	}
	details := decodeManagedToolBody(limited, resp.Header.Get("Content-Type"))
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return toolErrorPayload(fmt.Sprintf("http_%d", resp.StatusCode), fmt.Sprintf("Service returned HTTP %d", resp.StatusCode), resp.StatusCode, details), resp.StatusCode, nil
	}
	return toolSuccessPayload(resp.StatusCode, details), resp.StatusCode, nil
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

func buildManagedToolRequest(baseURL string, method string, path string, bodyKey string, args map[string]any) (string, io.Reader, error) {
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
	payload := any(args)
	if strings.TrimSpace(bodyKey) != "" {
		payload = map[string]any{strings.TrimSpace(bodyKey): args}
	}
	body, err := json.Marshal(payload)
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

func parseAnthropicToolResponse(body []byte) (map[string]any, []anthropicToolUse, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, nil, err
	}
	contentRaw, ok := payload["content"]
	if !ok {
		return nil, nil, fmt.Errorf("anthropic response missing content")
	}
	assistantMessage := map[string]any{
		"role": "assistant",
	}
	switch content := contentRaw.(type) {
	case string:
		assistantMessage["content"] = content
		return assistantMessage, nil, nil
	case []any:
		assistantMessage["content"] = content
		calls := make([]anthropicToolUse, 0)
		for i, raw := range content {
			block, _ := raw.(map[string]any)
			if block == nil {
				continue
			}
			if blockType, _ := block["type"].(string); blockType != "tool_use" {
				continue
			}
			callID, _ := block["id"].(string)
			if strings.TrimSpace(callID) == "" {
				callID = fmt.Sprintf("toolu_%d", i+1)
			}
			name, _ := block["name"].(string)
			input := block["input"]
			var (
				args     map[string]any
				parseErr error
				rawArgs  json.RawMessage
			)
			switch typed := input.(type) {
			case nil:
				args = map[string]any{}
				rawArgs = json.RawMessage([]byte(`{}`))
			case map[string]any:
				args = typed
				rawArgs, _ = json.Marshal(typed)
			default:
				parseErr = fmt.Errorf("tool input must decode to an object")
				rawArgs, _ = json.Marshal(typed)
			}
			calls = append(calls, anthropicToolUse{
				ID:           callID,
				Name:         strings.TrimSpace(name),
				Arguments:    args,
				ArgumentsRaw: rawArgs,
				ParseErr:     parseErr,
			})
		}
		return assistantMessage, calls, nil
	default:
		return nil, nil, fmt.Errorf("anthropic response content is %T", contentRaw)
	}
}

func partitionManagedOpenAIToolCalls(agentCtx *agentctx.AgentContext, calls []openAIToolCall) ([]openAIToolCall, []openAIToolCall) {
	if len(calls) == 0 {
		return nil, nil
	}
	managed := make([]openAIToolCall, 0, len(calls))
	native := make([]openAIToolCall, 0, len(calls))
	for _, call := range calls {
		if _, ok := resolveManagedTool(agentCtx, call.Name); ok {
			managed = append(managed, call)
			continue
		}
		native = append(native, call)
	}
	return managed, native
}

func partitionManagedAnthropicToolUses(agentCtx *agentctx.AgentContext, calls []anthropicToolUse) ([]anthropicToolUse, []anthropicToolUse) {
	if len(calls) == 0 {
		return nil, nil
	}
	managed := make([]anthropicToolUse, 0, len(calls))
	native := make([]anthropicToolUse, 0, len(calls))
	for _, call := range calls {
		if _, ok := resolveManagedTool(agentCtx, call.Name); ok {
			managed = append(managed, call)
			continue
		}
		native = append(native, call)
	}
	return managed, native
}

func appendOpenAIAssistantAndToolMessages(payload map[string]any, assistantMessage map[string]any, toolMessages []any) {
	messages, _ := payload["messages"].([]any)
	messages = append(messages, assistantMessage)
	messages = append(messages, toolMessages...)
	payload["messages"] = messages
}

func appendAnthropicAssistantAndToolResultMessages(payload map[string]any, assistantMessage map[string]any, toolResults []map[string]any) map[string]any {
	messages, _ := payload["messages"].([]any)
	messages = append(messages, assistantMessage)
	toolResultMessage := map[string]any{
		"role":    "user",
		"content": toolResults,
	}
	messages = append(messages, toolResultMessage)
	payload["messages"] = messages
	return toolResultMessage
}

func synthesizeOpenAIStream(finalBody []byte, upstreamModel string, usage managedUsageAggregate, includeUsage bool) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(finalBody, &payload); err != nil {
		return nil, err
	}
	id, _ := payload["id"].(string)
	if strings.TrimSpace(id) == "" {
		id = "chatcmpl-managed"
	}
	model, _ := payload["model"].(string)
	if strings.TrimSpace(model) == "" {
		model = upstreamModel
	}
	created := time.Now().Unix()
	if rawCreated, ok := payload["created"].(float64); ok {
		created = int64(rawCreated)
	}

	assistantMessage, toolCalls, err := parseOpenAIToolResponse(finalBody)
	if err != nil {
		return nil, err
	}
	if len(toolCalls) > 0 {
		return nil, fmt.Errorf("cannot synthesize streamed final text from tool_call payload")
	}
	content := openAIMessageContent(assistantMessage)

	var stream bytes.Buffer
	writeSSEChunk(&stream, map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"delta":         map[string]any{"role": "assistant"},
			"finish_reason": nil,
		}},
	})
	if content != "" {
		writeSSEChunk(&stream, map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         map[string]any{"content": content},
				"finish_reason": nil,
			}},
		})
	}
	writeSSEChunk(&stream, map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"delta":         map[string]any{},
			"finish_reason": "stop",
		}},
	})
	if includeUsage {
		writeSSEChunk(&stream, map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []any{},
			"usage": map[string]any{
				"prompt_tokens":     usage.PromptTokens,
				"completion_tokens": usage.CompletionTokens,
				"total_tokens":      usage.TotalTokens,
			},
		})
	}
	stream.WriteString("data: [DONE]\n\n")
	return stream.Bytes(), nil
}

func synthesizeOpenAIToolCallStream(finalBody []byte, upstreamModel string, usage managedUsageAggregate, includeUsage bool) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(finalBody, &payload); err != nil {
		return nil, err
	}
	id, _ := payload["id"].(string)
	if strings.TrimSpace(id) == "" {
		id = "chatcmpl-managed"
	}
	model, _ := payload["model"].(string)
	if strings.TrimSpace(model) == "" {
		model = upstreamModel
	}
	created := time.Now().Unix()
	if rawCreated, ok := payload["created"].(float64); ok {
		created = int64(rawCreated)
	}

	assistantMessage, toolCalls, err := parseOpenAIToolResponse(finalBody)
	if err != nil {
		return nil, err
	}
	if len(toolCalls) == 0 {
		return nil, fmt.Errorf("cannot synthesize streamed tool_call payload without tool calls")
	}

	var stream bytes.Buffer
	writeSSEChunk(&stream, map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"delta":         map[string]any{"role": "assistant"},
			"finish_reason": nil,
		}},
	})
	if content := openAIMessageContent(assistantMessage); content != "" {
		writeSSEChunk(&stream, map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index":         0,
				"delta":         map[string]any{"content": content},
				"finish_reason": nil,
			}},
		})
	}
	for i, call := range toolCalls {
		args := strings.TrimSpace(string(call.ArgumentsRaw))
		if args == "" {
			args = "{}"
		}
		writeSSEChunk(&stream, map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []map[string]any{{
				"index": 0,
				"delta": map[string]any{
					"tool_calls": []map[string]any{{
						"index": i,
						"id":    call.ID,
						"type":  "function",
						"function": map[string]any{
							"name":      call.Name,
							"arguments": args,
						},
					}},
				},
				"finish_reason": nil,
			}},
		})
	}
	writeSSEChunk(&stream, map[string]any{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]any{{
			"index":         0,
			"delta":         map[string]any{},
			"finish_reason": "tool_calls",
		}},
	})
	if includeUsage {
		writeSSEChunk(&stream, map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": []any{},
			"usage": map[string]any{
				"prompt_tokens":     usage.PromptTokens,
				"completion_tokens": usage.CompletionTokens,
				"total_tokens":      usage.TotalTokens,
			},
		})
	}
	stream.WriteString("data: [DONE]\n\n")
	return stream.Bytes(), nil
}

func synthesizeAnthropicStream(finalBody []byte, upstreamModel string, usage managedUsageAggregate) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(finalBody, &payload); err != nil {
		return nil, err
	}
	assistantMessage, toolUses, err := parseAnthropicToolResponse(finalBody)
	if err != nil {
		return nil, err
	}
	if len(toolUses) > 0 {
		return nil, fmt.Errorf("cannot synthesize streamed final text from tool_use payload")
	}
	id, _ := payload["id"].(string)
	if strings.TrimSpace(id) == "" {
		id = "msg_managed"
	}
	model, _ := payload["model"].(string)
	if strings.TrimSpace(model) == "" {
		model = upstreamModel
	}
	stopReason, _ := payload["stop_reason"].(string)
	if strings.TrimSpace(stopReason) == "" {
		stopReason = "end_turn"
	}

	var stream bytes.Buffer
	writeAnthropicSSEEvent(&stream, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            id,
			"type":          "message",
			"role":          "assistant",
			"model":         model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens": usage.PromptTokens,
			},
		},
	})

	for idx, text := range anthropicMessageTextBlocks(assistantMessage) {
		writeAnthropicSSEEvent(&stream, "content_block_start", map[string]any{
			"type":  "content_block_start",
			"index": idx,
			"content_block": map[string]any{
				"type": "text",
				"text": "",
			},
		})
		if text != "" {
			writeAnthropicSSEEvent(&stream, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{
					"type": "text_delta",
					"text": text,
				},
			})
		}
		writeAnthropicSSEEvent(&stream, "content_block_stop", map[string]any{
			"type":  "content_block_stop",
			"index": idx,
		})
	}

	writeAnthropicSSEEvent(&stream, "message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": payload["stop_sequence"],
		},
		"usage": map[string]any{
			"output_tokens": usage.CompletionTokens,
		},
	})
	writeAnthropicSSEEvent(&stream, "message_stop", map[string]any{
		"type": "message_stop",
	})
	return stream.Bytes(), nil
}

func synthesizeAnthropicToolUseStream(finalBody []byte, upstreamModel string, usage managedUsageAggregate) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(finalBody, &payload); err != nil {
		return nil, err
	}
	assistantMessage, toolUses, err := parseAnthropicToolResponse(finalBody)
	if err != nil {
		return nil, err
	}
	if len(toolUses) == 0 {
		return nil, fmt.Errorf("cannot synthesize streamed tool_use payload without tool calls")
	}

	id, _ := payload["id"].(string)
	if strings.TrimSpace(id) == "" {
		id = "msg_managed"
	}
	model, _ := payload["model"].(string)
	if strings.TrimSpace(model) == "" {
		model = upstreamModel
	}
	stopReason, _ := payload["stop_reason"].(string)
	if strings.TrimSpace(stopReason) == "" {
		stopReason = "tool_use"
	}

	var stream bytes.Buffer
	writeAnthropicSSEEvent(&stream, "message_start", map[string]any{
		"type": "message_start",
		"message": map[string]any{
			"id":            id,
			"type":          "message",
			"role":          "assistant",
			"model":         model,
			"content":       []any{},
			"stop_reason":   nil,
			"stop_sequence": nil,
			"usage": map[string]any{
				"input_tokens": usage.PromptTokens,
			},
		},
	})

	content, _ := assistantMessage["content"].([]any)
	for idx, raw := range content {
		block, _ := raw.(map[string]any)
		if block == nil {
			continue
		}
		blockType, _ := block["type"].(string)
		switch blockType {
		case "tool_use":
			blockID, _ := block["id"].(string)
			name, _ := block["name"].(string)
			input := "{}"
			if block["input"] != nil {
				encoded, err := json.Marshal(block["input"])
				if err != nil {
					return nil, err
				}
				input = string(encoded)
			}
			writeAnthropicSSEEvent(&stream, "content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": idx,
				"content_block": map[string]any{
					"type": "tool_use",
					"id":   blockID,
					"name": name,
				},
			})
			writeAnthropicSSEEvent(&stream, "content_block_delta", map[string]any{
				"type":  "content_block_delta",
				"index": idx,
				"delta": map[string]any{
					"type":         "input_json_delta",
					"partial_json": input,
				},
			})
			writeAnthropicSSEEvent(&stream, "content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": idx,
			})
		case "text", "":
			text, _ := block["text"].(string)
			writeAnthropicSSEEvent(&stream, "content_block_start", map[string]any{
				"type":  "content_block_start",
				"index": idx,
				"content_block": map[string]any{
					"type": "text",
					"text": "",
				},
			})
			if text != "" {
				writeAnthropicSSEEvent(&stream, "content_block_delta", map[string]any{
					"type":  "content_block_delta",
					"index": idx,
					"delta": map[string]any{
						"type": "text_delta",
						"text": text,
					},
				})
			}
			writeAnthropicSSEEvent(&stream, "content_block_stop", map[string]any{
				"type":  "content_block_stop",
				"index": idx,
			})
		}
	}

	writeAnthropicSSEEvent(&stream, "message_delta", map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   stopReason,
			"stop_sequence": payload["stop_sequence"],
		},
		"usage": map[string]any{
			"output_tokens": usage.CompletionTokens,
		},
	})
	writeAnthropicSSEEvent(&stream, "message_stop", map[string]any{
		"type": "message_stop",
	})
	return stream.Bytes(), nil
}

func openAIMessageContent(message map[string]any) string {
	if message == nil {
		return ""
	}
	if text, _ := message["content"].(string); strings.TrimSpace(text) != "" {
		return text
	}
	parts, _ := message["content"].([]any)
	var builder strings.Builder
	for _, raw := range parts {
		part, _ := raw.(map[string]any)
		if part == nil {
			continue
		}
		if partType, _ := part["type"].(string); partType != "" && partType != "text" {
			continue
		}
		text, _ := part["text"].(string)
		if text == "" {
			continue
		}
		if builder.Len() > 0 {
			builder.WriteByte('\n')
		}
		builder.WriteString(text)
	}
	return builder.String()
}

func anthropicMessageTextBlocks(message map[string]any) []string {
	if message == nil {
		return nil
	}
	switch content := message["content"].(type) {
	case string:
		if content == "" {
			return nil
		}
		return []string{content}
	case []any:
		parts := make([]string, 0, len(content))
		for _, raw := range content {
			block, _ := raw.(map[string]any)
			if block == nil {
				continue
			}
			blockType, _ := block["type"].(string)
			if blockType != "" && blockType != "text" {
				continue
			}
			text, _ := block["text"].(string)
			if text == "" {
				continue
			}
			parts = append(parts, text)
		}
		return parts
	default:
		return nil
	}
}

func writeSyntheticSSE(w http.ResponseWriter, stream []byte) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(stream)
}

func syntheticSSEHeader() http.Header {
	header := http.Header{}
	header.Set("Content-Type", "text/event-stream")
	header.Set("Cache-Control", "no-cache")
	header.Set("Connection", "keep-alive")
	return header
}

func newManagedStreamKeepalive(w http.ResponseWriter, enabled bool) *managedStreamKeepalive {
	if !enabled {
		return nil
	}
	return &managedStreamKeepalive{w: w}
}

func waitWithManagedKeepalive[T any](stream *managedStreamKeepalive, comment string, fn func() T) T {
	if stream == nil {
		return fn()
	}

	done := make(chan T, 1)
	go func() {
		done <- fn()
	}()

	ticker := time.NewTicker(managedKeepaliveInterval)
	defer ticker.Stop()

	for {
		select {
		case result := <-done:
			return result
		case <-ticker.C:
			stream.writeComment(comment)
		}
	}
}

func managedModelWaitComment(round int) string {
	return fmt.Sprintf("managed tool round %d waiting on model", round)
}

func managedToolWaitComment(round int, toolName string) string {
	toolName = strings.TrimSpace(toolName)
	if toolName == "" {
		return fmt.Sprintf("managed tool round %d executing tool", round)
	}
	return fmt.Sprintf("managed tool round %d executing %s", round, toolName)
}

func (s *managedStreamKeepalive) writeComment(comment string) {
	if s == nil {
		return
	}
	s.start()
	_, _ = io.WriteString(s.w, ": "+comment+"\n\n")
	flushManagedStream(s.w)
}

func (s *managedStreamKeepalive) writeFinal(stream []byte) {
	if s == nil {
		return
	}
	if !s.started {
		writeSyntheticSSE(s.w, stream)
		return
	}
	_, _ = s.w.Write(stream)
	flushManagedStream(s.w)
}

func (s *managedStreamKeepalive) writeOpenAIError(payload []byte) {
	if s == nil {
		return
	}
	s.start()
	raw := ensureJSONPayload(payload)
	if len(raw) == 0 {
		raw = jsonErrorPayload("managed tool mediation failed")
	}
	_, _ = io.WriteString(s.w, "data: ")
	_, _ = s.w.Write(raw)
	_, _ = io.WriteString(s.w, "\n\n")
	_, _ = io.WriteString(s.w, "data: [DONE]\n\n")
	flushManagedStream(s.w)
}

func (s *managedStreamKeepalive) writeAnthropicError(message string) {
	if s == nil {
		return
	}
	s.start()
	var stream bytes.Buffer
	writeAnthropicSSEEvent(&stream, "error", map[string]any{
		"type": "error",
		"error": map[string]any{
			"type":    "api_error",
			"message": message,
		},
	})
	_, _ = s.w.Write(stream.Bytes())
	flushManagedStream(s.w)
}

func (s *managedStreamKeepalive) start() {
	if s == nil || s.started {
		return
	}
	s.w.Header().Set("Content-Type", "text/event-stream")
	s.w.Header().Set("Cache-Control", "no-cache")
	s.w.Header().Set("Connection", "keep-alive")
	s.w.WriteHeader(http.StatusOK)
	s.started = true
	flushManagedStream(s.w)
}

func flushManagedStream(w http.ResponseWriter) {
	if flusher, ok := w.(http.Flusher); ok {
		flusher.Flush()
	}
}

func extractAnthropicErrorMessage(body []byte) string {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return "managed tool mediation failed"
	}
	if errObj, ok := payload["error"].(map[string]any); ok {
		if message, _ := errObj["message"].(string); strings.TrimSpace(message) != "" {
			return strings.TrimSpace(message)
		}
	}
	if message, _ := payload["message"].(string); strings.TrimSpace(message) != "" {
		return strings.TrimSpace(message)
	}
	return "managed tool mediation failed"
}

func writeSSEChunk(buf *bytes.Buffer, payload map[string]any) {
	raw, _ := json.Marshal(payload)
	buf.WriteString("data: ")
	buf.Write(raw)
	buf.WriteString("\n\n")
}

func writeAnthropicSSEEvent(buf *bytes.Buffer, event string, payload map[string]any) {
	raw, _ := json.Marshal(payload)
	buf.WriteString("event: ")
	buf.WriteString(event)
	buf.WriteString("\n")
	buf.WriteString("data: ")
	buf.Write(raw)
	buf.WriteString("\n\n")
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

func decodeManagedToolBody(body limitedReadResult, contentType string) any {
	trimmed := bytes.TrimSpace(body.Body)
	if len(trimmed) == 0 {
		return nil
	}
	if body.Truncated {
		return map[string]any{
			"data":           string(trimmed),
			"truncated":      true,
			"original_bytes": body.ObservedBytes,
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

func anthropicToolResultBlock(toolUseID string, raw []byte) map[string]any {
	block := map[string]any{
		"type":        "tool_result",
		"tool_use_id": toolUseID,
		"content":     string(raw),
	}
	if managedToolPayloadIsError(raw) {
		block["is_error"] = true
	}
	return block
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

func managedToolPayloadIsError(raw []byte) bool {
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		return false
	}
	ok, present := payload["ok"].(bool)
	return present && !ok
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

func readBodyLimited(r io.Reader, limit int) (limitedReadResult, error) {
	data, err := io.ReadAll(io.LimitReader(r, int64(limit)+1))
	if err != nil {
		return limitedReadResult{}, err
	}
	result := limitedReadResult{
		Body:          data,
		ObservedBytes: len(data),
	}
	if len(data) > limit {
		result.Truncated = true
		result.Body = data[:limit]
	}
	return result, nil
}

func formatDuration(d time.Duration) string {
	if d <= 0 {
		return "0s"
	}
	if d < time.Second {
		return d.Round(time.Millisecond).String()
	}
	return d.Round(time.Second).String()
}

func (a *managedUsageAggregate) AddRound(agentID string, usage cost.Usage, providerName, upstreamModel string, h *Handler) {
	a.TotalRounds++
	a.PromptTokens += usage.PromptTokens
	a.CompletionTokens += usage.CompletionTokens
	if usage.TotalTokens > 0 {
		a.TotalTokens += usage.TotalTokens
	} else {
		a.TotalTokens += usage.PromptTokens + usage.CompletionTokens
	}
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

func (h *Handler) recordManagedSuccess(agentID string, agentCtx *agentctx.AgentContext, providerName, requestedModel, upstreamModel string, requestPath string, requestOriginal []byte, requestEffective []byte, statusCode int, captured []byte, usage managedUsageAggregate, toolTrace []sessionhistory.ToolRoundTrace, downstreamStream bool, latencyMS int64) {
	if statusCode < 200 || statusCode >= 300 {
		return
	}
	responsePayload := sessionhistory.Payload{
		Format: "json",
		JSON:   json.RawMessage(captured),
	}
	if downstreamStream {
		responsePayload = sessionhistory.Payload{
			Format: "sse",
			Text:   string(captured),
		}
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
		Stream:            downstreamStream,
		RequestOriginal:   json.RawMessage(requestOriginal),
		RequestEffective:  json.RawMessage(requestEffective),
		Response:          responsePayload,
		Usage:             usage.sessionUsage(),
		ToolTrace:         toolTrace,
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
