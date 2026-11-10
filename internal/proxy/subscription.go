package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/mostlydev/cllama/internal/alert"
	"github.com/mostlydev/cllama/internal/provider"
	"github.com/mostlydev/cllama/internal/subscription"
	"github.com/mostlydev/cllama/internal/tracejournal"
)

const (
	// EnvSubscriptionLimitCooldownSeconds sets how long a plan route rests
	// after the vendor reports its usage limit. The vendor gives no reset
	// time, so this is a probe interval, not a prediction.
	EnvSubscriptionLimitCooldownSeconds = "CLLAMA_SUBSCRIPTION_LIMIT_COOLDOWN_SECONDS"

	defaultSubscriptionLimitCooldown = 15 * time.Minute
	// maxPlanStreamBytes bounds the buffered plan-route event stream.
	maxPlanStreamBytes = 64 << 20

	// planToolNamespace groups the agent's function tools on a plan route.
	planToolNamespace = "agent_tools"

	planLimitCode       = "subscription_sharing_usage_limit_exceeded"
	planUnsupportedCode = "subscription_sharing_unsupported_capability"
)

// WithSubscriptions gives the handler the plan sign-ins it may spend.
func WithSubscriptions(m *subscription.Manager) HandlerOption {
	return func(h *Handler) {
		h.subs = m
	}
}

// credentialClassFor names the kind of credential a provider is served with.
func credentialClassFor(providerName string) string {
	if subscription.IsPlanProvider(providerName) {
		return subscription.CredentialSubscription
	}
	return subscription.CredentialAPIKey
}

func subscriptionLimitCooldown() time.Duration {
	if raw := strings.TrimSpace(os.Getenv(EnvSubscriptionLimitCooldownSeconds)); raw != "" {
		if secs, err := strconv.Atoi(raw); err == nil && secs > 0 {
			return time.Duration(secs) * time.Second
		}
	}
	return defaultSubscriptionLimitCooldown
}

// planFieldsDropped are Responses fields the plan route refuses. They are
// removed and each removal is recorded as an intervention.
var planFieldsDropped = []string{
	"background", "conversation", "max_output_tokens", "max_tool_calls", "metadata",
	"moderation", "multi_agent", "prompt", "prompt_cache_retention", "safety_identifier",
	"service_tier", "temperature", "top_logprobs", "top_p", "truncation", "user",
}

// shapePlanResponsesRequest fits a translated Responses request to the plan
// route's documented limits. It returns the interventions it applied.
func shapePlanResponsesRequest(body map[string]any) ([]string, error) {
	if _, ok := body["previous_response_id"]; ok {
		return nil, &responsesAdapterRequestError{message: "previous_response_id is not supported on a plan route"}
	}
	var mutations []string
	for _, field := range planFieldsDropped {
		if _, ok := body[field]; ok {
			delete(body, field)
			mutations = append(mutations, "subscription_field_dropped:"+field)
		}
	}
	// System-role message items are refused; developer messages carry the
	// same instructions in the same position.
	if input, ok := body["input"].([]any); ok {
		moved := false
		for _, raw := range input {
			if item, ok := raw.(map[string]any); ok && item["role"] == "system" {
				item["role"] = "developer"
				moved = true
			}
		}
		if moved {
			mutations = append(mutations, "subscription_system_as_developer")
		}
	}
	// The route takes function and custom tools only inside a namespace.
	// Calls the model made earlier are replayed under the same namespace.
	if tools, ok := body["tools"].([]any); ok && len(tools) > 0 {
		for _, raw := range tools {
			tool, _ := raw.(map[string]any)
			if typ, _ := tool["type"].(string); typ != "function" && typ != "custom" {
				return nil, &responsesAdapterRequestError{message: fmt.Sprintf("tool type %q is not supported on a plan route", typ)}
			}
		}
		// How a tool_choice names one function inside a namespace is not
		// documented for this route. Forcing a tool is refused rather than
		// sent in a shape that may select nothing.
		if choice, ok := body["tool_choice"].(map[string]any); ok {
			if _, named := choice["name"]; named {
				return nil, &responsesAdapterRequestError{message: "a tool_choice that names a tool is not supported on a plan route"}
			}
		}
		body["tools"] = []any{map[string]any{
			"type":        "namespace",
			"name":        planToolNamespace,
			"description": "Tools available to the agent.",
			"tools":       tools,
		}}
		if input, ok := body["input"].([]any); ok {
			for _, raw := range input {
				if item, ok := raw.(map[string]any); ok && item["type"] == "function_call" {
					item["namespace"] = planToolNamespace
				}
			}
		}
		mutations = append(mutations, "subscription_tools_namespaced")
	}
	body["store"] = false
	body["stream"] = true
	return mutations, nil
}

// planCredential resolves the token for a plan provider selected from the
// pool. It returns the token it placed on prov, so a rejected token can be
// told apart from a newer one.
func (h *Handler) planCredential(ctx context.Context, prov *provider.Provider) (string, error) {
	if prov.Auth != provider.AuthSubscription {
		return "", nil
	}
	plan, ok := h.subs.Plan(prov.Name)
	if !ok {
		return "", &subscription.Error{Reason: subscription.ReasonSigninRequired, Terminal: true, Detail: "plan sign-in is not configured"}
	}
	token, err := h.subs.Token(ctx, prov.Name)
	if err != nil {
		return "", err
	}
	// The token may have been minted by this very request. It joins the
	// redaction set before it goes anywhere.
	tracejournal.FromContext(ctx).AddSecrets(h.subs.Secrets()...)
	// The endpoint is the plan's, never the registry's copy.
	prov.BaseURL = plan.InferenceBase
	prov.APIKey = token
	prov.Auth = "bearer"
	return token, nil
}

type planAction int

const (
	planPass    planAction = iota // not a plan matter; ordinary handling
	planRetry                     // token replaced; send again
	planAdvance                   // this route cannot serve now; next chain entry
	planFail                      // refuse the request; no other billing path
)

type planVerdict struct {
	Action planAction
	Reason string
	Code   string
}

// peekErrorCode reads a bounded prefix of an error body, restores it, and
// returns the vendor's error code.
func peekErrorCode(resp *http.Response) string {
	if resp.Body == nil {
		return ""
	}
	original := resp.Body
	prefix, _ := io.ReadAll(io.LimitReader(original, upstreamErrorPeekLimit))
	resp.Body = struct {
		io.Reader
		io.Closer
	}{Reader: io.MultiReader(bytes.NewReader(prefix), original), Closer: original}
	var shaped struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	if json.Unmarshal(prefix, &shaped) != nil {
		return ""
	}
	return shaped.Error.Code
}

func (h *Handler) coolPlan(lease provider.KeyLease) {
	until := time.Now().UTC().Add(subscriptionLimitCooldown())
	_ = h.registry.MarkCooldown(lease.ProviderName, lease.KeyID, subscription.ReasonLimitReached, until)
	stamp := until.Format(time.RFC3339)
	h.logger.LogProviderPool(lease.ProviderName, lease.KeyID, "cooldown", subscription.ReasonLimitReached, stamp)
	if h.notifier != nil {
		h.notifier.Notify(alert.PoolEvent{Provider: lease.ProviderName, KeyID: lease.KeyID, Action: "cooldown", Reason: subscription.ReasonLimitReached, CooldownUntil: stamp})
	}
}

// planResponseVerdict classifies a plan-route reply before the generic key
// rules see it. Those rules would mark the pool entry dead on any 401 or 403;
// a sign-in is not a key, and its state belongs to the saved record.
func (h *Handler) planResponseVerdict(ctx context.Context, resp *http.Response, lease provider.KeyLease, usedToken string, reauthorized *bool) planVerdict {
	if usedToken == "" || (resp.StatusCode >= 200 && resp.StatusCode < 300) {
		return planVerdict{}
	}
	code := peekErrorCode(resp)
	switch resp.StatusCode {
	case http.StatusTooManyRequests:
		if code == planLimitCode {
			h.coolPlan(lease)
			return planVerdict{Action: planAdvance, Reason: subscription.ReasonLimitReached, Code: code}
		}
		// An ordinary rate limit is not the plan's usage limit.
		until := time.Now().UTC().Add(parseCooldownDuration(resp))
		_ = h.registry.MarkCooldown(lease.ProviderName, lease.KeyID, "rate_limit", until)
		h.logger.LogProviderPool(lease.ProviderName, lease.KeyID, "cooldown", "rate_limit", until.Format(time.RFC3339))
		return planVerdict{Action: planAdvance, Reason: "rate_limit", Code: code}
	case http.StatusUnauthorized:
		if *reauthorized {
			// A fresh token was refused too. That is not proof of revocation.
			return planVerdict{Action: planAdvance, Reason: subscription.ReasonUnavailable, Code: code}
		}
		*reauthorized = true
		if _, err := h.subs.Reauthorize(ctx, lease.ProviderName, usedToken); err != nil {
			reason := subscription.ReasonOf(err)
			if reason == "" {
				reason = subscription.ReasonUnavailable
			}
			h.logger.LogProviderPool(lease.ProviderName, lease.KeyID, "signin", reason, "")
			return planVerdict{Action: planAdvance, Reason: reason, Code: code}
		}
		return planVerdict{Action: planRetry}
	case http.StatusForbidden:
		return planVerdict{Action: planFail, Reason: subscription.ReasonNotEligible, Code: code}
	case http.StatusBadRequest:
		if code == planUnsupportedCode {
			return planVerdict{Action: planFail, Reason: subscription.ReasonUnsupported, Code: code}
		}
	}
	return planVerdict{}
}

// planStreamError is a plan-route reply that opened as a success and did not
// complete.
type planStreamError struct {
	Code   string
	Detail string
}

func (e *planStreamError) Error() string {
	if e.Code != "" {
		return "plan route stream did not complete: " + e.Code
	}
	return "plan route stream did not complete: " + e.Detail
}

// collapsePlanStream reads a plan-route event stream to its end and replaces
// the body with the completed response object. Only response.completed is a
// success: a failed, incomplete or cut stream is an error, and nothing from
// it is passed on.
func collapsePlanStream(resp *http.Response) error {
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil
	}
	if !isSSE(resp.Header) {
		// A success that is not an event stream carries no completion proof.
		resp.Body.Close()
		return &planStreamError{Detail: "reply was not an event stream"}
	}
	defer resp.Body.Close()
	reader := bufio.NewReaderSize(io.LimitReader(resp.Body, maxPlanStreamBytes+1), 64<<10)
	var (
		data      bytes.Buffer
		completed json.RawMessage
		total     int
	)
	flush := func() error {
		defer data.Reset()
		if data.Len() == 0 {
			return nil
		}
		var event struct {
			Type     string          `json:"type"`
			Response json.RawMessage `json:"response"`
			Code     string          `json:"code"`
			Error    *struct {
				Code string `json:"code"`
			} `json:"error"`
		}
		if json.Unmarshal(data.Bytes(), &event) != nil {
			return nil
		}
		switch event.Type {
		case "response.completed":
			completed = append(json.RawMessage(nil), event.Response...)
		case "response.failed", "response.incomplete", "error":
			code := event.Code
			if event.Error != nil && event.Error.Code != "" {
				code = event.Error.Code
			}
			if code == "" && len(event.Response) > 0 {
				var failed struct {
					Error *struct {
						Code string `json:"code"`
					} `json:"error"`
				}
				if json.Unmarshal(event.Response, &failed) == nil && failed.Error != nil {
					code = failed.Error.Code
				}
			}
			return &planStreamError{Code: code, Detail: event.Type}
		}
		return nil
	}
	for {
		line, err := reader.ReadBytes('\n')
		total += len(line)
		if total > maxPlanStreamBytes {
			return &planStreamError{Detail: "stream too large"}
		}
		trimmed := bytes.TrimRight(line, "\r\n")
		switch {
		case len(trimmed) == 0:
			if ferr := flush(); ferr != nil {
				return ferr
			}
		case bytes.HasPrefix(trimmed, []byte("data:")):
			if data.Len() > 0 {
				data.WriteByte('\n')
			}
			data.Write(bytes.TrimPrefix(bytes.TrimPrefix(trimmed, []byte("data:")), []byte(" ")))
		}
		if completed != nil {
			break
		}
		if err != nil {
			if errors.Is(err, io.EOF) {
				if ferr := flush(); ferr != nil {
					return ferr
				}
				break
			}
			return &planStreamError{Detail: "stream interrupted"}
		}
	}
	if completed == nil {
		return &planStreamError{Detail: "stream ended before completion"}
	}
	resp.Body = io.NopCloser(bytes.NewReader(completed))
	resp.ContentLength = int64(len(completed))
	resp.Header.Set("Content-Type", "application/json")
	resp.Header.Del("Content-Length")
	resp.Header.Del("Content-Encoding")
	return nil
}

// planStreamVerdict turns a plan stream failure into a routing decision.
func (h *Handler) planStreamVerdict(err error, lease provider.KeyLease) (planVerdict, bool) {
	var streamErr *planStreamError
	if !errors.As(err, &streamErr) {
		return planVerdict{}, false
	}
	switch streamErr.Code {
	case planLimitCode:
		h.coolPlan(lease)
		return planVerdict{Action: planAdvance, Reason: subscription.ReasonLimitReached, Code: streamErr.Code}, true
	case planUnsupportedCode:
		return planVerdict{Action: planFail, Reason: subscription.ReasonUnsupported, Code: streamErr.Code}, true
	case "subscription_sharing_user_not_eligible":
		return planVerdict{Action: planFail, Reason: subscription.ReasonNotEligible, Code: streamErr.Code}, true
	}
	return planVerdict{Action: planAdvance, Reason: subscription.ReasonUnavailable, Code: streamErr.Code}, true
}

// planFailure is the client-facing status and message for a plan reason.
func planFailure(reason string) (int, string) {
	switch reason {
	case subscription.ReasonSigninRequired:
		return http.StatusServiceUnavailable, "plan sign-in is missing or expired; the operator must sign in again"
	case subscription.ReasonLimitReached:
		return http.StatusTooManyRequests, "plan usage limit reached"
	case subscription.ReasonNotEligible:
		return http.StatusForbidden, "plan usage is not permitted for this account"
	case subscription.ReasonUnsupported:
		return http.StatusBadRequest, "request uses a feature the plan route does not support"
	default:
		return http.StatusServiceUnavailable, "plan route is temporarily unavailable"
	}
}

func isPlanReason(reason string) bool {
	return strings.HasPrefix(reason, "subscription_")
}

// writePlanError answers a request the plan route could not serve, naming
// the reason in a stable error code.
func writePlanError(w http.ResponseWriter, reason string) (int, string) {
	status, msg := planFailure(reason)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{"message": msg, "type": "subscription_error", "code": reason},
	})
	return status, msg
}

func (h *Handler) failPlan(w http.ResponseWriter, reason, agentID, model string, start time.Time) {
	status, msg := writePlanError(w, reason)
	h.logger.LogError(agentID, model, status, time.Since(start).Milliseconds(), fmt.Errorf("%s: %s", reason, msg))
}

// planErrorBody is the JSON error body for a plan reason, for paths that
// return a status and message instead of writing the response.
func planErrorMessage(reason string) string {
	_, msg := planFailure(reason)
	return reason + ": " + msg
}

// planReasonOf maps a plan credential error to its reason.
func planReasonOf(err error) string {
	if reason := subscription.ReasonOf(err); reason != "" {
		return reason
	}
	return subscription.ReasonUnavailable
}

// cooldownReason names why a provider's whole pool is resting. A plan route
// rests only for the reason recorded on its entry.
func (h *Handler) cooldownReason(providerName string) string {
	if !h.registry.IsPlan(providerName) {
		return "provider_cooldown"
	}
	for _, key := range h.registry.All()[strings.ToLower(strings.TrimSpace(providerName))].Keys {
		if key.LastErrorReason == subscription.ReasonLimitReached {
			return subscription.ReasonLimitReached
		}
	}
	return "provider_cooldown"
}

const planRedaction = "[redacted]"

// scrubPlanBody removes sign-in token values from a buffered plan-route
// reply before it is recorded or returned. A provider that echoes the bearer
// must not hand it to the agent.
func (h *Handler) scrubPlanBody(resp *http.Response) error {
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxPlanStreamBytes+1))
	resp.Body.Close()
	if err != nil {
		return err
	}
	if len(body) > maxPlanStreamBytes {
		return &planStreamError{Detail: "reply too large"}
	}
	for _, secret := range h.subs.Secrets() {
		if len(secret) >= 8 {
			body = bytes.ReplaceAll(body, []byte(secret), []byte(planRedaction))
			// Response headers reach the worker too. Drop any header whose
			// name or value echoes a credential, including nonstandard debug
			// headers; a body-only scrub leaves that disclosure path open.
			for name, values := range resp.Header {
				if strings.Contains(name, secret) {
					delete(resp.Header, name)
					continue
				}
				for _, value := range values {
					if strings.Contains(value, secret) {
						delete(resp.Header, name)
						break
					}
				}
			}
		}
	}
	resp.Body = io.NopCloser(bytes.NewReader(body))
	resp.ContentLength = int64(len(body))
	resp.Header.Del("Content-Length")
	return nil
}
