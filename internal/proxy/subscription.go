package proxy

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
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
	// Diag is recorded when the verdict is a failure. It is also set on a
	// pass verdict for an error status the plan rules do not classify.
	Diag planDiag
	// Refresh describes a refresh that failed after a 401.
	Refresh planDiag
}

// peekErrorCode reads a bounded prefix of an error body, restores it, and
// returns the vendor's error code and type, and a fixed label for the body's
// top-level form. The vendor documents admission refusals shaped as
// {"detail": ...} beside the usual {"error": {...}}; the label says which
// arrived without recording any of its text.
func peekErrorCode(resp *http.Response) (code, kind, shape string) {
	if resp.Body == nil {
		return "", "", "empty"
	}
	original := resp.Body
	prefix, _ := io.ReadAll(io.LimitReader(original, upstreamErrorPeekLimit))
	resp.Body = struct {
		io.Reader
		io.Closer
	}{Reader: io.MultiReader(bytes.NewReader(prefix), original), Closer: original}
	var shaped struct {
		Error struct {
			Code any `json:"code"`
			Type any `json:"type"`
		} `json:"error"`
	}
	if len(bytes.TrimSpace(prefix)) == 0 {
		return "", "", "empty"
	}
	var top map[string]json.RawMessage
	if json.Unmarshal(prefix, &top) != nil {
		return "", "", "not_json_object"
	}
	switch {
	case top["error"] != nil:
		shape = "error"
	case top["detail"] != nil:
		shape = "detail"
	default:
		shape = "other_object"
	}
	if json.Unmarshal(prefix, &shaped) != nil {
		return "", "", shape
	}
	code, _ = shaped.Error.Code.(string)
	kind, _ = shaped.Error.Type.(string)
	return code, kind, shape
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
	code, kind, shape := peekErrorCode(resp)
	diag := func(branch string) planDiag {
		return planDiag{Branch: branch, Stage: planStageInference, Status: resp.StatusCode, Code: code, Kind: kind, Shape: shape}
	}
	switch resp.StatusCode {
	case http.StatusTooManyRequests:
		if code == planLimitCode {
			h.coolPlan(lease)
			return planVerdict{Action: planAdvance, Reason: subscription.ReasonLimitReached, Code: code, Diag: diag(planBranchLimit)}
		}
		// An ordinary rate limit is not the plan's usage limit.
		until := time.Now().UTC().Add(parseCooldownDuration(resp))
		_ = h.registry.MarkCooldown(lease.ProviderName, lease.KeyID, "rate_limit", until)
		h.logger.LogProviderPool(lease.ProviderName, lease.KeyID, "cooldown", "rate_limit", until.Format(time.RFC3339))
		return planVerdict{Action: planAdvance, Reason: "rate_limit", Code: code, Diag: diag(planBranchRateLimit)}
	case http.StatusUnauthorized:
		if *reauthorized {
			// A fresh token was refused too. That is not proof of revocation.
			return planVerdict{Action: planAdvance, Reason: subscription.ReasonUnavailable, Code: code, Diag: diag(planBranch401Repeated)}
		}
		*reauthorized = true
		if _, err := h.subs.Reauthorize(ctx, lease.ProviderName, usedToken); err != nil {
			reason := subscription.ReasonOf(err)
			if reason == "" {
				reason = subscription.ReasonUnavailable
			}
			h.logger.LogProviderPool(lease.ProviderName, lease.KeyID, "signin", reason, "")
			// The inference reply and the refresh failure are two facts; the
			// refresh one is logged by the caller through Refresh.
			d := diag(planBranch401RefreshFail)
			return planVerdict{Action: planAdvance, Reason: reason, Code: code, Diag: d, Refresh: credentialDiag(err)}
		}
		return planVerdict{Action: planRetry}
	case http.StatusForbidden:
		return planVerdict{Action: planFail, Reason: subscription.ReasonNotEligible, Code: code, Diag: diag(planBranch403)}
	case http.StatusBadRequest:
		if code == planUnsupportedCode {
			return planVerdict{Action: planFail, Reason: subscription.ReasonUnsupported, Code: code, Diag: diag(planBranch400Unsupported)}
		}
	}
	return planVerdict{Diag: diag(planBranchHTTPOther)}
}

// planStreamError is a plan-route reply that opened as a success and did not
// complete.
type planStreamError struct {
	Code   string
	Detail string
	// The rest is for diagnostics. Branch is a fixed label; the vendor
	// strings are reduced to listed values when they are logged.
	Branch      string
	Status      int
	Kind        string
	Event       string
	LastEvent   string
	Events      int
	ContentType string
	// Bytes is how much of the stream was read before it ended.
	Bytes int
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
	mediaType := planMediaType(resp.Header)
	if !isSSE(resp.Header) && resp.Header.Get("Content-Type") != "" {
		// A success that declares another type carries no completion proof.
		resp.Body.Close()
		return &planStreamError{Detail: "reply was not an event stream", Branch: planBranchNotEventStream, Status: resp.StatusCode, ContentType: mediaType}
	}
	// A reply that declares no type is read as a stream. The header was never
	// the proof: only a response.completed event below is, so an empty body,
	// a JSON object or an error page read this way still ends as a failure.
	defer resp.Body.Close()
	reader := bufio.NewReaderSize(io.LimitReader(resp.Body, maxPlanStreamBytes+1), 64<<10)
	var (
		data      bytes.Buffer
		completed json.RawMessage
		total     int
		events    int
		lastEvent string
	)
	failed := func(branch, detail string) *planStreamError {
		return &planStreamError{Detail: detail, Branch: branch, Status: resp.StatusCode, LastEvent: lastEvent, Events: events, ContentType: mediaType, Bytes: total}
	}
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
				Type any    `json:"type"`
			} `json:"error"`
		}
		if json.Unmarshal(data.Bytes(), &event) != nil {
			return nil
		}
		events++
		lastEvent = event.Type
		switch event.Type {
		case "response.completed":
			// The generic adapter deliberately passes non-response JSON through.
			// A plan completion must first prove it carries a completed response.
			var result map[string]any
			if json.Unmarshal(event.Response, &result) != nil || result["object"] != "response" || result["status"] != "completed" {
				return failed(planBranchInvalidCompleted, "invalid completed response")
			}
			if _, ok := result["output"].([]any); !ok {
				return failed(planBranchInvalidCompleted, "completed response has no output array")
			}
			completed = append(json.RawMessage(nil), event.Response...)
		case "response.failed", "response.incomplete", "error":
			code, kind := event.Code, ""
			if event.Error != nil {
				kind, _ = event.Error.Type.(string)
				if event.Error.Code != "" {
					code = event.Error.Code
				}
			}
			if len(event.Response) > 0 {
				var inner struct {
					Error *struct {
						Code string `json:"code"`
						Type any    `json:"type"`
					} `json:"error"`
					Incomplete any `json:"incomplete_details"`
				}
				if json.Unmarshal(event.Response, &inner) == nil {
					if inner.Error != nil {
						if code == "" {
							code = inner.Error.Code
						}
						if kind == "" {
							kind, _ = inner.Error.Type.(string)
						}
					}
					// An incomplete response gives its cause as a reason, not
					// an error. It is recorded only; routing ignores it.
					if code == "" {
						if details, ok := inner.Incomplete.(map[string]any); ok {
							kind, _ = details["reason"].(string)
						}
					}
				}
			}
			return &planStreamError{Code: code, Detail: event.Type, Branch: planBranchStreamTerminal, Status: resp.StatusCode, Kind: kind, Event: event.Type, Events: events, ContentType: mediaType, Bytes: total}
		}
		return nil
	}
	for {
		line, err := reader.ReadBytes('\n')
		total += len(line)
		if total > maxPlanStreamBytes {
			return failed(planBranchStreamTooLarge, "stream too large")
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
			return failed(planBranchStreamCut, "stream interrupted")
		}
	}
	if completed == nil {
		return failed(planBranchStreamEnded, "stream ended before completion")
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
		return &planStreamError{Detail: "reply too large", Branch: planBranchScrub, Status: resp.StatusCode}
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

// revocationAck is the acknowledgement of a revocation. Beside the durable
// record it reports, for the revoked identity, how many requests each
// credential class served and its last plan-route failure. The counts are
// held in memory by this process: boot_id names the process, so a caller
// that saw another boot_id when the identity started knows they are partial.
type revocationAck struct {
	revocationRecord
	BootID          string         `json:"boot_id"`
	ServedBy        map[string]int `json:"served_by,omitempty"`
	LastPlanFailure string         `json:"last_plan_failure,omitempty"`
}

func (h *Handler) revokeAck(record revocationRecord) revocationAck {
	ack := revocationAck{revocationRecord: record, BootID: h.bootID}
	if served := h.subs.Served()[record.AgentID]; len(served) > 0 {
		ack.ServedBy = served
	}
	ack.LastPlanFailure = h.subs.Failures()[record.AgentID]
	return ack
}

func newBootID() string {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return ""
	}
	return hex.EncodeToString(b)
}
