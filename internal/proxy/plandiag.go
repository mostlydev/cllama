package proxy

import (
	"errors"
	"mime"
	"net/http"
	"strconv"
	"strings"

	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/subscription"
)

// Branch labels for plan-route failures. Each names the code path that chose
// the outcome. They are constants: nothing from a vendor reply selects one
// except through the comparisons in this package.
const (
	planBranchCredential     = "credential_" // completed by a subscription stage
	planBranchTransport      = "transport"
	planBranchLimit          = "http_429_limit"
	planBranchRateLimit      = "http_429_rate"
	planBranch401Repeated    = "http_401_after_refresh"
	planBranch401RefreshFail = "http_401_refresh_failed"
	planBranch403            = "http_403"
	planBranch400Unsupported = "http_400_unsupported"
	planBranchHTTPOther      = "http_other"
	planBranchNotEventStream = "not_event_stream"
	planBranchStreamTerminal = "stream_terminal_event"
	planBranchStreamEnded    = "stream_ended_early"
	planBranchStreamCut      = "stream_interrupted"
	planBranchStreamTooLarge = "stream_too_large"
	planBranchScrub          = "scrub_failed"
	planBranchAdapter        = "adapter_error"

	planStageToken     = "token"
	planStageInference = "inference"

	// planUnlisted stands in for any vendor string that is not on a list
	// below. A vendor reply is untrusted text: it can reflect a credential or
	// an account name in any field, so no vendor string is logged as given.
	planUnlisted = "unlisted"
)

func labelSet(words string) map[string]bool {
	set := map[string]bool{}
	for _, w := range strings.Fields(words) {
		set[w] = true
	}
	return set
}

var (
	planKnownCodes = labelSet(`
		subscription_sharing_usage_limit_exceeded subscription_sharing_unsupported_capability
		subscription_sharing_user_not_eligible subscription_sharing_not_enabled
		subscription_sharing_usage_unavailable subscription_sharing_route_not_supported
		subscription_sharing_invalid_user subscription_sharing_user_unavailable
		chatpass_v2_scope_not_authorized chatpass_v2_invalid_authorization_context
		subscription_sharing_client_not_allowed subscription_sharing_model_not_supported
		usage_limit_reached usage_not_included plan_restricted
		unsupported_parameter unsupported_value invalid_value invalid_type
		missing_required_parameter string_above_max_length array_above_max_length
		integer_above_max_value invalid_tool_schema invalid_request_error
		invalid_prompt invalid_function_parameters unknown_parameter
		model_not_found unsupported_model model_not_supported context_length_exceeded
		insufficient_quota rate_limit_exceeded invalid_api_key invalid_token
		token_expired token_revoked invalid_scope insufficient_scope missing_scope
		invalid_audience invalid_issuer account_deactivated unsupported_country_region_territory
		content_policy_violation server_error server_is_overloaded internal_error
		bad_gateway timeout max_output_tokens content_filter
		invalid_grant invalid_client invalid_request unauthorized_client access_denied
		unsupported_grant_type invalid_target invalid_refresh_token
		refresh_token_expired refresh_token_invalidated refresh_token_reused
		temporarily_unavailable slow_down`)
	planKnownKinds = labelSet(`
		invalid_request_error authentication_error permission_error not_found_error
		rate_limit_error api_error overloaded_error server_error tokens
		insufficient_quota subscription_error
		max_output_tokens content_filter`)
	planKnownEvents = labelSet(`
		error keepalive ping
		response.created response.queued response.in_progress response.completed
		response.failed response.incomplete response.cancelled
		response.output_item.added response.output_item.done
		response.content_part.added response.content_part.done
		response.output_text.delta response.output_text.done
		response.output_text.annotation.added
		response.refusal.delta response.refusal.done
		response.function_call_arguments.delta response.function_call_arguments.done
		response.reasoning_summary_part.added response.reasoning_summary_part.done
		response.reasoning_summary_text.delta response.reasoning_summary_text.done
		response.reasoning_text.delta response.reasoning_text.done
		rate_limits.updated`)
	planKnownMediaTypes = labelSet(`
		application/json application/problem+json application/x-ndjson
		application/octet-stream text/event-stream text/html text/plain text/xml`)
)

// listed returns value when it is on the list, "" when it is empty, and
// planUnlisted otherwise.
func listed(set map[string]bool, value string) string {
	if value == "" {
		return ""
	}
	if set[value] {
		return value
	}
	return planUnlisted
}

// planMediaType is the listed media type of a reply, without parameters.
func planMediaType(header http.Header) string {
	raw := header.Get("Content-Type")
	if raw == "" {
		return "none"
	}
	mediaType, _, err := mime.ParseMediaType(raw)
	if err != nil {
		return planUnlisted
	}
	return listed(planKnownMediaTypes, mediaType)
}

// planDiag is what one plan-route failure is recorded with.
type planDiag struct {
	Branch      string
	Stage       string
	Status      int
	Code        string
	Kind        string
	Event       string
	LastEvent   string
	Events      int
	ContentType string
	// Shape classifies an error body by its top-level form, a fixed label.
	Shape string
}

// credentialDiag describes a failure of the sign-in step. The error's detail
// text is never used: it can carry transport strings.
func credentialDiag(err error) planDiag {
	d := planDiag{Branch: planBranchCredential + "other", Stage: planStageToken}
	var e *subscription.Error
	if errors.As(err, &e) {
		switch e.Stage {
		case subscription.StageSigninRequired, subscription.StageLock,
			subscription.StageRefreshTransport, subscription.StageRefreshHTTP,
			subscription.StageRefreshMalformed, subscription.StageRefreshNoRotation,
			subscription.StageSave:
			d.Branch = planBranchCredential + e.Stage
		}
		d.Status = e.Status
		d.Code = e.Code
	}
	return d
}

// streamDiag describes a plan stream that did not complete.
func streamDiag(err error) planDiag {
	d := planDiag{Branch: planBranchStreamEnded, Stage: planStageInference}
	var e *planStreamError
	if errors.As(err, &e) {
		if e.Branch != "" {
			d.Branch = e.Branch
		}
		d.Status = e.Status
		d.Code = e.Code
		d.Kind = e.Kind
		d.Event = e.Event
		d.LastEvent = e.LastEvent
		d.Events = e.Events
		d.ContentType = e.ContentType
	}
	return d
}

// logPlanFailure records why a plan route did not serve. Vendor strings are
// reduced to listed values here, at the single point where they are written.
func (h *Handler) logPlanFailure(agentID, model, providerName, reason string, d planDiag) {
	if h.logger == nil || d.Branch == "" {
		return
	}
	h.logger.LogPlanFailure(agentID, model, logging.PlanFailureInfo{
		Provider:        providerName,
		Reason:          reason,
		Branch:          d.Branch,
		Stage:           d.Stage,
		Status:          d.Status,
		Code:            listed(planKnownCodes, d.Code),
		Kind:            listed(planKnownKinds, d.Kind),
		StreamEvent:     listed(planKnownEvents, d.Event),
		StreamLastEvent: listed(planKnownEvents, d.LastEvent),
		StreamEvents:    d.Events,
		ContentType:     d.ContentType,
		BodyShape:       d.Shape,
	})
}

// logPlanVerdict records a classified plan-route reply. A retry is not a
// failure. A pass verdict on an error status is one the plan rules leave to
// the ordinary handling; it is recorded with the reason that handling uses.
func (h *Handler) logPlanVerdict(agentID, model, providerName string, v planVerdict, status int) {
	if v.Action == planRetry || v.Diag.Branch == "" {
		return
	}
	reason := v.Reason
	if reason == "" {
		reason = "http_" + strconv.Itoa(status)
	}
	if v.Refresh.Branch != "" {
		h.logPlanFailure(agentID, model, providerName, reason, v.Refresh)
	}
	h.logPlanFailure(agentID, model, providerName, reason, v.Diag)
}
