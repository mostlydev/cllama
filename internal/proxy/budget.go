package proxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/sessionhistory"
)

const (
	EnvBudgetFailMode = "CLLAMA_BUDGET_FAIL_MODE"

	budgetExceededIntervention          = "budget_exceeded"
	rateLimitedIntervention             = "rate_limited"
	budgetCheckUnavailableIntervention  = "budget_check_unavailable"
	budgetFailModeClosed                = "closed"
	budgetEnforcementUnavailableMessage = "budget enforcement unavailable"
)

type normalizedBudgetPolicy struct {
	LimitUSD    *float64
	MaxRequests *int
	Window      time.Duration
	WindowRaw   string
	Behavior    string
}

type budgetDecision struct {
	Reason  string
	Message string
	Policy  normalizedBudgetPolicy
	Summary sessionhistory.WindowSummary
	Soft    bool
}

func (h *Handler) enforceBudget(w http.ResponseWriter, agentID string, agentCtx *agentctx.AgentContext, requestedModel string, start time.Time) bool {
	decision, err := h.budgetDecision(agentID, agentCtx, time.Now().UTC())
	if err != nil {
		if budgetFailClosed() {
			h.fail(w, http.StatusServiceUnavailable, budgetEnforcementUnavailableMessage, agentID, requestedModel, start, err)
			return true
		}
		h.logger.LogIntervention(agentID, requestedModel, budgetCheckUnavailableIntervention)
		h.logger.LogError(agentID, requestedModel, 0, 0, err)
		return false
	}
	if decision == nil {
		return false
	}

	h.logger.LogIntervention(agentID, requestedModel, decision.Reason)
	if decision.Soft {
		return false
	}
	writeBudgetError(w, *decision)
	return true
}

func (h *Handler) budgetDecision(agentID string, agentCtx *agentctx.AgentContext, now time.Time) (*budgetDecision, error) {
	policy, err := h.effectiveBudgetPolicy(agentID, agentCtx)
	if err != nil {
		return nil, err
	}
	normalized, err := normalizeBudgetPolicy(policy)
	if err != nil || normalized == nil {
		return nil, err
	}

	historyDir := h.sessionHistoryDir()
	if historyDir == "" {
		return nil, fmt.Errorf("session history directory is not configured")
	}
	summary, err := sessionhistory.SummarizeWindow(historyDir, agentID, now.Add(-normalized.Window))
	if err != nil {
		return nil, fmt.Errorf("summarize session history: %w", err)
	}

	if normalized.LimitUSD != nil && summary.ReportedCostUSD >= *normalized.LimitUSD {
		return &budgetDecision{
			Reason:  budgetExceededIntervention,
			Message: "budget exceeded",
			Policy:  *normalized,
			Summary: summary,
			Soft:    normalized.Behavior == "soft_alert",
		}, nil
	}
	if normalized.MaxRequests != nil && summary.Requests >= *normalized.MaxRequests {
		return &budgetDecision{
			Reason:  rateLimitedIntervention,
			Message: "request rate limit exceeded",
			Policy:  *normalized,
			Summary: summary,
			Soft:    normalized.Behavior == "soft_alert",
		}, nil
	}
	return nil, nil
}

func budgetFailClosed() bool {
	return strings.EqualFold(strings.TrimSpace(os.Getenv(EnvBudgetFailMode)), budgetFailModeClosed)
}

func (h *Handler) sessionHistoryDir() string {
	if h == nil || h.sessionRecorder == nil {
		return ""
	}
	return h.sessionRecorder.BaseDir()
}

func (h *Handler) effectiveBudgetPolicy(agentID string, agentCtx *agentctx.AgentContext) (*agentctx.BudgetPolicy, error) {
	var policy *agentctx.BudgetPolicy
	if agentCtx != nil {
		policy = cloneBudgetPolicy(agentCtx.Budget)
		if agentCtx.ContextDir != "" {
			override, ok, err := loadBudgetPolicyFile(filepath.Join(agentCtx.ContextDir, "budget-override.json"))
			if err != nil {
				return nil, err
			}
			if ok {
				policy = mergeBudgetPolicy(policy, override)
			}
		}
	}
	if h == nil || h.governanceDir == "" || !safeBudgetAgentID(agentID) {
		return policy, nil
	}
	override, ok, err := loadBudgetPolicyFile(filepath.Join(h.governanceDir, agentID, "budget.json"))
	if err != nil {
		return nil, err
	}
	if ok {
		policy = mergeBudgetPolicy(policy, override)
	}
	return policy, nil
}

func cloneBudgetPolicy(policy *agentctx.BudgetPolicy) *agentctx.BudgetPolicy {
	if policy == nil {
		return nil
	}
	out := *policy
	if policy.LimitUSD != nil {
		v := *policy.LimitUSD
		out.LimitUSD = &v
	}
	if policy.MaxRequests != nil {
		v := *policy.MaxRequests
		out.MaxRequests = &v
	}
	return &out
}

func mergeBudgetPolicy(base, override *agentctx.BudgetPolicy) *agentctx.BudgetPolicy {
	if override == nil {
		return base
	}
	out := cloneBudgetPolicy(base)
	if out == nil {
		out = &agentctx.BudgetPolicy{}
	}
	if override.LimitUSD != nil {
		v := *override.LimitUSD
		out.LimitUSD = &v
	}
	if override.MaxRequests != nil {
		v := *override.MaxRequests
		out.MaxRequests = &v
	}
	if strings.TrimSpace(override.Window) != "" {
		out.Window = strings.TrimSpace(override.Window)
	}
	if strings.TrimSpace(override.Behavior) != "" {
		out.Behavior = strings.TrimSpace(override.Behavior)
	}
	return out
}

func loadBudgetPolicyFile(path string) (*agentctx.BudgetPolicy, bool, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("read budget policy %q: %w", path, err)
	}
	var policy agentctx.BudgetPolicy
	if err := json.Unmarshal(raw, &policy); err != nil {
		return nil, false, fmt.Errorf("parse budget policy %q: %w", path, err)
	}
	return &policy, true, nil
}

func normalizeBudgetPolicy(policy *agentctx.BudgetPolicy) (*normalizedBudgetPolicy, error) {
	if policy == nil || (policy.LimitUSD == nil && policy.MaxRequests == nil) {
		return nil, nil
	}
	if policy.LimitUSD != nil && *policy.LimitUSD <= 0 {
		return nil, fmt.Errorf("budget limit_usd must be > 0")
	}
	if policy.MaxRequests != nil && *policy.MaxRequests <= 0 {
		return nil, fmt.Errorf("budget max_requests must be > 0")
	}
	windowRaw := strings.TrimSpace(policy.Window)
	if windowRaw == "" {
		return nil, fmt.Errorf("budget window is required")
	}
	window, err := time.ParseDuration(windowRaw)
	if err != nil || window <= 0 {
		return nil, fmt.Errorf("budget window must be a positive duration")
	}
	behavior := strings.TrimSpace(policy.Behavior)
	if behavior == "" {
		behavior = "hard_stop"
	}
	switch behavior {
	case "rate_limit", "hard_stop", "soft_alert":
	default:
		return nil, fmt.Errorf("unknown budget behavior %q", behavior)
	}
	return &normalizedBudgetPolicy{
		LimitUSD:    policy.LimitUSD,
		MaxRequests: policy.MaxRequests,
		Window:      window,
		WindowRaw:   windowRaw,
		Behavior:    behavior,
	}, nil
}

func safeBudgetAgentID(agentID string) bool {
	agentID = strings.TrimSpace(agentID)
	return agentID != "" &&
		!strings.HasPrefix(agentID, ".") &&
		filepath.Base(agentID) == agentID &&
		!strings.Contains(agentID, "/") &&
		!strings.Contains(agentID, "\\")
}

func writeBudgetError(w http.ResponseWriter, decision budgetDecision) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Retry-After", budgetRetryAfter(decision.Policy.Window))
	w.WriteHeader(http.StatusTooManyRequests)

	budget := map[string]any{
		"behavior":              decision.Policy.Behavior,
		"window":                decision.Policy.WindowRaw,
		"requests":              decision.Summary.Requests,
		"reported_cost_usd":     decision.Summary.ReportedCostUSD,
		"unknown_cost_requests": decision.Summary.UnknownCost,
	}
	if decision.Policy.LimitUSD != nil {
		budget["limit_usd"] = *decision.Policy.LimitUSD
	}
	if decision.Policy.MaxRequests != nil {
		budget["max_requests"] = *decision.Policy.MaxRequests
	}

	_ = json.NewEncoder(w).Encode(map[string]any{
		"error": map[string]any{
			"message": decision.Message,
			"type":    decision.Reason,
			"code":    decision.Reason,
		},
		"budget": budget,
	})
}

func budgetRetryAfter(window time.Duration) string {
	seconds := int(window.Seconds())
	if seconds < 1 {
		seconds = 1
	}
	return fmt.Sprintf("%d", seconds)
}
