// Package subscription holds plan sign-ins that cllama uses in place of an
// API key. A sign-in belongs to the operator; cllama owns the record, refreshes
// it, and never hands it to an agent.
package subscription

import "strings"

// Credential classes recorded for every upstream attempt.
const (
	CredentialSubscription = "subscription"
	CredentialAPIKey       = "api_key"
)

// Reasons a plan route cannot serve a request. They are distinct from model
// and transport errors so an orchestrator can act on them.
const (
	ReasonSigninRequired = "subscription_signin_required"
	ReasonLimitReached   = "subscription_limit_reached"
	ReasonNotEligible    = "subscription_not_eligible"
	ReasonUnavailable    = "subscription_unavailable"
	ReasonUnsupported    = "subscription_unsupported_request"
)

// Plan is one vendor's documented plan-usage sign-in. Every URL is fixed: a
// sign-in token is sent only to these origins, whatever the operator has
// configured for the vendor's API-key provider.
type Plan struct {
	// Provider is the cllama provider name the plan is routed under.
	Provider string
	// Label names the plan in operator output.
	Label         string
	Issuer        string
	AuthorizeURL  string
	TokenURL      string
	DiscoveryURL  string
	Resource      string
	InferenceBase string
	Scopes        []string
	// PlanScope must be granted before any inference call is made.
	PlanScope string
	// RegistrationClientID is sent on first sign-in only. The issued client
	// ID that comes back is the one saved.
	RegistrationClientID string
	AgentNameHint        string
	CallbackPath         string
	DefaultCallbackPort  int
	// UsageURL is where the operator reviews plan limits.
	UsageURL string
}

// ChatGPT is OpenAI's "Sign in with ChatGPT" plan usage for open-source,
// locally hosted apps: https://developers.openai.com/siwc/token-sharing-open-source
var ChatGPT = Plan{
	Provider:             "chatgpt",
	Label:                "ChatGPT plan",
	Issuer:               "https://auth.openai.com",
	AuthorizeURL:         "https://auth.openai.com/api/accounts/authorize",
	TokenURL:             "https://auth.openai.com/api/accounts/oauth/token",
	DiscoveryURL:         "https://auth.openai.com/.well-known/openid-configuration",
	Resource:             "https://api.openai.com/v1",
	InferenceBase:        "https://api.openai.com/v1",
	Scopes:               []string{"openid", "profile", "email", "offline_access", "resource.invoke", "chatgpt.tokens.use.direct"},
	PlanScope:            "chatgpt.tokens.use.direct",
	RegistrationClientID: "dynamic_agent_client",
	AgentNameHint:        "cllama",
	CallbackPath:         "/auth/callback",
	DefaultCallbackPort:  1455,
	UsageURL:             "https://chatgpt.com/#settings/Usage",
}

var plans = map[string]Plan{ChatGPT.Provider: ChatGPT}

// Lookup returns the plan routed under a provider name.
func Lookup(provider string) (Plan, bool) {
	p, ok := plans[strings.ToLower(strings.TrimSpace(provider))]
	return p, ok
}

// IsPlanProvider reports whether a provider name is a plan route.
func IsPlanProvider(provider string) bool {
	_, ok := Lookup(provider)
	return ok
}

// Providers lists the plan provider names.
func Providers() []string {
	out := make([]string, 0, len(plans))
	for name := range plans {
		out = append(out, name)
	}
	return out
}
