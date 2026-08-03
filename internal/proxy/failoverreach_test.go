package proxy

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

// A policy may declare more than one fallback. Dropping every fallback after the
// first silently halves the protection an operator asked for.
func TestCandidateRefsUseEveryDeclaredFallback(t *testing.T) {
	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "openai/gpt-5"},
			{Slot: "fallback", Ref: "openrouter/gpt-4o-mini"},
			{Slot: "fallback", Ref: "xai/grok-4.1-fast"},
		},
	}

	got := candidateRefsFromPolicy(policy, policy.DefaultModel())
	want := []string{"openai/gpt-5", "openrouter/gpt-4o-mini", "xai/grok-4.1-fast"}
	if len(got) != len(want) {
		t.Fatalf("candidates: got %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("candidates: got %v, want %v", got, want)
		}
	}
}

// Slots carry intent: a model declared for a purpose (an "analysis" slot, say)
// is allowed but is not a failover target. Failover stays explicit.
func TestCandidateRefsExcludeNonFailoverSlots(t *testing.T) {
	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "openai/gpt-5"},
			{Slot: "analysis", Ref: "openai/gpt-4o-mini"},
		},
	}

	got := candidateRefsFromPolicy(policy, policy.DefaultModel())
	if len(got) != 1 || got[0] != "openai/gpt-5" {
		t.Fatalf("candidates: got %v, want [openai/gpt-5]", got)
	}
}

// The silent collapse is what made the production incident hard to read: the
// operator declared two models and got a 502 that pointed at credentials. If
// failover is unreachable, say so.
func TestHandlerWarnsWhenDeclaredModelsCannotFailOver(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hi"}}]}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "openai/gpt-5"},
			{Slot: "secondary", Ref: "openai/gpt-4o-mini"},
		},
	}

	var logs bytes.Buffer
	h := NewHandler(reg, stubContextLoaderWithPolicy("weston", "weston:dummy123", policy), logging.New(&logs))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer weston:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	assertInterventionLogged(t, logs.Bytes(), "policy_failover_unreachable")
}

func TestHandlerDoesNotWarnWhenFailoverIsReachable(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hi"}}]}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})
	reg.Set("openrouter", &provider.Provider{
		Name: "openrouter", BaseURL: backend.URL + "/v1", APIKey: "sk-or", Auth: "bearer",
	})

	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "openai/gpt-5"},
			{Slot: "fallback", Ref: "openrouter/gpt-4o-mini"},
		},
	}

	var logs bytes.Buffer
	h := NewHandler(reg, stubContextLoaderWithPolicy("weston", "weston:dummy123", policy), logging.New(&logs))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer weston:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if strings.Contains(logs.String(), "policy_failover_unreachable") {
		t.Errorf("a reachable fallback must not warn, got %s", logs.String())
	}
}

// A policy that legitimately declares a single model is not a misconfiguration.
func TestHandlerDoesNotWarnForSingleModelPolicy(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hi"}}]}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	policy := &agentctx.ModelPolicy{
		Mode:    "clamp",
		Allowed: []agentctx.AllowedModel{{Slot: "primary", Ref: "openai/gpt-5"}},
	}

	var logs bytes.Buffer
	h := NewHandler(reg, stubContextLoaderWithPolicy("weston", "weston:dummy123", policy), logging.New(&logs))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer weston:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if strings.Contains(logs.String(), "policy_failover_unreachable") {
		t.Errorf("a single-model policy must not warn, got %s", logs.String())
	}
}
