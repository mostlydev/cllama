package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

// bodyReadFailureBackend answers with valid headers and a Content-Length it
// never satisfies, then drops the connection — reproducing an upstream that
// commits headers and then fails mid-body.
func bodyReadFailureBackend(t *testing.T, onRequest func()) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if onRequest != nil {
			onRequest()
		}
		conn, buf, err := w.(http.Hijacker).Hijack()
		if err != nil {
			t.Errorf("hijack: %v", err)
			return
		}
		defer conn.Close()
		_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n")
		_, _ = buf.WriteString(`{"id":"chatcmpl-1","choices":[`)
		_ = buf.Flush()
	}))
}

func stubContextLoaderWithToolsAndPolicy(agentID, token string, tools *agentctx.ToolManifest, policy *agentctx.ModelPolicy) ContextLoader {
	return func(id string) (*agentctx.AgentContext, error) {
		if id != agentID {
			return nil, io.EOF
		}
		return &agentctx.AgentContext{
			AgentID:     id,
			ContextDir:  "/claw/context/" + id,
			AgentsMD:    []byte("# Contract"),
			ClawdapusMD: []byte("# Infra"),
			Metadata:    map[string]any{"token": token},
			Tools:       tools,
			ModelPolicy: policy,
		}, nil
	}
}

// Transport failures and 5xx already advance to the next declared candidate.
// A body read that fails after headers were committed must behave the same way:
// no downstream bytes were written, so the fallback is still safe.
func TestManagedDispatchAdvancesToFallbackOnResponseBodyReadError(t *testing.T) {
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	primaryCalls := 0
	primary := bodyReadFailureBackend(t, func() { primaryCalls++ })
	defer primary.Close()

	fallbackCalls := 0
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","choices":[{"message":{"role":"assistant","content":"fallback answered"}}]}`))
	}))
	defer fallback.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: primary.URL + "/v1", APIKey: "sk-openai", Auth: "bearer",
	})
	reg.Set("openrouter", &provider.Provider{
		Name: "openrouter", BaseURL: fallback.URL + "/v1", APIKey: "sk-or", Auth: "bearer",
	})

	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "openai/gpt-4o"},
			{Slot: "fallback", Ref: "openrouter/gpt-4o-mini"},
		},
	}

	var logs bytes.Buffer
	h := NewHandler(reg, stubContextLoaderWithToolsAndPolicy("weston", "weston:dummy123",
		managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", ""), policy),
		logging.New(&logs))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer weston:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected the declared fallback to answer with 200, got %d: %s", w.Code, w.Body.String())
	}
	if primaryCalls == 0 {
		t.Error("expected the primary candidate to be attempted")
	}
	if fallbackCalls != 1 {
		t.Fatalf("expected exactly one fallback dispatch, got %d", fallbackCalls)
	}

	var chat map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &chat); err != nil {
		t.Fatalf("unmarshal downstream body: %v", err)
	}
	choices, _ := chat["choices"].([]any)
	choice, _ := choices[0].(map[string]any)
	msg, _ := choice["message"].(map[string]any)
	if msg["content"] != "fallback answered" {
		t.Errorf("downstream content: got %#v", msg["content"])
	}

	// The failover must be visible in the audit trail, with a stable reason.
	if !strings.Contains(logs.String(), "response_read_error") {
		t.Errorf("expected a response_read_error fallback reason in the audit log, got %s", logs.String())
	}
}

// With no declared fallback the terminal 502 and its message are preserved.
func TestManagedDispatchKeepsTerminal502WhenNoFallbackRemains(t *testing.T) {
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	primary := bodyReadFailureBackend(t, nil)
	defer primary.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: primary.URL + "/v1", APIKey: "sk-openai", Auth: "bearer",
	})

	policy := &agentctx.ModelPolicy{
		Mode:    "clamp",
		Allowed: []agentctx.AllowedModel{{Slot: "primary", Ref: "openai/gpt-4o"}},
	}

	h := NewHandler(reg, stubContextLoaderWithToolsAndPolicy("weston", "weston:dummy123",
		managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", ""), policy),
		logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer weston:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "failed to read upstream response") {
		t.Errorf("expected the existing terminal message preserved, got %s", w.Body.String())
	}
}

// A response read failure can happen after an earlier key in the same provider
// was rate-limited. Preserve that cooldown signal while advancing so an
// exhausted candidate list retains the more accurate 503 classification.
func TestManagedDispatchPreservesCooldownAcrossResponseBodyReadFallback(t *testing.T) {
	var rateLimitedCalls, readFailureCalls int
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.Header.Get("Authorization") {
		case "Bearer sk-rate-limited":
			rateLimitedCalls++
			w.Header().Set("Retry-After", "60")
			w.WriteHeader(http.StatusTooManyRequests)
			_, _ = w.Write([]byte(`{"error":"rate limited"}`))
		case "Bearer sk-read-failure":
			readFailureCalls++
			conn, buf, err := w.(http.Hijacker).Hijack()
			if err != nil {
				t.Errorf("hijack: %v", err)
				return
			}
			defer conn.Close()
			_, _ = buf.WriteString("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 4096\r\n\r\n")
			_, _ = buf.WriteString(`{"id":"chatcmpl-1","choices":[`)
			_ = buf.Flush()
		default:
			t.Errorf("unexpected authorization header %q", r.Header.Get("Authorization"))
		}
	}))
	defer primary.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: primary.URL + "/v1", APIKey: "sk-rate-limited", Auth: "bearer",
	})
	if _, err := reg.AddRuntimeKey("openai", "read-failure", "sk-read-failure"); err != nil {
		t.Fatalf("AddRuntimeKey: %v", err)
	}

	h := NewHandler(reg, nil, logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Content-Type", "application/json")
	payload := map[string]any{
		"model":    "openai/gpt-4o",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}
	candidates := []dispatchCandidate{
		{ProviderName: "openai", UpstreamModel: "gpt-4o"},
		{ProviderName: "missing", UpstreamModel: "fallback-model"},
	}

	resp, status, msg, err := h.dispatchCandidatesJSON(context.Background(), req, "agent-1", "openai/gpt-4o", payload, candidates, nil)
	if err == nil {
		t.Fatal("expected exhausted candidate error")
	}
	if resp != nil {
		t.Fatalf("expected no response, got %+v", resp)
	}
	if status != http.StatusServiceUnavailable || msg != "all declared provider keys in cooldown" {
		t.Fatalf("cooldown classification was lost: status=%d msg=%q err=%v", status, msg, err)
	}
	if rateLimitedCalls != 1 || readFailureCalls != 1 {
		t.Fatalf("expected both primary keys once, got rate-limited=%d read-failure=%d", rateLimitedCalls, readFailureCalls)
	}
}

// Direct dispatch already advances when a Responses reply cannot be translated.
// Managed dispatch must preserve the same declared-fallback guarantee instead of
// turning a failed Responses object into a terminal adapter error.
func TestManagedDispatchAdvancesToFallbackOnResponsesAdapterError(t *testing.T) {
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	primaryCalls := 0
	primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		primaryCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_1","object":"response","status":"failed",
			"error":{"code":"server_error","message":"generation failed"},"output":[]
		}`))
	}))
	defer primary.Close()

	fallbackCalls := 0
	fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","choices":[{"message":{"role":"assistant","content":"fallback answered"}}]}`))
	}))
	defer fallback.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: primary.URL + "/v1", APIKey: "sk-openai", Auth: "bearer",
	})
	reg.Set("openrouter", &provider.Provider{
		Name: "openrouter", BaseURL: fallback.URL + "/v1", APIKey: "sk-or", Auth: "bearer",
	})
	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "openai/gpt-5.6-terra"},
			{Slot: "fallback", Ref: "openrouter/gpt-4o-mini"},
		},
	}

	var logs bytes.Buffer
	h := NewHandler(reg, stubContextLoaderWithToolsAndPolicy("weston", "weston:dummy123",
		managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", ""), policy),
		logging.New(&logs))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5.6-terra","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer weston:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected fallback 200, got %d: %s", w.Code, w.Body.String())
	}
	if primaryCalls != 1 || fallbackCalls != 1 {
		t.Fatalf("expected one primary and one fallback dispatch, got primary=%d fallback=%d", primaryCalls, fallbackCalls)
	}
	if !strings.Contains(logs.String(), "responses_adapter_error") {
		t.Fatalf("expected responses_adapter_error fallback reason, got %s", logs.String())
	}
}
