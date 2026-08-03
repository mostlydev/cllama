package proxy

import (
	"bytes"
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
