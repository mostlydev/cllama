package proxy

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/cost"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

func TestHandlerForwardsAndSwapsAuth(t *testing.T) {
	var gotAuth string
	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		var err error
		gotBody, err = io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), nil)
	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if gotAuth != "Bearer sk-real" {
		t.Errorf("expected real key forwarded, got %q", gotAuth)
	}
	if len(gotBody) == 0 {
		t.Fatal("backend received empty body")
	}
	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal backend body: %v", err)
	}
	if payload["model"] != "gpt-4o" {
		t.Errorf("expected stripped model gpt-4o, got %#v", payload["model"])
	}
}

func TestHandlerRejectsMissingBearer(t *testing.T) {
	h := NewHandler(provider.NewRegistry(""), nil, nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401, got %d", w.Code)
	}
}

func TestHandlerRejectsUnknownProvider(t *testing.T) {
	reg := provider.NewRegistry("")
	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), nil)
	body := `{"model":"unknown-provider/model","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusBadGateway {
		t.Errorf("expected 502 for unknown provider, got %d", w.Code)
	}
}

func TestHandlerFallsBackToOpenRouterForVendorPrefixedModel(t *testing.T) {
	var gotAuth string
	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		var err error
		gotBody, err = io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openrouter", &provider.Provider{
		Name: "openrouter", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), nil)
	body := `{"model":"anthropic/claude-sonnet-4","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if gotAuth != "Bearer sk-real" {
		t.Fatalf("expected openrouter auth forwarded, got %q", gotAuth)
	}
	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal backend body: %v", err)
	}
	if payload["model"] != "anthropic/claude-sonnet-4" {
		t.Fatalf("expected vendor-prefixed model preserved for openrouter fallback, got %#v", payload["model"])
	}
}

func TestHandlerRejectsWrongSecret(t *testing.T) {
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: "https://api.openai.com/v1", APIKey: "sk-real", Auth: "bearer"})
	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:correct-secret"), nil)
	body := `{"model":"openai/gpt-4o","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:wrong-secret")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusForbidden {
		t.Errorf("expected 403 for wrong secret, got %d", w.Code)
	}
}

func TestHandlerRecordsCost(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"id": "chatcmpl-1",
			"choices": [{"message": {"content": "hello"}}],
			"usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
		}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openrouter", &provider.Provider{
		Name: "openrouter", BaseURL: backend.URL, APIKey: "sk-real", Auth: "bearer",
	})

	acc := cost.NewAccumulator()
	pricing := cost.DefaultPricing()
	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard),
		WithCostTracking(acc, pricing))

	body := `{"model":"openrouter/anthropic/claude-sonnet-4","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	entries := acc.ByAgent("tiverton")
	if len(entries) == 0 {
		t.Fatal("expected cost entry recorded")
	}
	if entries[0].TotalInputTokens != 100 {
		t.Errorf("expected 100 input tokens, got %d", entries[0].TotalInputTokens)
	}
	if entries[0].TotalOutputTokens != 50 {
		t.Errorf("expected 50 output tokens, got %d", entries[0].TotalOutputTokens)
	}
	if entries[0].TotalCostUSD <= 0 {
		t.Error("expected positive cost")
	}
}

func TestHandlerRecordsCostFromSSE(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		w.Write([]byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":200,\"completion_tokens\":80,\"total_tokens\":280}}\n\n"))
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL, APIKey: "sk-real", Auth: "bearer",
	})

	acc := cost.NewAccumulator()
	pricing := cost.DefaultPricing()
	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard),
		WithCostTracking(acc, pricing))

	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	entries := acc.ByAgent("tiverton")
	if len(entries) == 0 {
		t.Fatal("expected cost entry recorded for SSE response")
	}
	if entries[0].TotalInputTokens != 200 {
		t.Errorf("expected 200 input tokens, got %d", entries[0].TotalInputTokens)
	}
	if entries[0].TotalCostUSD <= 0 {
		t.Error("expected positive cost")
	}
}

func TestHandlerEnablesUsageForStreamingChatCompletions(t *testing.T) {
	var sawIncludeUsage bool
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()

		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode upstream request: %v", err)
		}
		if opts, ok := payload["stream_options"].(map[string]any); ok {
			if includeUsage, ok := opts["include_usage"].(bool); ok && includeUsage {
				sawIncludeUsage = true
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n"))
		if sawIncludeUsage {
			w.Write([]byte("data: {\"choices\":[],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":30,\"total_tokens\":150}}\n\n"))
		}
		w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openrouter", &provider.Provider{
		Name: "openrouter", BaseURL: backend.URL, APIKey: "sk-real", Auth: "bearer",
	})

	acc := cost.NewAccumulator()
	pricing := cost.DefaultPricing()
	h := NewHandler(reg, stubContextLoaderWithToken("pc-roll", "pc-roll:dummy123"), logging.New(io.Discard),
		WithCostTracking(acc, pricing))

	body := `{"model":"openrouter/anthropic/claude-sonnet-4","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer pc-roll:dummy123")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !sawIncludeUsage {
		t.Fatal("expected include_usage=true in upstream streaming request")
	}

	entries := acc.ByAgent("pc-roll")
	if len(entries) != 1 {
		t.Fatalf("expected one cost entry, got %d", len(entries))
	}
	if entries[0].TotalInputTokens != 120 || entries[0].TotalOutputTokens != 30 {
		t.Fatalf("unexpected token counts: %+v", entries[0])
	}
	if entries[0].TotalCostUSD <= 0 {
		t.Fatalf("expected positive cost, got %+v", entries[0])
	}
}

func TestHandlerRecordsCostFromGzipJSONResponse(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Content-Encoding", "gzip")

		zw := gzip.NewWriter(w)
		_, err := zw.Write([]byte(`{
			"id": "chatcmpl-1",
			"choices": [{"message": {"content": "hello"}}],
			"usage": {"prompt_tokens": 90, "completion_tokens": 20, "total_tokens": 110}
		}`))
		if err != nil {
			t.Fatalf("gzip write: %v", err)
		}
		if err := zw.Close(); err != nil {
			t.Fatalf("gzip close: %v", err)
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openrouter", &provider.Provider{
		Name: "openrouter", BaseURL: backend.URL, APIKey: "sk-real", Auth: "bearer",
	})

	acc := cost.NewAccumulator()
	pricing := cost.DefaultPricing()
	h := NewHandler(reg, stubContextLoaderWithToken("pc-roll", "pc-roll:dummy123"), logging.New(io.Discard),
		WithCostTracking(acc, pricing))

	body := `{"model":"openrouter/anthropic/claude-sonnet-4","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer pc-roll:dummy123")
	req.Header.Set("Accept-Encoding", "gzip")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	entries := acc.ByAgent("pc-roll")
	if len(entries) != 1 {
		t.Fatalf("expected one cost entry, got %d", len(entries))
	}
	if entries[0].TotalInputTokens != 90 || entries[0].TotalOutputTokens != 20 {
		t.Fatalf("unexpected token counts: %+v", entries[0])
	}
	if entries[0].TotalCostUSD <= 0 {
		t.Fatalf("expected positive cost, got %+v", entries[0])
	}
}

func TestHandlerForwardsAnthropicMessages(t *testing.T) {
	var gotAPIKey string
	var gotVersion string
	var gotBeta string
	var gotAuth string
	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAPIKey = r.Header.Get("X-Api-Key")
		gotVersion = r.Header.Get("Anthropic-Version")
		gotBeta = r.Header.Get("Anthropic-Beta")
		gotAuth = r.Header.Get("Authorization")
		var err error
		gotBody, err = io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_01","type":"message","content":[{"type":"text","text":"hello"}]}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("nano-bot", "nano-bot:dummy456"), nil)
	body := `{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	req.Header.Set("Anthropic-Beta", "prompt-caching-2024-07-31")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	// Should use x-api-key, NOT bearer auth
	if gotAPIKey != "sk-ant-real" {
		t.Errorf("expected x-api-key=sk-ant-real, got %q", gotAPIKey)
	}
	if gotAuth != "" {
		t.Errorf("expected no Authorization header for x-api-key auth, got %q", gotAuth)
	}
	// Anthropic headers should be forwarded
	if gotVersion != "2023-06-01" {
		t.Errorf("expected anthropic-version=2023-06-01, got %q", gotVersion)
	}
	if gotBeta != "prompt-caching-2024-07-31" {
		t.Errorf("expected anthropic-beta forwarded, got %q", gotBeta)
	}
	// Model should NOT be provider-prefixed (Anthropic models have no prefix)
	if len(gotBody) == 0 {
		t.Fatal("backend received empty body")
	}
	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal backend body: %v", err)
	}
	if payload["model"] != "claude-sonnet-4-20250514" {
		t.Errorf("expected model unchanged, got %#v", payload["model"])
	}
}

func TestHandlerAnthropicRejectsUnknownAgent(t *testing.T) {
	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{Name: "anthropic", BaseURL: "https://api.anthropic.com/v1", APIKey: "sk-real", Auth: "x-api-key"})
	h := NewHandler(reg, stubContextLoaderWithToken("nano-bot", "nano-bot:correct"), nil)
	body := `{"model":"claude-sonnet-4-20250514","messages":[]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer unknown-agent:wrong-secret")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusForbidden {
		t.Errorf("expected 403 for unknown agent, got %d", w.Code)
	}
}

func stubContextLoaderWithToken(agentID, token string) ContextLoader {
	return func(id string) (*agentctx.AgentContext, error) {
		if id != agentID {
			return nil, io.EOF
		}
		return &agentctx.AgentContext{
			AgentID:     id,
			AgentsMD:    []byte("# Contract"),
			ClawdapusMD: []byte("# Infra"),
			Metadata: map[string]any{
				"token": token,
			},
		}, nil
	}
}
