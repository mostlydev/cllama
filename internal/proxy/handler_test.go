package proxy

import (
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

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

func TestHandlerNormalizesBareModelAgainstPolicy(t *testing.T) {
	var logs bytes.Buffer
	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: backend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "xai/grok-4.1-fast"},
		},
	}
	h := NewHandler(reg, stubContextLoaderWithPolicy("weston", "weston:dummy123", policy), logging.New(&logs))
	body := `{"model":"grok-4.1-fast","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer weston:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal backend body: %v", err)
	}
	if payload["model"] != "grok-4.1-fast" {
		t.Fatalf("expected normalized upstream model grok-4.1-fast, got %#v", payload["model"])
	}
	assertInterventionLogged(t, logs.Bytes(), "bare_model_normalized")
}

func TestHandlerFailsoverToDeclaredFallbackAndRebuildsBodyPerCandidate(t *testing.T) {
	var logs bytes.Buffer
	primaryBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "5")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"error":{"message":"rate limit exceeded"}}`))
	}))
	defer primaryBackend.Close()

	var fallbackBody []byte
	fallbackBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		fallbackBody, err = io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read fallback body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","choices":[{"message":{"content":"fallback"}}]}`))
	}))
	defer fallbackBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: primaryBackend.URL + "/v1", APIKey: "sk-openai", Auth: "bearer",
	})
	reg.Set("openrouter", &provider.Provider{
		Name: "openrouter", BaseURL: fallbackBackend.URL + "/v1", APIKey: "sk-or", Auth: "bearer",
	})

	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "openai/gpt-4o"},
			{Slot: "fallback", Ref: "openrouter/anthropic/claude-haiku-4-5"},
		},
	}
	h := NewHandler(reg, stubContextLoaderWithPolicy("weston", "weston:dummy123", policy), logging.New(&logs))
	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer weston:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 after declared fallback, got %d: %s", w.Code, w.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(fallbackBody, &payload); err != nil {
		t.Fatalf("unmarshal fallback body: %v", err)
	}
	if payload["model"] != "anthropic/claude-haiku-4-5" {
		t.Fatalf("expected fallback model in rebuilt body, got %#v", payload["model"])
	}
	assertInterventionLogged(t, logs.Bytes(), "provider_exhausted_failover")
}

func TestHandlerAnthropicPolicyClampsMissingModelToPrimary(t *testing.T) {
	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "anthropic/claude-sonnet-4-20250514"},
		},
	}
	h := NewHandler(reg, stubContextLoaderWithPolicy("nano-bot", "nano-bot:dummy456", policy), nil)
	body := `{"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("x-api-key", "nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal backend body: %v", err)
	}
	if payload["model"] != "claude-sonnet-4-20250514" {
		t.Fatalf("expected anthropic upstream model, got %#v", payload["model"])
	}
}

func TestHandlerAnthropicNoPolicyStripsProviderPrefix(t *testing.T) {
	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
	body := `{"model":"anthropic/claude-sonnet-4-20250514","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("x-api-key", "nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal backend body: %v", err)
	}
	if payload["model"] != "claude-sonnet-4-20250514" {
		t.Fatalf("expected stripped anthropic model, got %#v", payload["model"])
	}
}

func TestHandlerAnthropicPolicyRejectsUnsupportedProviderBridge(t *testing.T) {
	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: "https://api.anthropic.com/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "openrouter/anthropic/claude-sonnet-4"},
		},
	}
	h := NewHandler(reg, stubContextLoaderWithPolicy("nano-bot", "nano-bot:dummy456", policy), nil)
	body := `{"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("x-api-key", "nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "anthropic-compatible") {
		t.Fatalf("expected clear anthropic compatibility error, got %s", w.Body.String())
	}
}

func TestHandlerMissingModelClampsToDefaultAndLogsIntervention(t *testing.T) {
	var logs bytes.Buffer
	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "openai/gpt-4o"},
		},
	}
	h := NewHandler(reg, stubContextLoaderWithPolicy("weston", "weston:dummy123", policy), logging.New(&logs))
	body := `{"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer weston:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal backend body: %v", err)
	}
	if payload["model"] != "gpt-4o" {
		t.Fatalf("expected default upstream model gpt-4o, got %#v", payload["model"])
	}
	assertInterventionLogged(t, logs.Bytes(), "missing")
}

func TestHandlerDoesNotAdvanceCandidatesOnUpstream500(t *testing.T) {
	var fallbackCalls int
	primaryBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"upstream exploded"}}`))
	}))
	defer primaryBackend.Close()

	fallbackBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fallbackCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","choices":[{"message":{"content":"fallback"}}]}`))
	}))
	defer fallbackBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: primaryBackend.URL + "/v1", APIKey: "sk-openai", Auth: "bearer",
	})
	reg.Set("openrouter", &provider.Provider{
		Name: "openrouter", BaseURL: fallbackBackend.URL + "/v1", APIKey: "sk-or", Auth: "bearer",
	})

	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "openai/gpt-4o"},
			{Slot: "fallback", Ref: "openrouter/anthropic/claude-haiku-4-5"},
		},
	}
	h := NewHandler(reg, stubContextLoaderWithPolicy("weston", "weston:dummy123", policy), nil)
	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer weston:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Fatalf("expected upstream 500 to pass through, got %d: %s", w.Code, w.Body.String())
	}
	if fallbackCalls != 0 {
		t.Fatalf("expected no fallback calls on upstream 500, got %d", fallbackCalls)
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

func TestHandlerRecordsAnthropicJSONCost(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"id":"msg_01",
			"type":"message",
			"usage":{"input_tokens":180,"output_tokens":60}
		}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	acc := cost.NewAccumulator()
	pricing := cost.DefaultPricing()
	h := NewHandler(reg, stubContextLoaderWithToken("nano-bot", "nano-bot:dummy456"), logging.New(io.Discard),
		WithCostTracking(acc, pricing))

	body := `{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	entries := acc.ByAgent("nano-bot")
	if len(entries) != 1 {
		t.Fatalf("expected one cost entry, got %d", len(entries))
	}
	if entries[0].TotalInputTokens != 180 || entries[0].TotalOutputTokens != 60 {
		t.Fatalf("unexpected token counts: %+v", entries[0])
	}
	if entries[0].TotalCostUSD <= 0 {
		t.Fatalf("expected positive cost, got %+v", entries[0])
	}
	if entries[0].PricedRequests != 1 || entries[0].UnpricedRequests != 0 {
		t.Fatalf("unexpected pricing coverage counts: %+v", entries[0])
	}
}

func TestHandlerRecordsAnthropicSSECost(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Write([]byte("event: message_start\n"))
		w.Write([]byte("data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":140}}}\n\n"))
		w.Write([]byte("event: message_delta\n"))
		w.Write([]byte("data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":45}}\n\n"))
		w.Write([]byte("event: message_stop\n"))
		w.Write([]byte("data: {\"type\":\"message_stop\"}\n\n"))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	acc := cost.NewAccumulator()
	pricing := cost.DefaultPricing()
	h := NewHandler(reg, stubContextLoaderWithToken("nano-bot", "nano-bot:dummy456"), logging.New(io.Discard),
		WithCostTracking(acc, pricing))

	body := `{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	entries := acc.ByAgent("nano-bot")
	if len(entries) != 1 {
		t.Fatalf("expected one cost entry, got %d", len(entries))
	}
	if entries[0].TotalInputTokens != 140 || entries[0].TotalOutputTokens != 45 {
		t.Fatalf("unexpected token counts: %+v", entries[0])
	}
	if entries[0].TotalCostUSD <= 0 {
		t.Fatalf("expected positive cost, got %+v", entries[0])
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

func TestHandlerUsesProviderReportedCostWhenAvailable(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"id":"chatcmpl-1",
			"usage":{"prompt_tokens":100,"completion_tokens":25,"cost":0.1234}
		}`))
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

	body := `{"model":"openrouter/google/nonexistent-model","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer pc-roll:dummy123")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	entries := acc.ByAgent("pc-roll")
	if len(entries) != 1 {
		t.Fatalf("expected one cost entry, got %d", len(entries))
	}
	if entries[0].TotalCostUSD != 0.1234 {
		t.Fatalf("expected provider-reported cost 0.1234, got %+v", entries[0])
	}
	if entries[0].PricedRequests != 1 || entries[0].UnpricedRequests != 0 {
		t.Fatalf("unexpected pricing coverage counts: %+v", entries[0])
	}
}

func TestHandlerMarksUnknownPricingWithoutDroppingUsage(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{
			"id":"chatcmpl-1",
			"usage":{"prompt_tokens":240,"completion_tokens":90,"total_tokens":330}
		}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL, APIKey: "sk-real", Auth: "bearer",
	})

	acc := cost.NewAccumulator()
	pricing := cost.DefaultPricing()
	h := NewHandler(reg, stubContextLoaderWithToken("pc-roll", "pc-roll:dummy123"), logging.New(io.Discard),
		WithCostTracking(acc, pricing))

	body := `{"model":"openai/nonexistent-model","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer pc-roll:dummy123")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	entries := acc.ByAgent("pc-roll")
	if len(entries) != 1 {
		t.Fatalf("expected one cost entry, got %d", len(entries))
	}
	if entries[0].TotalInputTokens != 240 || entries[0].TotalOutputTokens != 90 {
		t.Fatalf("unexpected token counts: %+v", entries[0])
	}
	if entries[0].TotalCostUSD != 0 {
		t.Fatalf("expected zero known cost for unknown pricing, got %+v", entries[0])
	}
	if entries[0].PricedRequests != 0 || entries[0].UnpricedRequests != 1 {
		t.Fatalf("unexpected pricing coverage counts: %+v", entries[0])
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

func TestHandlerAnthropicAcceptsIncomingXAPIKeyAgentAuth(t *testing.T) {
	var gotAuth, gotAPIKey string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotAPIKey = r.Header.Get("x-api-key")
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
	req.Header.Set("x-api-key", "nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if gotAPIKey != "sk-ant-real" {
		t.Errorf("expected backend x-api-key=sk-ant-real, got %q", gotAPIKey)
	}
	if gotAuth != "" {
		t.Errorf("expected no backend Authorization header for anthropic auth, got %q", gotAuth)
	}
}

func TestHandlerInjectsFeedsIntoOpenAI(t *testing.T) {
	feedSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("Wallet: $5,000 cash | $20,000 invested"))
	}))
	defer feedSrv.Close()

	ctxDir := t.TempDir()
	agentDir := filepath.Join(ctxDir, "weston")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	feedsJSON := fmt.Sprintf(`[{"name":"market-context","source":"trading-api","path":"/api/v1/market_context/weston","ttl":300,"url":"%s"}]`, feedSrv.URL)
	if err := os.WriteFile(filepath.Join(agentDir, "feeds.json"), []byte(feedsJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "AGENTS.md"), []byte("# C"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "CLAWDAPUS.md"), []byte("# I"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "metadata.json"), []byte(`{"token":"weston:secret","pod":"test-pod","timezone":"America/New_York"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	contextLoader := func(agentID string) (*agentctx.AgentContext, error) {
		return agentctx.Load(ctxDir, agentID)
	}

	h := NewHandler(reg, contextLoader, nil, WithFeeds("test-pod"))
	body := `{"model":"openrouter/anthropic/claude-sonnet-4","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer weston:secret")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	messages, _ := payload["messages"].([]any)
	if len(messages) < 2 {
		t.Fatalf("expected >=2 messages (feed + user), got %d", len(messages))
	}
	first := messages[0].(map[string]any)
	content, _ := first["content"].(string)
	if !strings.HasPrefix(content, "Current time: ") {
		t.Errorf("expected current time at top of system content, got: %s", content)
	}
	if !strings.Contains(content, "Wallet: $5,000") {
		t.Errorf("expected feed content in first message, got: %s", content)
	}
	if !strings.Contains(content, "BEGIN FEED: market-context") {
		t.Errorf("expected feed delimiter, got: %s", content)
	}
}

func TestHandlerInjectsFeedsIntoAnthropic(t *testing.T) {
	feedSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("Fleet nominal"))
	}))
	defer feedSrv.Close()

	ctxDir := t.TempDir()
	agentDir := filepath.Join(ctxDir, "nano-bot")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	feedsJSON := fmt.Sprintf(`[{"name":"alerts","source":"claw-api","path":"/fleet/alerts","ttl":30,"url":"%s"}]`, feedSrv.URL)
	if err := os.WriteFile(filepath.Join(agentDir, "feeds.json"), []byte(feedsJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "AGENTS.md"), []byte("# C"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "CLAWDAPUS.md"), []byte("# I"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "metadata.json"), []byte(`{"token":"nano-bot:secret456","timezone":"America/New_York"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	contextLoader := func(agentID string) (*agentctx.AgentContext, error) {
		return agentctx.Load(ctxDir, agentID)
	}

	h := NewHandler(reg, contextLoader, nil, WithFeeds("test-pod"))
	body := `{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer nano-bot:secret456")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	sys, _ := payload["system"].(string)
	if !strings.HasPrefix(sys, "Current time: ") {
		t.Errorf("expected current time at top of system content, got: %q", sys)
	}
	if !strings.Contains(sys, "Fleet nominal") {
		t.Errorf("expected feed in system field, got: %q", sys)
	}
}

func TestHandlerNoFeedsStillWorks(t *testing.T) {
	ctxDir := t.TempDir()
	agentDir := filepath.Join(ctxDir, "bare-agent")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "AGENTS.md"), []byte("# C"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "CLAWDAPUS.md"), []byte("# I"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "metadata.json"), []byte(`{"token":"bare-agent:secret","timezone":"UTC"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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

	contextLoader := func(agentID string) (*agentctx.AgentContext, error) {
		return agentctx.Load(ctxDir, agentID)
	}

	h := NewHandler(reg, contextLoader, nil, WithFeeds("test-pod"))
	body := `{"model":"openrouter/anthropic/claude-sonnet-4","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer bare-agent:secret")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	messages, _ := payload["messages"].([]any)
	if len(messages) != 2 {
		t.Errorf("expected 2 messages (time + user), got %d", len(messages))
	}
	first := messages[0].(map[string]any)
	if first["role"] != "system" {
		t.Fatalf("expected first message to be system, got %q", first["role"])
	}
	content, _ := first["content"].(string)
	if !strings.HasPrefix(content, "Current time: ") {
		t.Errorf("expected current time system message, got: %q", content)
	}
}

// -- failure classification and retry tests ------------------------------------

func TestHandlerMarksKeyDeadOn401AndFallsBack(t *testing.T) {
	callCount := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		if callCount == 1 {
			w.WriteHeader(http.StatusUnauthorized)
			w.Write([]byte(`{"error":{"message":"invalid key"}}`))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"id":"chatcmpl-2","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer backend.Close()

	dir := t.TempDir()
	reg := provider.NewRegistry(dir)
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-dead", Auth: "bearer",
	})
	// Add a second key via AddRuntimeKey.
	secondID, err := reg.AddRuntimeKey("openai", "backup", "sk-alive")
	if err != nil {
		t.Fatalf("AddRuntimeKey: %v", err)
	}
	// Activate the second key so SelectKey has a fallback after marking first dead.
	_ = reg.ActivateKey("openai", secondID)

	h := NewHandler(reg, stubContextLoaderWithToken("agent", "agent:tok"), nil)
	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer agent:tok")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200 after fallback, got %d: %s", w.Code, w.Body.String())
	}
	if callCount < 2 {
		t.Errorf("expected at least 2 backend calls (dead + retry), got %d", callCount)
	}
}

func TestHandlerReturns429WhenAllKeysCooling(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Retry-After", "5")
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"rate limit exceeded"}}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-limited", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("agent", "agent:tok"), nil)
	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer agent:tok")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	// Only one key; after cooldown, SelectKey returns CooldownError → 503.
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("expected 503 when all keys cooling, got %d: %s", w.Code, w.Body.String())
	}
}

func TestHandlerMarksDeadOn429QuotaExhausted(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		w.Write([]byte(`{"error":{"message":"You exceeded your current quota, please check your plan"}}`))
	}))
	defer backend.Close()

	dir := t.TempDir()
	reg := provider.NewRegistry(dir)
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-quota-gone", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("agent", "agent:tok"), nil)
	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer agent:tok")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	// Quota-429 marks key dead; single key → no usable keys → 502.
	if w.Code != http.StatusBadGateway {
		t.Errorf("expected 502 when quota key is dead, got %d: %s", w.Code, w.Body.String())
	}

	// Verify the key is actually dead in the registry.
	all := reg.All()
	state, ok := all["openai"]
	if !ok {
		t.Fatal("openai not in registry after quota exhaustion")
	}
	if len(state.Keys) == 0 {
		t.Fatal("no keys in openai pool")
	}
	if state.Keys[0].State != provider.KeyStateDead {
		t.Errorf("expected key state=dead, got %q", state.Keys[0].State)
	}
}

func TestClassifyResponse(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
		want   responseClass
	}{
		{"401 → auth", 401, "", classAuth},
		{"403 → auth", 403, "", classAuth},
		{"402 → auth", 402, "", classAuth},
		{"429 rate-limit → cooldown", 429, `{"error":"rate limited"}`, classRateLimit},
		{"429 quota → auth", 429, `exceeded your current quota`, classAuth},
		{"429 insufficient_quota → auth", 429, `{"error":{"code":"insufficient_quota"}}`, classAuth},
		{"200 → ok", 200, "", classOK},
		{"500 → ok", 500, "", classOK},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := &http.Response{
				StatusCode: tc.status,
				Body:       io.NopCloser(strings.NewReader(tc.body)),
			}
			got := classifyResponse(resp)
			if got != tc.want {
				t.Errorf("classifyResponse(%d, %q) = %v, want %v", tc.status, tc.body, got, tc.want)
			}
		})
	}
}

func TestParseCooldownDuration(t *testing.T) {
	cases := []struct {
		retryAfter string
		wantSecs   int
	}{
		{"", 10},
		{"30", 30},
		{"120", 60}, // capped at 60
		{"abc", 10}, // invalid → default
	}
	for _, tc := range cases {
		hdr := http.Header{}
		if tc.retryAfter != "" {
			hdr.Set("Retry-After", tc.retryAfter)
		}
		resp := &http.Response{Header: hdr}
		got := parseCooldownDuration(resp)
		wantNs := int64(tc.wantSecs) * int64(time.Second)
		if int64(got) != wantNs {
			t.Errorf("Retry-After=%q: got %v, want %ds", tc.retryAfter, got, tc.wantSecs)
		}
	}
}

func TestHandlerRecordsSessionHistoryJSON(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hello"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	histDir := t.TempDir()
	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard),
		WithSessionHistory(histDir))

	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	histFile := filepath.Join(histDir, "tiverton", "history.jsonl")
	data, err := os.ReadFile(histFile)
	if err != nil {
		t.Fatalf("expected history file to exist: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("history file is empty")
	}

	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimRight(data, "\n"), &entry); err != nil {
		t.Fatalf("unmarshal history entry: %v", err)
	}
	if entry["claw_id"] != "tiverton" {
		t.Errorf("expected claw_id=tiverton, got %v", entry["claw_id"])
	}
	if entry["requested_model"] != "openai/gpt-4o" {
		t.Errorf("expected requested_model=openai/gpt-4o, got %v", entry["requested_model"])
	}
	resp, _ := entry["response"].(map[string]any)
	if resp == nil {
		t.Fatal("expected response field in entry")
	}
	if resp["format"] != "json" {
		t.Errorf("expected response.format=json, got %v", resp["format"])
	}
	if entry["ts"] == "" {
		t.Error("expected TS to be populated")
	}
}

func TestHandlerRecordsSessionHistorySSE(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	histDir := t.TempDir()
	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard),
		WithSessionHistory(histDir))

	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	histFile := filepath.Join(histDir, "tiverton", "history.jsonl")
	data, err := os.ReadFile(histFile)
	if err != nil {
		t.Fatalf("expected history file to exist: %v", err)
	}
	if len(data) == 0 {
		t.Fatal("history file is empty")
	}

	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimRight(data, "\n"), &entry); err != nil {
		t.Fatalf("unmarshal history entry: %v", err)
	}
	resp, _ := entry["response"].(map[string]any)
	if resp == nil {
		t.Fatal("expected response field in entry")
	}
	if resp["format"] != "sse" {
		t.Errorf("expected response.format=sse, got %v", resp["format"])
	}
	text, _ := resp["text"].(string)
	if text == "" {
		t.Errorf("expected non-empty response.text for SSE entry")
	}
}

func TestHandlerSkipsSessionHistoryOnUpstreamError(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":{"message":"internal error"}}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	histDir := t.TempDir()
	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard),
		WithSessionHistory(histDir))

	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	histFile := filepath.Join(histDir, "tiverton", "history.jsonl")
	if _, err := os.Stat(histFile); err == nil {
		t.Fatal("expected no history file created for upstream 500 response")
	}
}

func TestHandlerInjectsMemoryRecallIntoOpenAIRequests(t *testing.T) {
	var logs lockedBuffer
	var gotBody []byte
	var recallPayload map[string]any
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var err error
		gotBody, err = io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read backend body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer backend.Close()

	memorySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/recall" {
			t.Fatalf("unexpected memory path: %s", r.URL.Path)
		}
		if err := json.NewDecoder(r.Body).Decode(&recallPayload); err != nil {
			t.Fatalf("decode recall payload: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memories":[{"text":"User prefers concise answers","kind":"profile","source":"profile-store"}]}`))
	}))
	defer memorySrv.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, func(id string) (*agentctx.AgentContext, error) {
		if id != "tiverton" {
			return nil, io.EOF
		}
		return &agentctx.AgentContext{
			AgentID: "tiverton",
			Metadata: map[string]any{
				"token":    "tiverton:dummy123",
				"pod":      "ops",
				"service":  "tiverton",
				"type":     "openclaw",
				"timezone": "America/New_York",
			},
			Memory: &agentctx.MemoryManifest{
				Version: 1,
				Service: "team-memory",
				BaseURL: memorySrv.URL,
				Recall:  &agentctx.MemoryOp{Path: "/recall", TimeoutMS: 300},
			},
		}, nil
	}, logging.New(&logs))

	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal backend body: %v", err)
	}
	messages := payload["messages"].([]any)
	if len(messages) < 2 {
		t.Fatalf("expected injected system message, got %+v", payload)
	}
	first := messages[0].(map[string]any)
	if first["role"] != "system" {
		t.Fatalf("expected leading system message, got %+v", first)
	}
	content := first["content"].(string)
	if !strings.Contains(content, "User prefers concise answers") {
		t.Fatalf("expected recalled memory in injected content, got %q", content)
	}
	metadata, _ := recallPayload["metadata"].(map[string]any)
	if metadata == nil {
		t.Fatalf("expected recall metadata, got %+v", recallPayload)
	}
	if metadata["timezone"] != "America/New_York" {
		t.Fatalf("expected recall timezone metadata, got %+v", metadata)
	}
	if metadata["requested_model"] != "openai/gpt-4o" {
		t.Fatalf("expected requested_model metadata, got %+v", metadata)
	}
	recallLog := waitForLogEntry(t, &logs, func(entry map[string]any) bool {
		return entry["type"] == "memory_op" && entry["memory_op"] == "recall" && entry["memory_status"] == "succeeded"
	})
	if recallLog["memory_service"] != "team-memory" {
		t.Fatalf("expected recall telemetry memory_service=team-memory, got %+v", recallLog)
	}
	if recallLog["memory_blocks"].(float64) != 1 {
		t.Fatalf("expected recall telemetry memory_blocks=1, got %+v", recallLog)
	}
	if recallLog["memory_bytes"].(float64) <= 0 {
		t.Fatalf("expected recall telemetry memory_bytes > 0, got %+v", recallLog)
	}
}

func TestHandlerRetainsMemoryAfterSuccessfulTurn(t *testing.T) {
	var logs lockedBuffer
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hello"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`))
	}))
	defer backend.Close()

	retainCh := make(chan map[string]any, 1)
	memorySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/retain" {
			t.Fatalf("unexpected memory path: %s", r.URL.Path)
		}
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode retain payload: %v", err)
		}
		retainCh <- payload
		w.WriteHeader(http.StatusNoContent)
	}))
	defer memorySrv.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, func(id string) (*agentctx.AgentContext, error) {
		if id != "tiverton" {
			return nil, io.EOF
		}
		return &agentctx.AgentContext{
			AgentID: "tiverton",
			Metadata: map[string]any{
				"token":   "tiverton:dummy123",
				"pod":     "ops",
				"service": "tiverton",
				"type":    "openclaw",
			},
			Memory: &agentctx.MemoryManifest{
				Version: 1,
				Service: "team-memory",
				BaseURL: memorySrv.URL,
				Retain:  &agentctx.MemoryOp{Path: "/retain"},
			},
		}, nil
	}, logging.New(&logs))

	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}

	select {
	case payload := <-retainCh:
		if payload["agent_id"] != "tiverton" {
			t.Fatalf("unexpected retain payload: %+v", payload)
		}
		entry := payload["entry"].(map[string]any)
		if entry["claw_id"] != "tiverton" {
			t.Fatalf("unexpected retained entry: %+v", payload)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for retain request")
	}
	retainLog := waitForLogEntry(t, &logs, func(entry map[string]any) bool {
		return entry["type"] == "memory_op" && entry["memory_op"] == "retain" && entry["memory_status"] == "succeeded"
	})
	if retainLog["memory_service"] != "team-memory" {
		t.Fatalf("expected retain telemetry memory_service=team-memory, got %+v", retainLog)
	}
	if retainLog["status_code"].(float64) != http.StatusNoContent {
		t.Fatalf("expected retain telemetry status_code=%d, got %+v", http.StatusNoContent, retainLog)
	}
}

func TestHandlerLogsOversizedMemoryRecallResponseClearly(t *testing.T) {
	var logs bytes.Buffer
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer backend.Close()

	memorySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"memories":[{"text":"`+strings.Repeat("x", maxMemoryBlockBytes)+`"}]}`)
	}))
	defer memorySrv.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, func(id string) (*agentctx.AgentContext, error) {
		if id != "tiverton" {
			return nil, io.EOF
		}
		return &agentctx.AgentContext{
			AgentID: "tiverton",
			Metadata: map[string]any{
				"token": "tiverton:dummy123",
			},
			Memory: &agentctx.MemoryManifest{
				Version: 1,
				Service: "team-memory",
				BaseURL: memorySrv.URL,
				Recall:  &agentctx.MemoryOp{Path: "/recall", TimeoutMS: 300},
			},
		}, nil
	}, logging.New(&logs))

	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}
	entries := parseLogEntries(t, logs.Bytes())
	found := false
	for _, entry := range entries {
		if entry["type"] == "error" {
			if errText, _ := entry["error"].(string); strings.Contains(errText, "recall response exceeds") {
				found = true
				break
			}
		}
	}
	if !found {
		t.Fatalf("expected oversized recall error log, got %v", entries)
	}
}

func stubContextLoaderWithToken(agentID, token string) ContextLoader {
	return func(id string) (*agentctx.AgentContext, error) {
		if id != agentID {
			return nil, io.EOF
		}
		return &agentctx.AgentContext{
			AgentID:     id,
			ContextDir:  "/claw/context/" + id,
			AgentsMD:    []byte("# Contract"),
			ClawdapusMD: []byte("# Infra"),
			Metadata: map[string]any{
				"token": token,
			},
		}, nil
	}
}

func stubContextLoaderWithPolicy(agentID, token string, policy *agentctx.ModelPolicy) ContextLoader {
	return func(id string) (*agentctx.AgentContext, error) {
		if id != agentID {
			return nil, io.EOF
		}
		return &agentctx.AgentContext{
			AgentID:     id,
			ContextDir:  "/claw/context/" + id,
			AgentsMD:    []byte("# Contract"),
			ClawdapusMD: []byte("# Infra"),
			Metadata: map[string]any{
				"token": token,
			},
			ModelPolicy: policy,
		}, nil
	}
}

type lockedBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *lockedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *lockedBuffer) Bytes() []byte {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]byte(nil), b.buf.Bytes()...)
}

func waitForLogEntry(t *testing.T, logs *lockedBuffer, predicate func(map[string]any) bool) map[string]any {
	t.Helper()
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) {
		entries := parseLogEntries(t, logs.Bytes())
		for _, entry := range entries {
			if predicate(entry) {
				return entry
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatal("timed out waiting for log entry")
	return nil
}

func assertInterventionLogged(t *testing.T, raw []byte, reason string) {
	t.Helper()
	entries := parseLogEntries(t, raw)
	for _, entry := range entries {
		if entry["type"] == "intervention" && entry["intervention"] == reason {
			return
		}
	}
	t.Fatalf("expected intervention %q in logs, got %v", reason, entries)
}

func parseLogEntries(t *testing.T, raw []byte) []map[string]any {
	t.Helper()
	lines := bytes.Split(bytes.TrimSpace(raw), []byte("\n"))
	entries := make([]map[string]any, 0, len(lines))
	for _, line := range lines {
		if len(bytes.TrimSpace(line)) == 0 {
			continue
		}
		var entry map[string]any
		if err := json.Unmarshal(line, &entry); err != nil {
			t.Fatalf("unmarshal log entry: %v\nraw: %s", err, string(line))
		}
		entries = append(entries, entry)
	}
	return entries
}
