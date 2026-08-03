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
	"github.com/mostlydev/cllama/internal/sessionhistory"
)

func terminalSuccessManifest(baseURL string) *agentctx.ToolManifest {
	manifest := managedToolManifestForURL(baseURL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")
	manifest.Tools[0].Annotations = map[string]any{managedToolTerminalOnSuccessAnnotation: true}
	return manifest
}

func terminalSuccessToolServer(t *testing.T, calls *int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		(*calls)++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"DENIED","receipt_id":"receipt-1"}`))
	}))
}

func TestOpenAITerminalOnSuccessStopsAfterAuthoritativeReceipt(t *testing.T) {
	toolCalls := 0
	toolSrv := terminalSuccessToolServer(t, &toolCalls)
	defer toolSrv.Close()

	modelCalls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		modelCalls++
		w.Header().Set("Content-Type", "application/json")
		switch modelCalls {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-action","model":"grok-4.1-fast",
				"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[
					{"id":"call_action","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}}
				]}}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}
			}`))
		case 2:
			_, _ = w.Write([]byte(`{"id":"chatcmpl-bad","choices":[{"message":{"role":"assistant","content":"Approved; proceed."}}]}`))
		default:
			t.Fatalf("unexpected model round %d", modelCalls)
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{Name: "xai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer"})
	var logs bytes.Buffer
	histDir := t.TempDir()
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", terminalSuccessManifest(toolSrv.URL)), logging.New(&logs), WithSessionHistory(histDir))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"xai/grok-4.1-fast","messages":[{"role":"user","content":"act"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("Content-Length"); got != "" {
		t.Fatalf("synthetic terminal retained stale upstream content-length %q", got)
	}
	if modelCalls != 1 || toolCalls != 1 {
		t.Fatalf("terminal success must stop after one model/tool round, got model=%d tool=%d", modelCalls, toolCalls)
	}
	var completion map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &completion); err != nil {
		t.Fatalf("unmarshal terminal: %v", err)
	}
	choices, _ := completion["choices"].([]any)
	if len(choices) != 1 {
		t.Fatalf("choices: %#v", completion["choices"])
	}
	choice, _ := choices[0].(map[string]any)
	message, _ := choice["message"].(map[string]any)
	if message["content"] != "" || choice["finish_reason"] != "stop" {
		t.Fatalf("expected protocol-valid empty terminal, got %#v", choice)
	}
	if strings.Contains(w.Body.String(), "Approved") {
		t.Fatalf("contradictory model terminal leaked: %s", w.Body.String())
	}
	assertInterventionLogged(t, logs.Bytes(), "managed_tool_terminal_on_success:trading-api.get_market_context")

	entries, err := sessionhistory.ReadEntries(histDir, "tiverton", nil, 10)
	if err != nil {
		t.Fatalf("ReadEntries: %v", err)
	}
	if len(entries) != 1 || len(entries[0].ToolTrace) != 1 || len(entries[0].ToolTrace[0].ToolCalls) != 1 {
		t.Fatalf("expected complete terminal receipt trace, got %+v", entries)
	}
	if !bytes.Contains(entries[0].ToolTrace[0].ToolCalls[0].Result, []byte(`"receipt_id":"receipt-1"`)) {
		t.Fatalf("receipt missing from tool trace: %s", entries[0].ToolTrace[0].ToolCalls[0].Result)
	}
}

func TestAnthropicTerminalOnSuccessStopsAfterAuthoritativeReceipt(t *testing.T) {
	toolCalls := 0
	toolSrv := terminalSuccessToolServer(t, &toolCalls)
	defer toolSrv.Close()

	modelCalls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		modelCalls++
		_, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "application/json")
		switch modelCalls {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"msg_action","type":"message","role":"assistant","model":"claude-sonnet-4",
				"content":[{"type":"tool_use","id":"toolu_action","name":"trading-api.get_market_context","input":{}}],
				"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":3}
			}`))
		case 2:
			_, _ = w.Write([]byte(`{"id":"msg_bad","type":"message","role":"assistant","content":[{"type":"text","text":"Approved; proceed."}],"stop_reason":"end_turn"}`))
		default:
			t.Fatalf("unexpected model round %d", modelCalls)
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "x-api-key", APIFormat: "anthropic"})
	var logs bytes.Buffer
	h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", terminalSuccessManifest(toolSrv.URL)), logging.New(&logs))
	req := httptest.NewRequest(http.MethodPost, "/v1/messages",
		bytes.NewBufferString(`{"model":"anthropic/claude-sonnet-4","messages":[{"role":"user","content":"act"}]}`))
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("Content-Length"); got != "" {
		t.Fatalf("synthetic terminal retained stale upstream content-length %q", got)
	}
	if modelCalls != 1 || toolCalls != 1 {
		t.Fatalf("terminal success must stop after one model/tool round, got model=%d tool=%d", modelCalls, toolCalls)
	}
	var message map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &message); err != nil {
		t.Fatalf("unmarshal terminal: %v", err)
	}
	content, _ := message["content"].([]any)
	if len(content) != 1 || message["stop_reason"] != "end_turn" {
		t.Fatalf("expected protocol-valid empty terminal, got %#v", message)
	}
	block, _ := content[0].(map[string]any)
	if block["type"] != "text" || block["text"] != "" {
		t.Fatalf("expected one empty text block, got %#v", block)
	}
	assertInterventionLogged(t, logs.Bytes(), "managed_tool_terminal_on_success:trading-api.get_market_context")
}

func TestTerminalOnSuccessStreamingProtocolParity(t *testing.T) {
	t.Run("openai", func(t *testing.T) {
		toolCalls := 0
		toolSrv := terminalSuccessToolServer(t, &toolCalls)
		defer toolSrv.Close()
		modelCalls := 0
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			modelCalls++
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-action","model":"grok-4.1-fast",
				"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[
					{"id":"call_action","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}}
				]}}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}
			}`))
		}))
		defer backend.Close()
		reg := provider.NewRegistry("")
		reg.Set("xai", &provider.Provider{Name: "xai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer"})
		h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", terminalSuccessManifest(toolSrv.URL)), logging.New(io.Discard))
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			bytes.NewBufferString(`{"model":"xai/grok-4.1-fast","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"act"}]}`))
		req.Header.Set("Authorization", "Bearer tiverton:dummy123")
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		h.ServeHTTP(w, req)

		if modelCalls != 1 || toolCalls != 1 {
			t.Fatalf("expected one model/tool round, got model=%d tool=%d", modelCalls, toolCalls)
		}
		if got := w.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
			t.Fatalf("content-type: %q", got)
		}
		body := w.Body.String()
		if !strings.Contains(body, "data: [DONE]") || strings.Contains(body, "Approved") {
			t.Fatalf("invalid empty terminal stream: %s", body)
		}
		events := parseSSEEvents(t, body)
		if !sseHasUsage(events, 10, 3, 13) {
			t.Fatalf("expected aggregate usage in empty terminal stream: %+v", events)
		}
		for _, event := range events {
			choices, _ := event["choices"].([]any)
			if len(choices) == 0 {
				continue
			}
			choice, _ := choices[0].(map[string]any)
			if choice["finish_reason"] == "stop" {
				return
			}
		}
		t.Fatal("empty terminal stream did not finish with stop")
	})

	t.Run("anthropic", func(t *testing.T) {
		toolCalls := 0
		toolSrv := terminalSuccessToolServer(t, &toolCalls)
		defer toolSrv.Close()
		modelCalls := 0
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			modelCalls++
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{
				"id":"msg_action","type":"message","role":"assistant","model":"claude-sonnet-4",
				"content":[{"type":"tool_use","id":"toolu_action","name":"trading-api.get_market_context","input":{}}],
				"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":3}
			}`))
		}))
		defer backend.Close()
		reg := provider.NewRegistry("")
		reg.Set("anthropic", &provider.Provider{Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "x-api-key", APIFormat: "anthropic"})
		h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", terminalSuccessManifest(toolSrv.URL)), logging.New(io.Discard))
		req := httptest.NewRequest(http.MethodPost, "/v1/messages",
			bytes.NewBufferString(`{"model":"anthropic/claude-sonnet-4","stream":true,"messages":[{"role":"user","content":"act"}]}`))
		req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Anthropic-Version", "2023-06-01")
		w := httptest.NewRecorder()

		h.ServeHTTP(w, req)

		if modelCalls != 1 || toolCalls != 1 {
			t.Fatalf("expected one model/tool round, got model=%d tool=%d", modelCalls, toolCalls)
		}
		body := w.Body.String()
		for _, marker := range []string{"event: message_start", "event: content_block_start", "event: content_block_stop", `"stop_reason":"end_turn"`, "event: message_stop"} {
			if !strings.Contains(body, marker) {
				t.Fatalf("missing %q from empty terminal stream: %s", marker, body)
			}
		}
		for _, line := range strings.Split(body, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			var event map[string]any
			if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
				t.Fatalf("invalid Anthropic SSE JSON %q: %v", line, err)
			}
		}
	})
}

func TestTerminalOnSuccessInvalidAnnotationWarnsAndStaysDisabled(t *testing.T) {
	toolCalls := 0
	toolSrv := terminalSuccessToolServer(t, &toolCalls)
	defer toolSrv.Close()
	manifest := terminalSuccessManifest(toolSrv.URL)
	manifest.Tools[0].Annotations[managedToolTerminalOnSuccessAnnotation] = "true"
	modelCalls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		modelCalls++
		w.Header().Set("Content-Type", "application/json")
		if modelCalls == 1 {
			_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","choices":[{"message":{"role":"assistant","content":"normal final"},"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{Name: "xai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer"})
	var logs bytes.Buffer
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", manifest), logging.New(&logs))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"xai/grok-4.1-fast","messages":[{"role":"user","content":"act"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if modelCalls != 2 || toolCalls != 1 || !strings.Contains(w.Body.String(), "normal final") {
		t.Fatalf("invalid annotation must stay disabled, got model=%d tool=%d body=%s", modelCalls, toolCalls, w.Body.String())
	}
	assertInterventionLogged(t, logs.Bytes(), "managed_tool_terminal_annotation_invalid:trading-api.get_market_context")
}

func TestTerminalOnSuccessFailedReceiptKeepsRecoveryRound(t *testing.T) {
	toolCalls := 0
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolCalls++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		_, _ = w.Write([]byte(`{"status":"DENIED"}`))
	}))
	defer toolSrv.Close()
	modelCalls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		modelCalls++
		w.Header().Set("Content-Type", "application/json")
		if modelCalls == 1 {
			_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","choices":[{"message":{"role":"assistant","content":"action failed safely"},"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{Name: "xai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer"})
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", terminalSuccessManifest(toolSrv.URL)), logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"xai/grok-4.1-fast","messages":[{"role":"user","content":"act"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if modelCalls != 2 || toolCalls != 1 || !strings.Contains(w.Body.String(), "action failed safely") {
		t.Fatalf("failed receipt must preserve recovery, got model=%d tool=%d body=%s", modelCalls, toolCalls, w.Body.String())
	}
}

func TestTerminalOnSuccessMustBeLastBeforeAnyToolExecutes(t *testing.T) {
	toolCalls := 0
	toolSrv := terminalSuccessToolServer(t, &toolCalls)
	defer toolSrv.Close()
	manifest := terminalSuccessManifest(toolSrv.URL)
	second := manifest.Tools[0]
	second.Name = "trading-api.read_after_action"
	second.Annotations = nil
	manifest.Tools = append(manifest.Tools, second)
	modelCalls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		modelCalls++
		w.Header().Set("Content-Type", "application/json")
		if modelCalls == 1 {
			_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","tool_calls":[
				{"id":"call_terminal","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}},
				{"id":"call_after","type":"function","function":{"name":"trading-api.read_after_action","arguments":"{}"}}
			]},"finish_reason":"tool_calls"}]}`))
			return
		}
		if !bytes.Contains(body, []byte("terminal_on_success_order")) {
			t.Fatalf("retry round missing deterministic ordering error: %s", body)
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","choices":[{"message":{"role":"assistant","content":"recovered ordering"},"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{Name: "xai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer"})
	var logs bytes.Buffer
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", manifest), logging.New(&logs))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"xai/grok-4.1-fast","messages":[{"role":"user","content":"act"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if modelCalls != 2 || toolCalls != 0 || !strings.Contains(w.Body.String(), "recovered ordering") {
		t.Fatalf("invalid ordering must retry before side effects, got model=%d tool=%d body=%s", modelCalls, toolCalls, w.Body.String())
	}
	assertInterventionLogged(t, logs.Bytes(), "managed_tool_terminal_order_rejected")
}

func TestTerminalOnSuccessRejectsNativeSuffixBeforeAnyToolExecutes(t *testing.T) {
	toolCalls := 0
	toolSrv := terminalSuccessToolServer(t, &toolCalls)
	defer toolSrv.Close()
	modelCalls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		modelCalls++
		w.Header().Set("Content-Type", "application/json")
		if modelCalls == 1 {
			_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","tool_calls":[
				{"id":"call_terminal","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}},
				{"id":"call_native","type":"function","function":{"name":"runner_local","arguments":"{}"}}
			]},"finish_reason":"tool_calls"}]}`))
			return
		}
		if !bytes.Contains(body, []byte("terminal_on_success_order")) {
			t.Fatalf("retry round missing ordering error: %s", body)
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","choices":[{"message":{"role":"assistant","content":"native suffix removed"},"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{Name: "xai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer"})
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", terminalSuccessManifest(toolSrv.URL)), logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{
		"model":"xai/grok-4.1-fast","messages":[{"role":"user","content":"act"}],
		"tools":[{"type":"function","function":{"name":"runner_local","parameters":{"type":"object"}}}]
	}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if modelCalls != 2 || toolCalls != 0 || !strings.Contains(w.Body.String(), "native suffix removed") {
		t.Fatalf("native suffix must retry before side effects, got model=%d tool=%d body=%s", modelCalls, toolCalls, w.Body.String())
	}
}

func TestAnthropicTerminalOnSuccessMustBeLastBeforeAnyToolExecutes(t *testing.T) {
	toolCalls := 0
	toolSrv := terminalSuccessToolServer(t, &toolCalls)
	defer toolSrv.Close()
	manifest := terminalSuccessManifest(toolSrv.URL)
	second := manifest.Tools[0]
	second.Name = "trading-api.read_after_action"
	second.Annotations = nil
	manifest.Tools = append(manifest.Tools, second)
	modelCalls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		modelCalls++
		w.Header().Set("Content-Type", "application/json")
		if modelCalls == 1 {
			_, _ = w.Write([]byte(`{
				"id":"msg_1","type":"message","role":"assistant","content":[
					{"type":"tool_use","id":"toolu_terminal","name":"trading-api.get_market_context","input":{}},
					{"type":"tool_use","id":"toolu_after","name":"trading-api.read_after_action","input":{}}
				],"stop_reason":"tool_use"
			}`))
			return
		}
		if !bytes.Contains(body, []byte("terminal_on_success_order")) {
			t.Fatalf("retry round missing ordering error: %s", body)
		}
		_, _ = w.Write([]byte(`{"id":"msg_2","type":"message","role":"assistant","content":[{"type":"text","text":"recovered ordering"}],"stop_reason":"end_turn"}`))
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "x-api-key", APIFormat: "anthropic"})
	h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", manifest), logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"anthropic/claude-sonnet-4","messages":[{"role":"user","content":"act"}]}`))
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if modelCalls != 2 || toolCalls != 0 || !strings.Contains(w.Body.String(), "recovered ordering") {
		t.Fatalf("invalid ordering must retry before side effects, got model=%d tool=%d body=%s", modelCalls, toolCalls, w.Body.String())
	}
}

func TestOpenAITerminalOnSuccessContinuityRetainsReceipt(t *testing.T) {
	toolCalls := 0
	toolSrv := terminalSuccessToolServer(t, &toolCalls)
	defer toolSrv.Close()
	modelCalls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		modelCalls++
		w.Header().Set("Content-Type", "application/json")
		if modelCalls == 1 {
			_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","tool_calls":[{"id":"call_action","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`))
			return
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("unmarshal follow-up: %v", err)
		}
		messages, _ := payload["messages"].([]any)
		var sawCall, sawReceipt bool
		for _, raw := range messages {
			message, _ := raw.(map[string]any)
			if calls, _ := message["tool_calls"].([]any); len(calls) == 1 {
				call, _ := calls[0].(map[string]any)
				if call["id"] == "call_action" {
					sawCall = true
				}
			}
			if message["role"] == "tool" && message["tool_call_id"] == "call_action" && strings.Contains(message["content"].(string), "receipt-1") {
				sawReceipt = true
			}
		}
		if !sawCall || !sawReceipt {
			t.Fatalf("follow-up omitted authoritative hidden continuity: %s", body)
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl-2","choices":[{"message":{"role":"assistant","content":"receipt retained"},"finish_reason":"stop"}]}`))
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{Name: "xai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer"})
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", terminalSuccessManifest(toolSrv.URL)), logging.New(io.Discard))

	first := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"xai/grok-4.1-fast","messages":[{"role":"user","content":"act"}]}`))
	first.Header.Set("Authorization", "Bearer tiverton:dummy123")
	firstW := httptest.NewRecorder()
	h.ServeHTTP(firstW, first)
	if firstW.Code != http.StatusOK {
		t.Fatalf("first request: %d %s", firstW.Code, firstW.Body.String())
	}

	followup := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{
		"model":"xai/grok-4.1-fast","messages":[
			{"role":"user","content":"act"},
			{"role":"assistant","content":""},
			{"role":"user","content":"what happened?"}
		]
	}`))
	followup.Header.Set("Authorization", "Bearer tiverton:dummy123")
	followupW := httptest.NewRecorder()
	h.ServeHTTP(followupW, followup)

	if followupW.Code != http.StatusOK || !strings.Contains(followupW.Body.String(), "receipt retained") {
		t.Fatalf("follow-up failed: %d %s", followupW.Code, followupW.Body.String())
	}
	if toolCalls != 1 || modelCalls != 2 {
		t.Fatalf("continuity must prevent re-execution, got model=%d tool=%d", modelCalls, toolCalls)
	}
}

func TestAnthropicTerminalOnSuccessContinuityRetainsReceipt(t *testing.T) {
	toolCalls := 0
	toolSrv := terminalSuccessToolServer(t, &toolCalls)
	defer toolSrv.Close()
	modelCalls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		modelCalls++
		w.Header().Set("Content-Type", "application/json")
		if modelCalls == 1 {
			_, _ = w.Write([]byte(`{
				"id":"msg_1","type":"message","role":"assistant",
				"content":[{"type":"tool_use","id":"toolu_action","name":"trading-api.get_market_context","input":{}}],
				"stop_reason":"tool_use"
			}`))
			return
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("unmarshal follow-up: %v", err)
		}
		messages, _ := payload["messages"].([]any)
		var sawCall, sawReceipt bool
		for _, raw := range messages {
			message, _ := raw.(map[string]any)
			blocks, _ := message["content"].([]any)
			for _, rawBlock := range blocks {
				block, _ := rawBlock.(map[string]any)
				switch block["type"] {
				case "tool_use":
					if block["id"] == "toolu_action" {
						sawCall = true
					}
				case "tool_result":
					if block["tool_use_id"] == "toolu_action" && strings.Contains(block["content"].(string), "receipt-1") {
						sawReceipt = true
					}
				}
			}
		}
		if !sawCall || !sawReceipt {
			t.Fatalf("follow-up omitted authoritative hidden continuity: %s", body)
		}
		_, _ = w.Write([]byte(`{
			"id":"msg_2","type":"message","role":"assistant",
			"content":[{"type":"text","text":"receipt retained"}],"stop_reason":"end_turn"
		}`))
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "x-api-key", APIFormat: "anthropic"})
	h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", terminalSuccessManifest(toolSrv.URL)), logging.New(io.Discard))

	first := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"anthropic/claude-sonnet-4","messages":[{"role":"user","content":"act"}]}`))
	first.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	first.Header.Set("Anthropic-Version", "2023-06-01")
	firstW := httptest.NewRecorder()
	h.ServeHTTP(firstW, first)
	if firstW.Code != http.StatusOK {
		t.Fatalf("first request: %d %s", firstW.Code, firstW.Body.String())
	}

	followup := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{
		"model":"anthropic/claude-sonnet-4","messages":[
			{"role":"user","content":"act"},
			{"role":"assistant","content":[{"type":"text","text":""}]},
			{"role":"user","content":"what happened?"}
		]
	}`))
	followup.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	followup.Header.Set("Anthropic-Version", "2023-06-01")
	followupW := httptest.NewRecorder()
	h.ServeHTTP(followupW, followup)

	if followupW.Code != http.StatusOK || !strings.Contains(followupW.Body.String(), "receipt retained") {
		t.Fatalf("follow-up failed: %d %s", followupW.Code, followupW.Body.String())
	}
	if toolCalls != 1 || modelCalls != 2 {
		t.Fatalf("continuity must prevent re-execution, got model=%d tool=%d", modelCalls, toolCalls)
	}
}

func TestTerminalOnSuccessAfterEarlierToolErrorStillTerminates(t *testing.T) {
	var toolPaths []string
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolPaths = append(toolPaths, r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		if strings.HasPrefix(r.URL.Path, "/prepare/") {
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{"status":"DENIED","reason":"stale input"}`))
			return
		}
		_, _ = w.Write([]byte(`{"status":"COMMITTED","receipt_id":"receipt-2"}`))
	}))
	defer toolSrv.Close()

	manifest := terminalSuccessManifest(toolSrv.URL)
	manifest.Tools[0].Execution.Path = "/commit/{claw_id}"
	prepare := manifest.Tools[0]
	prepare.Name = "trading-api.prepare_action"
	prepare.Annotations = nil
	prepare.Execution.Path = "/prepare/{claw_id}"
	manifest.Tools = append([]agentctx.ToolManifestEntry{prepare}, manifest.Tools[0])

	modelCalls := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		modelCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-action","model":"grok-4.1-fast",
			"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[
				{"id":"call_prepare","type":"function","function":{"name":"trading-api.prepare_action","arguments":"{}"}},
				{"id":"call_commit","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}}
			]}}]
		}`))
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{Name: "xai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer"})
	histDir := t.TempDir()
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", manifest), logging.New(io.Discard), WithSessionHistory(histDir))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"xai/grok-4.1-fast","messages":[{"role":"user","content":"act"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK || modelCalls != 1 || len(toolPaths) != 2 {
		t.Fatalf("later terminal success must terminate after preserving earlier error, status=%d model=%d paths=%v body=%s", w.Code, modelCalls, toolPaths, w.Body.String())
	}
	if !strings.HasPrefix(toolPaths[0], "/prepare/") || !strings.HasPrefix(toolPaths[1], "/commit/") {
		t.Fatalf("tools executed out of order: %v", toolPaths)
	}
	entries, err := sessionhistory.ReadEntries(histDir, "tiverton", nil, 10)
	if err != nil {
		t.Fatalf("ReadEntries: %v", err)
	}
	if len(entries) != 1 || len(entries[0].ToolTrace) != 1 || len(entries[0].ToolTrace[0].ToolCalls) != 2 {
		t.Fatalf("expected both tool outcomes in trace, got %+v", entries)
	}
	if entries[0].ToolTrace[0].ToolCalls[0].StatusCode != http.StatusConflict || entries[0].ToolTrace[0].ToolCalls[1].StatusCode != http.StatusOK {
		t.Fatalf("unexpected ordered statuses: %+v", entries[0].ToolTrace[0].ToolCalls)
	}
}

func TestTerminalOnSuccessDuplicatePolicyMatrix(t *testing.T) {
	manifest := terminalSuccessManifest("http://tool.invalid")
	agentCtx, err := stubContextLoaderWithTools("tiverton", "tiverton:dummy123", manifest)("tiverton")
	if err != nil {
		t.Fatalf("load context: %v", err)
	}
	duplicate := &managedToolDuplicate{
		CanonicalName:   "trading-api.get_market_context",
		Service:         "trading-api",
		Arguments:       json.RawMessage(`{}`),
		FirstRound:      1,
		Count:           1,
		Streak:          1,
		CachedResult:    []byte(`{"ok":true,"status_code":200,"data":{"receipt_id":"receipt-1"}}`),
		HasCachedResult: true,
		Status:          "ok",
		StatusCode:      http.StatusOK,
	}

	replay := duplicateManagedToolOutcomeForCall(agentCtx, "trading-api.get_market_context", duplicate, managedDuplicatePolicyReplay)
	if !replay.TerminalSuccess {
		t.Fatal("replaying an authoritative terminal receipt must terminate")
	}
	reject := duplicateManagedToolOutcomeForCall(agentCtx, "trading-api.get_market_context", duplicate, managedDuplicatePolicyReject)
	if reject.TerminalSuccess {
		t.Fatal("rejecting a duplicate produces an error envelope and must not terminate")
	}
}
