package proxy

import (
	"bufio"
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

func TestHandlerInjectsManagedToolsIntoOpenAIRequests(t *testing.T) {
	expectedName := managedToolPresentedNameForCanonical("trading-api.get_market_context")
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

	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", managedToolManifest()), nil)
	body := `{
		"model":"openai/gpt-4o",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"runner_local"}}],
		"functions":[{"name":"legacy"}],
		"tool_choice":"auto",
		"parallel_tool_calls":true
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
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
	tools, _ := payload["tools"].([]any)
	if len(tools) != 3 {
		t.Fatalf("expected runner-local tools plus managed tools, got %+v", payload["tools"])
	}
	var names []string
	for _, raw := range tools {
		tool, _ := raw.(map[string]any)
		function, _ := tool["function"].(map[string]any)
		name, _ := function["name"].(string)
		names = append(names, name)
	}
	joined := strings.Join(names, ",")
	if !strings.Contains(joined, "runner_local") || !strings.Contains(joined, "legacy") || !strings.Contains(joined, expectedName) {
		t.Fatalf("expected merged tool list to include runner_local, legacy, and %q, got %+v", expectedName, payload["tools"])
	}
	if _, ok := payload["functions"]; ok {
		t.Fatalf("expected legacy functions field removed, got %+v", payload)
	}
	if payload["tool_choice"] != "auto" {
		t.Fatalf("expected tool_choice preserved, got %+v", payload)
	}
	if enabled, _ := payload["parallel_tool_calls"].(bool); !enabled {
		t.Fatalf("expected parallel_tool_calls preserved, got %+v", payload)
	}
}

func TestHandlerRestreamsManagedOpenAITools(t *testing.T) {
	var xaiBodies [][]byte
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	xaiBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read xai body: %v", err)
		}
		xaiBodies = append(xaiBodies, body)
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("unmarshal xai request: %v", err)
		}
		if stream, _ := payload["stream"].(bool); stream {
			t.Fatalf("expected managed upstream request to force stream=false, got %+v", payload)
		}
		if _, ok := payload["stream_options"]; ok {
			t.Fatalf("expected stream_options removed from managed upstream request, got %+v", payload)
		}
		w.Header().Set("Content-Type", "application/json")
		switch len(xaiBodies) {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-1",
				"choices":[{
					"finish_reason":"tool_calls",
					"message":{
						"role":"assistant",
						"tool_calls":[{"id":"call_1","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}}]
					}
				}],
				"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2",
				"model":"grok-4.1-fast",
				"choices":[{"message":{"role":"assistant","content":"market context loaded"}}],
				"usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12}
			}`))
		default:
			t.Fatalf("unexpected xai round %d", len(xaiBodies))
		}
	}))
	defer xaiBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: xaiBackend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	histDir := t.TempDir()
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard),
		WithSessionHistory(histDir))
	body := `{"model":"xai/grok-4.1-fast","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Fatalf("expected SSE content-type, got %q", got)
	}
	bodyText := w.Body.String()
	if !strings.Contains(bodyText, "data: [DONE]") {
		t.Fatalf("expected [DONE] marker, got %s", bodyText)
	}
	events := parseSSEEvents(t, bodyText)
	if !sseHasContent(events, "market context loaded") {
		t.Fatalf("expected synthetic SSE content chunk, got %+v", events)
	}
	if !sseHasUsage(events, 17, 8, 25) {
		t.Fatalf("expected aggregated usage chunk, got %+v", events)
	}

	histFile := filepath.Join(histDir, "tiverton", "history.jsonl")
	rawHist, err := os.ReadFile(histFile)
	if err != nil {
		t.Fatalf("read history: %v", err)
	}
	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimRight(rawHist, "\n"), &entry); err != nil {
		t.Fatalf("unmarshal history entry: %v", err)
	}
	if stream, _ := entry["stream"].(bool); !stream {
		t.Fatalf("expected history stream=true, got %+v", entry)
	}
	resp, _ := entry["response"].(map[string]any)
	if resp["format"] != "sse" {
		t.Fatalf("expected response.format=sse, got %+v", resp)
	}
}

func TestHandlerPassesThroughNativeOpenAIToolCallsWhenManagedToolsArePresent(t *testing.T) {
	var xaiBodies [][]byte
	toolCalls := 0
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	xaiBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read xai body: %v", err)
		}
		xaiBodies = append(xaiBodies, body)
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("unmarshal xai request: %v", err)
		}
		if stream, _ := payload["stream"].(bool); stream {
			t.Fatalf("expected managed/native upstream request to force stream=false, got %+v", payload)
		}
		tools, _ := payload["tools"].([]any)
		if len(tools) != 2 {
			t.Fatalf("expected merged tool list upstream, got %+v", payload["tools"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-native",
			"model":"grok-4.1-fast",
			"choices":[{
				"finish_reason":"tool_calls",
				"message":{
					"role":"assistant",
					"tool_calls":[{"id":"call_native_1","type":"function","function":{"name":"runner_local","arguments":"{\"ticker\":\"NVDA\"}"}}]
				}
			}],
			"usage":{"prompt_tokens":9,"completion_tokens":4,"total_tokens":13}
		}`))
	}))
	defer xaiBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: xaiBackend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard))
	body := `{
		"model":"xai/grok-4.1-fast",
		"stream":true,
		"stream_options":{"include_usage":true},
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"runner_local","description":"native runner tool","parameters":{"type":"object"}}}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if toolCalls != 0 {
		t.Fatalf("expected managed tool executor to stay idle for native tool call, got %d", toolCalls)
	}
	if len(xaiBodies) != 1 {
		t.Fatalf("expected one upstream model round, got %d", len(xaiBodies))
	}
	if got := w.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Fatalf("expected SSE content-type, got %q", got)
	}
	events := parseSSEEvents(t, w.Body.String())
	if !sseHasToolCall(events, "runner_local") {
		t.Fatalf("expected re-streamed native tool call, got %+v", events)
	}
	if !sseHasUsage(events, 9, 4, 13) {
		t.Fatalf("expected usage chunk in re-streamed native tool response, got %+v", events)
	}
}

func TestHandlerRetriesUnsafeMixedOpenAIToolCallsInternally(t *testing.T) {
	presentedName := managedToolPresentedNameForCanonical("trading-api.get_market_context")
	xaiCalls := 0
	toolCalls := 0
	var logs bytes.Buffer
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	xaiBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		xaiCalls++
		w.Header().Set("Content-Type", "application/json")
		switch xaiCalls {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-mixed",
				"choices":[{
					"finish_reason":"tool_calls",
					"message":{
						"role":"assistant",
						"content":"I will use the local runner first, then get market context.",
						"tool_calls":[
							{"id":"call_native_1","type":"function","function":{"name":"runner_local","arguments":"{}"}},
							{"id":"call_managed_1","type":"function","function":{"name":"` + presentedName + `","arguments":"{}"}}
						]
					}
				}]
			}`))
		case 2:
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("read xai body: %v", err)
			}
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal xai request: %v", err)
			}
			rawMessages, _ := payload["messages"].([]any)
			conversation := make([]map[string]any, 0, len(rawMessages))
			for _, raw := range rawMessages {
				msg, _ := raw.(map[string]any)
				if msg == nil {
					continue
				}
				if role, _ := msg["role"].(string); role == "system" {
					continue
				}
				conversation = append(conversation, msg)
			}
			if len(conversation) != 4 {
				t.Fatalf("expected unsafe mixed retry round to stay internal, got %+v", payload)
			}
			if conversation[0]["role"] != "user" || conversation[0]["content"] != "hi" {
				t.Fatalf("unexpected first conversation message: %+v", conversation[0])
			}
			mixedAssistant := conversation[1]
			mixedToolCalls, _ := mixedAssistant["tool_calls"].([]any)
			if mixedAssistant["role"] != "assistant" || len(mixedToolCalls) != 2 {
				t.Fatalf("expected full mixed tool-call assistant message, got %+v", mixedAssistant)
			}
			if mixedAssistant["content"] != "I will use the local runner first, then get market context." {
				t.Fatalf("expected assistant content preserved, got %+v", mixedAssistant)
			}
			rejectedNative := conversation[2]
			if rejectedNative["role"] != "tool" || rejectedNative["tool_call_id"] != "call_native_1" {
				t.Fatalf("expected native tool rejection message, got %+v", rejectedNative)
			}
			assertToolErrorContains(t, rejectedNative["content"], mixedToolOrderMessage)
			rejectedManaged := conversation[3]
			if rejectedManaged["role"] != "tool" || rejectedManaged["tool_call_id"] != "call_managed_1" {
				t.Fatalf("expected managed tool rejection message, got %+v", rejectedManaged)
			}
			assertToolErrorContains(t, rejectedManaged["content"], mixedToolOrderMessage)
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-replanned",
				"choices":[{"message":{"role":"assistant","content":"Understood. I need to fetch managed context before I ask for local runner work."}}]
			}`))
		default:
			t.Fatalf("unexpected xai round %d", xaiCalls)
		}
	}))
	defer xaiBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: xaiBackend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(&logs))
	body := `{
		"model":"xai/grok-4.1-fast",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"runner_local","description":"native runner tool","parameters":{"type":"object"}}}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "Understood. I need to fetch managed context before I ask for local runner work.") {
		t.Fatalf("expected internal retry reply, got %s", w.Body.String())
	}
	if toolCalls != 0 {
		t.Fatalf("expected no managed tool execution during unsafe mixed retry, got %d", toolCalls)
	}
	if xaiCalls != 2 {
		t.Fatalf("expected one internal retry round, got %d model calls", xaiCalls)
	}
	assertInterventionLogged(t, logs.Bytes(), mixedToolOrderInternalRetryIntervention)
}

func TestHandlerSerializesManagedPrefixBeforeNativeOpenAIToolCalls(t *testing.T) {
	presentedName := managedToolPresentedNameForCanonical("trading-api.get_market_context")
	xaiCalls := 0
	toolCalls := 0
	var logs bytes.Buffer
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	xaiBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		xaiCalls++
		w.Header().Set("Content-Type", "application/json")
		switch xaiCalls {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-mixed-first",
				"choices":[{
					"finish_reason":"tool_calls",
					"message":{
						"role":"assistant",
						"content":"Checking managed context first.",
						"tool_calls":[
							{"id":"call_managed_1","type":"function","function":{"name":"` + presentedName + `","arguments":"{}"}},
							{"id":"call_native_1","type":"function","function":{"name":"runner_local","arguments":"{}"}}
						]
					}
				}]
			}`))
		case 2:
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("read xai body: %v", err)
			}
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal xai request: %v", err)
			}
			rawMessages, _ := payload["messages"].([]any)
			conversation := make([]map[string]any, 0, len(rawMessages))
			for _, raw := range rawMessages {
				msg, _ := raw.(map[string]any)
				if msg == nil {
					continue
				}
				if role, _ := msg["role"].(string); role == "system" {
					continue
				}
				conversation = append(conversation, msg)
			}
			if len(conversation) != 3 {
				t.Fatalf("expected only managed prefix to be serialized before rerun, got %+v", payload)
			}
			if conversation[0]["role"] != "user" || conversation[0]["content"] != "hi" {
				t.Fatalf("unexpected first conversation message: %+v", conversation[0])
			}
			managedAssistant := conversation[1]
			managedToolCalls, _ := managedAssistant["tool_calls"].([]any)
			if managedAssistant["role"] != "assistant" || managedAssistant["content"] != "Checking managed context first." || len(managedToolCalls) != 1 {
				t.Fatalf("expected managed-only assistant message, got %+v", managedAssistant)
			}
			if managedAssistantCall := managedToolCalls[0].(map[string]any); managedAssistantCall["id"] != "call_managed_1" {
				t.Fatalf("expected managed tool call id preserved, got %+v", managedAssistantCall)
			}
			managedTool := conversation[2]
			if managedTool["role"] != "tool" || managedTool["tool_call_id"] != "call_managed_1" {
				t.Fatalf("expected managed tool result after serialization, got %+v", managedTool)
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-native-second",
				"choices":[{
					"finish_reason":"tool_calls",
					"message":{
						"role":"assistant",
						"tool_calls":[{"id":"call_native_1","type":"function","function":{"name":"runner_local","arguments":"{}"}}]
					}
				}]
			}`))
		case 3:
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("read xai body: %v", err)
			}
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal xai request: %v", err)
			}
			rawMessages, _ := payload["messages"].([]any)
			conversation := make([]map[string]any, 0, len(rawMessages))
			for _, raw := range rawMessages {
				msg, _ := raw.(map[string]any)
				if msg == nil {
					continue
				}
				if role, _ := msg["role"].(string); role == "system" {
					continue
				}
				conversation = append(conversation, msg)
			}
			if len(conversation) < 5 {
				t.Fatalf("expected managed handoff continuity injection, got %+v", payload)
			}
			hiddenAssistant := conversation[1]
			hiddenToolCalls, _ := hiddenAssistant["tool_calls"].([]any)
			if hiddenAssistant["role"] != "assistant" || len(hiddenToolCalls) != 1 {
				t.Fatalf("expected hidden managed assistant before native tool call, got %+v", hiddenAssistant)
			}
			hiddenTool := conversation[2]
			if hiddenTool["role"] != "tool" || hiddenTool["tool_call_id"] != "call_managed_1" {
				t.Fatalf("expected hidden managed tool result, got %+v", hiddenTool)
			}
			nativeAssistant := conversation[3]
			nativeToolCalls, _ := nativeAssistant["tool_calls"].([]any)
			if nativeAssistant["role"] != "assistant" || len(nativeToolCalls) != 1 {
				t.Fatalf("expected native tool call after hidden managed rounds, got %+v", nativeAssistant)
			}
			nativeToolResult := conversation[4]
			if nativeToolResult["role"] != "tool" || nativeToolResult["tool_call_id"] != "call_native_1" {
				t.Fatalf("expected native tool result after injected hidden rounds, got %+v", nativeToolResult)
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-final",
				"choices":[{"message":{"role":"assistant","content":"native handoff complete"}}]
			}`))
		default:
			t.Fatalf("unexpected xai round %d", xaiCalls)
		}
	}))
	defer xaiBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: xaiBackend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(&logs))
	body := `{
		"model":"xai/grok-4.1-fast",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"runner_local","description":"native runner tool","parameters":{"type":"object"}}}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "\"runner_local\"") {
		t.Fatalf("expected native tool call to be returned to runner, got %s", w.Body.String())
	}
	if xaiCalls != 2 {
		t.Fatalf("expected two model rounds before handoff, got %d", xaiCalls)
	}
	if toolCalls != 1 {
		t.Fatalf("expected one managed tool execution before handoff, got %d", toolCalls)
	}

	followupReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{
		"model":"xai/grok-4.1-fast",
		"messages":[
			{"role":"user","content":"hi"},
			{"role":"assistant","tool_calls":[{"id":"call_native_1","type":"function","function":{"name":"runner_local","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"call_native_1","content":"{\"native\":true}"}
		],
		"tools":[{"type":"function","function":{"name":"runner_local","description":"native runner tool","parameters":{"type":"object"}}}]
	}`))
	followupReq.Header.Set("Authorization", "Bearer tiverton:dummy123")
	followupReq.Header.Set("Content-Type", "application/json")
	followupW := httptest.NewRecorder()

	h.ServeHTTP(followupW, followupReq)

	if followupW.Code != http.StatusOK {
		t.Fatalf("expected follow-up request 200, got %d: %s", followupW.Code, followupW.Body.String())
	}
	if !strings.Contains(followupW.Body.String(), "native handoff complete") {
		t.Fatalf("expected final text after native handoff, got %s", followupW.Body.String())
	}
	if xaiCalls != 3 {
		t.Fatalf("expected third model round after runner tool result, got %d", xaiCalls)
	}
	assertInterventionLogged(t, logs.Bytes(), managedMixedPrefixSerializedIntervention)
}

func TestHandlerHandsOffNativeOpenAIToolCallsAfterManagedRounds(t *testing.T) {
	xaiCalls := 0
	toolCalls := 0
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	xaiBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		xaiCalls++
		w.Header().Set("Content-Type", "application/json")
		switch xaiCalls {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-managed-first",
				"choices":[{
					"finish_reason":"tool_calls",
					"message":{
						"role":"assistant",
						"tool_calls":[{"id":"call_managed_1","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}}]
					}
				}]
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-native-second",
				"choices":[{
					"finish_reason":"tool_calls",
					"message":{
						"role":"assistant",
						"tool_calls":[{"id":"call_native_1","type":"function","function":{"name":"runner_local","arguments":"{}"}}]
					}
				}]
			}`))
		case 3:
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("read xai body: %v", err)
			}
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal xai request: %v", err)
			}
			rawMessages, _ := payload["messages"].([]any)
			conversation := make([]map[string]any, 0, len(rawMessages))
			for _, raw := range rawMessages {
				msg, _ := raw.(map[string]any)
				if msg == nil {
					continue
				}
				if role, _ := msg["role"].(string); role == "system" {
					continue
				}
				conversation = append(conversation, msg)
			}
			if len(conversation) < 5 {
				t.Fatalf("expected managed handoff continuity injection, got %+v", payload)
			}
			if conversation[0]["role"] != "user" || conversation[0]["content"] != "hi" {
				t.Fatalf("unexpected first conversation message: %+v", conversation[0])
			}
			hiddenAssistant := conversation[1]
			if hiddenAssistant["role"] != "assistant" {
				t.Fatalf("expected hidden managed assistant before native tool call, got %+v", hiddenAssistant)
			}
			hiddenToolCalls, _ := hiddenAssistant["tool_calls"].([]any)
			if len(hiddenToolCalls) != 1 {
				t.Fatalf("expected hidden managed tool call, got %+v", hiddenAssistant)
			}
			if hiddenAssistantCall := hiddenToolCalls[0].(map[string]any); hiddenAssistantCall["id"] != "call_managed_1" {
				t.Fatalf("expected hidden managed tool call id, got %+v", hiddenAssistantCall)
			}
			hiddenTool := conversation[2]
			if hiddenTool["role"] != "tool" || hiddenTool["tool_call_id"] != "call_managed_1" {
				t.Fatalf("expected hidden managed tool result, got %+v", hiddenTool)
			}
			nativeAssistant := conversation[3]
			nativeToolCalls, _ := nativeAssistant["tool_calls"].([]any)
			if nativeAssistant["role"] != "assistant" || len(nativeToolCalls) != 1 {
				t.Fatalf("expected native tool call after hidden managed rounds, got %+v", nativeAssistant)
			}
			if nativeAssistantCall := nativeToolCalls[0].(map[string]any); nativeAssistantCall["id"] != "call_native_1" {
				t.Fatalf("expected native tool call id preserved, got %+v", nativeAssistantCall)
			}
			nativeToolResult := conversation[4]
			if nativeToolResult["role"] != "tool" || nativeToolResult["tool_call_id"] != "call_native_1" {
				t.Fatalf("expected native tool result after injected hidden rounds, got %+v", nativeToolResult)
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-final",
				"choices":[{"message":{"role":"assistant","content":"native handoff complete"}}]
			}`))
		default:
			t.Fatalf("unexpected xai round %d", xaiCalls)
		}
	}))
	defer xaiBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: xaiBackend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard))
	body := `{
		"model":"xai/grok-4.1-fast",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"runner_local","description":"native runner tool","parameters":{"type":"object"}}}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "\"runner_local\"") {
		t.Fatalf("expected native tool call to be returned to runner, got %s", w.Body.String())
	}
	if xaiCalls != 2 {
		t.Fatalf("expected two model rounds before handoff, got %d", xaiCalls)
	}
	if toolCalls != 1 {
		t.Fatalf("expected one managed tool execution before handoff, got %d", toolCalls)
	}

	followupReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{
		"model":"xai/grok-4.1-fast",
		"messages":[
			{"role":"user","content":"hi"},
			{"role":"assistant","tool_calls":[{"id":"call_native_1","type":"function","function":{"name":"runner_local","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"call_native_1","content":"{\"native\":true}"}
		],
		"tools":[{"type":"function","function":{"name":"runner_local","description":"native runner tool","parameters":{"type":"object"}}}]
	}`))
	followupReq.Header.Set("Authorization", "Bearer tiverton:dummy123")
	followupReq.Header.Set("Content-Type", "application/json")
	followupW := httptest.NewRecorder()

	h.ServeHTTP(followupW, followupReq)

	if followupW.Code != http.StatusOK {
		t.Fatalf("expected follow-up request 200, got %d: %s", followupW.Code, followupW.Body.String())
	}
	if !strings.Contains(followupW.Body.String(), "native handoff complete") {
		t.Fatalf("expected final text after native handoff, got %s", followupW.Body.String())
	}
	if xaiCalls != 3 {
		t.Fatalf("expected third model round after runner tool result, got %d", xaiCalls)
	}
}

func TestHandlerStreamsManagedOpenAIKeepaliveComments(t *testing.T) {
	presentedName := managedToolPresentedNameForCanonical("trading-api.get_market_context")
	toolRelease := make(chan struct{})
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-toolRelease
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	var xaiBodies [][]byte
	xaiBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read xai body: %v", err)
		}
		xaiBodies = append(xaiBodies, body)
		w.Header().Set("Content-Type", "application/json")
		switch len(xaiBodies) {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-1",
				"choices":[{
					"finish_reason":"tool_calls",
					"message":{
						"role":"assistant",
						"tool_calls":[{"id":"call_1","type":"function","function":{"name":"` + presentedName + `","arguments":"{}"}}]
					}
				}],
				"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2",
				"model":"grok-4.1-fast",
				"choices":[{"message":{"role":"assistant","content":"market context loaded"}}],
				"usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12}
			}`))
		default:
			t.Fatalf("unexpected xai round %d", len(xaiBodies))
		}
	}))
	defer xaiBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: xaiBackend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	histDir := t.TempDir()
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard),
		WithSessionHistory(histDir))
	proxySrv := httptest.NewServer(h)
	defer proxySrv.Close()

	req, err := http.NewRequest(http.MethodPost, proxySrv.URL+"/v1/chat/completions", bytes.NewBufferString(`{"model":"xai/grok-4.1-fast","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()

	bodyText, sawComment := readManagedKeepaliveStream(t, resp.Body, ": managed tool round 1 executing trading-api.get_market_context", func() {
		close(toolRelease)
	})
	if !sawComment {
		t.Fatalf("expected keepalive comment in managed stream, got %s", bodyText)
	}
	if !strings.Contains(bodyText, "data: [DONE]") {
		t.Fatalf("expected managed stream completion, got %s", bodyText)
	}
	if !sseHasContent(parseSSEEvents(t, bodyText), "market context loaded") {
		t.Fatalf("expected final managed SSE content, got %s", bodyText)
	}

	histFile := filepath.Join(histDir, "tiverton", "history.jsonl")
	rawHist, err := os.ReadFile(histFile)
	if err != nil {
		t.Fatalf("read history: %v", err)
	}
	if bytes.Contains(rawHist, []byte(": managed tool round")) {
		t.Fatalf("expected keepalive comments to stay out of session history, got %s", rawHist)
	}
}

func TestHandlerExecutesManagedToolsViaXAI(t *testing.T) {
	presentedName := managedToolPresentedNameForCanonical("trading-api.get_market_context")
	var xaiBodies [][]byte
	var toolAuth string
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolAuth = r.Header.Get("Authorization")
		if r.URL.Path != "/api/v1/market_context/tiverton" {
			t.Fatalf("unexpected tool path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000,"positions":[{"symbol":"NVDA","qty":2}]}`))
	}))
	defer toolSrv.Close()

	xaiBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read xai body: %v", err)
		}
		xaiBodies = append(xaiBodies, body)
		w.Header().Set("Content-Type", "application/json")
		switch len(xaiBodies) {
		case 1:
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal xai request: %v", err)
			}
			tools, _ := payload["tools"].([]any)
			if len(tools) != 1 {
				t.Fatalf("expected 1 managed xai tool, got %+v", payload["tools"])
			}
			first, _ := tools[0].(map[string]any)
			function, _ := first["function"].(map[string]any)
			if function["name"] != presentedName {
				t.Fatalf("expected provider-safe managed tool name %q, got %+v", presentedName, first)
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-1",
				"choices":[{
					"finish_reason":"tool_calls",
					"message":{
						"role":"assistant",
						"tool_calls":[{"id":"call_1","type":"function","function":{"name":"` + presentedName + `","arguments":"{}"}}]
					}
				}],
				"usage":{"prompt_tokens":10,"completion_tokens":3}
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2",
				"choices":[{"message":{"role":"assistant","content":"market context loaded"}}],
				"usage":{"prompt_tokens":7,"completion_tokens":5}
			}`))
		default:
			t.Fatalf("unexpected xai round %d", len(xaiBodies))
		}
	}))
	defer xaiBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: xaiBackend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	histDir := t.TempDir()
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "svc-token")), logging.New(io.Discard),
		WithSessionHistory(histDir))

	body := `{"model":"xai/grok-4.1-fast","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "market context loaded") {
		t.Fatalf("expected final xai text response, got %s", w.Body.String())
	}
	if len(xaiBodies) != 2 {
		t.Fatalf("expected 2 xai rounds, got %d", len(xaiBodies))
	}
	if toolAuth != "Bearer svc-token" {
		t.Fatalf("expected projected tool auth, got %q", toolAuth)
	}

	var secondPayload map[string]any
	if err := json.Unmarshal(xaiBodies[1], &secondPayload); err != nil {
		t.Fatalf("unmarshal second xai request: %v", err)
	}
	if secondPayload["model"] != "grok-4.1-fast" {
		t.Fatalf("expected stripped xai model, got %#v", secondPayload["model"])
	}
	messages, _ := secondPayload["messages"].([]any)
	if len(messages) < 3 {
		t.Fatalf("expected tool round reflected in follow-up request, got %+v", secondPayload)
	}
	last := messages[len(messages)-1].(map[string]any)
	if last["role"] != "tool" {
		t.Fatalf("expected final follow-up message to be tool result, got %+v", last)
	}
	var toolResult map[string]any
	if err := json.Unmarshal([]byte(last["content"].(string)), &toolResult); err != nil {
		t.Fatalf("unmarshal tool result: %v", err)
	}
	if ok, _ := toolResult["ok"].(bool); !ok {
		t.Fatalf("expected successful tool result, got %+v", toolResult)
	}
	data, _ := toolResult["data"].(map[string]any)
	if data["balance"].(float64) != 5000 {
		t.Fatalf("expected tool payload balance, got %+v", toolResult)
	}

	histFile := filepath.Join(histDir, "tiverton", "history.jsonl")
	rawHist, err := os.ReadFile(histFile)
	if err != nil {
		t.Fatalf("read history: %v", err)
	}
	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimRight(rawHist, "\n"), &entry); err != nil {
		t.Fatalf("unmarshal history entry: %v", err)
	}
	if entry["status"] != "ok" {
		t.Fatalf("expected history status=ok, got %+v", entry)
	}
	usage, _ := entry["usage"].(map[string]any)
	if usage["total_rounds"].(float64) != 2 {
		t.Fatalf("expected total_rounds=2, got %+v", usage)
	}
	trace, _ := entry["tool_trace"].([]any)
	if len(trace) != 1 {
		t.Fatalf("expected one tool trace round, got %+v", entry)
	}
	round := trace[0].(map[string]any)
	toolCalls := round["tool_calls"].([]any)
	if len(toolCalls) != 1 {
		t.Fatalf("expected one tool call in trace, got %+v", round)
	}
	call := toolCalls[0].(map[string]any)
	if call["name"] != "trading-api.get_market_context" {
		t.Fatalf("unexpected tool trace name: %+v", call)
	}
}

func TestHandlerReinjectsManagedToolContinuityOnNextTurn(t *testing.T) {
	var xaiBodies [][]byte
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000,"positions":[{"symbol":"NVDA","qty":2}]}`))
	}))
	defer toolSrv.Close()

	xaiBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read xai body: %v", err)
		}
		xaiBodies = append(xaiBodies, body)
		w.Header().Set("Content-Type", "application/json")
		switch len(xaiBodies) {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-1",
				"choices":[{
					"finish_reason":"tool_calls",
					"message":{
						"role":"assistant",
						"tool_calls":[{"id":"call_1","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}}]
					}
				}]
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2",
				"choices":[{"message":{"role":"assistant","content":"market context loaded"}}]
			}`))
		case 3:
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal xai request: %v", err)
			}
			rawMessages, _ := payload["messages"].([]any)
			conversation := make([]map[string]any, 0, len(rawMessages))
			for _, raw := range rawMessages {
				msg, _ := raw.(map[string]any)
				if msg == nil {
					continue
				}
				if role, _ := msg["role"].(string); role == "system" {
					continue
				}
				conversation = append(conversation, msg)
			}
			if len(conversation) < 5 {
				t.Fatalf("expected continuity-injected conversation, got %+v", payload)
			}
			if conversation[0]["role"] != "user" || conversation[0]["content"] != "hi" {
				t.Fatalf("unexpected first conversation message: %+v", conversation[0])
			}
			if conversation[1]["role"] != "assistant" {
				t.Fatalf("expected hidden assistant tool call before visible reply, got %+v", conversation[1])
			}
			toolCalls, _ := conversation[1]["tool_calls"].([]any)
			if len(toolCalls) != 1 {
				t.Fatalf("expected continuity assistant tool_calls, got %+v", conversation[1])
			}
			if conversation[2]["role"] != "tool" || conversation[2]["tool_call_id"] != "call_1" {
				t.Fatalf("expected continuity tool result before visible reply, got %+v", conversation[2])
			}
			if conversation[3]["role"] != "assistant" || conversation[3]["content"] != "market context loaded" {
				t.Fatalf("expected original visible assistant reply preserved after hidden turns, got %+v", conversation[3])
			}
			if conversation[4]["role"] != "user" || conversation[4]["content"] != "what next?" {
				t.Fatalf("expected new user turn to remain after injected continuity, got %+v", conversation[4])
			}
			_, _ = w.Write([]byte(`{"id":"chatcmpl-3","choices":[{"message":{"role":"assistant","content":"next step is hedge"}}]}`))
		default:
			t.Fatalf("unexpected xai round %d", len(xaiBodies))
		}
	}))
	defer xaiBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: xaiBackend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard))

	firstReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"xai/grok-4.1-fast","messages":[{"role":"user","content":"hi"}]}`))
	firstReq.Header.Set("Authorization", "Bearer tiverton:dummy123")
	firstReq.Header.Set("Content-Type", "application/json")
	firstW := httptest.NewRecorder()
	h.ServeHTTP(firstW, firstReq)
	if firstW.Code != http.StatusOK {
		t.Fatalf("expected first request 200, got %d: %s", firstW.Code, firstW.Body.String())
	}
	if !strings.Contains(firstW.Body.String(), "market context loaded") {
		t.Fatalf("expected first response to complete mediated tool turn, got %s", firstW.Body.String())
	}

	secondReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{
		"model":"xai/grok-4.1-fast",
		"messages":[
			{"role":"user","content":"hi"},
			{"role":"assistant","content":"market context loaded"},
			{"role":"user","content":"what next?"}
		]
	}`))
	secondReq.Header.Set("Authorization", "Bearer tiverton:dummy123")
	secondReq.Header.Set("Content-Type", "application/json")
	secondW := httptest.NewRecorder()
	h.ServeHTTP(secondW, secondReq)
	if secondW.Code != http.StatusOK {
		t.Fatalf("expected second request 200, got %d: %s", secondW.Code, secondW.Body.String())
	}
	if !strings.Contains(secondW.Body.String(), "next step is hedge") {
		t.Fatalf("expected second response text, got %s", secondW.Body.String())
	}
	if len(xaiBodies) != 3 {
		t.Fatalf("expected 3 xai calls across both turns, got %d", len(xaiBodies))
	}
}

func TestHandlerManagedToolHTTPErrorFeedsBackToModel(t *testing.T) {
	var xaiBodies [][]byte
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_, _ = w.Write([]byte(`{"message":"backend unavailable"}`))
	}))
	defer toolSrv.Close()

	xaiBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read xai body: %v", err)
		}
		xaiBodies = append(xaiBodies, body)
		w.Header().Set("Content-Type", "application/json")
		switch len(xaiBodies) {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-1",
				"choices":[{
					"finish_reason":"tool_calls",
					"message":{
						"role":"assistant",
						"tool_calls":[{"id":"call_1","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}}]
					}
				}]
			}`))
		case 2:
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal xai follow-up: %v", err)
			}
			messages := payload["messages"].([]any)
			last := messages[len(messages)-1].(map[string]any)
			var toolResult map[string]any
			if err := json.Unmarshal([]byte(last["content"].(string)), &toolResult); err != nil {
				t.Fatalf("unmarshal tool result: %v", err)
			}
			if ok, _ := toolResult["ok"].(bool); ok {
				t.Fatalf("expected failed tool result, got %+v", toolResult)
			}
			errPayload := toolResult["error"].(map[string]any)
			if errPayload["code"] != "http_502" {
				t.Fatalf("expected http_502 tool error, got %+v", toolResult)
			}
			_, _ = w.Write([]byte(`{"id":"chatcmpl-2","choices":[{"message":{"role":"assistant","content":"tool failed cleanly"}}]}`))
		default:
			t.Fatalf("unexpected xai round %d", len(xaiBodies))
		}
	}))
	defer xaiBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: xaiBackend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard))

	body := `{"model":"xai/grok-4.1-fast","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "tool failed cleanly") {
		t.Fatalf("expected recovery text after tool failure, got %s", w.Body.String())
	}
	if len(xaiBodies) != 2 {
		t.Fatalf("expected 2 xai rounds, got %d", len(xaiBodies))
	}
}

func TestHandlerManagedToolMaxRoundsFailsClosed(t *testing.T) {
	xaiCalls := 0
	toolCalls := 0
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	xaiBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		xaiCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"chatcmpl-loop",
			"choices":[{
				"finish_reason":"tool_calls",
				"message":{
					"role":"assistant",
					"tool_calls":[{"id":"call_1","type":"function","function":{"name":"trading-api.get_market_context","arguments":"{}"}}]
				}
			}]
		}`))
	}))
	defer xaiBackend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: xaiBackend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	tools := managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")
	tools.Policy.MaxRounds = 1
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", tools), logging.New(io.Discard))

	body := `{"model":"xai/grok-4.1-fast","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadGateway {
		t.Fatalf("expected 502, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "managed tool max rounds exceeded") {
		t.Fatalf("expected max_rounds error, got %s", w.Body.String())
	}
	if xaiCalls != 2 {
		t.Fatalf("expected 2 xai calls before max_rounds failure, got %d", xaiCalls)
	}
	if toolCalls != 1 {
		t.Fatalf("expected 1 tool execution before max_rounds failure, got %d", toolCalls)
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

func TestHandlerExecutesManagedAnthropicTools(t *testing.T) {
	presentedName := managedToolPresentedNameForCanonical("trading-api.get_market_context")
	var anthropicBodies [][]byte
	var toolAuth string
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolAuth = r.Header.Get("Authorization")
		if r.URL.Path != "/api/v1/market_context/nano-bot" {
			t.Fatalf("unexpected tool path: %s", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000,"positions":[{"symbol":"NVDA","qty":2}]}`))
	}))
	defer toolSrv.Close()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read anthropic body: %v", err)
		}
		anthropicBodies = append(anthropicBodies, body)
		w.Header().Set("Content-Type", "application/json")
		switch len(anthropicBodies) {
		case 1:
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal anthropic request: %v", err)
			}
			tools, _ := payload["tools"].([]any)
			if len(tools) != 2 {
				t.Fatalf("expected runner-native plus managed anthropic tools, got %+v", payload["tools"])
			}
			if !anthropicToolsIncludeName(tools, "runner_local") || !anthropicToolsIncludeName(tools, presentedName) {
				t.Fatalf("expected merged anthropic tool list to include runner_local and %q, got %+v", presentedName, payload["tools"])
			}
			toolChoice, _ := payload["tool_choice"].(map[string]any)
			if toolChoice == nil || toolChoice["type"] != "tool" || toolChoice["name"] != "runner_local" {
				t.Fatalf("expected native anthropic tool_choice preserved, got %+v", payload)
			}
			_, _ = w.Write([]byte(`{
				"id":"msg_01",
				"type":"message",
				"content":[{"type":"tool_use","id":"toolu_1","name":"` + presentedName + `","input":{}}],
				"stop_reason":"tool_use",
				"usage":{"input_tokens":11,"output_tokens":4}
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"msg_02",
				"type":"message",
				"model":"claude-sonnet-4-20250514",
				"content":[{"type":"text","text":"market context loaded"}],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":7,"output_tokens":5}
			}`))
		default:
			t.Fatalf("unexpected anthropic round %d", len(anthropicBodies))
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	histDir := t.TempDir()
	h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "svc-token")), logging.New(io.Discard),
		WithSessionHistory(histDir))
	body := `{
		"model":"claude-sonnet-4-20250514",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"runner_local","description":"native runner tool","input_schema":{"type":"object"}}],
		"tool_choice":{"type":"tool","name":"runner_local"}
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "market context loaded") {
		t.Fatalf("expected final anthropic text response, got %s", w.Body.String())
	}
	if len(anthropicBodies) != 2 {
		t.Fatalf("expected 2 anthropic rounds, got %d", len(anthropicBodies))
	}
	if toolAuth != "Bearer svc-token" {
		t.Fatalf("expected projected tool auth, got %q", toolAuth)
	}

	var secondPayload map[string]any
	if err := json.Unmarshal(anthropicBodies[1], &secondPayload); err != nil {
		t.Fatalf("unmarshal second anthropic request: %v", err)
	}
	if secondPayload["model"] != "claude-sonnet-4-20250514" {
		t.Fatalf("expected anthropic model unchanged, got %#v", secondPayload["model"])
	}
	messages, _ := secondPayload["messages"].([]any)
	if len(messages) < 3 {
		t.Fatalf("expected mediated anthropic follow-up messages, got %+v", secondPayload)
	}
	last := messages[len(messages)-1].(map[string]any)
	if last["role"] != "user" {
		t.Fatalf("expected final follow-up message to be user tool_result, got %+v", last)
	}
	blocks, _ := last["content"].([]any)
	if len(blocks) != 1 {
		t.Fatalf("expected one tool_result block, got %+v", last)
	}
	resultBlock := blocks[0].(map[string]any)
	if resultBlock["type"] != "tool_result" || resultBlock["tool_use_id"] != "toolu_1" {
		t.Fatalf("unexpected tool_result block: %+v", resultBlock)
	}
	var toolResult map[string]any
	if err := json.Unmarshal([]byte(resultBlock["content"].(string)), &toolResult); err != nil {
		t.Fatalf("unmarshal tool result: %v", err)
	}
	if ok, _ := toolResult["ok"].(bool); !ok {
		t.Fatalf("expected successful tool result, got %+v", toolResult)
	}
	data, _ := toolResult["data"].(map[string]any)
	if data["balance"].(float64) != 5000 {
		t.Fatalf("expected tool payload balance, got %+v", toolResult)
	}

	histFile := filepath.Join(histDir, "nano-bot", "history.jsonl")
	rawHist, err := os.ReadFile(histFile)
	if err != nil {
		t.Fatalf("read history: %v", err)
	}
	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimRight(rawHist, "\n"), &entry); err != nil {
		t.Fatalf("unmarshal history entry: %v", err)
	}
	if entry["status"] != "ok" {
		t.Fatalf("expected history status=ok, got %+v", entry)
	}
	usage, _ := entry["usage"].(map[string]any)
	if usage["total_rounds"].(float64) != 2 {
		t.Fatalf("expected total_rounds=2, got %+v", usage)
	}
	trace, _ := entry["tool_trace"].([]any)
	if len(trace) != 1 {
		t.Fatalf("expected one tool trace round, got %+v", entry)
	}
	round := trace[0].(map[string]any)
	toolCalls := round["tool_calls"].([]any)
	if len(toolCalls) != 1 {
		t.Fatalf("expected one tool call in trace, got %+v", round)
	}
	call := toolCalls[0].(map[string]any)
	if call["name"] != "trading-api.get_market_context" {
		t.Fatalf("unexpected tool trace name: %+v", call)
	}
}

func TestHandlerRestreamsManagedAnthropicTools(t *testing.T) {
	var anthropicBodies [][]byte
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read anthropic body: %v", err)
		}
		anthropicBodies = append(anthropicBodies, body)
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("unmarshal anthropic request: %v", err)
		}
		if stream, _ := payload["stream"].(bool); stream {
			t.Fatalf("expected managed anthropic upstream request to force stream=false, got %+v", payload)
		}
		w.Header().Set("Content-Type", "application/json")
		switch len(anthropicBodies) {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"msg_01",
				"type":"message",
				"content":[{"type":"tool_use","id":"toolu_1","name":"trading-api.get_market_context","input":{}}],
				"stop_reason":"tool_use",
				"usage":{"input_tokens":10,"output_tokens":3}
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"msg_02",
				"type":"message",
				"model":"claude-sonnet-4-20250514",
				"content":[{"type":"text","text":"market context loaded"}],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":7,"output_tokens":5}
			}`))
		default:
			t.Fatalf("unexpected anthropic round %d", len(anthropicBodies))
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	histDir := t.TempDir()
	h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard),
		WithSessionHistory(histDir))
	body := `{"model":"claude-sonnet-4-20250514","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Fatalf("expected SSE content-type, got %q", got)
	}
	bodyText := w.Body.String()
	if !strings.Contains(bodyText, "event: message_start") || !strings.Contains(bodyText, "event: message_stop") {
		t.Fatalf("expected anthropic SSE envelope, got %s", bodyText)
	}
	events := parseSSEEvents(t, bodyText)
	var sawText bool
	var sawInput bool
	var sawOutput bool
	for _, event := range events {
		if event["type"] == "content_block_delta" {
			delta, _ := event["delta"].(map[string]any)
			if delta["text"] == "market context loaded" {
				sawText = true
			}
		}
		if event["type"] == "message_start" {
			msg, _ := event["message"].(map[string]any)
			usage, _ := msg["usage"].(map[string]any)
			if usage != nil && int(usage["input_tokens"].(float64)) == 17 {
				sawInput = true
			}
		}
		if event["type"] == "message_delta" {
			usage, _ := event["usage"].(map[string]any)
			if usage != nil && int(usage["output_tokens"].(float64)) == 8 {
				sawOutput = true
			}
		}
	}
	if !sawText || !sawInput || !sawOutput {
		t.Fatalf("expected synthetic anthropic text and aggregated usage, got %+v", events)
	}

	histFile := filepath.Join(histDir, "nano-bot", "history.jsonl")
	rawHist, err := os.ReadFile(histFile)
	if err != nil {
		t.Fatalf("read history: %v", err)
	}
	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimRight(rawHist, "\n"), &entry); err != nil {
		t.Fatalf("unmarshal history entry: %v", err)
	}
	if stream, _ := entry["stream"].(bool); !stream {
		t.Fatalf("expected history stream=true, got %+v", entry)
	}
	resp, _ := entry["response"].(map[string]any)
	if resp["format"] != "sse" {
		t.Fatalf("expected response.format=sse, got %+v", resp)
	}
}

func TestHandlerPassesThroughNativeAnthropicToolUseWhenManagedToolsArePresent(t *testing.T) {
	var anthropicBodies [][]byte
	toolCalls := 0
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read anthropic body: %v", err)
		}
		anthropicBodies = append(anthropicBodies, body)
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("unmarshal anthropic request: %v", err)
		}
		if stream, _ := payload["stream"].(bool); stream {
			t.Fatalf("expected managed/native anthropic upstream request to force stream=false, got %+v", payload)
		}
		tools, _ := payload["tools"].([]any)
		if len(tools) != 2 {
			t.Fatalf("expected merged anthropic tool list upstream, got %+v", payload["tools"])
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"msg_native",
			"type":"message",
			"model":"claude-sonnet-4-20250514",
			"content":[{"type":"tool_use","id":"toolu_native_1","name":"runner_local","input":{"ticker":"NVDA"}}],
			"stop_reason":"tool_use",
			"usage":{"input_tokens":8,"output_tokens":3}
		}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard))
	body := `{
		"model":"claude-sonnet-4-20250514",
		"stream":true,
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"runner_local","description":"native runner tool","input_schema":{"type":"object"}}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if toolCalls != 0 {
		t.Fatalf("expected managed tool executor to stay idle for native anthropic tool_use, got %d", toolCalls)
	}
	if len(anthropicBodies) != 1 {
		t.Fatalf("expected one upstream anthropic round, got %d", len(anthropicBodies))
	}
	if got := w.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Fatalf("expected SSE content-type, got %q", got)
	}
	if !anthropicSSEHasToolUse(parseSSEEvents(t, w.Body.String()), "runner_local") {
		t.Fatalf("expected re-streamed native anthropic tool_use, got %s", w.Body.String())
	}
}

func TestHandlerRetriesUnsafeMixedAnthropicToolUsesInternally(t *testing.T) {
	presentedName := managedToolPresentedNameForCanonical("trading-api.get_market_context")
	anthropicCalls := 0
	toolCalls := 0
	var logs bytes.Buffer
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		anthropicCalls++
		w.Header().Set("Content-Type", "application/json")
		switch anthropicCalls {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"msg_unsafe_mixed",
				"type":"message",
				"role":"assistant",
				"content":[
					{"type":"text","text":"I will use the local runner first, then fetch managed context."},
					{"type":"tool_use","id":"toolu_native_1","name":"runner_local","input":{}},
					{"type":"tool_use","id":"toolu_managed_1","name":"` + presentedName + `","input":{}}
				],
				"stop_reason":"tool_use",
				"usage":{"input_tokens":8,"output_tokens":3}
			}`))
		case 2:
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("read anthropic body: %v", err)
			}
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal anthropic request: %v", err)
			}
			rawMessages, _ := payload["messages"].([]any)
			if len(rawMessages) != 3 {
				t.Fatalf("expected unsafe mixed retry messages to stay internal, got %+v", payload)
			}
			mixedAssistant, _ := rawMessages[1].(map[string]any)
			if mixedAssistant["role"] != "assistant" {
				t.Fatalf("expected mixed assistant message preserved, got %+v", mixedAssistant)
			}
			assistantBlocks, _ := mixedAssistant["content"].([]any)
			if len(assistantBlocks) != 3 {
				t.Fatalf("expected assistant content plus both tool_use blocks, got %+v", mixedAssistant)
			}
			rejectedResultMsg, _ := rawMessages[2].(map[string]any)
			if rejectedResultMsg["role"] != "user" {
				t.Fatalf("expected user tool_result message, got %+v", rejectedResultMsg)
			}
			resultBlocks, _ := rejectedResultMsg["content"].([]any)
			if len(resultBlocks) != 2 {
				t.Fatalf("expected two rejection tool_result blocks, got %+v", rejectedResultMsg)
			}
			rejectedNative, _ := resultBlocks[0].(map[string]any)
			if rejectedNative["type"] != "tool_result" || rejectedNative["tool_use_id"] != "toolu_native_1" || rejectedNative["is_error"] != true {
				t.Fatalf("expected native rejection tool_result, got %+v", rejectedNative)
			}
			assertToolErrorContains(t, rejectedNative["content"], mixedToolOrderMessage)
			rejectedManaged, _ := resultBlocks[1].(map[string]any)
			if rejectedManaged["type"] != "tool_result" || rejectedManaged["tool_use_id"] != "toolu_managed_1" || rejectedManaged["is_error"] != true {
				t.Fatalf("expected managed rejection tool_result, got %+v", rejectedManaged)
			}
			assertToolErrorContains(t, rejectedManaged["content"], mixedToolOrderMessage)
			_, _ = w.Write([]byte(`{
				"id":"msg_replanned",
				"type":"message",
				"role":"assistant",
				"content":[{"type":"text","text":"Understood. I need to fetch managed context before I ask for local runner work."}],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":12,"output_tokens":4}
			}`))
		default:
			t.Fatalf("unexpected anthropic round %d", anthropicCalls)
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(&logs))
	body := `{
		"model":"claude-sonnet-4-20250514",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"runner_local","description":"native runner tool","input_schema":{"type":"object"}}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "Understood. I need to fetch managed context before I ask for local runner work.") {
		t.Fatalf("expected internal anthropic retry reply, got %s", w.Body.String())
	}
	if toolCalls != 0 {
		t.Fatalf("expected no managed tool execution during unsafe mixed retry, got %d", toolCalls)
	}
	if anthropicCalls != 2 {
		t.Fatalf("expected one internal anthropic retry round, got %d model calls", anthropicCalls)
	}
	assertInterventionLogged(t, logs.Bytes(), mixedToolOrderInternalRetryIntervention)
}

func TestHandlerSerializesManagedPrefixBeforeNativeAnthropicToolUse(t *testing.T) {
	presentedName := managedToolPresentedNameForCanonical("trading-api.get_market_context")
	anthropicCalls := 0
	toolCalls := 0
	var logs bytes.Buffer
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		anthropicCalls++
		w.Header().Set("Content-Type", "application/json")
		switch anthropicCalls {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"msg_mixed_first",
				"type":"message",
				"content":[
					{"type":"text","text":"Checking managed context first."},
					{"type":"tool_use","id":"toolu_managed_1","name":"` + presentedName + `","input":{}},
					{"type":"tool_use","id":"toolu_native_1","name":"runner_local","input":{}}
				],
				"stop_reason":"tool_use",
				"usage":{"input_tokens":10,"output_tokens":3}
			}`))
		case 2:
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("read anthropic body: %v", err)
			}
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal anthropic request: %v", err)
			}
			rawMessages, _ := payload["messages"].([]any)
			if len(rawMessages) != 3 {
				t.Fatalf("expected only managed prefix to be serialized before rerun, got %+v", payload)
			}
			first := rawMessages[0].(map[string]any)
			if first["role"] != "user" || first["content"] != "hi" {
				t.Fatalf("unexpected first conversation message: %+v", first)
			}
			managedAssistant := rawMessages[1].(map[string]any)
			managedBlocks, _ := managedAssistant["content"].([]any)
			if managedAssistant["role"] != "assistant" || len(managedBlocks) != 2 {
				t.Fatalf("expected managed-only anthropic assistant, got %+v", managedAssistant)
			}
			if managedBlocks[0].(map[string]any)["type"] != "text" || managedBlocks[1].(map[string]any)["id"] != "toolu_managed_1" {
				t.Fatalf("expected managed prefix preserved, got %+v", managedBlocks)
			}
			managedResult := rawMessages[2].(map[string]any)
			managedResultBlocks, _ := managedResult["content"].([]any)
			if managedResult["role"] != "user" || len(managedResultBlocks) != 1 || managedResultBlocks[0].(map[string]any)["tool_use_id"] != "toolu_managed_1" {
				t.Fatalf("expected managed tool_result after serialization, got %+v", managedResult)
			}
			_, _ = w.Write([]byte(`{
				"id":"msg_native_second",
				"type":"message",
				"content":[{"type":"tool_use","id":"toolu_native_1","name":"runner_local","input":{}}],
				"stop_reason":"tool_use",
				"usage":{"input_tokens":7,"output_tokens":4}
			}`))
		case 3:
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("read anthropic body: %v", err)
			}
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal anthropic request: %v", err)
			}
			rawMessages, _ := payload["messages"].([]any)
			if len(rawMessages) < 5 {
				t.Fatalf("expected managed handoff continuity injection, got %+v", payload)
			}
			hiddenAssistant := rawMessages[1].(map[string]any)
			hiddenAssistantBlocks, _ := hiddenAssistant["content"].([]any)
			if hiddenAssistant["role"] != "assistant" || len(hiddenAssistantBlocks) != 2 {
				t.Fatalf("expected hidden managed assistant before native tool use, got %+v", hiddenAssistant)
			}
			hiddenUser := rawMessages[2].(map[string]any)
			hiddenUserBlocks, _ := hiddenUser["content"].([]any)
			if hiddenUser["role"] != "user" || len(hiddenUserBlocks) != 1 || hiddenUserBlocks[0].(map[string]any)["tool_use_id"] != "toolu_managed_1" {
				t.Fatalf("expected hidden managed tool_result after tool_use, got %+v", hiddenUser)
			}
			nativeAssistant := rawMessages[3].(map[string]any)
			nativeAssistantBlocks, _ := nativeAssistant["content"].([]any)
			if nativeAssistant["role"] != "assistant" || len(nativeAssistantBlocks) != 1 {
				t.Fatalf("expected native tool_use after injected hidden rounds, got %+v", nativeAssistant)
			}
			nativeUser := rawMessages[4].(map[string]any)
			nativeUserBlocks, _ := nativeUser["content"].([]any)
			if nativeUser["role"] != "user" || len(nativeUserBlocks) != 1 || nativeUserBlocks[0].(map[string]any)["tool_use_id"] != "toolu_native_1" {
				t.Fatalf("expected native tool_result after injected hidden rounds, got %+v", nativeUser)
			}
			_, _ = w.Write([]byte(`{
				"id":"msg_final",
				"type":"message",
				"content":[{"type":"text","text":"native handoff complete"}],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":6,"output_tokens":5}
			}`))
		default:
			t.Fatalf("unexpected anthropic round %d", anthropicCalls)
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(&logs))
	body := `{
		"model":"claude-sonnet-4-20250514",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"runner_local","description":"native runner tool","input_schema":{"type":"object"}}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "\"runner_local\"") {
		t.Fatalf("expected native anthropic tool_use to be returned to runner, got %s", w.Body.String())
	}
	if anthropicCalls != 2 {
		t.Fatalf("expected two anthropic rounds before handoff, got %d", anthropicCalls)
	}
	if toolCalls != 1 {
		t.Fatalf("expected one managed tool execution before handoff, got %d", toolCalls)
	}

	followupReq := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{
		"model":"claude-sonnet-4-20250514",
		"messages":[
			{"role":"user","content":"hi"},
			{"role":"assistant","content":[{"type":"tool_use","id":"toolu_native_1","name":"runner_local","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_native_1","content":"{\"native\":true}"}]}
		],
		"tools":[{"name":"runner_local","description":"native runner tool","input_schema":{"type":"object"}}]
	}`))
	followupReq.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	followupReq.Header.Set("Content-Type", "application/json")
	followupReq.Header.Set("Anthropic-Version", "2023-06-01")
	followupW := httptest.NewRecorder()

	h.ServeHTTP(followupW, followupReq)

	if followupW.Code != http.StatusOK {
		t.Fatalf("expected follow-up request 200, got %d: %s", followupW.Code, followupW.Body.String())
	}
	if !strings.Contains(followupW.Body.String(), "native handoff complete") {
		t.Fatalf("expected final text after native handoff, got %s", followupW.Body.String())
	}
	if anthropicCalls != 3 {
		t.Fatalf("expected third anthropic round after runner tool result, got %d", anthropicCalls)
	}
	assertInterventionLogged(t, logs.Bytes(), managedMixedPrefixSerializedIntervention)
}

func TestBuildAnthropicAssistantMessageRetainsManagedBlockWhenUpstreamIDMissing(t *testing.T) {
	assistantMessage, toolUses, err := parseAnthropicToolResponse([]byte(`{
		"content":[
			{"type":"text","text":"Checking managed context first."},
			{"type":"tool_use","name":"trading-api.get_market_context","input":{}},
			{"type":"tool_use","id":"toolu_native_1","name":"runner_local","input":{}}
		]
	}`))
	if err != nil {
		t.Fatalf("parse anthropic response: %v", err)
	}
	if len(toolUses) != 2 {
		t.Fatalf("expected two parsed tool uses, got %d", len(toolUses))
	}
	if toolUses[0].ID != "toolu_2" {
		t.Fatalf("expected synthetic id for id-less managed tool_use, got %q", toolUses[0].ID)
	}

	managedAssistant := buildAnthropicAssistantMessage(assistantMessage, toolUses[:1], true)
	blocks, _ := managedAssistant["content"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("expected text plus managed tool_use after filtering, got %+v", managedAssistant["content"])
	}

	first, _ := blocks[0].(map[string]any)
	if first == nil || first["type"] != "text" {
		t.Fatalf("expected leading text block preserved, got %+v", blocks[0])
	}
	second, _ := blocks[1].(map[string]any)
	if second == nil || second["type"] != "tool_use" || second["name"] != "trading-api.get_market_context" {
		t.Fatalf("expected managed tool_use preserved after synthetic-id filtering, got %+v", blocks[1])
	}
	if _, ok := second["id"]; ok {
		t.Fatalf("expected filtered managed tool_use to preserve missing upstream id, got %+v", second)
	}
}

func TestHandlerHandsOffNativeAnthropicToolUseAfterManagedRounds(t *testing.T) {
	anthropicCalls := 0
	toolCalls := 0
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolCalls++
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		anthropicCalls++
		w.Header().Set("Content-Type", "application/json")
		switch anthropicCalls {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"msg_managed_first",
				"type":"message",
				"content":[{"type":"tool_use","id":"toolu_managed_1","name":"trading-api.get_market_context","input":{}}],
				"stop_reason":"tool_use",
				"usage":{"input_tokens":10,"output_tokens":3}
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"msg_native_second",
				"type":"message",
				"content":[{"type":"tool_use","id":"toolu_native_1","name":"runner_local","input":{}}],
				"stop_reason":"tool_use",
				"usage":{"input_tokens":7,"output_tokens":4}
			}`))
		case 3:
			body, err := io.ReadAll(r.Body)
			if err != nil {
				t.Fatalf("read anthropic body: %v", err)
			}
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal anthropic request: %v", err)
			}
			rawMessages, _ := payload["messages"].([]any)
			if len(rawMessages) < 5 {
				t.Fatalf("expected managed handoff continuity injection, got %+v", payload)
			}
			first := rawMessages[0].(map[string]any)
			if first["role"] != "user" || first["content"] != "hi" {
				t.Fatalf("unexpected first conversation message: %+v", first)
			}
			hiddenAssistant := rawMessages[1].(map[string]any)
			hiddenAssistantBlocks, _ := hiddenAssistant["content"].([]any)
			if hiddenAssistant["role"] != "assistant" || len(hiddenAssistantBlocks) != 1 {
				t.Fatalf("expected hidden managed assistant before native tool use, got %+v", hiddenAssistant)
			}
			if hiddenAssistantBlocks[0].(map[string]any)["id"] != "toolu_managed_1" {
				t.Fatalf("expected hidden managed tool_use block, got %+v", hiddenAssistantBlocks[0])
			}
			hiddenUser := rawMessages[2].(map[string]any)
			hiddenUserBlocks, _ := hiddenUser["content"].([]any)
			if hiddenUser["role"] != "user" || len(hiddenUserBlocks) != 1 || hiddenUserBlocks[0].(map[string]any)["type"] != "tool_result" {
				t.Fatalf("expected hidden managed tool_result after tool_use, got %+v", hiddenUser)
			}
			nativeAssistant := rawMessages[3].(map[string]any)
			nativeAssistantBlocks, _ := nativeAssistant["content"].([]any)
			if nativeAssistant["role"] != "assistant" || len(nativeAssistantBlocks) != 1 {
				t.Fatalf("expected native tool_use after injected hidden rounds, got %+v", nativeAssistant)
			}
			if nativeAssistantBlocks[0].(map[string]any)["id"] != "toolu_native_1" {
				t.Fatalf("expected native tool_use id preserved, got %+v", nativeAssistantBlocks[0])
			}
			nativeUser := rawMessages[4].(map[string]any)
			nativeUserBlocks, _ := nativeUser["content"].([]any)
			if nativeUser["role"] != "user" || len(nativeUserBlocks) != 1 || nativeUserBlocks[0].(map[string]any)["tool_use_id"] != "toolu_native_1" {
				t.Fatalf("expected native tool_result after injected hidden rounds, got %+v", nativeUser)
			}
			_, _ = w.Write([]byte(`{
				"id":"msg_final",
				"type":"message",
				"content":[{"type":"text","text":"native handoff complete"}],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":6,"output_tokens":5}
			}`))
		default:
			t.Fatalf("unexpected anthropic round %d", anthropicCalls)
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard))
	body := `{
		"model":"claude-sonnet-4-20250514",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"name":"runner_local","description":"native runner tool","input_schema":{"type":"object"}}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "\"runner_local\"") {
		t.Fatalf("expected native anthropic tool_use to be returned to runner, got %s", w.Body.String())
	}
	if anthropicCalls != 2 {
		t.Fatalf("expected two anthropic rounds before handoff, got %d", anthropicCalls)
	}
	if toolCalls != 1 {
		t.Fatalf("expected one managed tool execution before handoff, got %d", toolCalls)
	}

	followupReq := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{
		"model":"claude-sonnet-4-20250514",
		"messages":[
			{"role":"user","content":"hi"},
			{"role":"assistant","content":[{"type":"tool_use","id":"toolu_native_1","name":"runner_local","input":{}}]},
			{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_native_1","content":"{\"native\":true}"}]}
		],
		"tools":[{"name":"runner_local","description":"native runner tool","input_schema":{"type":"object"}}]
	}`))
	followupReq.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	followupReq.Header.Set("Content-Type", "application/json")
	followupReq.Header.Set("Anthropic-Version", "2023-06-01")
	followupW := httptest.NewRecorder()

	h.ServeHTTP(followupW, followupReq)

	if followupW.Code != http.StatusOK {
		t.Fatalf("expected follow-up request 200, got %d: %s", followupW.Code, followupW.Body.String())
	}
	if !strings.Contains(followupW.Body.String(), "native handoff complete") {
		t.Fatalf("expected final anthropic text after native handoff, got %s", followupW.Body.String())
	}
	if anthropicCalls != 3 {
		t.Fatalf("expected third anthropic round after runner tool result, got %d", anthropicCalls)
	}
}

func TestHandlerStreamsManagedAnthropicKeepaliveComments(t *testing.T) {
	toolRelease := make(chan struct{})
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		<-toolRelease
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	var anthropicBodies [][]byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read anthropic body: %v", err)
		}
		anthropicBodies = append(anthropicBodies, body)
		w.Header().Set("Content-Type", "application/json")
		switch len(anthropicBodies) {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"msg_01",
				"type":"message",
				"content":[{"type":"tool_use","id":"toolu_1","name":"trading-api.get_market_context","input":{}}],
				"stop_reason":"tool_use",
				"usage":{"input_tokens":10,"output_tokens":3}
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"msg_02",
				"type":"message",
				"model":"claude-sonnet-4-20250514",
				"content":[{"type":"text","text":"market context loaded"}],
				"stop_reason":"end_turn",
				"usage":{"input_tokens":7,"output_tokens":5}
			}`))
		default:
			t.Fatalf("unexpected anthropic round %d", len(anthropicBodies))
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard))
	proxySrv := httptest.NewServer(h)
	defer proxySrv.Close()

	req, err := http.NewRequest(http.MethodPost, proxySrv.URL+"/v1/messages", bytes.NewBufferString(`{"model":"claude-sonnet-4-20250514","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	if err != nil {
		t.Fatalf("NewRequest: %v", err)
	}
	req.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("Do: %v", err)
	}
	defer resp.Body.Close()

	bodyText, sawComment := readManagedKeepaliveStream(t, resp.Body, ": managed tool round 1 executing trading-api.get_market_context", func() {
		close(toolRelease)
	})
	if !sawComment {
		t.Fatalf("expected keepalive comment in anthropic stream, got %s", bodyText)
	}
	if !strings.Contains(bodyText, "event: message_stop") {
		t.Fatalf("expected anthropic stream completion, got %s", bodyText)
	}
	var sawText bool
	for _, event := range parseSSEEvents(t, bodyText) {
		if event["type"] == "content_block_delta" {
			delta, _ := event["delta"].(map[string]any)
			if delta["text"] == "market context loaded" {
				sawText = true
			}
		}
	}
	if !sawText {
		t.Fatalf("expected anthropic final text, got %s", bodyText)
	}
}

func TestHandlerReinjectsManagedAnthropicContinuityOnNextTurn(t *testing.T) {
	var anthropicBodies [][]byte
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000,"positions":[{"symbol":"NVDA","qty":2}]}`))
	}))
	defer toolSrv.Close()

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read anthropic body: %v", err)
		}
		anthropicBodies = append(anthropicBodies, body)
		w.Header().Set("Content-Type", "application/json")
		switch len(anthropicBodies) {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"msg_01",
				"type":"message",
				"content":[{"type":"tool_use","id":"toolu_1","name":"trading-api.get_market_context","input":{}}],
				"stop_reason":"tool_use"
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"msg_02",
				"type":"message",
				"content":[{"type":"text","text":"market context loaded"}],
				"stop_reason":"end_turn"
			}`))
		case 3:
			var payload map[string]any
			if err := json.Unmarshal(body, &payload); err != nil {
				t.Fatalf("unmarshal anthropic request: %v", err)
			}
			rawMessages, _ := payload["messages"].([]any)
			if len(rawMessages) < 5 {
				t.Fatalf("expected continuity-injected anthropic conversation, got %+v", payload)
			}
			first := rawMessages[0].(map[string]any)
			if first["role"] != "user" || first["content"] != "hi" {
				t.Fatalf("unexpected first conversation message: %+v", first)
			}
			hiddenAssistant := rawMessages[1].(map[string]any)
			if hiddenAssistant["role"] != "assistant" {
				t.Fatalf("expected hidden assistant tool_use before visible reply, got %+v", hiddenAssistant)
			}
			hiddenAssistantBlocks, _ := hiddenAssistant["content"].([]any)
			if len(hiddenAssistantBlocks) != 1 {
				t.Fatalf("expected hidden assistant tool_use block, got %+v", hiddenAssistant)
			}
			if hiddenAssistantBlocks[0].(map[string]any)["type"] != "tool_use" {
				t.Fatalf("expected tool_use block, got %+v", hiddenAssistantBlocks[0])
			}
			hiddenUser := rawMessages[2].(map[string]any)
			if hiddenUser["role"] != "user" {
				t.Fatalf("expected hidden user tool_result after tool_use, got %+v", hiddenUser)
			}
			hiddenUserBlocks, _ := hiddenUser["content"].([]any)
			if len(hiddenUserBlocks) != 1 || hiddenUserBlocks[0].(map[string]any)["type"] != "tool_result" {
				t.Fatalf("expected tool_result block, got %+v", hiddenUser)
			}
			visibleAssistant := rawMessages[3].(map[string]any)
			if visibleAssistant["role"] != "assistant" {
				t.Fatalf("expected visible assistant reply preserved after hidden turns, got %+v", visibleAssistant)
			}
			if text := strings.Join(anthropicMessageTextBlocks(visibleAssistant), "\n"); text != "market context loaded" {
				t.Fatalf("expected preserved visible assistant reply, got %+v", visibleAssistant)
			}
			last := rawMessages[4].(map[string]any)
			if last["role"] != "user" || last["content"] != "what next?" {
				t.Fatalf("expected new user turn to remain after continuity injection, got %+v", last)
			}
			_, _ = w.Write([]byte(`{
				"id":"msg_03",
				"type":"message",
				"content":[{"type":"text","text":"next step is hedge"}],
				"stop_reason":"end_turn"
			}`))
		default:
			t.Fatalf("unexpected anthropic round %d", len(anthropicBodies))
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("nano-bot", "nano-bot:dummy456", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard))

	firstReq := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"claude-sonnet-4-20250514","messages":[{"role":"user","content":"hi"}]}`))
	firstReq.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	firstReq.Header.Set("Content-Type", "application/json")
	firstReq.Header.Set("Anthropic-Version", "2023-06-01")
	firstW := httptest.NewRecorder()
	h.ServeHTTP(firstW, firstReq)
	if firstW.Code != http.StatusOK {
		t.Fatalf("expected first request 200, got %d: %s", firstW.Code, firstW.Body.String())
	}
	if !strings.Contains(firstW.Body.String(), "market context loaded") {
		t.Fatalf("expected first response to complete mediated anthropic tool turn, got %s", firstW.Body.String())
	}

	secondReq := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{
		"model":"claude-sonnet-4-20250514",
		"messages":[
			{"role":"user","content":"hi"},
			{"role":"assistant","content":"market context loaded"},
			{"role":"user","content":"what next?"}
		]
	}`))
	secondReq.Header.Set("Authorization", "Bearer nano-bot:dummy456")
	secondReq.Header.Set("Content-Type", "application/json")
	secondReq.Header.Set("Anthropic-Version", "2023-06-01")
	secondW := httptest.NewRecorder()
	h.ServeHTTP(secondW, secondReq)
	if secondW.Code != http.StatusOK {
		t.Fatalf("expected second request 200, got %d: %s", secondW.Code, secondW.Body.String())
	}
	if !strings.Contains(secondW.Body.String(), "next step is hedge") {
		t.Fatalf("expected second response text, got %s", secondW.Body.String())
	}
	if len(anthropicBodies) != 3 {
		t.Fatalf("expected 3 anthropic calls across both turns, got %d", len(anthropicBodies))
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
	if !strings.Contains(content, "Current time: ") {
		t.Errorf("expected current time in system content, got: %s", content)
	}
	if !strings.Contains(content, "Wallet: $5,000") {
		t.Errorf("expected feed content in first message, got: %s", content)
	}
	if !strings.Contains(content, "BEGIN FEED: market-context") {
		t.Errorf("expected feed delimiter, got: %s", content)
	}
	// Feed content should appear before time (feeds set first, time appended).
	feedIdx := strings.Index(content, "BEGIN FEED:")
	timeIdx := strings.Index(content, "Current time:")
	if feedIdx > timeIdx {
		t.Errorf("expected feeds before time (append order), feed@%d time@%d", feedIdx, timeIdx)
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
	if !strings.Contains(sys, "Current time: ") {
		t.Errorf("expected current time in system content, got: %q", sys)
	}
	if !strings.Contains(sys, "Fleet nominal") {
		t.Errorf("expected feed in system field, got: %q", sys)
	}
	// Feed content should appear before time (feeds set first, time appended).
	feedIdx := strings.Index(sys, "BEGIN FEED:")
	timeIdx := strings.Index(sys, "Current time:")
	if feedIdx > timeIdx {
		t.Errorf("expected feeds before time (append order), feed@%d time@%d", feedIdx, timeIdx)
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

func TestHandlerLogsManagedToolManifestState(t *testing.T) {
	var logs lockedBuffer

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("weston", "weston:secret", managedToolManifest()), logging.New(&logs))
	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer weston:secret")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	entry := waitForLogEntry(t, &logs, func(entry map[string]any) bool {
		return entry["type"] == "tool_manifest_loaded"
	})
	if entry["claw_id"] != "weston" {
		t.Fatalf("expected claw_id=weston, got %+v", entry)
	}
	if entry["model"] != "openai/gpt-4o" {
		t.Fatalf("expected model openai/gpt-4o, got %+v", entry)
	}
	if entry["manifest_present"] != true {
		t.Fatalf("expected manifest_present=true, got %+v", entry)
	}
	if entry["tools_count"].(float64) != 1 {
		t.Fatalf("expected tools_count=1, got %+v", entry)
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
	if entry["id"] == "" {
		t.Errorf("expected stable history entry ID, got %+v", entry)
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

func TestHandlerAppliesRecallMemoryPolicyAndLogsRemovals(t *testing.T) {
	var logs lockedBuffer
	var gotBody []byte
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
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memories":[` +
			`{"text":"raw transcript should not inject","source":"raw_transcript"},` +
			`{"text":"   ","kind":"profile"},` +
			`{"text":"secret note sk-live-123456789","kind":"profile","source":"profile-store"}` +
			`]}`))
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
	var payload map[string]any
	if err := json.Unmarshal(gotBody, &payload); err != nil {
		t.Fatalf("unmarshal backend body: %v", err)
	}
	messages := payload["messages"].([]any)
	first := messages[0].(map[string]any)
	content := first["content"].(string)
	if strings.Contains(content, "raw transcript should not inject") {
		t.Fatalf("expected blocked recall source to be removed, got %q", content)
	}
	if strings.Contains(content, "sk-live-123456789") {
		t.Fatalf("expected recalled secret to be scrubbed, got %q", content)
	}
	if !strings.Contains(content, "[redacted-secret]") {
		t.Fatalf("expected recalled secret to be redacted, got %q", content)
	}

	recallLog := waitForLogEntry(t, &logs, func(entry map[string]any) bool {
		return entry["type"] == "memory_op" && entry["memory_op"] == "recall" && entry["memory_status"] == "succeeded"
	})
	if recallLog["memory_blocks"].(float64) != 3 {
		t.Fatalf("expected recall telemetry memory_blocks=3, got %+v", recallLog)
	}
	if recallLog["memory_removed"].(float64) != 2 {
		t.Fatalf("expected recall telemetry memory_removed=2, got %+v", recallLog)
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
		if entry["id"] == "" {
			t.Fatalf("expected retained entry ID, got %+v", payload)
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

func TestHandlerAppliesRetainMemoryPolicyAndLogsRedactions(t *testing.T) {
	var logs lockedBuffer
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"Bearer supersecrettoken"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`))
	}))
	defer backend.Close()

	retainBodyCh := make(chan []byte, 1)
	memorySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read retain body: %v", err)
		}
		retainBodyCh <- body
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
				"token": "tiverton:dummy123",
			},
			Memory: &agentctx.MemoryManifest{
				Version: 1,
				Service: "team-memory",
				BaseURL: memorySrv.URL,
				Retain:  &agentctx.MemoryOp{Path: "/retain"},
			},
		}, nil
	}, logging.New(&logs))

	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"my key is sk-live-123456789"}]}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d body=%s", w.Code, w.Body.String())
	}

	var retainBody []byte
	select {
	case retainBody = <-retainBodyCh:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for retain request")
	}
	if bytes.Contains(retainBody, []byte("sk-live-123456789")) {
		t.Fatalf("expected retain payload to scrub request secret, got %s", string(retainBody))
	}
	if bytes.Contains(retainBody, []byte("supersecrettoken")) {
		t.Fatalf("expected retain payload to scrub response bearer token, got %s", string(retainBody))
	}
	if !bytes.Contains(retainBody, []byte("[redacted-secret]")) || !bytes.Contains(retainBody, []byte("Bearer [redacted]")) {
		t.Fatalf("expected retain payload to include redacted placeholders, got %s", string(retainBody))
	}

	var payload map[string]any
	if err := json.Unmarshal(retainBody, &payload); err != nil {
		t.Fatalf("unmarshal retain payload: %v", err)
	}
	metadata := payload["metadata"].(map[string]any)
	if metadata["policy_removed"].(float64) != 3 {
		t.Fatalf("expected retain metadata policy_removed=3, got %+v", metadata)
	}

	retainLog := waitForLogEntry(t, &logs, func(entry map[string]any) bool {
		return entry["type"] == "memory_op" && entry["memory_op"] == "retain" && entry["memory_status"] == "succeeded"
	})
	if retainLog["memory_removed"].(float64) != 3 {
		t.Fatalf("expected retain telemetry memory_removed=3, got %+v", retainLog)
	}
}

func TestHandlerLogsOversizedMemoryRecallResponseClearly(t *testing.T) {
	var logs lockedBuffer
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
	failureLog := waitForLogEntry(t, &logs, func(entry map[string]any) bool {
		return entry["type"] == "memory_op" && entry["memory_op"] == "recall" && entry["memory_status"] == "failed"
	})
	if errText, _ := failureLog["error"].(string); !strings.Contains(errText, "recall response exceeds") {
		t.Fatalf("expected oversized recall error text in memory_op log, got %+v", failureLog)
	}
	if hasLogEntry(t, logs.Bytes(), func(entry map[string]any) bool { return entry["type"] == "error" }) {
		t.Fatalf("expected no generic error log for oversized recall, got %v", parseLogEntries(t, logs.Bytes()))
	}
}

func TestHandlerLogsSkippedMemoryRecallWithoutLatency(t *testing.T) {
	var logs lockedBuffer
	var gotBody []byte
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
	for _, raw := range messages {
		msg := raw.(map[string]any)
		if msg["role"] == "system" {
			if content, _ := msg["content"].(string); strings.Contains(content, "BEGIN MEMORY") {
				t.Fatalf("did not expect memory injection when recall is skipped, got %+v", msg)
			}
		}
	}
	skippedLog := waitForLogEntry(t, &logs, func(entry map[string]any) bool {
		return entry["type"] == "memory_op" && entry["memory_op"] == "recall" && entry["memory_status"] == "skipped"
	})
	if _, ok := skippedLog["latency_ms"]; ok {
		t.Fatalf("expected skipped recall log to omit latency_ms, got %+v", skippedLog)
	}
}

func TestHandlerLogsTimedOutMemoryRecallWithoutGenericError(t *testing.T) {
	var logs lockedBuffer
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer backend.Close()

	memorySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memories":[{"text":"late memory"}]}`))
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
				Recall:  &agentctx.MemoryOp{Path: "/recall", TimeoutMS: 10},
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
	timeoutLog := waitForLogEntry(t, &logs, func(entry map[string]any) bool {
		return entry["type"] == "memory_op" && entry["memory_op"] == "recall" && entry["memory_status"] == "timed_out"
	})
	if _, ok := timeoutLog["error"]; !ok {
		t.Fatalf("expected timed_out recall log to include error text, got %+v", timeoutLog)
	}
	if hasLogEntry(t, logs.Bytes(), func(entry map[string]any) bool { return entry["type"] == "error" }) {
		t.Fatalf("expected no generic error log for timed out recall, got %v", parseLogEntries(t, logs.Bytes()))
	}
}

func TestHandlerLogsFailedMemoryRetainWithoutGenericError(t *testing.T) {
	var logs lockedBuffer
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hello"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`))
	}))
	defer backend.Close()

	memorySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
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
	failedLog := waitForLogEntry(t, &logs, func(entry map[string]any) bool {
		return entry["type"] == "memory_op" && entry["memory_op"] == "retain" && entry["memory_status"] == "failed"
	})
	if failedLog["status_code"].(float64) != http.StatusBadGateway {
		t.Fatalf("expected retain failure status_code=%d, got %+v", http.StatusBadGateway, failedLog)
	}
	if hasLogEntry(t, logs.Bytes(), func(entry map[string]any) bool { return entry["type"] == "error" }) {
		t.Fatalf("expected no generic error log for failed retain, got %v", parseLogEntries(t, logs.Bytes()))
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

func stubContextLoaderWithTools(agentID, token string, tools *agentctx.ToolManifest) ContextLoader {
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
			Tools: tools,
		}, nil
	}
}

func managedToolManifest() *agentctx.ToolManifest {
	return managedToolManifestForURL("http://trading-api:4000", http.MethodGet, "/api/v1/market_context/{claw_id}", "")
}

func managedToolManifestForURL(baseURL, method, path, token string) *agentctx.ToolManifest {
	var auth *agentctx.AuthEntry
	if token != "" {
		auth = &agentctx.AuthEntry{
			Type:  "bearer",
			Token: token,
		}
	}
	return &agentctx.ToolManifest{
		Version: 1,
		Tools: []agentctx.ToolManifestEntry{{
			Name:        "trading-api.get_market_context",
			Description: "Retrieve market context",
			InputSchema: map[string]any{"type": "object"},
			Execution: agentctx.ToolExecution{
				Transport: "http",
				Service:   "trading-api",
				BaseURL:   baseURL,
				Method:    method,
				Path:      path,
				Auth:      auth,
			},
		}},
		Policy: agentctx.ToolPolicy{
			MaxRounds:        8,
			TimeoutPerToolMS: 30000,
			TotalTimeoutMS:   120000,
		},
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

func hasLogEntry(t *testing.T, raw []byte, predicate func(map[string]any) bool) bool {
	t.Helper()
	for _, entry := range parseLogEntries(t, raw) {
		if predicate(entry) {
			return true
		}
	}
	return false
}

func parseSSEEvents(t *testing.T, raw string) []map[string]any {
	t.Helper()
	lines := strings.Split(raw, "\n")
	events := make([]map[string]any, 0)
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			continue
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(payload), &event); err != nil {
			t.Fatalf("unmarshal SSE event %q: %v", payload, err)
		}
		events = append(events, event)
	}
	return events
}

func readManagedKeepaliveStream(t *testing.T, body io.Reader, commentPrefix string, release func()) (string, bool) {
	t.Helper()
	reader := bufio.NewReader(body)
	var out strings.Builder
	sawComment := false
	for {
		line, err := reader.ReadString('\n')
		if line != "" {
			out.WriteString(line)
			if !sawComment && strings.HasPrefix(strings.TrimSpace(line), commentPrefix) {
				sawComment = true
				if release != nil {
					release()
				}
			}
		}
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("read managed stream: %v\n%s", err, out.String())
		}
	}
	return out.String(), sawComment
}

func sseHasContent(events []map[string]any, want string) bool {
	for _, event := range events {
		choices, _ := event["choices"].([]any)
		for _, rawChoice := range choices {
			choice, _ := rawChoice.(map[string]any)
			if choice == nil {
				continue
			}
			delta, _ := choice["delta"].(map[string]any)
			if delta == nil {
				continue
			}
			if delta["content"] == want {
				return true
			}
		}
	}
	return false
}

func sseHasToolCall(events []map[string]any, want string) bool {
	for _, event := range events {
		choices, _ := event["choices"].([]any)
		for _, rawChoice := range choices {
			choice, _ := rawChoice.(map[string]any)
			if choice == nil {
				continue
			}
			delta, _ := choice["delta"].(map[string]any)
			if delta == nil {
				continue
			}
			toolCalls, _ := delta["tool_calls"].([]any)
			for _, rawTool := range toolCalls {
				tool, _ := rawTool.(map[string]any)
				function, _ := tool["function"].(map[string]any)
				if function["name"] == want {
					return true
				}
			}
		}
	}
	return false
}

func anthropicSSEHasToolUse(events []map[string]any, want string) bool {
	for _, event := range events {
		if event["type"] != "content_block_start" {
			continue
		}
		block, _ := event["content_block"].(map[string]any)
		if block == nil {
			continue
		}
		if block["type"] == "tool_use" && block["name"] == want {
			return true
		}
	}
	return false
}

func sseHasUsage(events []map[string]any, prompt, completion, total int) bool {
	for _, event := range events {
		usage, _ := event["usage"].(map[string]any)
		if usage == nil {
			continue
		}
		if int(usage["prompt_tokens"].(float64)) == prompt &&
			int(usage["completion_tokens"].(float64)) == completion &&
			int(usage["total_tokens"].(float64)) == total {
			return true
		}
	}
	return false
}

func anthropicToolsIncludeName(tools []any, want string) bool {
	for _, raw := range tools {
		tool, _ := raw.(map[string]any)
		if tool == nil {
			continue
		}
		if tool["name"] == want {
			return true
		}
	}
	return false
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

func assertToolErrorContains(t *testing.T, raw any, want string) {
	t.Helper()
	text, ok := raw.(string)
	if !ok {
		t.Fatalf("expected tool payload string, got %T", raw)
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(text), &payload); err != nil {
		t.Fatalf("unmarshal tool payload: %v", err)
	}
	if okValue, _ := payload["ok"].(bool); okValue {
		t.Fatalf("expected tool error payload, got %+v", payload)
	}
	errPayload, _ := payload["error"].(map[string]any)
	if errPayload == nil {
		t.Fatalf("expected nested error payload, got %+v", payload)
	}
	message, _ := errPayload["message"].(string)
	if !strings.Contains(message, want) {
		t.Fatalf("expected tool error message to contain %q, got %+v", want, payload)
	}
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
