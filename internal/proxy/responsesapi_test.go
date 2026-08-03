package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/cost"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

func TestHandlerDispatchesResponsesOnlyModelThroughAdapter(t *testing.T) {
	var gotPath string
	var gotBody []byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		var err error
		gotBody, err = io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read body: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_1","object":"response","status":"completed","model":"gpt-5.6-terra",
			"output":[
				{"type":"reasoning","id":"rs_1"},
				{"type":"function_call","id":"fc_1","call_id":"call_abc","name":"ping","arguments":"{}"}
			],
			"usage":{"input_tokens":11,"output_tokens":7,"total_tokens":18}
		}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), nil)
	body := `{
		"model":"openai/gpt-5.6-terra",
		"messages":[{"role":"user","content":"ping"}],
		"reasoning_effort":"high",
		"tools":[{"type":"function","function":{"name":"ping","description":"ping","parameters":{"type":"object"}}}]
	}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if gotPath != "/v1/responses" {
		t.Errorf("expected upstream dispatch to /v1/responses, got %q", gotPath)
	}

	var upstream map[string]any
	if err := json.Unmarshal(gotBody, &upstream); err != nil {
		t.Fatalf("unmarshal upstream body: %v", err)
	}
	if upstream["model"] != "gpt-5.6-terra" {
		t.Errorf("upstream model: got %#v", upstream["model"])
	}
	if _, ok := upstream["messages"]; ok {
		t.Error("upstream body must use input, not messages")
	}
	if _, ok := upstream["input"].([]any); !ok {
		t.Errorf("upstream input: got %#v", upstream["input"])
	}
	tools, _ := upstream["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("upstream tools: got %#v", upstream["tools"])
	}
	if tool, _ := tools[0].(map[string]any); tool["name"] != "ping" {
		t.Errorf("upstream tool must be flattened: got %#v", tools[0])
	}

	// The agent must still see the chat/completions shape it asked for.
	var chat map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &chat); err != nil {
		t.Fatalf("unmarshal downstream body: %v", err)
	}
	if chat["object"] != "chat.completion" {
		t.Errorf("downstream object: got %#v", chat["object"])
	}
	choices, _ := chat["choices"].([]any)
	if len(choices) != 1 {
		t.Fatalf("downstream choices: got %#v", chat["choices"])
	}
	choice, _ := choices[0].(map[string]any)
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("downstream finish_reason: got %#v", choice["finish_reason"])
	}
	msg, _ := choice["message"].(map[string]any)
	calls, _ := msg["tool_calls"].([]any)
	if len(calls) != 1 {
		t.Fatalf("downstream tool_calls: got %#v", msg["tool_calls"])
	}
	usage, _ := chat["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(11) || usage["completion_tokens"] != float64(7) {
		t.Errorf("downstream usage must use chat field names: got %#v", chat["usage"])
	}
}

// The capability issue #31 is about is managed tools on responses-only models,
// so the mediation loop — not just a single dispatch — has to run through the
// adapter, including replaying the executed round back to the model.
func TestHandlerRunsManagedToolMediationThroughAdapter(t *testing.T) {
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	var paths []string
	var bodies [][]byte
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read body: %v", err)
		}
		paths = append(paths, r.URL.Path)
		bodies = append(bodies, body)
		w.Header().Set("Content-Type", "application/json")
		switch len(bodies) {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"resp_1","status":"completed","model":"gpt-5.6-terra",
				"output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"` +
				managedToolHashlessAliasForCanonical("trading-api.get_market_context") + `","arguments":"{}"}],
				"usage":{"input_tokens":10,"output_tokens":3,"total_tokens":13}
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"resp_2","status":"completed","model":"gpt-5.6-terra",
				"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"market context loaded"}]}],
				"usage":{"input_tokens":7,"output_tokens":5,"total_tokens":12}
			}`))
		default:
			t.Errorf("unexpected upstream round %d", len(bodies))
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123",
		managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")),
		logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5.6-terra","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if len(paths) != 2 {
		t.Fatalf("expected two mediated rounds, got %d (%v)", len(paths), paths)
	}
	for i, path := range paths {
		if path != "/v1/responses" {
			t.Errorf("round %d dispatched to %q, want /v1/responses", i+1, path)
		}
	}

	// Round two must replay the executed tool round in the Responses item shape,
	// or the model has no idea its call was answered.
	var second map[string]any
	if err := json.Unmarshal(bodies[1], &second); err != nil {
		t.Fatalf("unmarshal round two: %v", err)
	}
	input, _ := second["input"].([]any)
	var sawCall, sawOutput bool
	for _, raw := range input {
		item, _ := raw.(map[string]any)
		switch item["type"] {
		case "function_call":
			if item["call_id"] == "call_1" {
				sawCall = true
			}
		case "function_call_output":
			if item["call_id"] == "call_1" && strings.Contains(item["output"].(string), "5000") {
				sawOutput = true
			}
		}
	}
	if !sawCall {
		t.Errorf("round two input missing the function_call item: %s", bodies[1])
	}
	if !sawOutput {
		t.Errorf("round two input missing the function_call_output item: %s", bodies[1])
	}

	var chat map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &chat); err != nil {
		t.Fatalf("unmarshal downstream body: %v", err)
	}
	choices, _ := chat["choices"].([]any)
	choice, _ := choices[0].(map[string]any)
	msg, _ := choice["message"].(map[string]any)
	if msg["content"] != "market context loaded" {
		t.Errorf("downstream content: got %#v", msg["content"])
	}
}

func TestHandlerSynthesizesSSEForStreamedResponsesOnlyModel(t *testing.T) {
	var gotStream any
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read body: %v", err)
		}
		var payload map[string]any
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("unmarshal upstream body: %v", err)
		}
		gotStream = payload["stream"]
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_1","status":"completed","model":"gpt-5.6-terra",
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"streamed hello"}]}],
			"usage":{"input_tokens":4,"output_tokens":2,"total_tokens":6}
		}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5.6-terra","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if gotStream != nil {
		t.Errorf("adapter must dispatch non-streaming upstream, got stream=%#v", gotStream)
	}
	if got := w.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Fatalf("agent asked for a stream, got content-type %q", got)
	}
	body := w.Body.String()
	if !strings.Contains(body, "data: [DONE]") {
		t.Fatalf("expected [DONE] marker, got %s", body)
	}
	chunks := parseSSEChunks(t, w.Body.Bytes())
	var sawContent bool
	for _, chunk := range chunks {
		if _, ok := chunk["usage"]; ok {
			t.Errorf("usage chunk must be omitted unless stream_options.include_usage was requested: %#v", chunk)
		}
		if sseDelta(t, chunk)["content"] == "streamed hello" {
			sawContent = true
		}
	}
	if !sawContent {
		t.Errorf("expected a content delta chunk, got %s", body)
	}
}

// The built-in list cannot anticipate every model OpenAI moves to /v1/responses.
// When upstream says so explicitly, retry through the adapter rather than
// surfacing a rejection that reads like a credentials problem.
func TestHandlerRetriesThroughAdapterOnResponsesOnlyUpstreamSignal(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
	}{
		{
			name:   "400 function tools with reasoning",
			status: http.StatusBadRequest,
			body:   `{"error":{"message":"Function tools with reasoning_effort are not supported for gpt-9-experimental in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'.","type":"invalid_request_error"}}`,
		},
		{
			name:   "404 model is responses only",
			status: http.StatusNotFound,
			body:   `{"error":{"message":"This model is only supported in v1/responses and not in v1/chat/completions.","type":"invalid_request_error"}}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var paths []string
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				paths = append(paths, r.URL.Path)
				w.Header().Set("Content-Type", "application/json")
				if r.URL.Path == "/v1/chat/completions" {
					w.WriteHeader(tc.status)
					_, _ = w.Write([]byte(tc.body))
					return
				}
				_, _ = w.Write([]byte(`{
					"id":"resp_1","status":"completed","model":"gpt-9-experimental",
					"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recovered"}]}]
				}`))
			}))
			defer backend.Close()

			reg := provider.NewRegistry("")
			reg.Set("openai", &provider.Provider{
				Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
			})

			h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard))
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
				bytes.NewBufferString(`{"model":"openai/gpt-9-experimental","messages":[{"role":"user","content":"hi"}]}`))
			req.Header.Set("Authorization", "Bearer tiverton:dummy123")
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			h.ServeHTTP(w, req)

			if w.Code != http.StatusOK {
				t.Fatalf("expected the retry to succeed with 200, got %d: %s", w.Code, w.Body.String())
			}
			want := []string{"/v1/chat/completions", "/v1/responses"}
			if len(paths) != 2 || paths[0] != want[0] || paths[1] != want[1] {
				t.Fatalf("expected dispatch sequence %v, got %v", want, paths)
			}

			var chat map[string]any
			if err := json.Unmarshal(w.Body.Bytes(), &chat); err != nil {
				t.Fatalf("unmarshal downstream body: %v", err)
			}
			choices, _ := chat["choices"].([]any)
			choice, _ := choices[0].(map[string]any)
			msg, _ := choice["message"].(map[string]any)
			if msg["content"] != "recovered" {
				t.Errorf("downstream content: got %#v", msg["content"])
			}
		})
	}
}

func TestManagedMediationRetriesThroughAdapterOnResponsesOnlyUpstreamSignal(t *testing.T) {
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	var paths []string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		if r.URL.Path == "/v1/chat/completions" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"To use function tools, use /v1/responses."}}`))
			return
		}
		_, _ = w.Write([]byte(`{
			"id":"resp_1","status":"completed","model":"gpt-9-experimental",
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"recovered"}]}]
		}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123",
		managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")),
		logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-9-experimental","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected the retry to succeed with 200, got %d: %s", w.Code, w.Body.String())
	}
	want := []string{"/v1/chat/completions", "/v1/responses"}
	if len(paths) != 2 || paths[0] != want[0] || paths[1] != want[1] {
		t.Fatalf("expected dispatch sequence %v, got %v", want, paths)
	}
}

// The adapter is an OpenAI chat/completions concept. The Anthropic path must
// never be retried through it, whatever an upstream rejection happens to say.
func TestHandlerNeverRetriesAnthropicPathThroughAdapter(t *testing.T) {
	var paths []string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"To use function tools, use /v1/responses."}}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/messages",
		bytes.NewBufferString(`{"model":"anthropic/claude-sonnet-5","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if len(paths) != 1 || paths[0] != "/v1/messages" {
		t.Fatalf("expected a single /v1/messages dispatch, got %v", paths)
	}
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected the upstream 400 to reach the agent, got %d", w.Code)
	}
}

// An ordinary 400 must still reach the agent unchanged; retrying every
// rejection through the adapter would double every bad request upstream.
func TestHandlerDoesNotRetryUnrelatedBadRequests(t *testing.T) {
	var paths []string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"Unknown parameter: 'foo'.","type":"invalid_request_error"}}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected the upstream 400 to reach the agent, got %d", w.Code)
	}
	if len(paths) != 1 {
		t.Errorf("expected a single upstream dispatch, got %v", paths)
	}
	if !strings.Contains(w.Body.String(), "Unknown parameter") {
		t.Errorf("expected the upstream error body preserved, got %s", w.Body.String())
	}
}

func TestResponsesOnlyUpstreamSignalPreservesEntireUnrelatedBody(t *testing.T) {
	body := []byte(`{"error":{"message":"` + strings.Repeat("x", 8192) + `"}}`)
	resp := &http.Response{
		StatusCode: http.StatusBadRequest,
		Body:       io.NopCloser(bytes.NewReader(body)),
	}

	if responsesOnlyUpstreamSignal(resp) {
		t.Fatal("unrelated error must not trigger the adapter")
	}
	got, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read restored response body: %v", err)
	}
	if !bytes.Equal(got, body) {
		t.Fatalf("response body was not preserved: got %d bytes, want %d", len(got), len(body))
	}
}

// A streamed request is not bounded by the dispatch-candidate timeout natively;
// it gets the much longer stream first-byte budget. The adapter buffers the
// reply, but it must not silently shrink that budget to 60s — responses-only
// models are the slowest ones cllama dispatches to.
func TestHandlerKeepsStreamBudgetForBufferedAdapterRequests(t *testing.T) {
	t.Setenv(EnvDispatchCandidateTimeoutMS, "150")
	t.Setenv(EnvStreamFirstByteTimeoutMS, "5000")

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(400 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_1","status":"completed","model":"gpt-5.6-terra",
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"slow but fine"}]}]
		}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5.6-terra","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "slow but fine") {
		t.Errorf("expected the buffered reply, got %s", w.Body.String())
	}
}

// The non-streamed budget is unchanged, which is what makes the test above
// meaningful rather than a timeout that never applied.
func TestHandlerAppliesDispatchTimeoutToNonStreamedAdapterRequests(t *testing.T) {
	t.Setenv(EnvDispatchCandidateTimeoutMS, "150")

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(400 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"resp_1","status":"completed","output":[]}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5.6-terra","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code == http.StatusOK {
		t.Errorf("expected the dispatch timeout to apply, got 200")
	}
}

func TestHandlerLeavesChatCompletionsModelsOnChatCompletions(t *testing.T) {
	var gotPath string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"hello"}}]}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), nil)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if gotPath != "/v1/chat/completions" {
		t.Errorf("models that work on chat/completions must not be rerouted, got %q", gotPath)
	}
}

func TestChatToResponsesRequestMapsCoreFields(t *testing.T) {
	payload := map[string]any{
		"model":            "gpt-5.6-terra",
		"messages":         []any{map[string]any{"role": "user", "content": "hi"}},
		"reasoning_effort": "high",
	}

	got, err := chatToResponsesRequest(payload)
	if err != nil {
		t.Fatalf("chatToResponsesRequest: %v", err)
	}

	if got["model"] != "gpt-5.6-terra" {
		t.Errorf("model: got %#v", got["model"])
	}
	if _, ok := got["messages"]; ok {
		t.Error("messages must not survive translation")
	}
	input, ok := got["input"].([]any)
	if !ok || len(input) != 1 {
		t.Fatalf("input: got %#v", got["input"])
	}
	item, _ := input[0].(map[string]any)
	if item["role"] != "user" || item["content"] != "hi" {
		t.Errorf("input[0]: got %#v", item)
	}
	reasoning, ok := got["reasoning"].(map[string]any)
	if !ok || reasoning["effort"] != "high" {
		t.Errorf("reasoning: got %#v", got["reasoning"])
	}
	if _, ok := got["reasoning_effort"]; ok {
		t.Error("reasoning_effort must not survive translation")
	}
}

// Managed tool mediation replays prior rounds as an assistant message carrying
// tool_calls followed by role:"tool" results. Responses represents both as
// top-level input items, so a wrong mapping here silently breaks every
// multi-round mediated call — the exact capability issue #31 is about.
func TestChatToResponsesRequestTranslatesToolCallRounds(t *testing.T) {
	payload := map[string]any{
		"model": "gpt-5.6-terra",
		"messages": []any{
			map[string]any{"role": "user", "content": "ping example.com"},
			map[string]any{
				"role":    "assistant",
				"content": nil,
				"tool_calls": []any{
					map[string]any{
						"id":       "call_abc",
						"type":     "function",
						"function": map[string]any{"name": "ping", "arguments": `{"host":"example.com"}`},
					},
				},
			},
			map[string]any{"role": "tool", "tool_call_id": "call_abc", "content": "pong"},
		},
	}

	got, err := chatToResponsesRequest(payload)
	if err != nil {
		t.Fatalf("chatToResponsesRequest: %v", err)
	}

	input, ok := got["input"].([]any)
	if !ok || len(input) != 3 {
		t.Fatalf("input: expected 3 items, got %#v", got["input"])
	}

	call, _ := input[1].(map[string]any)
	if call["type"] != "function_call" {
		t.Errorf("assistant tool_calls must become a function_call item: got %#v", call["type"])
	}
	if call["call_id"] != "call_abc" {
		t.Errorf("call_id: got %#v", call["call_id"])
	}
	if call["name"] != "ping" {
		t.Errorf("name: got %#v", call["name"])
	}
	if call["arguments"] != `{"host":"example.com"}` {
		t.Errorf("arguments: got %#v", call["arguments"])
	}
	if _, ok := call["tool_calls"]; ok {
		t.Error("tool_calls must not survive translation")
	}

	result, _ := input[2].(map[string]any)
	if result["type"] != "function_call_output" {
		t.Errorf("role:tool must become a function_call_output item: got %#v", result["type"])
	}
	if result["call_id"] != "call_abc" {
		t.Errorf("call_id: got %#v", result["call_id"])
	}
	if result["output"] != "pong" {
		t.Errorf("output: got %#v", result["output"])
	}
	if _, ok := result["role"]; ok {
		t.Error("role must not survive on a function_call_output item")
	}
}

func TestChatToResponsesRequestSplitsAssistantTextAndToolCalls(t *testing.T) {
	payload := map[string]any{
		"model": "gpt-5.6-terra",
		"messages": []any{
			map[string]any{
				"role":    "assistant",
				"content": "let me check",
				"tool_calls": []any{
					map[string]any{
						"id":       "call_1",
						"function": map[string]any{"name": "ping", "arguments": "{}"},
					},
					map[string]any{
						"id":       "call_2",
						"function": map[string]any{"name": "pong", "arguments": "{}"},
					},
				},
			},
		},
	}

	got, err := chatToResponsesRequest(payload)
	if err != nil {
		t.Fatalf("chatToResponsesRequest: %v", err)
	}

	input, _ := got["input"].([]any)
	if len(input) != 3 {
		t.Fatalf("expected assistant text plus two function_call items, got %d: %#v", len(input), input)
	}
	text, _ := input[0].(map[string]any)
	if text["role"] != "assistant" || text["content"] != "let me check" {
		t.Errorf("assistant text item: got %#v", text)
	}
	for i, wantName := range []string{"ping", "pong"} {
		call, _ := input[i+1].(map[string]any)
		if call["type"] != "function_call" || call["name"] != wantName {
			t.Errorf("input[%d]: got %#v", i+1, call)
		}
	}
}

func TestChatToResponsesRequestTranslatesContentParts(t *testing.T) {
	payload := map[string]any{
		"model": "gpt-5.6-terra",
		"messages": []any{
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{"type": "text", "text": "what is this"},
					map[string]any{"type": "image_url", "image_url": map[string]any{"url": "https://example.com/a.png"}},
				},
			},
			map[string]any{
				"role":    "assistant",
				"content": []any{map[string]any{"type": "text", "text": "a picture"}},
			},
		},
	}

	got, err := chatToResponsesRequest(payload)
	if err != nil {
		t.Fatalf("chatToResponsesRequest: %v", err)
	}

	input, _ := got["input"].([]any)
	if len(input) != 2 {
		t.Fatalf("input: got %#v", input)
	}

	user, _ := input[0].(map[string]any)
	userParts, _ := user["content"].([]any)
	if len(userParts) != 2 {
		t.Fatalf("user content parts: got %#v", user["content"])
	}
	textPart, _ := userParts[0].(map[string]any)
	if textPart["type"] != "input_text" || textPart["text"] != "what is this" {
		t.Errorf("user text part: got %#v", textPart)
	}
	imagePart, _ := userParts[1].(map[string]any)
	if imagePart["type"] != "input_image" || imagePart["image_url"] != "https://example.com/a.png" {
		t.Errorf("user image part: got %#v", imagePart)
	}

	assistant, _ := input[1].(map[string]any)
	assistantParts, _ := assistant["content"].([]any)
	assistantText, _ := assistantParts[0].(map[string]any)
	if assistantText["type"] != "output_text" || assistantText["text"] != "a picture" {
		t.Errorf("assistant text part: got %#v", assistantText)
	}
}

func TestChatToResponsesRequestFlattensFunctionTools(t *testing.T) {
	payload := map[string]any{
		"model": "gpt-5.6-terra",
		"tools": []any{
			map[string]any{
				"type": "function",
				"function": map[string]any{
					"name":        "ping",
					"description": "ping a host",
					"parameters":  map[string]any{"type": "object"},
					"strict":      true,
				},
			},
		},
	}

	got, err := chatToResponsesRequest(payload)
	if err != nil {
		t.Fatalf("chatToResponsesRequest: %v", err)
	}

	tools, ok := got["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools: got %#v", got["tools"])
	}
	tool, _ := tools[0].(map[string]any)
	if tool["type"] != "function" {
		t.Errorf("tool type: got %#v", tool["type"])
	}
	if tool["name"] != "ping" {
		t.Errorf("tool name must be flattened to top level: got %#v", tool["name"])
	}
	if tool["description"] != "ping a host" {
		t.Errorf("tool description: got %#v", tool["description"])
	}
	if _, ok := tool["parameters"].(map[string]any); !ok {
		t.Errorf("tool parameters: got %#v", tool["parameters"])
	}
	if tool["strict"] != true {
		t.Errorf("tool strict: got %#v", tool["strict"])
	}
	if _, ok := tool["function"]; ok {
		t.Error("nested function object must not survive translation")
	}
}

// Chat Completions leaves strict disabled when it is omitted, while Responses
// enables strict validation by default. The adapter must preserve the caller's
// original non-strict tool contract.
func TestChatToResponsesRequestDefaultsOmittedFunctionStrictToFalse(t *testing.T) {
	got, err := chatToResponsesRequest(map[string]any{
		"model": "gpt-5.6-terra",
		"tools": []any{map[string]any{
			"type": "function",
			"function": map[string]any{
				"name":       "ping",
				"parameters": map[string]any{"type": "object"},
			},
		}},
	})
	if err != nil {
		t.Fatalf("chatToResponsesRequest: %v", err)
	}

	tools, _ := got["tools"].([]any)
	tool, _ := tools[0].(map[string]any)
	if tool["strict"] != false {
		t.Errorf("omitted Chat strict must become explicit false for Responses: got %#v", tool["strict"])
	}
}

func TestChatToResponsesRequestMapsTokenAndToolChoiceParams(t *testing.T) {
	cases := []struct {
		name    string
		payload map[string]any
		want    map[string]any
		absent  []string
	}{
		{
			name:    "max_completion_tokens",
			payload: map[string]any{"max_completion_tokens": float64(256)},
			want:    map[string]any{"max_output_tokens": float64(256)},
			absent:  []string{"max_completion_tokens", "max_tokens"},
		},
		{
			name:    "legacy max_tokens",
			payload: map[string]any{"max_tokens": float64(128)},
			want:    map[string]any{"max_output_tokens": float64(128)},
			absent:  []string{"max_tokens"},
		},
		{
			name:    "tool_choice passthrough",
			payload: map[string]any{"tool_choice": "auto"},
			want:    map[string]any{"tool_choice": "auto"},
		},
		{
			name: "named tool_choice flattened",
			payload: map[string]any{"tool_choice": map[string]any{
				"type":     "function",
				"function": map[string]any{"name": "ping"},
			}},
			want: map[string]any{"tool_choice": map[string]any{"type": "function", "name": "ping"}},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.payload["model"] = "gpt-5.6-terra"
			got, err := chatToResponsesRequest(tc.payload)
			if err != nil {
				t.Fatalf("chatToResponsesRequest: %v", err)
			}
			for key, want := range tc.want {
				gotJSON, _ := json.Marshal(got[key])
				wantJSON, _ := json.Marshal(want)
				if string(gotJSON) != string(wantJSON) {
					t.Errorf("%s: got %s, want %s", key, gotJSON, wantJSON)
				}
			}
			for _, key := range tc.absent {
				if _, ok := got[key]; ok {
					t.Errorf("%s must not survive translation", key)
				}
			}
		})
	}
}

func TestChatToResponsesRequestDropsTransportAndNeutralParams(t *testing.T) {
	payload := map[string]any{
		"model":               "gpt-5.6-terra",
		"stream":              true,
		"stream_options":      map[string]any{"include_usage": true},
		"frequency_penalty":   float64(0),
		"presence_penalty":    float64(0),
		"logprobs":            false,
		"logit_bias":          map[string]any{},
		"parallel_tool_calls": true,
	}

	got, err := chatToResponsesRequest(payload)
	if err != nil {
		t.Fatalf("chatToResponsesRequest: %v", err)
	}

	for _, key := range []string{"stream", "stream_options", "frequency_penalty", "presence_penalty", "logprobs", "logit_bias"} {
		if _, ok := got[key]; ok {
			t.Errorf("%s must not survive translation, got %#v", key, got[key])
		}
	}
	if got["parallel_tool_calls"] != true {
		t.Errorf("parallel_tool_calls is supported by responses and must survive: got %#v", got["parallel_tool_calls"])
	}
}

func TestChatToResponsesRequestRejectsUnsupportedSemanticParams(t *testing.T) {
	cases := []struct {
		name  string
		field string
		value any
	}{
		{name: "frequency penalty", field: "frequency_penalty", value: float64(0.5)},
		{name: "presence penalty", field: "presence_penalty", value: float64(0.5)},
		{name: "log probabilities", field: "logprobs", value: true},
		{name: "logit bias", field: "logit_bias", value: map[string]any{"1": float64(1)}},
		{name: "stop sequence", field: "stop", value: "END"},
		{name: "seed", field: "seed", value: float64(42)},
		{name: "upstream storage", field: "store", value: true},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := chatToResponsesRequest(map[string]any{
				"model":  "gpt-5.6-terra",
				tc.field: tc.value,
			})
			if err == nil {
				t.Fatalf("expected %s to be rejected rather than silently dropped", tc.field)
			}
		})
	}
}

func TestChatToResponsesRequestPreservesSharedOptionalFields(t *testing.T) {
	payload := map[string]any{
		"model":                  "gpt-5.6-terra",
		"user":                   "user-123",
		"safety_identifier":      "safety-123",
		"prompt_cache_key":       "cache-123",
		"prompt_cache_retention": "24h",
		"service_tier":           "priority",
	}

	got, err := chatToResponsesRequest(payload)
	if err != nil {
		t.Fatalf("chatToResponsesRequest: %v", err)
	}
	for _, field := range []string{"user", "safety_identifier", "prompt_cache_key", "prompt_cache_retention", "service_tier"} {
		if got[field] != payload[field] {
			t.Errorf("%s: got %#v, want %#v", field, got[field], payload[field])
		}
	}
}

// Responses cannot generate multiple choices. Silently dropping n would turn
// a request for two completions into one successful-looking completion.
func TestChatToResponsesRequestRejectsMultipleChoices(t *testing.T) {
	_, err := chatToResponsesRequest(map[string]any{
		"model": "gpt-5.6-terra",
		"n":     float64(2),
	})
	if err == nil {
		t.Fatal("expected n > 1 to be rejected")
	}
}

func TestHandlerRejectsMultipleChoicesBeforeResponsesDispatch(t *testing.T) {
	dispatches := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		dispatches++
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})
	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5.6-terra","n":2,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
	}
	if dispatches != 0 {
		t.Fatalf("unsupported request reached upstream %d time(s)", dispatches)
	}
}

// response_format constrains what the model may emit. Dropping it would let the
// model return prose where the runner parses JSON, so it must be translated
// rather than silently discarded.
func TestChatToResponsesRequestTranslatesResponseFormat(t *testing.T) {
	cases := []struct {
		name  string
		given any
		want  string
	}{
		{
			name:  "json_object",
			given: map[string]any{"type": "json_object"},
			want:  `{"format":{"type":"json_object"}}`,
		},
		{
			name: "json_schema",
			given: map[string]any{
				"type": "json_schema",
				"json_schema": map[string]any{
					"name":   "reply",
					"schema": map[string]any{"type": "object"},
					"strict": true,
				},
			},
			want: `{"format":{"name":"reply","schema":{"type":"object"},"strict":true,"type":"json_schema"}}`,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := chatToResponsesRequest(map[string]any{
				"model":           "gpt-5.6-terra",
				"response_format": tc.given,
			})
			if err != nil {
				t.Fatalf("chatToResponsesRequest: %v", err)
			}
			if _, ok := got["response_format"]; ok {
				t.Error("response_format must not survive translation")
			}
			gotJSON, _ := json.Marshal(got["text"])
			if string(gotJSON) != tc.want {
				t.Errorf("text: got %s, want %s", gotJSON, tc.want)
			}
		})
	}
}

func TestChatToResponsesRequestDisablesUpstreamStorage(t *testing.T) {
	got, err := chatToResponsesRequest(map[string]any{"model": "gpt-5.6-terra"})
	if err != nil {
		t.Fatalf("chatToResponsesRequest: %v", err)
	}
	if got["store"] != false {
		t.Errorf("store must default to false so responses are not retained upstream: got %#v", got["store"])
	}
}

// The Responses event taxonomy is not the chat SSE one, so an agent that asked
// for a stream gets synthetic chat SSE built from the buffered completion.
func TestChatCompletionToSSEEmitsTextStream(t *testing.T) {
	body := []byte(`{
		"id":"resp_1","object":"chat.completion","created":1750000000,"model":"gpt-5.6-terra",
		"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
	}`)

	sse, err := chatCompletionToSSE(body, true)
	if err != nil {
		t.Fatalf("chatCompletionToSSE: %v", err)
	}

	chunks := parseSSEChunks(t, sse)
	if len(chunks) < 4 {
		t.Fatalf("expected role, content, finish, and usage chunks, got %d: %s", len(chunks), sse)
	}
	for _, chunk := range chunks {
		if chunk["object"] != "chat.completion.chunk" {
			t.Errorf("object: got %#v", chunk["object"])
		}
		if chunk["id"] != "resp_1" || chunk["model"] != "gpt-5.6-terra" {
			t.Errorf("chunk identity: got %#v", chunk)
		}
	}

	if role := sseDelta(t, chunks[0])["role"]; role != "assistant" {
		t.Errorf("first chunk must open the assistant role: got %#v", role)
	}
	if content := sseDelta(t, chunks[1])["content"]; content != "hello" {
		t.Errorf("content delta: got %#v", content)
	}

	finish := chunks[2]
	choices, _ := finish["choices"].([]any)
	choice, _ := choices[0].(map[string]any)
	if choice["finish_reason"] != "stop" {
		t.Errorf("finish_reason: got %#v", choice["finish_reason"])
	}

	usage, _ := chunks[3]["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(10) || usage["completion_tokens"] != float64(5) {
		t.Errorf("usage chunk: got %#v", chunks[3]["usage"])
	}

	if !bytes.HasSuffix(sse, []byte("data: [DONE]\n\n")) {
		t.Errorf("stream must terminate with [DONE]: got %s", sse)
	}
}

func TestChatCompletionToSSEEmitsToolCallStream(t *testing.T) {
	body := []byte(`{
		"id":"resp_1","object":"chat.completion","created":1750000000,"model":"gpt-5.6-terra",
		"choices":[{"index":0,"message":{"role":"assistant","content":null,"tool_calls":[
			{"index":0,"id":"call_abc","type":"function","function":{"name":"ping","arguments":"{\"host\":\"a\"}"}}
		]},"finish_reason":"tool_calls"}]
	}`)

	sse, err := chatCompletionToSSE(body, false)
	if err != nil {
		t.Fatalf("chatCompletionToSSE: %v", err)
	}

	chunks := parseSSEChunks(t, sse)
	var sawToolCall bool
	var finishReason any
	for _, chunk := range chunks {
		delta := sseDelta(t, chunk)
		if calls, ok := delta["tool_calls"].([]any); ok && len(calls) == 1 {
			call, _ := calls[0].(map[string]any)
			fn, _ := call["function"].(map[string]any)
			if call["id"] != "call_abc" || call["index"] != float64(0) || fn["name"] != "ping" {
				t.Errorf("tool_call delta: got %#v", call)
			}
			if fn["arguments"] != `{"host":"a"}` {
				t.Errorf("tool_call arguments: got %#v", fn["arguments"])
			}
			sawToolCall = true
		}
		choices, _ := chunk["choices"].([]any)
		if len(choices) == 1 {
			choice, _ := choices[0].(map[string]any)
			if choice["finish_reason"] != nil {
				finishReason = choice["finish_reason"]
			}
		}
	}
	if !sawToolCall {
		t.Errorf("expected a tool_calls delta chunk: got %s", sse)
	}
	if finishReason != "tool_calls" {
		t.Errorf("finish_reason: got %#v", finishReason)
	}
}

func parseSSEChunks(t *testing.T, sse []byte) []map[string]any {
	t.Helper()
	var chunks []map[string]any
	for _, line := range bytes.Split(sse, []byte("\n")) {
		payload, ok := bytes.CutPrefix(bytes.TrimSpace(line), []byte("data: "))
		if !ok || string(payload) == "[DONE]" {
			continue
		}
		var chunk map[string]any
		if err := json.Unmarshal(payload, &chunk); err != nil {
			t.Fatalf("unmarshal sse chunk %q: %v", payload, err)
		}
		chunks = append(chunks, chunk)
	}
	return chunks
}

func sseDelta(t *testing.T, chunk map[string]any) map[string]any {
	t.Helper()
	choices, _ := chunk["choices"].([]any)
	if len(choices) == 0 {
		return map[string]any{}
	}
	choice, _ := choices[0].(map[string]any)
	delta, _ := choice["delta"].(map[string]any)
	return delta
}

func TestResponsesAPIRequiredCoversKnownResponsesOnlyModels(t *testing.T) {
	cases := []struct {
		provider string
		model    string
		want     bool
	}{
		{"openai", "gpt-5.6-terra", true},
		{"openai", "gpt-5.6-luna", true},
		{"openai", "gpt-5-pro", true},
		{"openai", "gpt-5-pro-2026-01-01", true},
		{"openai", "gpt-5", false},
		{"openai", "gpt-5-mini", false},
		{"openai", "gpt-4o", false},
		// Other providers expose these model names through their own
		// chat/completions surface, which works. Rerouting them would break
		// what currently succeeds.
		{"openrouter", "gpt-5.6-terra", false},
		{"anthropic", "claude-sonnet-5", false},
	}

	for _, tc := range cases {
		if got := responsesAPIRequired(tc.provider, tc.model); got != tc.want {
			t.Errorf("responsesAPIRequired(%q, %q) = %v, want %v", tc.provider, tc.model, got, tc.want)
		}
	}
}

func TestResponsesAPIRequiredHonoursEnvOverrides(t *testing.T) {
	t.Setenv(EnvResponsesAPIModels, "openai/gpt-6, custom/weird-model")
	if !responsesAPIRequired("openai", "gpt-6-turbo") {
		t.Error("expected declared override openai/gpt-6 to route through the adapter")
	}
	if !responsesAPIRequired("custom", "weird-model") {
		t.Error("expected declared override custom/weird-model to route through the adapter")
	}
	if responsesAPIRequired("other", "gpt-6-turbo") {
		t.Error("override must stay scoped to its declared provider")
	}
	if !responsesAPIRequired("openai", "gpt-5.6-terra") {
		t.Error("override must add to the built-in set, not replace it")
	}
}

func TestResponsesAPIRequiredCanBeDisabledEntirely(t *testing.T) {
	t.Setenv(EnvResponsesAPIDisabled, "1")
	if responsesAPIRequired("openai", "gpt-5.6-terra") {
		t.Error("expected the adapter to be disabled by the escape hatch")
	}
}

func TestResponsesToChatCompletionMapsTextAndUsage(t *testing.T) {
	body := []byte(`{
		"id":"resp_123",
		"object":"response",
		"created_at":1750000000,
		"status":"completed",
		"model":"gpt-5.6-terra",
		"output":[
			{"type":"reasoning","id":"rs_1","summary":[]},
			{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"hello"}]}
		],
		"usage":{
			"input_tokens":10,
			"output_tokens":5,
			"total_tokens":15,
			"input_tokens_details":{"cached_tokens":2},
			"output_tokens_details":{"reasoning_tokens":3}
		}
	}`)

	got, err := responsesToChatCompletion(body)
	if err != nil {
		t.Fatalf("responsesToChatCompletion: %v", err)
	}

	var chat map[string]any
	if err := json.Unmarshal(got, &chat); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if chat["id"] != "resp_123" {
		t.Errorf("id: got %#v", chat["id"])
	}
	if chat["object"] != "chat.completion" {
		t.Errorf("object: got %#v", chat["object"])
	}
	if chat["created"] != float64(1750000000) {
		t.Errorf("created: got %#v", chat["created"])
	}
	if chat["model"] != "gpt-5.6-terra" {
		t.Errorf("model: got %#v", chat["model"])
	}

	choices, ok := chat["choices"].([]any)
	if !ok || len(choices) != 1 {
		t.Fatalf("choices: got %#v", chat["choices"])
	}
	choice, _ := choices[0].(map[string]any)
	if choice["index"] != float64(0) {
		t.Errorf("index: got %#v", choice["index"])
	}
	if choice["finish_reason"] != "stop" {
		t.Errorf("finish_reason: got %#v", choice["finish_reason"])
	}
	msg, _ := choice["message"].(map[string]any)
	if msg["role"] != "assistant" || msg["content"] != "hello" {
		t.Errorf("message: got %#v", msg)
	}
	if _, ok := msg["tool_calls"]; ok {
		t.Error("tool_calls must be absent when the model emitted none")
	}

	// Budget enforcement reads the chat field names, so the usage mapping is
	// what keeps spend accounting correct through the adapter.
	usage, _ := chat["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(10) {
		t.Errorf("prompt_tokens: got %#v", usage["prompt_tokens"])
	}
	if usage["completion_tokens"] != float64(5) {
		t.Errorf("completion_tokens: got %#v", usage["completion_tokens"])
	}
	if usage["total_tokens"] != float64(15) {
		t.Errorf("total_tokens: got %#v", usage["total_tokens"])
	}
	promptDetails, _ := usage["prompt_tokens_details"].(map[string]any)
	if promptDetails["cached_tokens"] != float64(2) {
		t.Errorf("prompt_tokens_details.cached_tokens: got %#v", usage["prompt_tokens_details"])
	}
	completionDetails, _ := usage["completion_tokens_details"].(map[string]any)
	if completionDetails["reasoning_tokens"] != float64(3) {
		t.Errorf("completion_tokens_details.reasoning_tokens: got %#v", usage["completion_tokens_details"])
	}
}

func TestResponsesToChatCompletionMapsFunctionCalls(t *testing.T) {
	body := []byte(`{
		"id":"resp_1","status":"completed","model":"gpt-5.6-terra",
		"output":[
			{"type":"reasoning","id":"rs_1"},
			{"type":"function_call","id":"fc_1","call_id":"call_abc","name":"ping","arguments":"{\"host\":\"example.com\"}"}
		]
	}`)

	got, err := responsesToChatCompletion(body)
	if err != nil {
		t.Fatalf("responsesToChatCompletion: %v", err)
	}

	var chat map[string]any
	if err := json.Unmarshal(got, &chat); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	choices, _ := chat["choices"].([]any)
	choice, _ := choices[0].(map[string]any)
	if choice["finish_reason"] != "tool_calls" {
		t.Errorf("finish_reason: got %#v", choice["finish_reason"])
	}
	msg, _ := choice["message"].(map[string]any)
	if msg["content"] != nil {
		t.Errorf("content must be null when only tool calls were emitted: got %#v", msg["content"])
	}
	calls, ok := msg["tool_calls"].([]any)
	if !ok || len(calls) != 1 {
		t.Fatalf("tool_calls: got %#v", msg["tool_calls"])
	}
	call, _ := calls[0].(map[string]any)
	if call["id"] != "call_abc" {
		t.Errorf("tool call id must be the call_id the model will be replayed with: got %#v", call["id"])
	}
	if call["type"] != "function" {
		t.Errorf("type: got %#v", call["type"])
	}
	if call["index"] != float64(0) {
		t.Errorf("index: got %#v", call["index"])
	}
	fn, _ := call["function"].(map[string]any)
	if fn["name"] != "ping" || fn["arguments"] != `{"host":"example.com"}` {
		t.Errorf("function: got %#v", fn)
	}
}

func TestResponsesToChatCompletionMapsIncompleteToLengthFinishReason(t *testing.T) {
	body := []byte(`{
		"id":"resp_1","status":"incomplete","model":"gpt-5.6-terra",
		"incomplete_details":{"reason":"max_output_tokens"},
		"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"partial"}]}]
	}`)

	got, err := responsesToChatCompletion(body)
	if err != nil {
		t.Fatalf("responsesToChatCompletion: %v", err)
	}
	var chat map[string]any
	if err := json.Unmarshal(got, &chat); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	choices, _ := chat["choices"].([]any)
	choice, _ := choices[0].(map[string]any)
	if choice["finish_reason"] != "length" {
		t.Errorf("finish_reason: got %#v", choice["finish_reason"])
	}
}

func TestResponsesToChatCompletionMapsContentFilterFinishReason(t *testing.T) {
	body := []byte(`{
		"id":"resp_1","status":"incomplete","model":"gpt-5.6-terra",
		"incomplete_details":{"reason":"content_filter"},
		"output":[{"type":"message","role":"assistant","content":[]}]
	}`)

	got, err := responsesToChatCompletion(body)
	if err != nil {
		t.Fatalf("responsesToChatCompletion: %v", err)
	}
	var chat map[string]any
	if err := json.Unmarshal(got, &chat); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	choices, _ := chat["choices"].([]any)
	choice, _ := choices[0].(map[string]any)
	if choice["finish_reason"] != "content_filter" {
		t.Fatalf("finish_reason: got %#v", choice["finish_reason"])
	}
}

func TestResponsesToChatCompletionMapsRefusal(t *testing.T) {
	body := []byte(`{
		"id":"resp_1","status":"completed","model":"gpt-5.6-terra",
		"output":[{"type":"message","role":"assistant","content":[{"type":"refusal","refusal":"I cannot help with that."}]}]
	}`)

	got, err := responsesToChatCompletion(body)
	if err != nil {
		t.Fatalf("responsesToChatCompletion: %v", err)
	}
	var chat map[string]any
	if err := json.Unmarshal(got, &chat); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	choices, _ := chat["choices"].([]any)
	choice, _ := choices[0].(map[string]any)
	msg, _ := choice["message"].(map[string]any)
	if msg["refusal"] != "I cannot help with that." {
		t.Fatalf("refusal was lost: got %#v", msg)
	}
}

func TestResponsesToChatCompletionRejectsFailedResponse(t *testing.T) {
	body := []byte(`{
		"id":"resp_1","object":"response","status":"failed",
		"error":{"code":"server_error","message":"generation failed"},
		"output":[]
	}`)

	if _, err := responsesToChatCompletion(body); err == nil {
		t.Fatal("failed Responses object must not become an empty successful completion")
	}
}

func TestResponsesToChatCompletionRejectsResponseWithoutOutput(t *testing.T) {
	body := []byte(`{"id":"resp_1","object":"response","status":"completed","output":null}`)

	if _, err := responsesToChatCompletion(body); err == nil {
		t.Fatal("malformed Responses object must not leak through the chat/completions endpoint")
	}
}

// An upstream error body is not a Responses object. Rewriting it into an empty
// chat completion would turn a visible failure into a silent empty answer, so
// it must pass through untouched.
func TestResponsesToChatCompletionPassesThroughErrorBodies(t *testing.T) {
	body := []byte(`{"error":{"message":"bad request","type":"invalid_request_error"}}`)

	got, err := responsesToChatCompletion(body)
	if err != nil {
		t.Fatalf("responsesToChatCompletion: %v", err)
	}
	if string(got) != string(body) {
		t.Errorf("error body must pass through unchanged: got %s", got)
	}
}

func TestResponsesToChatCompletionRejectsUnparseableBody(t *testing.T) {
	if _, err := responsesToChatCompletion([]byte("not json")); err == nil {
		t.Error("expected an error for a body that is not JSON")
	}
}

func TestChatToResponsesRequestDoesNotMutateInput(t *testing.T) {
	payload := map[string]any{
		"model":    "gpt-5.6-terra",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}
	before, _ := json.Marshal(payload)

	if _, err := chatToResponsesRequest(payload); err != nil {
		t.Fatalf("chatToResponsesRequest: %v", err)
	}

	after, _ := json.Marshal(payload)
	if string(before) != string(after) {
		t.Errorf("input payload mutated:\nbefore %s\nafter  %s", before, after)
	}
}

// The unit-level rejection of failed Responses objects is not enough: the
// direct dispatch path recovers from translation errors by passing the body
// through, which would hand the agent a 200 whose body is a raw failed
// Responses object. A recognized-but-failed reply must surface as a failure
// (or a declared-fallback advance), never as a success.
func TestHandlerDoesNotForwardFailedResponsesObjectAsSuccess(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_1","object":"response","status":"failed",
			"error":{"code":"server_error","message":"generation failed"},"output":[]
		}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5.6-terra","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code == http.StatusOK {
		t.Fatalf("a failed Responses object must not reach the agent as a 200: %s", w.Body.String())
	}
}

// Same guarantee for a streamed request: a failed Responses object must not be
// synthesized into an SSE stream that ends with a clean stop and no content.
func TestHandlerDoesNotStreamFailedResponsesObjectAsSuccess(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_1","object":"response","status":"failed",
			"error":{"code":"server_error","message":"generation failed"},"output":[]
		}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-5.6-terra","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if strings.Contains(w.Body.String(), "data: [DONE]") && w.Code == http.StatusOK {
		t.Fatalf("a failed Responses object must not become a clean synthetic stream: %s", w.Body.String())
	}
}

// The owner's pre-merge ask #1: a turn dispatched through the adapter must land
// in session history with the same shape as a chat-path turn — chat-shaped
// request_effective and response, populated usage, and populated
// reported_cost_usd when pricing knows the model. Proven with a priced model
// routed through the adapter via the env override, since cost parity cannot be
// shown on a model the pricing table does not know.
func TestAdapterTurnRecordsChatShapedSessionHistoryWithCost(t *testing.T) {
	t.Setenv(EnvResponsesAPIModels, "openai/gpt-4o")

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Errorf("expected adapter dispatch, got %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"resp_1","object":"response","status":"completed","model":"gpt-4o",
			"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hello"}]}],
			"usage":{"input_tokens":1000,"output_tokens":500,"total_tokens":1500}
		}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{
		Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	histDir := t.TempDir()
	h := NewHandler(reg, stubContextLoaderWithToken("tiverton", "tiverton:dummy123"), logging.New(io.Discard),
		WithCostTracking(cost.NewAccumulator(), cost.DefaultPricing()),
		WithSessionHistory(histDir))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
		bytes.NewBufferString(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	raw, err := os.ReadFile(filepath.Join(histDir, "tiverton", "history.jsonl"))
	if err != nil {
		t.Fatalf("read history: %v", err)
	}
	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimRight(raw, "\n"), &entry); err != nil {
		t.Fatalf("unmarshal history entry: %v", err)
	}

	if entry["path"] != "/v1/chat/completions" {
		t.Errorf("history path must be the agent-facing path: got %#v", entry["path"])
	}
	reqEff, _ := entry["request_effective"].(map[string]any)
	if _, ok := reqEff["messages"]; !ok {
		t.Errorf("request_effective must stay chat-shaped (messages present): got %#v", reqEff)
	}
	if _, ok := reqEff["input"]; ok {
		t.Error("request_effective must not carry the translated Responses shape")
	}
	respPayload, _ := entry["response"].(map[string]any)
	if respPayload["format"] != "json" {
		t.Errorf("response format: got %#v", respPayload["format"])
	}
	respJSON, _ := respPayload["json"].(map[string]any)
	if respJSON["object"] != "chat.completion" {
		t.Errorf("recorded response must be chat-shaped: got %#v", respJSON["object"])
	}
	usage, _ := entry["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(1000) || usage["completion_tokens"] != float64(500) {
		t.Errorf("usage tokens must be populated from the translated reply: got %#v", entry["usage"])
	}
	costUSD, ok := usage["reported_cost_usd"].(float64)
	if !ok || costUSD <= 0 {
		t.Errorf("reported_cost_usd must be populated when pricing knows the model: got %#v", usage["reported_cost_usd"])
	}
	// gpt-4o: 1000 in @ $2.50/M + 500 out @ $10/M
	if want := 0.0075; costUSD < want*0.99 || costUSD > want*1.01 {
		t.Errorf("reported_cost_usd: got %v, want ~%v", costUSD, want)
	}
}

// The owner's pre-merge ask #2: a multi-round managed-tool turn through the
// adapter must produce the same tool_trace as the identical turn on the chat
// path. Both paths run against equivalent backends and the traces are compared
// structurally, with volatile fields normalized.
func TestAdapterMediatedToolTraceMatchesChatPath(t *testing.T) {
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	alias := managedToolHashlessAliasForCanonical("trading-api.get_market_context")

	runTurn := func(t *testing.T, adapter bool) []any {
		t.Helper()
		var rounds int
		backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			rounds++
			w.Header().Set("Content-Type", "application/json")
			if adapter {
				if r.URL.Path != "/v1/responses" {
					t.Errorf("expected adapter dispatch, got %q", r.URL.Path)
				}
				if rounds == 1 {
					_, _ = w.Write([]byte(`{
						"id":"resp_1","status":"completed","model":"gpt-4o",
						"output":[{"type":"function_call","id":"fc_1","call_id":"call_1","name":"` + alias + `","arguments":"{}"}],
						"usage":{"input_tokens":10,"output_tokens":3,"total_tokens":13}
					}`))
					return
				}
				_, _ = w.Write([]byte(`{
					"id":"resp_2","status":"completed","model":"gpt-4o",
					"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}],
					"usage":{"input_tokens":7,"output_tokens":5,"total_tokens":12}
				}`))
				return
			}
			if r.URL.Path != "/v1/chat/completions" {
				t.Errorf("expected chat dispatch, got %q", r.URL.Path)
			}
			if rounds == 1 {
				_, _ = w.Write([]byte(`{
					"id":"chatcmpl-1","model":"gpt-4o",
					"choices":[{"finish_reason":"tool_calls","message":{"role":"assistant","tool_calls":[
						{"id":"call_1","type":"function","function":{"name":"` + alias + `","arguments":"{}"}}
					]}}],
					"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}
				}`))
				return
			}
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2","model":"gpt-4o",
				"choices":[{"message":{"role":"assistant","content":"done"}}],
				"usage":{"prompt_tokens":7,"completion_tokens":5,"total_tokens":12}
			}`))
		}))
		defer backend.Close()

		if adapter {
			t.Setenv(EnvResponsesAPIModels, "openai/gpt-4o")
		} else {
			t.Setenv(EnvResponsesAPIModels, "")
		}

		reg := provider.NewRegistry("")
		reg.Set("openai", &provider.Provider{
			Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
		})

		histDir := t.TempDir()
		h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123",
			managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")),
			logging.New(io.Discard), WithSessionHistory(histDir))
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions",
			bytes.NewBufferString(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("Authorization", "Bearer tiverton:dummy123")
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()

		h.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
		}
		if rounds != 2 {
			t.Fatalf("expected two model rounds, got %d", rounds)
		}

		raw, err := os.ReadFile(filepath.Join(histDir, "tiverton", "history.jsonl"))
		if err != nil {
			t.Fatalf("read history: %v", err)
		}
		var entry map[string]any
		if err := json.Unmarshal(bytes.TrimRight(raw, "\n"), &entry); err != nil {
			t.Fatalf("unmarshal history entry: %v", err)
		}
		trace, _ := entry["tool_trace"].([]any)
		if len(trace) == 0 {
			t.Fatalf("expected a tool_trace in history, got %#v", entry["tool_trace"])
		}
		return trace
	}

	normalize := func(trace []any) string {
		for _, rawRound := range trace {
			round, _ := rawRound.(map[string]any)
			for _, volatile := range []string{"latency_ms", "started_at", "finished_at", "duration_ms"} {
				delete(round, volatile)
			}
			calls, _ := round["tool_calls"].([]any)
			for _, rawCall := range calls {
				call, _ := rawCall.(map[string]any)
				for _, volatile := range []string{"latency_ms", "started_at", "finished_at", "duration_ms"} {
					delete(call, volatile)
				}
			}
		}
		out, _ := json.Marshal(trace)
		return string(out)
	}

	chatTrace := normalize(runTurn(t, false))
	adapterTrace := normalize(runTurn(t, true))
	if !strings.Contains(chatTrace, "get_market_context") {
		t.Fatalf("trace comparison would be vacuous — tool call missing from chat trace: %s", chatTrace)
	}
	if chatTrace != adapterTrace {
		t.Errorf("tool_trace diverges by request shape:\nchat:    %s\nadapter: %s", chatTrace, adapterTrace)
	}
}
