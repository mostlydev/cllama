package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

type anthropicBackend struct {
	mu      sync.Mutex
	paths   []string
	bodies  [][]byte
	headers []http.Header
	replies []string
}

func (b *anthropicBackend) handler(t *testing.T) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("read body: %v", err)
		}
		b.mu.Lock()
		b.paths = append(b.paths, r.URL.Path)
		b.bodies = append(b.bodies, body)
		b.headers = append(b.headers, r.Header.Clone())
		round := len(b.bodies)
		b.mu.Unlock()
		if round > len(b.replies) {
			t.Errorf("unexpected upstream round %d", round)
			http.Error(w, "unexpected", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(b.replies[round-1]))
	}
}

func (b *anthropicBackend) body(t *testing.T, i int) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal(b.bodies[i], &out); err != nil {
		t.Fatalf("unmarshal upstream body %d: %v", i, err)
	}
	return out
}

func newAnthropicBridgeHandler(t *testing.T, backend *anthropicBackend, loader ContextLoader) (*Handler, func()) {
	t.Helper()
	srv := httptest.NewServer(backend.handler(t))
	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: srv.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})
	if loader == nil {
		loader = stubContextLoaderWithToken("worker", "worker:dummy123")
	}
	return NewHandler(reg, loader, logging.New(io.Discard)), srv.Close
}

func serveChat(t *testing.T, h *Handler, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer worker:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w
}

const anthropicTextReply = `{
	"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5-5",
	"content":[{"type":"text","text":"hello from claude"}],
	"stop_reason":"end_turn",
	"usage":{"input_tokens":10,"cache_read_input_tokens":4,"cache_creation_input_tokens":2,"output_tokens":5}
}`

func TestHandlerServesOpenAIRequestFromAnthropicProvider(t *testing.T) {
	backend := &anthropicBackend{replies: []string{anthropicTextReply}}
	h, done := newAnthropicBridgeHandler(t, backend, nil)
	defer done()

	w := serveChat(t, h, `{
		"model":"anthropic/claude-sonnet-5-5",
		"messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],
		"temperature":0.2
	}`)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if backend.paths[0] != "/v1/messages" {
		t.Fatalf("expected dispatch to /v1/messages, got %q", backend.paths[0])
	}
	hdr := backend.headers[0]
	if hdr.Get("X-Api-Key") != "sk-ant-real" || hdr.Get("Authorization") != "" {
		t.Errorf("upstream must authenticate with x-api-key only, got %v", hdr)
	}
	if hdr.Get("Anthropic-Version") != anthropicAPIVersion {
		t.Errorf("anthropic-version: got %q", hdr.Get("Anthropic-Version"))
	}
	upstream := backend.body(t, 0)
	if upstream["model"] != "claude-sonnet-5-5" {
		t.Errorf("upstream model: got %#v", upstream["model"])
	}
	if upstream["max_tokens"] != float64(anthropicDefaultMaxTokens) {
		t.Errorf("max_tokens must default, got %#v", upstream["max_tokens"])
	}
	if _, ok := upstream["stream"]; ok {
		t.Errorf("adapter must dispatch non-streaming upstream")
	}
	system, _ := upstream["system"].([]any)
	if len(system) == 0 || system[0].(map[string]any)["text"] != "be brief" {
		t.Errorf("system: got %#v", upstream["system"])
	}
	if _, ok := upstream["temperature"]; ok {
		t.Errorf("current models reject sampling; temperature must be dropped")
	}

	var chat map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &chat); err != nil {
		t.Fatalf("downstream body: %v", err)
	}
	choice := chat["choices"].([]any)[0].(map[string]any)
	if choice["finish_reason"] != "stop" || choice["message"].(map[string]any)["content"] != "hello from claude" {
		t.Errorf("downstream choice: got %#v", choice)
	}
	usage := chat["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(16) || usage["completion_tokens"] != float64(5) {
		t.Errorf("usage must count all input tokens: got %#v", usage)
	}
	details := usage["prompt_tokens_details"].(map[string]any)
	if details["cached_tokens"] != float64(4) || details["cache_write_tokens"] != float64(2) {
		t.Errorf("cache split must survive: got %#v", details)
	}
}

func TestHandlerStreamsAnthropicAdapterReplyAsChatSSE(t *testing.T) {
	backend := &anthropicBackend{replies: []string{anthropicTextReply}}
	h, done := newAnthropicBridgeHandler(t, backend, nil)
	defer done()

	w := serveChat(t, h, `{"model":"anthropic/claude-sonnet-5-5","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if got := w.Header().Get("Content-Type"); !strings.Contains(got, "text/event-stream") {
		t.Fatalf("content-type: %q", got)
	}
	var sawContent, sawUsage bool
	for _, chunk := range parseSSEChunks(t, w.Body.Bytes()) {
		if _, ok := chunk["usage"]; ok {
			sawUsage = true
			continue
		}
		if sseDelta(t, chunk)["content"] == "hello from claude" {
			sawContent = true
		}
	}
	if !sawContent || !sawUsage {
		t.Errorf("expected content and usage chunks, got %s", w.Body.String())
	}
}

func TestHandlerRejectsUnsupportedSemanticsBeforeAnthropicDispatch(t *testing.T) {
	cases := map[string]string{
		"json mode":   `"response_format":{"type":"json_object"}`,
		"audio input": `"messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{"data":"AAAA","format":"wav"}}]}]`,
		"bad args":    `"messages":[{"role":"user","content":"hi"},{"role":"assistant","tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{not json"}}]}]`,
		"n":           `"n":2`,
		"unknown":     `"some_future_field":true`,
	}
	for name, extra := range cases {
		t.Run(name, func(t *testing.T) {
			backend := &anthropicBackend{}
			h, done := newAnthropicBridgeHandler(t, backend, nil)
			defer done()
			body := `{"model":"anthropic/claude-sonnet-5-5",` + extra
			if !strings.Contains(extra, `"messages"`) {
				body += `,"messages":[{"role":"user","content":"hi"}]`
			}
			body += `}`
			w := serveChat(t, h, body)
			if w.Code != http.StatusBadRequest {
				t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
			}
			if len(backend.bodies) != 0 {
				t.Fatalf("unsupported request must not reach upstream")
			}
		})
	}
}

func TestHandlerRunsManagedToolMediationThroughAnthropicAdapter(t *testing.T) {
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	alias := managedToolHashlessAliasForCanonical("trading-api.get_market_context")
	backend := &anthropicBackend{replies: []string{
		`{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5-5",
		  "content":[{"type":"tool_use","id":"toolu_1","name":"` + alias + `","input":{}}],
		  "stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":3}}`,
		`{"id":"msg_2","type":"message","role":"assistant","model":"claude-sonnet-5-5",
		  "content":[{"type":"text","text":"market context loaded"}],
		  "stop_reason":"end_turn","usage":{"input_tokens":7,"output_tokens":5}}`,
	}}
	loader := stubContextLoaderWithTools("worker", "worker:dummy123",
		managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", ""))
	h, done := newAnthropicBridgeHandler(t, backend, loader)
	defer done()

	w := serveChat(t, h, `{"model":"anthropic/claude-sonnet-5-5","messages":[{"role":"user","content":"hi"}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if len(backend.paths) != 2 || backend.paths[1] != "/v1/messages" {
		t.Fatalf("expected two rounds through /v1/messages, got %v", backend.paths)
	}
	first := backend.body(t, 0)
	if tools, _ := first["tools"].([]any); len(tools) == 0 || tools[0].(map[string]any)["input_schema"] == nil {
		t.Fatalf("managed tools must be translated to Anthropic tools: %s", backend.bodies[0])
	}
	second := backend.body(t, 1)
	messages := second["messages"].([]any)
	var sawUse, sawResult bool
	for _, raw := range messages {
		for _, rawBlock := range raw.(map[string]any)["content"].([]any) {
			block := rawBlock.(map[string]any)
			switch block["type"] {
			case "tool_use":
				sawUse = block["id"] == "toolu_1"
			case "tool_result":
				sawResult = block["tool_use_id"] == "toolu_1" && strings.Contains(block["content"].(string), "5000")
			}
		}
	}
	if !sawUse || !sawResult {
		t.Fatalf("round two must replay tool_use and tool_result: %s", backend.bodies[1])
	}
	var chat map[string]any
	_ = json.Unmarshal(w.Body.Bytes(), &chat)
	msg := chat["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if msg["content"] != "market context loaded" {
		t.Errorf("downstream content: got %#v", msg["content"])
	}
}

func TestHandlerPassesAnthropicErrorsThrough(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}`))
	}))
	defer srv.Close()
	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{Name: "anthropic", BaseURL: srv.URL + "/v1", APIKey: "k", Auth: "x-api-key", APIFormat: "anthropic"})
	h := NewHandler(reg, stubContextLoaderWithToken("worker", "worker:dummy123"), logging.New(io.Discard))

	w := serveChat(t, h, `{"model":"anthropic/claude-sonnet-5-5","messages":[{"role":"user","content":"hi"}]}`)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "invalid_request_error") {
		t.Fatalf("upstream rejection must pass through, got %d: %s", w.Code, w.Body.String())
	}
}

func TestOpenAIPathRoutesRegisteredAnthropicProviderNatively(t *testing.T) {
	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{Name: "anthropic", BaseURL: "https://api.anthropic.com/v1", Auth: "x-api-key", APIFormat: "anthropic"})
	reg.Set("openrouter", &provider.Provider{Name: "openrouter", BaseURL: "https://openrouter.ai/api/v1", APIKey: "or", Auth: "bearer"})
	h := NewHandler(reg, stubContextLoaderWithToken("worker", "worker:dummy123"), nil)

	candidate, err := h.openAICandidateFromRef("anthropic/claude-sonnet-5-5")
	if err != nil {
		t.Fatal(err)
	}
	if candidate.ProviderName != "anthropic" || candidate.UpstreamModel != "claude-sonnet-5-5" || candidate.APIFormat != "anthropic" {
		t.Fatalf("registered anthropic provider must route natively, even with OpenRouter present: %+v", candidate)
	}
	candidate, err = h.openAICandidateFromRef("mistral/large")
	if err != nil || candidate.ProviderName != "openrouter" || candidate.UpstreamModel != "mistral/large" {
		t.Fatalf("unknown vendor prefix must still bridge through OpenRouter: %+v %v", candidate, err)
	}
}

func TestChatToAnthropicRequestTranslatesToolRounds(t *testing.T) {
	payload := map[string]any{}
	if err := json.Unmarshal([]byte(`{
		"model":"claude-sonnet-4-6","max_completion_tokens":512,"temperature":0.5,"top_p":0.9,"stop":"END","user":"u1",
		"parallel_tool_calls":false,"tool_choice":"required",
		"tools":[{"type":"function","function":{"name":"lookup","description":"d","parameters":{"type":"object","properties":{"q":{"type":"string"}}}}}],
		"messages":[
			{"role":"developer","content":[{"type":"text","text":"rules"}]},
			{"role":"user","content":[{"type":"text","text":"look"},{"type":"image_url","image_url":{"url":"data:image/png;base64,iVBOR"}}]},
			{"role":"assistant","content":"checking","tool_calls":[
				{"id":"c1","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"a\"}"}},
				{"id":"c2","type":"function","function":{"name":"lookup","arguments":"{\"q\":\"b\"}"}}]},
			{"role":"tool","tool_call_id":"c1","content":"A"},
			{"role":"tool","tool_call_id":"c2","content":"B"}
		]}`), &payload); err != nil {
		t.Fatal(err)
	}
	out, mutations, err := chatToAnthropicRequest(payload)
	if err != nil {
		t.Fatal(err)
	}
	if out["max_tokens"] != float64(512) || out["temperature"] != 0.5 {
		t.Errorf("sampling: %#v", out)
	}
	if _, ok := out["top_p"]; ok || len(mutations) != 1 || mutations[0] != anthropicTopPDroppedIntervention {
		t.Errorf("top_p must be dropped beside temperature and logged, got %#v %v", out["top_p"], mutations)
	}
	if stops := out["stop_sequences"].([]any); len(stops) != 1 || stops[0] != "END" {
		t.Errorf("stop_sequences: %#v", out["stop_sequences"])
	}
	if out["metadata"].(map[string]any)["user_id"] != "u1" {
		t.Errorf("metadata: %#v", out["metadata"])
	}
	choice := out["tool_choice"].(map[string]any)
	if choice["type"] != "any" || choice["disable_parallel_tool_use"] != true {
		t.Errorf("tool_choice: %#v", choice)
	}
	tool := out["tools"].([]any)[0].(map[string]any)
	if tool["name"] != "lookup" || tool["input_schema"] == nil {
		t.Errorf("tool: %#v", tool)
	}
	messages := out["messages"].([]any)
	if len(messages) != 3 {
		t.Fatalf("expected user, assistant, merged tool results; got %#v", messages)
	}
	userBlocks := messages[0].(map[string]any)["content"].([]any)
	if image := userBlocks[1].(map[string]any); image["type"] != "image" || image["source"].(map[string]any)["media_type"] != "image/png" {
		t.Errorf("image block: %#v", userBlocks[1])
	}
	assistant := messages[1].(map[string]any)["content"].([]any)
	if len(assistant) != 3 || assistant[1].(map[string]any)["input"].(map[string]any)["q"] != "a" {
		t.Errorf("assistant blocks: %#v", assistant)
	}
	results := messages[2].(map[string]any)
	blocks := results["content"].([]any)
	if results["role"] != "user" || len(blocks) != 2 ||
		blocks[0].(map[string]any)["tool_use_id"] != "c1" || blocks[1].(map[string]any)["tool_use_id"] != "c2" {
		t.Errorf("tool results must merge into one user turn in order: %#v", results)
	}
}

func TestAnthropicToChatCompletionMapsToolUseAndStopReasons(t *testing.T) {
	body := []byte(`{"id":"msg_1","type":"message","model":"m","content":[
		{"type":"thinking","thinking":"hidden"},
		{"type":"text","text":"calling"},
		{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":"a"}}],
		"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":2}}`)
	out, converted, err := anthropicToChatCompletion(body)
	if err != nil || !converted {
		t.Fatalf("convert: %v %v", converted, err)
	}
	var chat map[string]any
	_ = json.Unmarshal(out, &chat)
	choice := chat["choices"].([]any)[0].(map[string]any)
	msg := choice["message"].(map[string]any)
	if choice["finish_reason"] != "tool_calls" || msg["content"] != "calling" {
		t.Errorf("choice: %#v", choice)
	}
	call := msg["tool_calls"].([]any)[0].(map[string]any)
	if call["id"] != "toolu_1" || call["function"].(map[string]any)["arguments"] != `{"q":"a"}` {
		t.Errorf("tool call: %#v", call)
	}
	for reason, want := range map[string]string{"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length", "refusal": "content_filter"} {
		if got := anthropicFinishReason(reason, false); got != want {
			t.Errorf("%s: got %s want %s", reason, got, want)
		}
	}
	if _, converted, _ := anthropicToChatCompletion([]byte(`{"type":"error","error":{"type":"x"}}`)); converted {
		t.Errorf("error bodies must pass through unconverted")
	}
	if _, _, err := anthropicToChatCompletion([]byte(`{"type":"message","content":[{"type":"server_tool_use"}]}`)); err == nil {
		t.Errorf("unknown response blocks must fail rather than vanish")
	}
}

func TestAnthropicModelAcceptsSampling(t *testing.T) {
	for model, want := range map[string]bool{
		"claude-sonnet-4-6": true, "claude-haiku-4-5": true, "claude-sonnet-4-5-20250929": true,
		"claude-opus-4": true, "claude-opus-4-20250514": true, "claude-3-5-haiku-latest": true,
		"claude-opus-4-7": false, "claude-opus-4-8": false, "claude-sonnet-5-5": false,
		"claude-haiku-5-5": false, "claude-opus-5": false, "claude-fable-5-1": false, "some-proxy-model": false,
	} {
		if got := anthropicModelAcceptsSampling(model); got != want {
			t.Errorf("%s: got %v want %v", model, got, want)
		}
	}
	out, mutations, err := chatToAnthropicRequest(map[string]any{
		"model": "claude-sonnet-5-5", "temperature": 0.7, "messages": []any{map[string]any{"role": "user", "content": "hi"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := out["temperature"]; ok || len(mutations) != 2 || mutations[1] != anthropicSamplingDroppedIntervention {
		t.Errorf("sampling must be dropped and logged for current models: %#v %v", out, mutations)
	}
}
