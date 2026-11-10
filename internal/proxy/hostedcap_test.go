package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

// hostedRig is a proxy whose agent has a fully permissive data policy, in
// front of a vendor that would serve anything it is sent.
type hostedRig struct {
	t      *testing.T
	h      *Handler
	logs   bytes.Buffer
	mu     sync.Mutex
	hits   int
	bodies []map[string]any
	paths  []string

	// Set before the first request.
	chain                 []string               // model chain; default below
	tools                 *agentctx.ToolManifest // makes requests take the managed path
	failFirstModel        string                 // this upstream model answers 500
	rejectMaxTokensOnChat bool                   // chat completions refuses max_tokens
}

const (
	hostedChatRef      = "openai/gpt-4o"        // dispatched as chat completions
	hostedResponsesRef = "openai/gpt-5.6-terra" // dispatched through the Responses adapter
	hostedFallbackRef  = "xai/grok-4"
)

func newHostedRig(t *testing.T, policy *agentctx.DataPolicy, grant any, hasGrant bool) *hostedRig {
	t.Helper()
	rig := &hostedRig{t: t}
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		rig.mu.Lock()
		rig.hits++
		rig.bodies = append(rig.bodies, body)
		rig.paths = append(rig.paths, r.URL.Path)
		rig.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if model, _ := body["model"].(string); rig.failFirstModel != "" && model == rig.failFirstModel {
			w.WriteHeader(http.StatusInternalServerError)
			_, _ = io.WriteString(w, `{"error":{"message":"unavailable"}}`)
			return
		}
		if rig.rejectMaxTokensOnChat && strings.HasSuffix(r.URL.Path, "/chat/completions") {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = io.WriteString(w, rejectsMaxTok)
			return
		}
		if strings.HasSuffix(r.URL.Path, "/responses") {
			_, _ = fmt.Fprintf(w, completedResponse, "ok")
			return
		}
		_, _ = io.WriteString(w, `{"id":"c1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	t.Cleanup(backend.Close)
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-api-key", Auth: "bearer"})
	reg.Set("xai", &provider.Provider{Name: "xai", BaseURL: backend.URL + "/v1", APIKey: "sk-xai-key", Auth: "bearer"})
	loader := func(id string) (*agentctx.AgentContext, error) {
		if id != "worker" {
			return nil, io.EOF
		}
		chain := rig.chain
		if chain == nil {
			chain = []string{hostedChatRef, hostedResponsesRef, hostedFallbackRef}
		}
		models := &agentctx.ModelPolicy{Mode: "clamp"}
		for i, ref := range chain {
			slot := "fallback"
			if i == 0 {
				slot = "primary"
			}
			models.Allowed = append(models.Allowed, agentctx.AllowedModel{Slot: slot, Ref: ref})
		}
		meta := map[string]any{"token": "worker:secret"}
		if hasGrant {
			meta["hosted_capabilities"] = grant
		}
		return &agentctx.AgentContext{DataPolicy: policy, AgentID: id, ContextDir: "/claw/context/" + id,
			AgentsMD: []byte("# Contract"), ClawdapusMD: []byte("# Infra"), Metadata: meta, ModelPolicy: models, Tools: rig.tools}, nil
	}
	rig.h = NewHandler(reg, loader, logging.New(&rig.logs))
	return rig
}

func (rig *hostedRig) direct(model string, extra map[string]any) *httptest.ResponseRecorder {
	rig.t.Helper()
	body := map[string]any{"model": model, "messages": []any{map[string]any{"role": "user", "content": "hi"}}}
	for k, v := range extra {
		body[k] = v
	}
	raw, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(raw))
	req.Header.Set("Authorization", "Bearer worker:secret")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	rig.h.ServeHTTP(w, req)
	return w
}

func (rig *hostedRig) managed(model string, grants map[string]bool, extra map[string]any) (int, string, error) {
	rig.t.Helper()
	var candidates []dispatchCandidate
	for _, ref := range []string{model, hostedFallbackRef} {
		c, err := rig.h.openAICandidateFromRef(ref)
		if err != nil {
			rig.t.Fatal(err)
		}
		candidates = append(candidates, c)
	}
	ctx := context.WithValue(context.Background(), dataPolicyKey{}, *testPermissiveDataPolicy())
	ctx = context.WithValue(ctx, hostedGrantKey{}, grants)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx)
	payload := map[string]any{"model": model, "messages": []any{map[string]any{"role": "user", "content": "hi"}}}
	for k, v := range extra {
		payload[k] = v
	}
	_, status, msg, err := rig.h.dispatchCandidatesJSON(ctx, req, "worker", model, payload, candidates, nil)
	return status, msg, err
}

func (rig *hostedRig) refusals() []string {
	var out []string
	for _, entry := range parseLogEntries(rig.t, rig.logs.Bytes()) {
		if v := fmt.Sprint(entry["intervention"]); strings.HasPrefix(v, "hosted_capability_refused:") {
			out = append(out, strings.TrimPrefix(v, "hosted_capability_refused:"))
		}
	}
	return out
}

func tools(entries ...map[string]any) map[string]any {
	list := make([]any, 0, len(entries))
	for _, e := range entries {
		list = append(list, e)
	}
	return map[string]any{"tools": list}
}

var functionTool = map[string]any{"type": "function", "function": map[string]any{"name": "lookup", "parameters": map[string]any{"type": "object"}}}

// hostedRequests are requests that ask the vendor to act on the agent's
// behalf, with the capability each is refused for.
var hostedRequests = map[string]struct {
	extra map[string]any
	// dropped names a body key the chat-to-Responses adapter never
	// translates. On that path the request is sent without it, which the
	// test asserts, instead of being refused.
	dropped string
	want    string
}{
	"web search tool":        {tools(map[string]any{"type": "web_search"}), "", "web_search"},
	"web search preview":     {tools(map[string]any{"type": "web_search_preview_2025_03_11"}), "", "web_search"},
	"web search options":     {map[string]any{"web_search_options": map[string]any{}}, "web_search_options", "web_search"},
	"code interpreter":       {tools(map[string]any{"type": "code_interpreter", "container": map[string]any{"type": "auto"}}), "", "code_interpreter"},
	"file search":            {tools(map[string]any{"type": "file_search", "vector_store_ids": []any{"vs_1"}}), "", "file_search"},
	"image generation":       {tools(map[string]any{"type": "image_generation"}), "", "image_generation"},
	"remote mcp tool":        {tools(map[string]any{"type": "mcp", "server_label": "x", "server_url": "https://mcp.example/sse"}), "", "mcp"},
	"mcp servers":            {map[string]any{"mcp_servers": []any{map[string]any{"url": "https://mcp.example"}}}, "mcp_servers", "mcp"},
	"untyped remote server":  {tools(map[string]any{"name": "x", "server_url": "https://mcp.example/sse"}), "", "mcp"},
	"plugins":                {map[string]any{"plugins": []any{map[string]any{"id": "web"}}}, "plugins", "plugins"},
	"unknown hosted type":    {tools(map[string]any{"type": "browser_agent"}), "", "unlisted"},
	"type not a string":      {tools(map[string]any{"type": []any{"function"}}), "", "unlisted"},
	"tools not a list":       {map[string]any{"tools": map[string]any{"type": "web_search"}}, "tools", "unlisted"},
	"client tool look-alike": {tools(map[string]any{"type": "bash_agent"}), "", "unlisted"},
	"computer look-alike":    {tools(map[string]any{"type": "computer_agent_hosted"}), "", "unlisted"},
	"namespace member with a remote server": {tools(map[string]any{"type": "namespace", "name": "n", "tools": []any{
		map[string]any{"type": "function", "name": "x", "server_url": "https://mcp.example/sse"}}}), "", "unlisted"},
	"namespace hiding a tool":  {tools(map[string]any{"type": "namespace", "name": "n", "tools": []any{functionTool, map[string]any{"type": "web_search"}}}), "", "unlisted"},
	"malformed namespace":      {tools(map[string]any{"type": "namespace", "name": "n", "tools": "x"}), "", "unlisted"},
	"hosted beside a function": {tools(functionTool, map[string]any{"type": "web_search"}), "", "web_search"},
}

// Under a fully permissive data policy and with no grant, nothing hosted
// reaches the vendor, on either dispatch path and in either wire shape.
func TestHostedCapabilitiesAreRefusedWithoutAGrant(t *testing.T) {
	for name, tc := range hostedRequests {
		for _, model := range []string{hostedChatRef, hostedResponsesRef} {
			if tc.dropped != "" && model == hostedResponsesRef {
				// The adapter does not carry this key, so the wire has
				// nothing hosted. What matters is that it is not sent.
				t.Run(name+"/adapted", func(t *testing.T) {
					rig := newHostedRig(t, testPermissiveDataPolicy(), nil, false)
					if w := rig.direct(model, tc.extra); w.Code != 200 || rig.hits != 1 {
						t.Fatalf("status %d hits %d: %s", w.Code, rig.hits, w.Body)
					}
					if _, sent := rig.bodies[0][tc.dropped]; sent {
						t.Fatalf("%s reached the vendor: %v", tc.dropped, rig.bodies[0])
					}
				})
				continue
			}
			t.Run(name+"/direct/"+model, func(t *testing.T) {
				rig := newHostedRig(t, testPermissiveDataPolicy(), nil, false)
				w := rig.direct(model, tc.extra)
				if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "hosted capability not granted: "+tc.want) {
					t.Fatalf("status %d: %s", w.Code, w.Body)
				}
				if rig.hits != 0 {
					t.Fatalf("the vendor was called %d times: %v", rig.hits, rig.paths)
				}
				if got := rig.refusals(); len(got) != 1 || got[0] != tc.want {
					t.Fatalf("refusals = %v", got)
				}
			})
			t.Run(name+"/managed/"+model, func(t *testing.T) {
				rig := newHostedRig(t, testPermissiveDataPolicy(), nil, false)
				status, msg, err := rig.managed(model, nil, tc.extra)
				if err == nil || status != http.StatusBadRequest || !strings.Contains(msg, "hosted capability not granted: "+tc.want) {
					t.Fatalf("status %d %q %v", status, msg, err)
				}
				if rig.hits != 0 {
					t.Fatalf("the vendor was called %d times, fallback candidate included: %v", rig.hits, rig.paths)
				}
			})
		}
	}
	// The :online model variant is asked for through the model name.
	rig := newHostedRig(t, testPermissiveDataPolicy(), nil, false)
	if got := hostedCapabilities(map[string]any{"model": "x/y:online"}); len(got) != 1 || got[0] != "web_search" {
		t.Fatalf(":online = %v", got)
	}
	_ = rig
}

// The agent's own tools are not hosted capabilities.
func TestOwnToolsNeedNoGrant(t *testing.T) {
	own := map[string]map[string]any{
		"function":             tools(functionTool),
		"custom":               tools(map[string]any{"type": "custom", "name": "grammar"}),
		"untyped definition":   tools(map[string]any{"name": "lookup", "input_schema": map[string]any{"type": "object"}}),
		"client bash tool":     tools(map[string]any{"type": "bash_20250124", "name": "bash"}),
		"client computer tool": tools(map[string]any{"type": "computer_use_preview", "display_width": 1024}),
		"no tools":             {},
		"empty lists":          {"tools": []any{}, "mcp_servers": []any{}, "plugins": []any{}},
	}
	for name, extra := range own {
		t.Run(name, func(t *testing.T) {
			rig := newHostedRig(t, testPermissiveDataPolicy(), nil, false)
			if w := rig.direct(hostedChatRef, extra); w.Code != 200 || rig.hits != 1 {
				t.Fatalf("status %d hits %d: %s", w.Code, rig.hits, w.Body)
			}
		})
	}
	// A namespace of functions is the agent's own; checked on the wire body.
	wire, _ := json.Marshal(tools(map[string]any{"type": "namespace", "name": "agent_tools", "tools": []any{functionTool}}))
	if err := enforceHostedCapabilities(nil, wire); err != nil {
		t.Fatalf("namespace of functions: %v", err)
	}
}

// A grant names capabilities exactly. It covers nothing else, and it never
// gets past a data policy that refuses.
func TestHostedCapabilityGrant(t *testing.T) {
	search := tools(functionTool, map[string]any{"type": "web_search"})

	rig := newHostedRig(t, testPermissiveDataPolicy(), []any{"web_search"}, true)
	if w := rig.direct(hostedResponsesRef, search); w.Code != 200 || rig.hits != 1 {
		t.Fatalf("granted web search: status %d hits %d: %s", w.Code, rig.hits, w.Body)
	}
	sent, _ := json.Marshal(rig.bodies[0]["tools"])
	if !strings.Contains(string(sent), `"web_search"`) {
		t.Fatalf("the granted tool was not sent: %s", sent)
	}

	// The same grant does not cover another capability in the same request.
	rig = newHostedRig(t, testPermissiveDataPolicy(), []any{"web_search"}, true)
	w := rig.direct(hostedResponsesRef, tools(map[string]any{"type": "web_search"}, map[string]any{"type": "code_interpreter"}))
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "hosted capability not granted: code_interpreter") || rig.hits != 0 {
		t.Fatalf("partial grant: status %d hits %d: %s", w.Code, rig.hits, w.Body)
	}

	// A remote server cannot be granted at all.
	rig = newHostedRig(t, testPermissiveDataPolicy(), []any{"web_search", "code_interpreter", "file_search", "image_generation"}, true)
	w = rig.direct(hostedResponsesRef, tools(map[string]any{"type": "mcp", "server_url": "https://mcp.example/sse"}))
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "hosted capability not granted: mcp") || rig.hits != 0 {
		t.Fatalf("mcp with every grant: status %d hits %d: %s", w.Code, rig.hits, w.Body)
	}

	// With a grant, a constrained data policy still refuses, in its own words.
	constrained := agentctx.DefaultDataPolicy()
	rig = newHostedRig(t, &constrained, []any{"web_search"}, true)
	w = rig.direct(hostedChatRef, search)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "data policy requirement not established") || rig.hits != 0 {
		t.Fatalf("grant under a constrained policy: status %d hits %d: %s", w.Code, rig.hits, w.Body)
	}
	if got := rig.refusals(); len(got) != 0 {
		t.Fatalf("the data policy refusal was reported as a hosted refusal: %v", got)
	}

	// The managed path honours the grant the same way.
	rig = newHostedRig(t, testPermissiveDataPolicy(), nil, false)
	if status, msg, err := rig.managed(hostedResponsesRef, map[string]bool{"web_search": true}, search); err != nil || rig.hits != 1 {
		t.Fatalf("managed granted: %d %q %v hits %d", status, msg, err, rig.hits)
	}
}

// A grant that is not a list of known names is an invalid context. It is
// never read as a partial grant or as no grant.
func TestMalformedHostedGrantRefusesTheRequest(t *testing.T) {
	for name, grant := range map[string]any{
		"a string":            "web_search",
		"null":                nil,
		"an object":           map[string]any{"web_search": true},
		"a non-string entry":  []any{7},
		"an unknown name":     []any{"web_search", "browser_agent"},
		"mcp":                 []any{"mcp"},
		"a wildcard":          []any{"*"},
		"a differently cased": []any{"Web_Search"},
	} {
		t.Run(name, func(t *testing.T) {
			rig := newHostedRig(t, testPermissiveDataPolicy(), grant, true)
			// Even a request with no hosted feature is refused: the context is invalid.
			w := rig.direct(hostedChatRef, nil)
			if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "invalid hosted capability grant") || rig.hits != 0 {
				t.Fatalf("status %d hits %d: %s", w.Code, rig.hits, w.Body)
			}
		})
	}
	// An empty list is a valid grant of nothing.
	rig := newHostedRig(t, testPermissiveDataPolicy(), []any{}, true)
	if w := rig.direct(hostedChatRef, nil); w.Code != 200 {
		t.Fatalf("empty grant: %d %s", w.Code, w.Body)
	}
	if w := rig.direct(hostedChatRef, tools(map[string]any{"type": "web_search"})); w.Code != http.StatusBadRequest {
		t.Fatalf("empty grant let a hosted tool through: %d", w.Code)
	}
}

// The request cannot grant itself anything.
func TestRequestCannotGrantItself(t *testing.T) {
	rig := newHostedRig(t, testPermissiveDataPolicy(), nil, false)
	extra := tools(map[string]any{"type": "web_search"})
	extra["hosted_capabilities"] = []any{"web_search"}
	extra["metadata"] = map[string]any{"hosted_capabilities": []any{"web_search"}}
	if w := rig.direct(hostedChatRef, extra); w.Code != http.StatusBadRequest || rig.hits != 0 {
		t.Fatalf("status %d hits %d: %s", w.Code, rig.hits, w.Body)
	}
}

// The guard is applied to every candidate, not only the first: a chain whose
// first entry is harmless and unavailable does not let the next one through.
func TestHostedGuardCoversTheFallbackCandidate(t *testing.T) {
	rig := newHostedRig(t, testPermissiveDataPolicy(), nil, false)
	rig.chain = []string{hostedChatRef, "xai/grok-4:online"}
	rig.failFirstModel = "gpt-4o"
	w := rig.direct(hostedChatRef, nil)
	if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "hosted capability not granted: web_search") {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	// One call, to the first candidate. The fallback was refused before any.
	if rig.hits != 1 || rig.bodies[0]["model"] != "gpt-4o" {
		t.Fatalf("hits = %d, bodies = %v", rig.hits, rig.bodies)
	}
	if got := rig.refusals(); len(got) != 1 || got[0] != "web_search" {
		t.Fatalf("refusals = %v", got)
	}

	// With the grant, the same fallback is served.
	rig = newHostedRig(t, testPermissiveDataPolicy(), []any{"web_search"}, true)
	rig.chain = []string{hostedChatRef, "xai/grok-4:online"}
	rig.failFirstModel = "gpt-4o"
	if w := rig.direct(hostedChatRef, nil); w.Code != 200 || rig.hits != 2 || rig.bodies[1]["model"] != "grok-4:online" {
		t.Fatalf("granted fallback: status %d hits %d: %s", w.Code, rig.hits, w.Body)
	}
}

// A request that is re-sent in another wire shape is judged again in that
// shape, with the same grant.
func TestHostedGuardCoversTheShapeRetry(t *testing.T) {
	const unknown = "openai/gpt-9.9-nova"
	search := tools(functionTool, map[string]any{"type": "web_search"})
	search["max_tokens"] = 50

	// Granted: chat is tried, refuses max_tokens, and the Responses retry
	// carries the granted tool.
	rig := newHostedRig(t, testPermissiveDataPolicy(), []any{"web_search"}, true)
	rig.chain, rig.rejectMaxTokensOnChat = []string{unknown}, true
	if w := rig.direct(unknown, search); w.Code != 200 {
		t.Fatalf("granted retry: status %d: %s", w.Code, w.Body)
	}
	if len(rig.paths) != 2 || !strings.HasSuffix(rig.paths[0], "/chat/completions") || !strings.HasSuffix(rig.paths[1], "/responses") {
		t.Fatalf("paths = %v", rig.paths)
	}
	if sent, _ := json.Marshal(rig.bodies[1]["tools"]); !strings.Contains(string(sent), `"web_search"`) {
		t.Fatalf("retry wire lost the granted tool: %s", sent)
	}

	// The retry's wire body is what is judged: enforceHostedCapabilities on
	// that exact body refuses it without the grant.
	wire, _ := json.Marshal(rig.bodies[1])
	if err := enforceHostedCapabilities(nil, wire); err == nil || !strings.Contains(err.Error(), "web_search") {
		t.Fatalf("the adapted wire body was not classified as hosted: %v", err)
	}

	// Not granted: refused before the first attempt, so no retry happens.
	rig = newHostedRig(t, testPermissiveDataPolicy(), nil, false)
	rig.chain, rig.rejectMaxTokensOnChat = []string{unknown}, true
	if w := rig.direct(unknown, search); w.Code != http.StatusBadRequest || rig.hits != 0 {
		t.Fatalf("ungranted: status %d hits %d", w.Code, rig.hits)
	}
}

// The managed-tool path, entered through the HTTP handler with a compiled
// tool manifest, refuses the same way.
func TestHostedGuardOnTheManagedHTTPPath(t *testing.T) {
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"ok":true}`)
	}))
	defer toolSrv.Close()
	for _, model := range []string{hostedChatRef, hostedResponsesRef} {
		rig := newHostedRig(t, testPermissiveDataPolicy(), nil, false)
		rig.tools = managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/context/{claw_id}", "")
		w := rig.direct(model, tools(map[string]any{"type": "web_search"}))
		if w.Code != http.StatusBadRequest || !strings.Contains(w.Body.String(), "hosted capability not granted: web_search") || rig.hits != 0 {
			t.Fatalf("%s: status %d hits %d: %s", model, w.Code, rig.hits, w.Body)
		}
		// An ordinary request on the same path is served.
		if w := rig.direct(model, nil); w.Code != 200 || rig.hits != 1 {
			t.Fatalf("%s plain: status %d hits %d: %s", model, w.Code, rig.hits, w.Body)
		}
	}
}
