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

// shapeRig is a proxy in front of a vendor that behaves like a model this
// build has never heard of: chat completions refuses max_tokens, and the
// Responses API refuses sampling controls.
type shapeRig struct {
	t        *testing.T
	h        *Handler
	reg      *provider.Registry
	logs     bytes.Buffer
	mu       sync.Mutex
	paths    []string
	auth     []string
	bodies   []map[string]any
	chatFail string // JSON error body for chat completions; "" serves chat
	respFail string // JSON error body for every Responses call, when set
	backend  *httptest.Server
}

const (
	futureModel   = "gpt-9.9-nova"
	futureRef     = "openai/" + futureModel
	rejectsMaxTok = `{"error":{"message":"Unsupported parameter: 'max_tokens'.","type":"invalid_request_error","param":"max_tokens","code":"unsupported_parameter"}}`
)

func newShapeRig(t *testing.T, providerName string) *shapeRig {
	t.Helper()
	rig := &shapeRig{t: t, chatFail: rejectsMaxTok}
	rig.backend = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		rig.mu.Lock()
		rig.paths = append(rig.paths, r.URL.Path)
		rig.auth = append(rig.auth, r.Header.Get("Authorization"))
		rig.bodies = append(rig.bodies, body)
		chatFail, respFail := rig.chatFail, rig.respFail
		rig.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/chat/completions":
			if chatFail != "" {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = io.WriteString(w, chatFail)
				return
			}
			_, _ = io.WriteString(w, `{"id":"c1","object":"chat.completion","model":"`+futureModel+`","choices":[{"index":0,"message":{"role":"assistant","content":"from chat"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
		case "/v1/responses":
			if respFail != "" {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = io.WriteString(w, respFail)
				return
			}
			for _, field := range []string{"temperature", "top_p"} {
				if _, ok := body[field]; ok {
					w.WriteHeader(http.StatusBadRequest)
					_, _ = fmt.Fprintf(w, `{"error":{"message":"Unsupported parameter.","type":"invalid_request_error","param":%q,"code":"unsupported_parameter"}}`, field)
					return
				}
			}
			_, _ = fmt.Fprintf(w, completedResponse, "from responses")
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(rig.backend.Close)
	rig.reg = provider.NewRegistry("")
	rig.reg.Set(providerName, &provider.Provider{Name: providerName, BaseURL: rig.backend.URL + "/v1", APIKey: "sk-api-key", Auth: "bearer"})
	policy := &agentctx.ModelPolicy{Mode: "clamp", Allowed: []agentctx.AllowedModel{{Slot: "primary", Ref: providerName + "/" + futureModel}}}
	rig.h = NewHandler(rig.reg, stubContextLoaderWithPolicy("worker", "worker:secret", policy), logging.New(&rig.logs))
	return rig
}

func (rig *shapeRig) calls() []string {
	rig.mu.Lock()
	defer rig.mu.Unlock()
	return append([]string(nil), rig.paths...)
}

func (rig *shapeRig) direct(ref string) *httptest.ResponseRecorder {
	rig.t.Helper()
	body := `{"model":"` + ref + `","messages":[{"role":"user","content":"hi"}],"temperature":0.2,"top_p":0.9,"max_tokens":50}`
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer worker:secret")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	rig.h.ServeHTTP(w, req)
	return w
}

func (rig *shapeRig) managed(ref string) (int, string) {
	rig.t.Helper()
	candidate, err := rig.h.openAICandidateFromRef(ref)
	if err != nil {
		rig.t.Fatal(err)
	}
	ctx := context.WithValue(context.Background(), dataPolicyKey{}, *testPermissiveDataPolicy())
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx)
	payload := map[string]any{"model": ref, "temperature": 0.2, "top_p": 0.9, "max_tokens": 50.0,
		"messages": []any{map[string]any{"role": "user", "content": "hi"}}}
	resp, status, _, _ := rig.h.dispatchCandidatesJSON(ctx, req, "worker", ref, payload, []dispatchCandidate{candidate}, nil)
	if resp != nil {
		return resp.StatusCode, string(resp.Body)
	}
	return status, ""
}

func (rig *shapeRig) send(path, ref string) (int, string) {
	if path == "managed" {
		return rig.managed(ref)
	}
	w := rig.direct(ref)
	return w.Code, w.Body.String()
}

func (rig *shapeRig) interventions() []string {
	var out []string
	for _, entry := range parseLogEntries(rig.t, rig.logs.Bytes()) {
		if entry["type"] == "intervention" {
			out = append(out, fmt.Sprint(entry["intervention"]))
		}
	}
	return out
}

func count(items []string, want string) int {
	n := 0
	for _, item := range items {
		if item == want {
			n++
		}
	}
	return n
}

// A model this build does not know is served by following the vendor's own
// rejections, and the next request goes straight to the shape that worked.
func TestUnknownModelLearnsItsShape(t *testing.T) {
	for _, path := range []string{"direct", "managed"} {
		t.Run(path, func(t *testing.T) {
			rig := newShapeRig(t, "openai")
			status, body := rig.send(path, futureRef)
			if status != 200 || !strings.Contains(body, "from responses") {
				t.Fatalf("first request: %d %s", status, body)
			}
			want := []string{"/v1/chat/completions", "/v1/responses", "/v1/responses"}
			if got := rig.calls(); fmt.Sprint(got) != fmt.Sprint(want) {
				t.Fatalf("first request calls = %v, want %v", got, want)
			}
			for _, auth := range rig.auth {
				if auth != "Bearer sk-api-key" {
					t.Fatalf("a retry changed credential: %v", rig.auth)
				}
			}
			for _, sent := range rig.bodies {
				if sent["model"] != futureModel {
					t.Fatalf("a retry changed model: %v", sent["model"])
				}
			}
			last := rig.bodies[2]
			if _, ok := last["temperature"]; ok {
				t.Fatal("sampling control was sent again")
			}
			if last["max_output_tokens"] != 50.0 {
				t.Fatalf("max_tokens was not translated: %v", last)
			}

			status, body = rig.send(path, futureRef)
			if status != 200 || !strings.Contains(body, "from responses") {
				t.Fatalf("second request: %d %s", status, body)
			}
			if got := rig.calls(); len(got) != 4 || got[3] != "/v1/responses" {
				t.Fatalf("second request did not go straight to the learned shape: %v", got)
			}
			// The dropped controls are logged on the learned path too.
			if n := count(rig.interventions(), "responses_adapter_dropped_sampling"); n != 2 {
				t.Fatalf("responses_adapter_dropped_sampling logged %d times, want 2: %v", n, rig.interventions())
			}
			if n := count(rig.interventions(), "responses_api_adapter_retry"); n != 2 {
				t.Fatalf("retries logged %d times, want 2 (both on the first request)", n)
			}
		})
	}
}

// Nothing but the exact structured signal changes a request's shape.
func TestOtherRejectionsAreForwardedUnchanged(t *testing.T) {
	cases := map[string]string{
		"another parameter":   `{"error":{"message":"x","type":"invalid_request_error","param":"logit_bias","code":"unsupported_parameter"}}`,
		"another code":        `{"error":{"message":"x","type":"invalid_request_error","param":"max_tokens","code":"invalid_value"}}`,
		"only the message":    `{"error":{"message":"unsupported_parameter max_tokens","type":"invalid_request_error"}}`,
		"non-string fields":   `{"error":{"code":{"a":"unsupported_parameter"},"param":["max_tokens"]}}`,
		"not an error object": `"unsupported_parameter max_tokens"`,
	}
	for name, reply := range cases {
		for _, path := range []string{"direct", "managed"} {
			t.Run(name+"/"+path, func(t *testing.T) {
				rig := newShapeRig(t, "openai")
				rig.chatFail = reply
				status, _ := rig.send(path, futureRef)
				if status != http.StatusBadRequest {
					t.Fatalf("status = %d, want the vendor's 400", status)
				}
				if got := rig.calls(); len(got) != 1 {
					t.Fatalf("an unrelated rejection was retried: %v", got)
				}
			})
		}
	}
}

// Another provider's endpoint need not offer the Responses API.
func TestOtherProvidersAreNotMovedToResponses(t *testing.T) {
	rig := newShapeRig(t, "vercel")
	if w := rig.direct("vercel/" + futureModel); w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d", w.Code)
	}
	if got := rig.calls(); len(got) != 1 {
		t.Fatalf("calls = %v", got)
	}
}

// A shape that was tried and refused is not remembered.
func TestFailedShapeIsNotLearned(t *testing.T) {
	rig := newShapeRig(t, "openai")
	rig.respFail = `{"error":{"message":"x","type":"invalid_request_error","param":"input","code":"invalid_value"}}`
	if w := rig.direct(futureRef); w.Code != http.StatusBadRequest {
		t.Fatalf("status = %d", w.Code)
	}
	rig.mu.Lock()
	rig.chatFail, rig.respFail = "", ""
	rig.mu.Unlock()
	w := rig.direct(futureRef)
	if w.Code != 200 || !strings.Contains(w.Body.String(), "from chat") {
		t.Fatalf("second request: %d %s", w.Code, w.Body)
	}
	if got := rig.calls(); got[len(got)-1] != "/v1/chat/completions" {
		t.Fatalf("a refused shape was reused: %v", got)
	}
}

// A provider pointed at another endpoint starts without the old hints, and
// the operator's switch turns learned shapes off as well.
func TestLearnedShapeIsScoped(t *testing.T) {
	rig := newShapeRig(t, "openai")
	if w := rig.direct(futureRef); w.Code != 200 {
		t.Fatalf("status = %d", w.Code)
	}
	candidate, err := rig.h.openAICandidateFromRef(futureRef)
	if err != nil {
		t.Fatal(err)
	}
	if got := rig.h.hinted(candidate); !got.ViaResponses || !got.DropSampling {
		t.Fatalf("shape was not learned: %+v", got)
	}
	t.Run("disabled", func(t *testing.T) {
		t.Setenv(EnvResponsesAPIDisabled, "1")
		if got := rig.h.hinted(candidate); got.ViaResponses || got.DropSampling {
			t.Fatalf("the disable switch did not cover learned shapes: %+v", got)
		}
		before := len(rig.calls())
		if w := rig.direct(futureRef); w.Code != http.StatusBadRequest {
			t.Fatalf("status = %d", w.Code)
		}
		if got := rig.calls(); len(got) != before+1 || got[before] != "/v1/chat/completions" {
			t.Fatalf("calls = %v", got[before:])
		}
	})
	rig.reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: rig.backend.URL + "/v1/", APIKey: "sk-api-key", Auth: "bearer"})
	if got := rig.h.hinted(candidate); got.ViaResponses || got.DropSampling {
		t.Fatalf("another endpoint inherited the hint: %+v", got)
	}
}

func TestLearnedShapesAreBounded(t *testing.T) {
	var hints modelHints
	for i := 0; i < maxModelHints+50; i++ {
		hints.learn("http://x/v1", dispatchCandidate{ProviderName: "openai", UpstreamModel: fmt.Sprintf("m-%d", i), ViaResponses: true})
	}
	if len(hints.m) != maxModelHints {
		t.Fatalf("hints = %d, want %d", len(hints.m), maxModelHints)
	}
}

func TestModelHintDoesNotLearnFailedResponseWith200(t *testing.T) {
	for _, path := range []string{"direct", "managed"} {
		t.Run(path, func(t *testing.T) {
			rig := newShapeRig(t, "openai")
			rig.backend.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				if r.URL.Path == "/v1/chat/completions" {
					w.WriteHeader(400)
					_, _ = io.WriteString(w, rejectsMaxTok)
					return
				}
				_, _ = io.WriteString(w, `{"object":"response","status":"failed","error":{"code":"server_error"},"output":[]}`)
			})
			status, _ := rig.send(path, futureRef)
			if status != 502 {
				t.Fatalf("status = %d", status)
			}
			candidate, _ := rig.h.openAICandidateFromRef(futureRef)
			if got := rig.h.hinted(candidate); got.ViaResponses || got.DropSampling {
				t.Fatal("failed response taught a future request to use the speculative shape")
			}
		})
	}
}

func TestDisabledModelHintDoesNotRetryEndpointSignal(t *testing.T) {
	t.Setenv(EnvResponsesAPIDisabled, "true")
	for _, path := range []string{"direct", "managed"} {
		t.Run(path, func(t *testing.T) {
			rig := newShapeRig(t, "openai")
			rig.chatFail = `{"error":{"message":"use /v1/responses"}}`
			status, _ := rig.send(path, futureRef)
			if status != 400 || len(rig.calls()) != 1 {
				t.Fatalf("disabled adapter retried: status=%d calls=%v", status, rig.calls())
			}
		})
	}
}
