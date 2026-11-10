package proxy

import (
	"bytes"
	"context"
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
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
	"github.com/mostlydev/cllama/internal/subscription"
	"github.com/mostlydev/cllama/internal/tracejournal"
)

const completedResponse = `{"id":"resp_1","object":"response","status":"completed","model":"gpt-5.6-terra","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"%s"}]}],"usage":{"input_tokens":7,"output_tokens":3,"total_tokens":10}}`

func planSSE(events ...string) string {
	var b strings.Builder
	for _, e := range events {
		b.WriteString("data: " + e + "\n\n")
	}
	return b.String()
}

func planCompleted(text string) string {
	return planSSE(`{"type":"response.created"}`, `{"type":"response.output_text.delta","delta":"x"}`,
		`{"type":"response.completed","response":`+fmt.Sprintf(completedResponse, text)+`}`)
}

// planRig is a proxy with a plan route and an API-key route, each backed by
// a local server.
type planRig struct {
	t        *testing.T
	h        *Handler
	reg      *provider.Registry
	subs     *subscription.Manager
	store    *subscription.Store
	logs     bytes.Buffer
	mu       sync.Mutex
	planAuth []string
	planBody []map[string]any
	planHits int
	keyHits  int
	keyAuth  []string
	// planReply answers a plan-route request. nil serves a completed stream.
	planReply func(w http.ResponseWriter, r *http.Request, hit int)
	tokenHits int
	tokenFail string
}

func newPlanRig(t *testing.T, chain ...string) *planRig {
	t.Helper()
	rig := &planRig{t: t}
	planBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]any
		_ = json.Unmarshal(raw, &body)
		rig.mu.Lock()
		rig.planHits++
		hit := rig.planHits
		rig.planAuth = append(rig.planAuth, r.Header.Get("Authorization"))
		rig.planBody = append(rig.planBody, body)
		reply := rig.planReply
		rig.mu.Unlock()
		if r.URL.Path != "/v1/responses" {
			http.Error(w, "wrong route", http.StatusForbidden)
			return
		}
		if reply != nil {
			reply(w, r, hit)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, planCompleted("from the plan"))
	}))
	t.Cleanup(planBackend.Close)
	keyBackend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rig.mu.Lock()
		rig.keyHits++
		rig.keyAuth = append(rig.keyAuth, r.Header.Get("Authorization"))
		rig.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, completedResponse, "from the key")
	}))
	t.Cleanup(keyBackend.Close)
	issuer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rig.mu.Lock()
		rig.tokenHits++
		n := rig.tokenHits
		fail := rig.tokenFail
		rig.mu.Unlock()
		if fail != "" {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = fmt.Fprintf(w, `{"error":%q}`, fail)
			return
		}
		_, _ = fmt.Fprintf(w, `{"access_token":"plan-access-%d-canary","refresh_token":"plan-refresh-%d-canary","expires_in":3600,"scope":"offline_access chatgpt.tokens.use.direct"}`, n, n)
	}))
	t.Cleanup(issuer.Close)

	plan := subscription.ChatGPT
	plan.InferenceBase = planBackend.URL + "/v1"
	plan.TokenURL = issuer.URL + "/token"
	plan.Issuer = issuer.URL
	rig.store = subscription.NewStore(t.TempDir())
	rig.subs = subscription.NewManagerWithClients(subscription.NewClient(plan, rig.store))

	rig.reg = provider.NewRegistry("")
	// The registry's copy of the plan endpoint is deliberately wrong: the
	// token must go to the plan's own endpoint and nowhere else.
	rig.reg.RegisterPlan("chatgpt", "http://127.0.0.1:1/v1")
	rig.reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: keyBackend.URL + "/v1", APIKey: "sk-api-key", Auth: "bearer"})

	allowed := make([]agentctx.AllowedModel, 0, len(chain))
	for i, ref := range chain {
		slot := "fallback"
		if i == 0 {
			slot = "primary"
		}
		allowed = append(allowed, agentctx.AllowedModel{Slot: slot, Ref: ref})
	}
	policy := &agentctx.ModelPolicy{Mode: "clamp", Allowed: allowed}
	rig.h = NewHandler(rig.reg, stubContextLoaderWithPolicy("worker", "worker:secret", policy), logging.New(&rig.logs), WithSubscriptions(rig.subs))
	return rig
}

func (rig *planRig) signIn(access string, expiresIn time.Duration) {
	rig.t.Helper()
	err := rig.store.Save(&subscription.Record{
		Provider: "chatgpt", ClientID: "oaiapp_one", HostID: "urn:uuid:h", Subject: "user-1",
		AccessToken: access, RefreshToken: "plan-refresh-0-canary", Scopes: []string{"offline_access", "chatgpt.tokens.use.direct"},
		ExpiresAt: time.Now().Add(expiresIn), State: subscription.StateActive,
	})
	if err != nil {
		rig.t.Fatal(err)
	}
}

func (rig *planRig) chat(body string) *httptest.ResponseRecorder {
	rig.t.Helper()
	if body == "" {
		body = `{"model":"chatgpt/gpt-5.6-terra","messages":[{"role":"system","content":"be brief"},{"role":"user","content":"hi"}],"temperature":0.2,"max_tokens":50,"user":"u1"}`
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer worker:secret")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	rig.h.ServeHTTP(w, req)
	return w
}

func (rig *planRig) responseCredentials() []string {
	var out []string
	for _, entry := range parseLogEntries(rig.t, rig.logs.Bytes()) {
		if entry["type"] == "response" {
			out = append(out, fmt.Sprint(entry["credential"]))
		}
	}
	return out
}

func errorCode(t *testing.T, w *httptest.ResponseRecorder) string {
	t.Helper()
	var body struct {
		Error struct {
			Code string `json:"code"`
		} `json:"error"`
	}
	_ = json.Unmarshal(w.Body.Bytes(), &body)
	return body.Error.Code
}

const (
	planRef = "chatgpt/gpt-5.6-terra"
	keyRef  = "openai/gpt-5.6-terra"
)

func TestPlanRouteServesWithTheSignIn(t *testing.T) {
	rig := newPlanRig(t, planRef, keyRef)
	rig.signIn("plan-access-0-canary", time.Hour)
	w := rig.chat("")
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "from the plan") {
		t.Fatalf("reply: %d %s", w.Code, w.Body.String())
	}
	if rig.keyHits != 0 || rig.planHits != 1 {
		t.Fatalf("hits: plan %d key %d", rig.planHits, rig.keyHits)
	}
	if rig.planAuth[0] != "Bearer plan-access-0-canary" {
		t.Fatalf("plan auth: %q", rig.planAuth[0])
	}
	body := rig.planBody[0]
	if body["store"] != false || body["stream"] != true || body["model"] != "gpt-5.6-terra" {
		t.Fatalf("plan request shape: %v", body)
	}
	for _, field := range []string{"temperature", "max_output_tokens", "user", "previous_response_id"} {
		if _, ok := body[field]; ok {
			t.Fatalf("field %s must not reach the plan route: %v", field, body)
		}
	}
	for _, raw := range body["input"].([]any) {
		if raw.(map[string]any)["role"] == "system" {
			t.Fatalf("system-role item reached the plan route: %v", body["input"])
		}
	}
	assertInterventionLogged(t, rig.logs.Bytes(), "subscription_system_as_developer")
	assertInterventionLogged(t, rig.logs.Bytes(), "subscription_field_dropped:max_output_tokens")
	if got := rig.responseCredentials(); len(got) != 1 || got[0] != "subscription" {
		t.Fatalf("credential recorded: %v", got)
	}
	if served := rig.subs.Served()["worker"]; served["subscription"] != 1 || served["api_key"] != 0 {
		t.Fatalf("served: %v", served)
	}
	if strings.Contains(rig.logs.String(), "canary") || strings.Contains(w.Body.String(), "canary") {
		t.Fatal("a token value reached the log or the agent")
	}
}

func TestPlanLimitFallsBackToTheKeyAndRests(t *testing.T) {
	rig := newPlanRig(t, planRef, keyRef)
	rig.signIn("plan-access-0-canary", time.Hour)
	rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"code":"subscription_sharing_usage_limit_exceeded","message":"limit"}}`)
	}
	w := rig.chat("")
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "from the key") {
		t.Fatalf("reply: %d %s", w.Code, w.Body.String())
	}
	if rig.keyAuth[0] != "Bearer sk-api-key" {
		t.Fatalf("the key route must use the key: %q", rig.keyAuth[0])
	}
	assertFailoverLogged(t, rig.logs.Bytes(), map[string]string{"from_provider": "chatgpt", "to_provider": "openai", "reason": "subscription_limit_reached"})

	// The plan rests: the next request does not touch it.
	if w := rig.chat(""); w.Code != http.StatusOK {
		t.Fatalf("second reply: %d", w.Code)
	}
	if rig.planHits != 1 || rig.keyHits != 2 {
		t.Fatalf("hits: plan %d key %d", rig.planHits, rig.keyHits)
	}
	if got := rig.responseCredentials(); len(got) != 2 || got[0] != "api_key" || got[1] != "api_key" {
		t.Fatalf("credentials recorded: %v", got)
	}
	// Only requests that were served count, never the failed attempt.
	if served := rig.subs.Served()["worker"]; served["subscription"] != 0 || served["api_key"] != 2 {
		t.Fatalf("served: %v", served)
	}
	if rig.reg.CooldownUntil("chatgpt").IsZero() {
		t.Fatal("the plan route must be resting")
	}
	if state := rig.reg.All()["chatgpt"].Keys[0].State; state == provider.KeyStateDead {
		t.Fatal("a sign-in is never marked dead")
	}
}

func TestPlanOnlyChainFailsWithItsOwnReason(t *testing.T) {
	cases := map[string]struct {
		prepare func(rig *planRig)
		status  int
		code    string
		hits    int
	}{
		"not signed in": {prepare: func(rig *planRig) {}, status: http.StatusServiceUnavailable, code: "subscription_signin_required"},
		"sign-in expired": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", -time.Minute)
			rig.tokenFail = "invalid_grant"
		}, status: http.StatusServiceUnavailable, code: "subscription_signin_required"},
		"limit reached": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
				w.WriteHeader(http.StatusTooManyRequests)
				_, _ = io.WriteString(w, `{"error":{"code":"subscription_sharing_usage_limit_exceeded"}}`)
			}
		}, status: http.StatusTooManyRequests, code: "subscription_limit_reached", hits: 1},
		"limit reported in the stream": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, planSSE(`{"type":"response.created"}`, `{"type":"response.failed","response":{"error":{"code":"subscription_sharing_usage_limit_exceeded"}}}`))
			}
		}, status: http.StatusTooManyRequests, code: "subscription_limit_reached", hits: 1},
		"not eligible": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
				w.WriteHeader(http.StatusForbidden)
				_, _ = io.WriteString(w, `{"error":{"code":"subscription_sharing_user_not_eligible"}}`)
			}
		}, status: http.StatusForbidden, code: "subscription_not_eligible", hits: 1},
		"unsupported capability": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = io.WriteString(w, `{"error":{"code":"subscription_sharing_unsupported_capability","param":"tools"}}`)
			}
		}, status: http.StatusBadRequest, code: "subscription_unsupported_request", hits: 1},
		"stream cut before completion": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, planSSE(`{"type":"response.created"}`, `{"type":"response.output_text.delta","delta":"partial"}`))
			}
		}, status: http.StatusServiceUnavailable, code: "subscription_unavailable", hits: 1},
		"incomplete stream": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = io.WriteString(w, planSSE(`{"type":"response.incomplete","response":{"status":"incomplete"}}`))
			}
		}, status: http.StatusServiceUnavailable, code: "subscription_unavailable", hits: 1},
		"success that is not a stream": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
				w.Header().Set("Content-Type", "application/json")
				_, _ = fmt.Fprintf(w, completedResponse, "unproven")
			}
		}, status: http.StatusServiceUnavailable, code: "subscription_unavailable", hits: 1},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			rig := newPlanRig(t, planRef)
			tc.prepare(rig)
			w := rig.chat("")
			if w.Code != tc.status || errorCode(t, w) != tc.code {
				t.Fatalf("got %d %q, want %d %q: %s", w.Code, errorCode(t, w), tc.status, tc.code, w.Body.String())
			}
			if strings.Contains(w.Body.String(), "partial") || strings.Contains(w.Body.String(), "unproven") {
				t.Fatalf("an unfinished reply reached the agent: %s", w.Body.String())
			}
			// A plan-only chain never touches the API key.
			if rig.keyHits != 0 || rig.planHits != tc.hits {
				t.Fatalf("hits: plan %d (want %d) key %d", rig.planHits, tc.hits, rig.keyHits)
			}
			if got := rig.subs.Failures()["worker"]; got != tc.code {
				t.Fatalf("last failure %q, want %q", got, tc.code)
			}
			if len(rig.subs.Served()["worker"]) != 0 {
				t.Fatalf("nothing was served: %v", rig.subs.Served())
			}
		})
	}
}

// Eligibility and capability refusals do not move the request to another
// billing path, even when the chain declares one.
func TestPlanRefusalsDoNotChangeBilling(t *testing.T) {
	for name, reply := range map[string]struct {
		status int
		body   string
	}{
		"not eligible":           {http.StatusForbidden, `{"error":{"code":"subscription_sharing_user_not_eligible"}}`},
		"permission context":     {http.StatusForbidden, `{"error":{"code":"chatpass_v2_scope_not_authorized"}}`},
		"unsupported capability": {http.StatusBadRequest, `{"error":{"code":"subscription_sharing_unsupported_capability"}}`},
	} {
		t.Run(name, func(t *testing.T) {
			rig := newPlanRig(t, planRef, keyRef)
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
				w.WriteHeader(reply.status)
				_, _ = io.WriteString(w, reply.body)
			}
			w := rig.chat("")
			if w.Code != reply.status || rig.keyHits != 0 {
				t.Fatalf("status %d, key hits %d: %s", w.Code, rig.keyHits, w.Body.String())
			}
		})
	}
}

func TestMissingSignInFallsBackToTheKey(t *testing.T) {
	rig := newPlanRig(t, planRef, keyRef)
	w := rig.chat("")
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "from the key") || rig.planHits != 0 {
		t.Fatalf("reply: %d %s (plan hits %d)", w.Code, w.Body.String(), rig.planHits)
	}
	assertFailoverLogged(t, rig.logs.Bytes(), map[string]string{"from_provider": "chatgpt", "to_provider": "openai", "reason": "subscription_signin_required"})
	if got := rig.subs.Failures()["worker"]; got != "subscription_signin_required" {
		t.Fatalf("last failure: %q", got)
	}
}

// An ordinary rate limit is not reported as a spent plan.
func TestOrdinaryRateLimitIsNotAPlanLimit(t *testing.T) {
	rig := newPlanRig(t, planRef)
	rig.signIn("plan-access-0-canary", time.Hour)
	rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = io.WriteString(w, `{"error":{"code":"rate_limit_exceeded"}}`)
	}
	w := rig.chat("")
	if code := errorCode(t, w); code == "subscription_limit_reached" || w.Code == http.StatusOK {
		t.Fatalf("got %d %q", w.Code, code)
	}
}

// A token refreshed while the request is in flight is used for the retry and
// is redacted wherever the provider echoes it.
func TestRejectedTokenIsRefreshedOnceAndNeverEchoed(t *testing.T) {
	rig := newPlanRig(t, planRef)
	rig.signIn("plan-access-0-canary", time.Hour)
	rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
		if hit == 1 {
			w.WriteHeader(http.StatusUnauthorized)
			_, _ = io.WriteString(w, `{"error":{"code":"subscription_sharing_invalid_user"}}`)
			return
		}
		echoed := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, planCompleted("your token is "+echoed))
	}
	dir := t.TempDir()
	journal, err := tracejournal.Open(tracejournal.Config{HistoryDir: dir, OwnerUID: -1, OwnerGID: -1,
		MaxFieldBytes: tracejournal.DefaultMaxFieldBytes, MaxSegmentBytes: 64 << 20, Version: "test", Revision: "test", Pod: "test"})
	if err != nil {
		t.Fatal(err)
	}
	defer journal.Close()
	rig.h.journal = journal

	w := rig.chat("")
	if w.Code != http.StatusOK {
		t.Fatalf("reply: %d %s", w.Code, w.Body.String())
	}
	if rig.planHits != 2 || rig.tokenHits != 1 {
		t.Fatalf("plan hits %d, refreshes %d", rig.planHits, rig.tokenHits)
	}
	if rig.planAuth[1] != "Bearer plan-access-1-canary" {
		t.Fatalf("retry auth: %q", rig.planAuth[1])
	}
	if strings.Contains(w.Body.String(), "canary") {
		t.Fatalf("the agent saw a token: %s", w.Body.String())
	}
	if strings.Contains(rig.logs.String(), "canary") {
		t.Fatal("the log holds a token")
	}
	assertNoCanaryUnder(t, dir)

	// A second rejection after a fresh token is not proof of revocation.
	rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
		w.WriteHeader(http.StatusUnauthorized)
	}
	if w := rig.chat(""); errorCode(t, w) != "subscription_unavailable" {
		t.Fatalf("got %d %q", w.Code, errorCode(t, w))
	}
	if st := rig.subs.Status()[0]; st.State != subscription.StatusSignedIn {
		t.Fatalf("sign-in must survive an unconfirmed rejection: %+v", st)
	}
}

func TestPlanProviderTakesNoKeysAndIsNotPersisted(t *testing.T) {
	dir := t.TempDir()
	reg := provider.NewRegistry(dir)
	reg.RegisterPlan("chatgpt", "https://api.openai.com/v1")
	if _, err := reg.AddRuntimeKey("chatgpt", "x", "sk-stray"); err == nil {
		t.Fatal("a plan route must refuse an API key")
	}
	if err := reg.DeleteKey("chatgpt", provider.SubscriptionKeyID); err == nil {
		t.Fatal("a plan route's entry is not deletable")
	}
	if err := reg.SaveToFile(); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(dir + "/providers.json")
	if strings.Contains(string(data), "chatgpt") {
		t.Fatalf("plan route persisted: %s", data)
	}
}

func TestPlanRouteRefusesContinuationByID(t *testing.T) {
	if _, err := shapePlanResponsesRequest(map[string]any{"previous_response_id": "resp_1"}); err == nil {
		t.Fatal("previous_response_id must be refused")
	}
}

func assertNoCanaryUnder(t *testing.T, dir string) {
	t.Helper()
	files := 0
	_ = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err != nil || d.IsDir() {
			return nil
		}
		files++
		data, _ := os.ReadFile(path)
		if bytes.Contains(data, []byte("canary")) {
			t.Fatalf("%s holds a token value", path)
		}
		return nil
	})
	if files == 0 {
		t.Fatal("the journal wrote nothing; the check proves nothing")
	}
	classSeen := false
	_ = filepath.WalkDir(dir, func(path string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			data, _ := os.ReadFile(path)
			classSeen = classSeen || bytes.Contains(data, []byte(`"credential":"subscription"`))
		}
		return nil
	})
	if !classSeen {
		t.Fatal("the journal does not name the credential class of the attempt")
	}
}

func TestPlanResponseHeadersCannotExposeCredentials(t *testing.T) {
	rig := newPlanRig(t, planRef)
	rig.signIn("plan-access-0-canary", time.Hour)
	rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
		w.Header().Set("X-Debug-Echo", r.Header.Get("Authorization"))
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, planCompleted("done"))
	}
	w := rig.chat("")
	if w.Code != http.StatusOK {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	for name, values := range w.Header() {
		if strings.Contains(name+strings.Join(values, " "), "canary") {
			t.Fatalf("credential leaked in response header %q", name)
		}
	}
}

// Exercise the separate managed-tool dispatch loop; native-tool tests do not
// cover its retry, SSE completion, or credential scrubbing branches.
func TestManagedPlanDispatchSecurity(t *testing.T) {
	for _, scenario := range []string{"completed", "limit", "ineligible", "cut", "refresh"} {
		t.Run(scenario, func(t *testing.T) {
			rig := newPlanRig(t, planRef, keyRef)
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
				switch scenario {
				case "limit":
					w.WriteHeader(429)
					_, _ = io.WriteString(w, `{"error":{"code":"subscription_sharing_usage_limit_exceeded"}}`)
					return
				case "ineligible":
					w.WriteHeader(403)
					return
				case "refresh":
					if hit == 1 {
						w.WriteHeader(401)
						return
					}
				}
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("X-Echo", r.Header.Get("Authorization"))
				if scenario == "cut" {
					_, _ = io.WriteString(w, planSSE(`{"type":"response.output_text.delta","delta":"partial"}`))
					return
				}
				_, _ = io.WriteString(w, planCompleted(strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")))
			}
			var candidates []dispatchCandidate
			for _, ref := range []string{planRef, keyRef} {
				candidate, err := rig.h.openAICandidateFromRef(ref)
				if err != nil {
					t.Fatal(err)
				}
				candidates = append(candidates, candidate)
			}
			ctx := context.WithValue(context.Background(), dataPolicyKey{}, *testPermissiveDataPolicy())
			req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx)
			payload := map[string]any{"model": planRef, "messages": []any{map[string]any{"role": "user", "content": "hello"}}}
			resp, status, msg, err := rig.h.dispatchCandidatesJSON(ctx, req, "worker", planRef, payload, candidates, nil)
			if scenario == "ineligible" {
				if err == nil || status != 403 || rig.keyHits != 0 {
					t.Fatalf("eligibility refusal changed billing: %d %s %v, key hits %d", status, msg, err, rig.keyHits)
				}
				return
			}
			if err != nil || resp == nil {
				t.Fatalf("dispatch: %d %s %v", status, msg, err)
			}
			if strings.Contains(string(resp.Body)+fmt.Sprint(resp.Header), "canary") {
				t.Fatal("managed dispatch exposed a credential")
			}
			expectedKeyHits := 0
			if scenario == "limit" || scenario == "cut" {
				expectedKeyHits = 1
			}
			if rig.keyHits != expectedKeyHits {
				t.Fatalf("key calls: %d want %d", rig.keyHits, expectedKeyHits)
			}
			if scenario == "refresh" && (rig.tokenHits != 1 || rig.planHits != 2) {
				t.Fatalf("refresh %d plan %d", rig.tokenHits, rig.planHits)
			}
		})
	}
}

// The plan route takes function tools only inside a namespace, and replays
// earlier calls under it. The agent still sees plain tool calls.
func TestPlanRouteNamespacesFunctionTools(t *testing.T) {
	rig := newPlanRig(t, planRef)
	rig.signIn("plan-access-0-canary", time.Hour)
	rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, planSSE(`{"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","model":"gpt-5.6-terra","output":[{"type":"function_call","id":"fc_1","call_id":"call_2","name":"ping","namespace":"agent_tools","arguments":"{}"}],"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`))
	}
	w := rig.chat(`{"model":"chatgpt/gpt-5.6-terra","messages":[
		{"role":"user","content":"go"},
		{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"ping","arguments":"{}"}}]},
		{"role":"tool","tool_call_id":"call_1","content":"pong"}],
		"tools":[{"type":"function","function":{"name":"ping","description":"p","parameters":{"type":"object","properties":{}}}}]}`)
	if w.Code != http.StatusOK {
		t.Fatalf("reply: %d %s", w.Code, w.Body.String())
	}
	tools := rig.planBody[0]["tools"].([]any)
	ns := tools[0].(map[string]any)
	if len(tools) != 1 || ns["type"] != "namespace" || ns["name"] != "agent_tools" {
		t.Fatalf("tools must be one namespace: %v", tools)
	}
	member := ns["tools"].([]any)[0].(map[string]any)
	if member["type"] != "function" || member["name"] != "ping" {
		t.Fatalf("namespace member: %v", member)
	}
	replayed := false
	for _, raw := range rig.planBody[0]["input"].([]any) {
		item := raw.(map[string]any)
		if item["type"] == "function_call" {
			replayed = item["namespace"] == "agent_tools" && item["name"] == "ping"
		}
	}
	if !replayed {
		t.Fatalf("earlier call not replayed under the namespace: %v", rig.planBody[0]["input"])
	}
	var reply struct {
		Choices []struct {
			Message struct {
				ToolCalls []struct {
					Function struct {
						Name string `json:"name"`
					} `json:"function"`
				} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &reply); err != nil || len(reply.Choices) != 1 || len(reply.Choices[0].Message.ToolCalls) != 1 || reply.Choices[0].Message.ToolCalls[0].Function.Name != "ping" {
		t.Fatalf("agent-facing tool call: %s", w.Body.String())
	}
	assertInterventionLogged(t, rig.logs.Bytes(), "subscription_tools_namespaced")
}

func TestPlanRouteRefusesHostedTools(t *testing.T) {
	body := map[string]any{"tools": []any{map[string]any{"type": "web_search"}}}
	if _, err := shapePlanResponsesRequest(body); err == nil {
		t.Fatal("a hosted tool must be refused on a plan route")
	}
}

// A namespace is judged by its members, so grouping cannot hide a hosted
// tool from the data policy.
func TestDataPolicySeesInsideNamespaces(t *testing.T) {
	fn := map[string]any{"type": "function", "name": "ping"}
	ns := func(members ...any) map[string]any {
		return map[string]any{"tools": []any{map[string]any{"type": "namespace", "name": "agent_tools", "tools": members}}}
	}
	if hasUnscopedFeature(ns(fn)) {
		t.Fatal("a namespace of function tools is not a hosted feature")
	}
	if !hasUnscopedFeature(ns(fn, map[string]any{"type": "web_search"})) {
		t.Fatal("a hosted tool inside a namespace must be seen")
	}
	if !hasUnscopedFeature(ns(map[string]any{"type": "namespace", "tools": []any{fn}})) {
		t.Fatal("a nested namespace must not be accepted")
	}
	if !hasUnscopedFeature(map[string]any{"tools": []any{map[string]any{"type": "namespace"}}}) {
		t.Fatal("a namespace without members must not be accepted")
	}
}
