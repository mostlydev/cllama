package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func (rig *planRig) planFailures() []map[string]any {
	var out []map[string]any
	for _, entry := range parseLogEntries(rig.t, rig.logs.Bytes()) {
		if entry["type"] == "plan_failure" {
			out = append(out, entry)
		}
	}
	return out
}

func sseReply(body string) func(http.ResponseWriter, *http.Request, int) {
	return func(w http.ResponseWriter, r *http.Request, hit int) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, body)
	}
}

func jsonReply(status int, body string) func(http.ResponseWriter, *http.Request, int) {
	return func(w http.ResponseWriter, r *http.Request, hit int) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

// Every way a plan route can fail to serve is recorded with the branch that
// decided it and what the vendor answered.
func TestPlanFailureNamesItsBranch(t *testing.T) {
	type want map[string]any
	cases := map[string]struct {
		prepare func(rig *planRig)
		want    []want
	}{
		"not signed in": {prepare: func(rig *planRig) {},
			want: []want{{"branch": "credential_signin_required", "stage": "token", "reason": "subscription_signin_required"}}},
		"refresh refused, terminal": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", -time.Minute)
			rig.tokenFail = "invalid_grant"
		}, want: []want{{"branch": "credential_refresh_http", "stage": "token", "reason": "subscription_signin_required", "status_code": 400.0, "upstream_error_code": "invalid_grant"}}},
		"refresh refused, not terminal": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", -time.Minute)
			rig.tokenFail = "temporarily_unavailable"
		}, want: []want{{"branch": "credential_refresh_http", "stage": "token", "reason": "subscription_unavailable", "status_code": 400.0, "upstream_error_code": "temporarily_unavailable"}}},
		"plan limit": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = jsonReply(429, `{"error":{"code":"subscription_sharing_usage_limit_exceeded","type":"rate_limit_error"}}`)
		}, want: []want{{"branch": "http_429_limit", "stage": "inference", "reason": "subscription_limit_reached", "status_code": 429.0, "upstream_error_code": "subscription_sharing_usage_limit_exceeded", "upstream_error_type": "rate_limit_error"}}},
		"ordinary rate limit": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = jsonReply(429, `{"error":{"code":"rate_limit_exceeded"}}`)
		}, want: []want{{"branch": "http_429_rate", "reason": "rate_limit", "status_code": 429.0, "upstream_error_code": "rate_limit_exceeded"}}},
		"401 after a refresh": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = jsonReply(401, `{"error":{"code":"invalid_token","type":"authentication_error"}}`)
		}, want: []want{{"body_shape": "error", "branch": "http_401_after_refresh", "reason": "subscription_unavailable", "status_code": 401.0, "upstream_error_code": "invalid_token", "upstream_error_type": "authentication_error"}}},
		"401 and the refresh fails": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.tokenFail = "temporarily_unavailable"
			rig.planReply = jsonReply(401, `{"error":{"code":"invalid_token"}}`)
		}, want: []want{
			{"branch": "credential_refresh_http", "stage": "token", "status_code": 400.0, "upstream_error_code": "temporarily_unavailable"},
			{"branch": "http_401_refresh_failed", "stage": "inference", "status_code": 401.0, "upstream_error_code": "invalid_token"},
		}},
		"not eligible": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = jsonReply(403, `{"error":{"code":"subscription_sharing_user_not_eligible"}}`)
		}, want: []want{{"branch": "http_403", "reason": "subscription_not_eligible", "status_code": 403.0, "upstream_error_code": "subscription_sharing_user_not_eligible"}}},
		"unsupported capability": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = jsonReply(400, `{"error":{"code":"subscription_sharing_unsupported_capability"}}`)
		}, want: []want{{"branch": "http_400_unsupported", "reason": "subscription_unsupported_request", "status_code": 400.0}}},
		"an error status the plan rules do not classify": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = jsonReply(400, `{"error":{"code":"unsupported_parameter","type":"invalid_request_error"}}`)
		}, want: []want{{"branch": "http_other", "reason": "http_400", "status_code": 400.0, "upstream_error_code": "unsupported_parameter", "upstream_error_type": "invalid_request_error"}}},
		"refusal shaped as a detail body": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = jsonReply(403, `{"detail":"plan-access-0-canary is not admitted"}`)
		}, want: []want{{"branch": "http_403", "status_code": 403.0, "body_shape": "detail"}}},
		"success that is not an event stream": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = jsonReply(200, `{"id":"resp_1"}`)
		}, want: []want{{"branch": "not_event_stream", "reason": "subscription_unavailable", "status_code": 200.0, "content_type": "application/json"}}},
		"stream fails with an unmapped code": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = sseReply(planSSE(`{"type":"response.created"}`, `{"type":"response.failed","response":{"error":{"code":"server_error","type":"server_error"}}}`))
		}, want: []want{{"branch": "stream_terminal_event", "reason": "subscription_unavailable", "status_code": 200.0, "stream_event": "response.failed", "stream_events": 2.0, "upstream_error_code": "server_error", "upstream_error_type": "server_error"}}},
		"stream ends incomplete": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = sseReply(planSSE(`{"type":"response.incomplete","response":{"incomplete_details":{"reason":"max_output_tokens"}}}`))
		}, want: []want{{"branch": "stream_terminal_event", "stream_event": "response.incomplete", "upstream_error_type": "max_output_tokens"}}},
		"stream error event": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = sseReply(planSSE(`{"type":"error","code":"model_not_found"}`))
		}, want: []want{{"branch": "stream_terminal_event", "stream_event": "error", "upstream_error_code": "model_not_found"}}},
		"stream ends before completion": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = sseReply(planSSE(`{"type":"response.created"}`, `{"type":"response.output_text.delta","delta":"x"}`))
		}, want: []want{{"branch": "stream_ended_early", "reason": "subscription_unavailable", "stream_last_event": "response.output_text.delta", "stream_events": 2.0}}},
		"stream with no readable events": {prepare: func(rig *planRig) {
			rig.signIn("plan-access-0-canary", time.Hour)
			rig.planReply = sseReply("event: response.completed\n\n")
		}, want: []want{{"branch": "stream_ended_early", "status_code": 200.0}}},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			rig := newPlanRig(t, planRef)
			tc.prepare(rig)
			rig.chat("")
			got := rig.planFailures()
			if len(got) != len(tc.want) {
				t.Fatalf("plan_failure events = %v, want %d", got, len(tc.want))
			}
			for i, w := range tc.want {
				if got[i]["provider"] != "chatgpt" {
					t.Fatalf("provider = %v", got[i]["provider"])
				}
				for field, value := range w {
					if got[i][field] != value {
						t.Fatalf("event %d %s = %v, want %v (event %v)", i, field, got[i][field], value, got[i])
					}
				}
			}
		})
	}
}

// A vendor reply is untrusted. Whatever it puts in a code, type, event name
// or content type, none of it reaches the log unless it is a listed value.
func TestPlanFailureLogsNoVendorText(t *testing.T) {
	hostile := []string{
		"plan-access-0-canary",             // the sign-in's own token
		"plan-refresh-0-canary",            // its refresh token
		"abcdefghijklmnopqrstuvwxyzabcdef", // an all-letter token
		"someone@example.com",              // an account identifier
		"accountname",                      // a bare account name
		"server_error\r\nforged: line",     // a log-forging value
		strings.Repeat("a", 5000),          // an oversized value
		"Server_Error",                     // a near miss of a listed value
	}
	replies := map[string]func(v string) func(http.ResponseWriter, *http.Request, int){
		"http code and type": func(v string) func(http.ResponseWriter, *http.Request, int) {
			return func(w http.ResponseWriter, r *http.Request, hit int) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(500)
				_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": v, "type": v, "message": v, "param": v}})
			}
		},
		"401 code": func(v string) func(http.ResponseWriter, *http.Request, int) {
			return func(w http.ResponseWriter, r *http.Request, hit int) {
				w.WriteHeader(401)
				_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": v, "type": v}})
			}
		},
		"content type": func(v string) func(http.ResponseWriter, *http.Request, int) {
			return func(w http.ResponseWriter, r *http.Request, hit int) {
				w.Header()["Content-Type"] = []string{strings.NewReplacer("\r", "", "\n", "").Replace(v)}
				w.WriteHeader(200)
			}
		},
		"terminal event": func(v string) func(http.ResponseWriter, *http.Request, int) {
			return func(w http.ResponseWriter, r *http.Request, hit int) {
				w.Header().Set("Content-Type", "text/event-stream")
				event, _ := json.Marshal(map[string]any{"type": "response.failed", "code": v, "response": map[string]any{
					"error": map[string]any{"code": v, "type": v, "message": v}, "incomplete_details": map[string]any{"reason": v}}})
				_, _ = io.WriteString(w, planSSE(string(event)))
			}
		},
		"event name": func(v string) func(http.ResponseWriter, *http.Request, int) {
			return func(w http.ResponseWriter, r *http.Request, hit int) {
				w.Header().Set("Content-Type", "text/event-stream")
				event, _ := json.Marshal(map[string]any{"type": v, "delta": v})
				_, _ = io.WriteString(w, planSSE(string(event)))
			}
		},
	}
	for name, reply := range replies {
		for _, value := range hostile {
			t.Run(name, func(t *testing.T) {
				rig := newPlanRig(t, planRef)
				rig.signIn("plan-access-0-canary", time.Hour)
				rig.planReply = reply(value)
				rig.chat("")
				if len(rig.planFailures()) == 0 {
					t.Fatal("no plan_failure event")
				}
				logs := rig.logs.String()
				needle := value
				if len(needle) > 64 {
					needle = needle[:64]
				}
				for _, part := range strings.Split(needle, "\r\n") {
					// "server_error" alone is a listed value and may appear.
					if part == "server_error" {
						continue
					}
					if strings.Contains(logs, part) {
						t.Fatalf("vendor value %q reached the log:\n%s", part, logs)
					}
				}
				if strings.Contains(logs, "forged") {
					t.Fatalf("forged text reached the log:\n%s", logs)
				}
			})
		}
	}
}

// A token minted by this very request never reaches the log either.
func TestPlanFailureDoesNotLogARefreshedToken(t *testing.T) {
	rig := newPlanRig(t, planRef)
	rig.signIn("plan-access-0-canary", time.Hour)
	rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
		// Reflect whatever bearer arrived, including the one minted by the
		// refresh that follows the first 401.
		token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(401)
		_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"code": token, "type": token}})
	}
	rig.chat("")
	if rig.tokenHits != 1 {
		t.Fatalf("refreshes = %d, want 1", rig.tokenHits)
	}
	events := rig.planFailures()
	if len(events) != 1 || events[0]["branch"] != "http_401_after_refresh" || events[0]["upstream_error_code"] != "unlisted" {
		t.Fatalf("events = %v", events)
	}
	if logs := rig.logs.String(); strings.Contains(logs, "canary") {
		t.Fatalf("a token reached the log:\n%s", logs)
	}
}

// The managed-tool path records the same diagnostics as the direct path.
func TestManagedPlanFailureNamesItsBranch(t *testing.T) {
	cases := map[string]struct {
		reply  func(http.ResponseWriter, *http.Request, int)
		expire bool
		branch string
	}{
		"malformed completed": {reply: sseReply(planSSE(`{"type":"response.completed","response":{"object":"response","status":"completed","output":"invalid"}}`)), branch: "stream_invalid_completed"},
		"response limit":      {reply: sseReply(planCompleted(strings.Repeat("x", maxManagedLLMResponseBytes))), branch: "response_too_large"},
		"credential":          {expire: true, branch: "credential_refresh_http"},
		"http":                {reply: jsonReply(401, `{"error":{"code":"invalid_token"}}`), branch: "http_401_after_refresh"},
		"not a stream":        {reply: jsonReply(200, `{}`), branch: "not_event_stream"},
		"terminal event":      {reply: sseReply(planSSE(`{"type":"response.failed","response":{"error":{"code":"server_error"}}}`)), branch: "stream_terminal_event"},
		"cut":                 {reply: sseReply(planSSE(`{"type":"response.created"}`)), branch: "stream_ended_early"},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			rig := newPlanRig(t, planRef, keyRef)
			if tc.expire {
				rig.signIn("plan-access-0-canary", -time.Minute)
				rig.tokenFail = "temporarily_unavailable"
			} else {
				rig.signIn("plan-access-0-canary", time.Hour)
			}
			rig.planReply = tc.reply
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
			_, _, _, _ = rig.h.dispatchCandidatesJSON(ctx, req, "worker", planRef, payload, candidates, nil)
			events := rig.planFailures()
			if len(events) != 1 || events[0]["branch"] != tc.branch {
				t.Fatalf("events = %v, want branch %s", events, tc.branch)
			}
		})
	}
}

// Diagnostic-only fields must not turn a refusal into billable API fallback.
func TestPlanDiagnosticsMalformedAdvisoryCannotChangeRouting(t *testing.T) {
	events := map[string]string{
		"incomplete_container": `{"type":"response.failed","response":{"error":{"code":"subscription_sharing_unsupported_capability"},"incomplete_details":42}}`,
		"outer_type":           `{"type":"response.failed","error":{"code":"subscription_sharing_unsupported_capability","type":{"account":"private"}}}`,
		"nested_type":          `{"type":"response.failed","response":{"error":{"code":"subscription_sharing_unsupported_capability","type":42}}}`,
		"incomplete_reason":    `{"type":"response.failed","response":{"error":{"code":"subscription_sharing_unsupported_capability"},"incomplete_details":{"reason":["private"]}}}`,
	}
	for name, event := range events {
		for _, managed := range []bool{false, true} {
			mode := "direct"
			if managed {
				mode = "managed"
			}
			t.Run(name+"/"+mode, func(t *testing.T) {
				rig := newPlanRig(t, planRef, keyRef)
				rig.signIn("plan-access-0-canary", time.Hour)
				rig.planReply = sseReply(planSSE(event))
				if !managed {
					w := rig.chat("")
					if w.Code != http.StatusBadRequest || errorCode(t, w) != "subscription_unsupported_request" {
						t.Fatalf("refusal changed: status=%d body=%s", w.Code, w.Body.String())
					}
				} else {
					var candidates []dispatchCandidate
					for _, ref := range []string{planRef, keyRef} {
						c, err := rig.h.openAICandidateFromRef(ref)
						if err != nil {
							t.Fatal(err)
						}
						candidates = append(candidates, c)
					}
					ctx := context.WithValue(context.Background(), dataPolicyKey{}, *testPermissiveDataPolicy())
					req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil).WithContext(ctx)
					payload := map[string]any{"model": planRef, "messages": []any{map[string]any{"role": "user", "content": "hello"}}}
					_, status, _, _ := rig.h.dispatchCandidatesJSON(ctx, req, "worker", planRef, payload, candidates, nil)
					if status != http.StatusBadRequest {
						t.Fatalf("refusal changed: status=%d", status)
					}
				}
				if rig.keyHits != 0 {
					t.Fatalf("diagnostic field triggered %d API calls", rig.keyHits)
				}
			})
		}
	}
}

func bareReply(status int, body string) func(http.ResponseWriter, *http.Request, int) {
	return func(w http.ResponseWriter, r *http.Request, hit int) {
		// A nil entry stops net/http from sniffing a type into the header.
		w.Header()["Content-Type"] = nil
		w.WriteHeader(status)
		_, _ = io.WriteString(w, body)
	}
}

// The vendor's live reply carries no Content-Type. It is read as a stream,
// and only a completed response makes it a success.
func TestPlanReplyWithoutContentType(t *testing.T) {
	cases := map[string]struct {
		body   string
		served bool
		branch string
		events float64
	}{
		"invalid completed null":           {body: planSSE(`{"type":"response.completed","response":null}`), branch: "stream_invalid_completed", events: 1},
		"invalid completed empty_object":   {body: planSSE(`{"type":"response.completed","response":{}}`), branch: "stream_invalid_completed", events: 1},
		"invalid completed array":          {body: planSSE(`{"type":"response.completed","response":[]}`), branch: "stream_invalid_completed", events: 1},
		"invalid completed number":         {body: planSSE(`{"type":"response.completed","response":42}`), branch: "stream_invalid_completed", events: 1},
		"invalid completed missing_output": {body: planSSE(`{"type":"response.completed","response":{"object":"response","status":"completed"}}`), branch: "stream_invalid_completed", events: 1},
		"invalid completed wrong_status":   {body: planSSE(`{"type":"response.completed","response":{"object":"response","status":"failed","output":[]}}`), branch: "stream_invalid_completed", events: 1},
		"invalid completed null_output":    {body: planSSE(`{"type":"response.completed","response":{"object":"response","status":"completed","output":null}}`), branch: "stream_invalid_completed", events: 1},
		"invalid completed wrong_object":   {body: planSSE(`{"type":"response.completed","response":{"object":"other","status":"completed","output":[]}}`), branch: "stream_invalid_completed", events: 1},
		"completed stream":                 {body: planCompleted("from the plan"), served: true},
		"empty body":                       {body: "", branch: "stream_ended_early"},
		"json object":                      {body: fmt.Sprintf(completedResponse, "not a stream"), branch: "stream_ended_early"},
		"html page":                        {body: "<html><body>data: sign in</body></html>\n", branch: "stream_ended_early"},
		"cut stream":                       {body: planSSE(`{"type":"response.created"}`), branch: "stream_ended_early", events: 1},
		"failed stream":                    {body: planSSE(`{"type":"response.failed","response":{"error":{"code":"server_error"}}}`), branch: "stream_terminal_event", events: 1},
	}
	for name, tc := range cases {
		for _, path := range []string{"direct", "managed"} {
			t.Run(name+"/"+path, func(t *testing.T) {
				rig := newPlanRig(t, planRef, keyRef)
				rig.signIn("plan-access-0-canary", time.Hour)
				rig.planReply = bareReply(200, tc.body)
				if path == "direct" {
					if w := rig.chat(""); w.Code != 200 {
						t.Fatalf("status %d: %s", w.Code, w.Body)
					}
				} else {
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
					if resp, status, msg, err := rig.h.dispatchCandidatesJSON(ctx, req, "worker", planRef, payload, candidates, nil); err != nil || resp == nil {
						t.Fatalf("dispatch: %d %s %v", status, msg, err)
					}
				}
				served := rig.subs.Served()["worker"]
				events := rig.planFailures()
				if tc.served {
					if rig.keyHits != 0 || served["subscription"] != 1 || served["api_key"] != 0 || len(events) != 0 {
						t.Fatalf("plan did not serve: key hits %d, served %v, failures %v", rig.keyHits, served, events)
					}
					return
				}
				if rig.keyHits != 1 || served["subscription"] != 0 || served["api_key"] != 1 {
					t.Fatalf("unproven reply counted as plan-served: key hits %d, served %v", rig.keyHits, served)
				}
				if len(events) != 1 || events[0]["branch"] != tc.branch || events[0]["content_type"] != "none" || events[0]["stream_bytes"] != float64(len(tc.body)) {
					t.Fatalf("events = %v, want branch %s over %d bytes", events, tc.branch, len(tc.body))
				}
				if tc.events > 0 && events[0]["stream_events"] != tc.events {
					t.Fatalf("stream_events = %v", events[0]["stream_events"])
				}
			})
		}
	}
}

// A 2xx that declares another type is refused without being read.
func TestPlanReplyDeclaringAnotherTypeIsRefusedUnread(t *testing.T) {
	rig := newPlanRig(t, planRef)
	rig.signIn("plan-access-0-canary", time.Hour)
	rig.planReply = func(w http.ResponseWriter, r *http.Request, hit int) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, planCompleted("a stream under the wrong label"))
	}
	if w := rig.chat(""); w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d", w.Code)
	}
	events := rig.planFailures()
	if len(events) != 1 || events[0]["branch"] != "not_event_stream" || events[0]["content_type"] != "application/json" {
		t.Fatalf("events = %v", events)
	}
	if _, ok := events[0]["stream_bytes"]; ok {
		t.Fatalf("an unread reply recorded bytes: %v", events[0])
	}
}
