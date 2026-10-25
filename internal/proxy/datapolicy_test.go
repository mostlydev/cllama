package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/provider"
	"github.com/mostlydev/cllama/internal/sessionhistory"
)

// Existing behavioral fixtures opt in explicitly; production has no bypass.
func testPermissiveDataPolicy() *agentctx.DataPolicy {
	return &agentctx.DataPolicy{Version: 1, Regions: []string{"global"}, Training: "allowed", Retention: "provider-default", LocalRetention: "allowed"}
}

func policyForEndpoint(name, endpoint, region string) agentctx.DataPolicy {
	p := agentctx.DefaultDataPolicy()
	p.Attestations = []agentctx.DataAttestation{{Provider: name, Endpoint: endpoint, Guarantees: []string{"processing:" + region, "no-training", "zero-data-retention"}}}
	return p
}

func TestDataPolicyAdapterControlsAndScope(t *testing.T) {
	base := agentctx.DefaultDataPolicy()
	cases := []struct {
		name, endpoint, path, model string
		p                           agentctx.DataPolicy
		body                        string
		denied                      string
	}{
		{"openrouter", "https://us.openrouter.ai/api/v1", "/v1/chat/completions", "openai/gpt-4.1", base, `{"provider":{"only":["openai"],"allow_fallbacks":false}}`, ""},
		{"openrouter", "https://us.openrouter.ai/api/v1", "/v1/chat/completions", "m", base, `{"provider":{"zdr":false}}`, "retention"},
		{"openrouter", "https://us.openrouter.ai/api/v1", "/v1/chat/completions", "m", base, `{"provider":{"data_collection":"allow"}}`, "training"},
		{"openrouter", "https://openrouter.ai/api/v1", "/v1/chat/completions", "m", base, `{}`, "processing_region"},
		{"openrouter", "https://us.openrouter.ai.evil/api/v1", "/v1/chat/completions", "m", base, `{}`, "processing_region"},
		{"openai", "https://us.api.openai.com/v1", "/v1/responses", "gpt-6.1-sol", policyForEndpoint("openai", "", "us"), `{}`, ""},
		{"openai", "https://us.api.openai.com/v1", "/v1/responses", "gpt-6.1-sol", base, `{}`, "training"},
		{"openai", "https://custom.example/v1", "/v1/responses", "gpt-6.1-sol", policyForEndpoint("openai", "", "us"), `{}`, "processing_region"},
		{"openai", "https://custom.example/v1", "/v1/responses", "gpt-6.1-sol", policyForEndpoint("openai", "https://custom.example/v1", "us"), `{}`, ""},
		{"openai", "https://jp.api.openai.com/v1", "/v1/responses", "m", base, `{}`, "processing_region"},
		{"anthropic", "https://api.anthropic.com/v1", "/v1/messages", "claude-sonnet-5-5", policyForEndpoint("anthropic", "", "us"), `{}`, ""},
		{"anthropic", "https://api.anthropic.com/v1", "/v1/messages", "claude-haiku-4-5", base, `{}`, "processing_region"},
		{"anthropic", "https://api.anthropic.com/v1", "/v1/messages", "claude-fable-5", policyForEndpoint("anthropic", "", "us"), `{}`, "retention_model"},
		{"vercel", "https://ai-gateway.vercel.sh/v1", "/v1/chat/completions", "openai/gpt-4.1", policyForEndpoint("vercel", "", "us"), `{}`, ""},
		{"google", "https://generativelanguage.googleapis.com/v1beta/openai", "/v1/chat/completions", "gemini-3.6-pro", base, `{}`, "processing_region"},
		{"ollama", "http://ollama:11434/v1", "/v1/chat/completions", "m", base, `{}`, "processing_region"},
		{"ollama", "http://ollama:11434/v1", "/v1/chat/completions", "m", policyForEndpoint("ollama", "http://ollama:11434/v1", "us"), `{}`, ""},
		{"openai", "https://us.api.openai.com/v1", "/v1/responses", "m", policyForEndpoint("openai", "", "us"), `{"background":true}`, "retention_feature"},
		{"openai", "https://us.api.openai.com/v1", "/v1/responses", "m", policyForEndpoint("openai", "", "us"), `{"store":true}`, "retention_feature"},
		{"openai", "https://us.api.openai.com/v1", "/v1/files", "m", policyForEndpoint("openai", "", "us"), `{}`, "retention_endpoint"},
	}
	for _, tc := range cases {
		t.Run(tc.name+"/"+tc.model+"/"+tc.denied, func(t *testing.T) {
			var body map[string]any
			_ = json.Unmarshal([]byte(tc.body), &body)
			body["model"] = tc.model
			raw, _ := json.Marshal(body)
			got, err := enforceProviderDataPolicy(tc.p, &provider.Provider{Name: tc.name, BaseURL: tc.endpoint}, upstreamEncoding{Path: tc.path, Body: raw})
			if tc.denied != "" {
				if err == nil || !strings.HasSuffix(err.Error(), tc.denied) {
					t.Fatalf("want %s got %v", tc.denied, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			_ = json.Unmarshal(got.Body, &body)
			if tc.name == "openrouter" {
				prefs := body["provider"].(map[string]any)
				if prefs["zdr"] != true || prefs["data_collection"] != "deny" {
					t.Fatal(prefs)
				}
				if _, exists := prefs["only"]; !exists {
					t.Fatal("caller narrowing was lost")
				}
			}
			if tc.name == "anthropic" && body["inference_geo"] != "us" {
				t.Fatal(body)
			}
			if tc.name == "vercel" && body["providerOptions"].(map[string]any)["gateway"].(map[string]any)["zeroDataRetention"] != true {
				t.Fatal(body)
			}
		})
	}
}

func TestDataPolicyCannotMistakeToolSchemaForStoredProviderState(t *testing.T) {
	p := policyForEndpoint("openai", "", "us")
	raw := []byte(`{"model":"gpt-4.1","tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{"store":{"type":"boolean"},"file_id":{"type":"string"}}}}}]}`)
	if _, err := enforceProviderDataPolicy(p, &provider.Provider{Name: "openai", BaseURL: "https://us.api.openai.com/v1"}, upstreamEncoding{Path: "/v1/chat/completions", Body: raw}); err != nil {
		t.Fatal(err)
	}
}

func dataPolicyLoader(p *agentctx.DataPolicy) ContextLoader {
	return func(string) (*agentctx.AgentContext, error) {
		return &agentctx.AgentContext{AgentID: "policy-agent", Metadata: map[string]any{"token": "policy-agent:secret"}, DataPolicy: p}, nil
	}
}

func TestDataPolicyDefaultRefusesBeforeProviderDispatch(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { hits.Add(1) }))
	defer srv.Close()
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{BaseURL: srv.URL + "/v1", APIKey: "provider-key"})
	for _, p := range []*agentctx.DataPolicy{nil, {Version: 2}} {
		h := NewHandler(reg, dataPolicyLoader(p), nil)
		req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"openai/gpt-4.1","messages":[]}`))
		req.Header.Set("Authorization", "Bearer policy-agent:secret")
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if w.Code != 400 || hits.Load() != 0 {
			t.Fatalf("status %d, hits %d", w.Code, hits.Load())
		}
	}
}

func TestDataPolicyLocalDefaultKeepsAccountingWithoutContent(t *testing.T) {
	const private = "PRIVATE-PROMPT-AND-TOOL-RESULT"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[{"message":{"content":"`+private+`"}}],"usage":{"prompt_tokens":3,"completion_tokens":2}}`)
	}))
	defer srv.Close()
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{BaseURL: srv.URL + "/v1", APIKey: "provider-key"})
	p := policyForEndpoint("openai", srv.URL+"/v1", "us")
	root := t.TempDir()
	h := NewHandler(reg, dataPolicyLoader(&p), nil, WithSessionHistory(root))
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"openai/gpt-4.1","messages":[{"role":"user","content":"`+private+`"}]}`))
	req.Header.Set("Authorization", "Bearer policy-agent:secret")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("%d: %s", w.Code, w.Body.String())
	}
	entries, err := sessionhistory.ReadEntries(root, "policy-agent", nil, 10)
	if err != nil || len(entries) != 1 {
		t.Fatalf("entries %v %v", entries, err)
	}
	raw, _ := json.Marshal(entries[0])
	if bytes.Contains(raw, []byte(private)) || len(entries[0].RequestOriginal) != 0 || entries[0].Response.Format != "metadata" || entries[0].Usage.PromptTokens != 3 {
		t.Fatalf("history %s", raw)
	}
	if _, ok := h.snapshots.Get("policy-agent"); ok {
		t.Fatal("snapshot retained content")
	}
}

type policyRoundTrip func(*http.Request) (*http.Response, error)

func (f policyRoundTrip) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestDataPolicyControlReachesActualWireBody(t *testing.T) {
	var got map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewDecoder(r.Body).Decode(&got)
		io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer srv.Close()
	reg := provider.NewRegistry("")
	reg.Set("openrouter", &provider.Provider{BaseURL: "https://us.openrouter.ai/api/v1", APIKey: "provider-key"})
	p := agentctx.DefaultDataPolicy()
	h := NewHandler(reg, dataPolicyLoader(&p), nil)
	target, _ := url.Parse(srv.URL)
	h.client.Transport = policyRoundTrip(func(r *http.Request) (*http.Response, error) {
		copy := r.Clone(r.Context())
		u := *r.URL
		u.Scheme = target.Scheme
		u.Host = target.Host
		copy.URL = &u
		return http.DefaultTransport.RoundTrip(copy)
	})
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"openrouter/openai/gpt-4.1","provider":{"only":["openai"],"allow_fallbacks":false},"messages":[]}`))
	req.Header.Set("Authorization", "Bearer policy-agent:secret")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != 200 {
		t.Fatalf("%d: %s", w.Code, w.Body.String())
	}
	prefs := got["provider"].(map[string]any)
	if prefs["zdr"] != true || prefs["data_collection"] != "deny" || prefs["allow_fallbacks"] != false || len(prefs["only"].([]any)) != 1 || got["store"] != false {
		t.Fatal(got)
	}
}

func TestDataPolicyRefusesProviderRedirectEscape(t *testing.T) {
	var leaked atomic.Int32
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { leaked.Add(1) }))
	defer other.Close()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Location", other.URL)
		w.WriteHeader(307)
	}))
	defer srv.Close()
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{BaseURL: srv.URL + "/v1", APIKey: "provider-key"})
	p := policyForEndpoint("openai", srv.URL+"/v1", "us")
	h := NewHandler(reg, dataPolicyLoader(&p), nil)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"openai/gpt-4.1","messages":[]}`))
	req.Header.Set("Authorization", "Bearer policy-agent:secret")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if leaked.Load() != 0 || w.Code < 400 {
		t.Fatalf("leaked %d, status %d", leaked.Load(), w.Code)
	}
}

func TestDataPolicyCustomGatewayCannotContradictPolicy(t *testing.T) {
	for _, name := range []string{"openrouter", "vercel"} {
		p := policyForEndpoint(name, "https://custom.example/v1", "us")
		body := `{"model":"m","provider":{"zdr":false},"providerOptions":{"gateway":{"zeroDataRetention":false}}}`
		if _, err := enforceProviderDataPolicy(p, &provider.Provider{Name: name, BaseURL: "https://custom.example/v1"}, upstreamEncoding{Path: "/v1/chat/completions", Body: []byte(body)}); err == nil {
			t.Fatal("custom route accepted contradictory policy", name)
		}
	}
	p := policyForEndpoint("openrouter", "", "us")
	if _, err := enforceProviderDataPolicy(p, &provider.Provider{Name: "openrouter", BaseURL: "https://us.openrouter.ai/api/v1"}, upstreamEncoding{Path: "/v1/chat/completions", Body: []byte(`{"model":"safe","models":["anthropic/claude-fable-5"]}`)}); err == nil {
		t.Fatal("retention-ineligible fallback escaped")
	}
}

func TestDataPolicyManagedAdapterRoundsAndStream(t *testing.T) {
	for _, name := range []string{"openai", "anthropic"} {
		for _, stream := range []bool{false, true} {
			t.Run(name+"/stream="+fmt.Sprint(stream), func(t *testing.T) {
				tool := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { io.WriteString(w, `{"value":"synthetic-result"}`) }))
				defer tool.Close()
				alias := managedToolHashlessAliasForCanonical("trading-api.get_market_context")
				var hits int
				backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					hits++
					var body map[string]any
					json.NewDecoder(r.Body).Decode(&body)
					w.Header().Set("Content-Type", "application/json")
					if name == "openai" {
						if r.URL.Path != "/v1/responses" || body["store"] != false {
							t.Errorf("unsafe Responses wire %s: %v", r.URL.Path, body)
						}
						if hits == 1 {
							io.WriteString(w, `{"id":"r1","status":"completed","output":[{"type":"function_call","call_id":"c1","name":"`+alias+`","arguments":"{}"}]}`)
						} else {
							io.WriteString(w, `{"id":"r2","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}`)
						}
					} else {
						if r.URL.Path != "/v1/messages" || body["inference_geo"] != "us" {
							t.Errorf("unsafe Messages wire %s: %v", r.URL.Path, body)
						}
						if hits == 1 {
							io.WriteString(w, `{"id":"m1","type":"message","role":"assistant","content":[{"type":"tool_use","id":"c1","name":"`+alias+`","input":{}}],"stop_reason":"tool_use"}`)
						} else {
							io.WriteString(w, `{"id":"m2","type":"message","role":"assistant","content":[{"type":"text","text":"done"}],"stop_reason":"end_turn"}`)
						}
					}
				}))
				defer backend.Close()
				origin, model := "https://us.api.openai.com/v1", "gpt-6.1-sol"
				if name == "anthropic" {
					origin, model = "https://api.anthropic.com/v1", "claude-sonnet-5-5"
				}
				reg := provider.NewRegistry("")
				reg.Set(name, &provider.Provider{BaseURL: origin, APIKey: "synthetic-key", APIFormat: map[string]string{"openai": "openai", "anthropic": "anthropic"}[name]})
				p := policyForEndpoint(name, "", "us")
				loader := stubContextLoaderWithTools("policy-agent", "policy-agent:secret", managedToolManifestForURL(tool.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", ""))
				historyDir := t.TempDir()
				h := NewHandler(reg, func(id string) (*agentctx.AgentContext, error) { a, e := loader(id); a.DataPolicy = &p; return a, e }, nil, WithSessionHistory(historyDir))
				target, _ := url.Parse(backend.URL)
				h.client.Transport = policyRoundTrip(func(r *http.Request) (*http.Response, error) {
					if r.URL.Host != "us.api.openai.com" && r.URL.Host != "api.anthropic.com" {
						return http.DefaultTransport.RoundTrip(r)
					}
					c := r.Clone(r.Context())
					u := *r.URL
					u.Scheme = target.Scheme
					u.Host = target.Host
					c.URL = &u
					return http.DefaultTransport.RoundTrip(c)
				})
				req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(fmt.Sprintf(`{"model":%q,"stream":%v,"messages":[{"role":"user","content":"synthetic"}]}`, name+"/"+model, stream)))
				req.Header.Set("Authorization", "Bearer policy-agent:secret")
				w := httptest.NewRecorder()
				h.ServeHTTP(w, req)
				if w.Code != 200 || hits != 2 || !strings.Contains(w.Body.String(), "done") {
					t.Fatalf("%d rounds=%d: %s", w.Code, hits, w.Body.String())
				}
				entries, err := sessionhistory.ReadEntries(historyDir, "policy-agent", nil, 10)
				if err != nil || len(entries) != 1 {
					t.Fatalf("history: %v %v", entries, err)
				}
				raw, _ := json.Marshal(entries[0])
				if entries[0].Response.Format != "metadata" || len(entries[0].ToolTrace) != 0 || bytes.Contains(raw, []byte("synthetic-result")) {
					t.Fatalf("managed history retained content: %s", raw)
				}

			})
		}
	}
}

func TestDataPolicyRechecksFallbackRoute(t *testing.T) {
	for _, allowFallback := range []bool{false, true} {
		t.Run(fmt.Sprint(allowFallback), func(t *testing.T) {
			var fallbackHits atomic.Int32
			primary := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(502)
				io.WriteString(w, `{"error":{"message":"unavailable"}}`)
			}))
			defer primary.Close()
			fallback := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				fallbackHits.Add(1)
				var body map[string]any
				json.NewDecoder(r.Body).Decode(&body)
				prefs, _ := body["provider"].(map[string]any)
				if prefs["zdr"] != true || prefs["data_collection"] != "deny" || body["store"] != false {
					t.Errorf("unsafe fallback %v", body)
				}
				io.WriteString(w, `{"choices":[{"message":{"content":"ok"}}]}`)
			}))
			defer fallback.Close()
			reg := provider.NewRegistry("")
			reg.Set("openai", &provider.Provider{BaseURL: primary.URL + "/v1", APIKey: "primary"})
			reg.Set("openrouter", &provider.Provider{BaseURL: fallback.URL + "/v1", APIKey: "fallback"})
			p := policyForEndpoint("openai", primary.URL+"/v1", "us")
			if allowFallback {
				p.Attestations = append(p.Attestations, policyForEndpoint("openrouter", fallback.URL+"/v1", "eu").Attestations...)
			}
			loader := dataPolicyLoader(&p)
			h := NewHandler(reg, func(id string) (*agentctx.AgentContext, error) {
				a, e := loader(id)
				a.ModelPolicy = &agentctx.ModelPolicy{Mode: "clamp", Allowed: []agentctx.AllowedModel{{Slot: "primary", Ref: "openai/gpt-4.1"}, {Slot: "fallback", Ref: "openrouter/other/model"}}}
				return a, e
			}, nil)
			req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"openai/gpt-4.1","messages":[]}`))
			req.Header.Set("Authorization", "Bearer policy-agent:secret")
			w := httptest.NewRecorder()
			h.ServeHTTP(w, req)
			if allowFallback {
				if w.Code != 200 || fallbackHits.Load() != 1 {
					t.Fatalf("%d hits=%d: %s", w.Code, fallbackHits.Load(), w.Body.String())
				}
			} else if w.Code != 400 || fallbackHits.Load() != 0 {
				t.Fatalf("unattested fallback dispatched: %d hits=%d", w.Code, fallbackHits.Load())
			}
		})
	}
}

func TestDataPolicyReappliesOnFormatRetry(t *testing.T) {
	var paths []string
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		paths = append(paths, r.URL.Path)
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["store"] != false {
			t.Errorf("retry wire omitted store:false: %v", body)
		}
		w.Header().Set("Content-Type", "application/json")
		if len(paths) == 1 {
			w.WriteHeader(400)
			io.WriteString(w, `{"error":{"message":"This model is only supported in v1/responses endpoint and not in v1/chat/completions."}}`)
			return
		}
		io.WriteString(w, `{"id":"r1","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}`)
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{BaseURL: backend.URL + "/v1", APIKey: "synthetic"})
	p := policyForEndpoint("openai", backend.URL+"/v1", "us")
	h := NewHandler(reg, dataPolicyLoader(&p), nil)
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"openai/future-model","messages":[]}`))
	req.Header.Set("Authorization", "Bearer policy-agent:secret")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != 200 || len(paths) != 2 || paths[0] != "/v1/chat/completions" || paths[1] != "/v1/responses" {
		t.Fatalf("%d %v: %s", w.Code, paths, w.Body.String())
	}
}
