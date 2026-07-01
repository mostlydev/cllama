package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

type stubPolicyEvaluator struct {
	gateRequest  func(context.Context, PolicyGateRequest) (*PolicyGateRequestResult, error)
	decorate     func(context.Context, PolicyDecorateRequest) (*PolicyDecorateResult, error)
	gateResponse func(context.Context, PolicyGateResponseRequest) (*PolicyGateResponseResult, error)
	score        func(context.Context, PolicyScoreRequest) error
}

func (s stubPolicyEvaluator) GateRequest(ctx context.Context, req PolicyGateRequest) (*PolicyGateRequestResult, error) {
	if s.gateRequest != nil {
		return s.gateRequest(ctx, req)
	}
	return &PolicyGateRequestResult{Verdict: policyVerdictAllow}, nil
}

func (s stubPolicyEvaluator) Decorate(ctx context.Context, req PolicyDecorateRequest) (*PolicyDecorateResult, error) {
	if s.decorate != nil {
		return s.decorate(ctx, req)
	}
	return nil, nil
}

func (s stubPolicyEvaluator) GateResponse(ctx context.Context, req PolicyGateResponseRequest) (*PolicyGateResponseResult, error) {
	if s.gateResponse != nil {
		return s.gateResponse(ctx, req)
	}
	return &PolicyGateResponseResult{Verdict: policyVerdictAllow}, nil
}

func (s stubPolicyEvaluator) Score(ctx context.Context, req PolicyScoreRequest) error {
	if s.score != nil {
		return s.score(ctx, req)
	}
	return nil
}

func TestPolicyNilEvaluatorPassthroughConformance(t *testing.T) {
	t.Setenv(EnvPolicyURL, "")
	cases := []struct {
		name    string
		format  string
		managed bool
		stream  bool
	}{
		{name: "openai plain nonstream", format: "openai"},
		{name: "openai plain stream", format: "openai", stream: true},
		{name: "openai managed nonstream", format: "openai", managed: true},
		{name: "openai managed stream", format: "openai", managed: true, stream: true},
		{name: "anthropic plain nonstream", format: "anthropic"},
		{name: "anthropic plain stream", format: "anthropic", stream: true},
		{name: "anthropic managed nonstream", format: "anthropic", managed: true},
		{name: "anthropic managed stream", format: "anthropic", managed: true, stream: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			baseline := runPolicyPassthroughCase(t, tc.format, tc.managed, tc.stream)
			withNil := runPolicyPassthroughCase(t, tc.format, tc.managed, tc.stream, WithPolicyEvaluator(nil))
			if withNil.status != baseline.status {
				t.Fatalf("status changed: baseline=%d nil=%d body=%s", baseline.status, withNil.status, withNil.proxyBody)
			}
			if withNil.backendBody != baseline.backendBody {
				t.Fatalf("backend request changed:\nbaseline=%s\nnil=%s", baseline.backendBody, withNil.backendBody)
			}
			if withNil.proxyBody != baseline.proxyBody {
				t.Fatalf("proxy response changed:\nbaseline=%s\nnil=%s", baseline.proxyBody, withNil.proxyBody)
			}
		})
	}
}

func TestPolicyRequestGateDeniesBeforeBackend(t *testing.T) {
	var backendHits int
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		backendHits++
		w.WriteHeader(http.StatusOK)
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer"})
	policy := stubPolicyEvaluator{
		gateRequest: func(context.Context, PolicyGateRequest) (*PolicyGateRequestResult, error) {
			return &PolicyGateRequestResult{Verdict: policyVerdictDeny, Reason: "blocked"}, nil
		},
	}
	h := NewHandler(reg, policyContextLoader("agent", "agent:tok", nil, nil), logging.New(io.Discard), WithPolicyEvaluator(policy))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer agent:tok")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d: %s", w.Code, w.Body.String())
	}
	if backendHits != 0 {
		t.Fatalf("policy-denied request reached backend %d time(s)", backendHits)
	}
	if !strings.Contains(w.Body.String(), "blocked") {
		t.Fatalf("expected denial reason in body, got %s", w.Body.String())
	}
}

func TestPolicyDecorationOpenAIAndAnthropic(t *testing.T) {
	t.Run("openai", func(t *testing.T) {
		gotBody := policyCapturedBodyBackend()
		reg, backend := policyRegistryWithCapture(t, "openai", gotBody)
		defer backend.Close()
		policy := stubPolicyEvaluator{
			decorate: func(context.Context, PolicyDecorateRequest) (*PolicyDecorateResult, error) {
				return &PolicyDecorateResult{
					MessagesPatch: []map[string]any{{"role": "system", "content": "policy context"}},
				}, nil
			},
		}
		h := NewHandler(reg, policyContextLoader("agent", "agent:tok", nil, nil), logging.New(io.Discard), WithPolicyEvaluator(policy))
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("Authorization", "Bearer agent:tok")
		w := httptest.NewRecorder()

		h.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
		}
		var payload map[string]any
		if err := json.Unmarshal(*gotBody, &payload); err != nil {
			t.Fatalf("unmarshal backend body: %v", err)
		}
		messages := payload["messages"].([]any)
		last := messages[len(messages)-1].(map[string]any)
		if last["role"] != "system" || last["content"] != "policy context" {
			t.Fatalf("expected policy system message, got %+v", messages)
		}
	})

	t.Run("anthropic", func(t *testing.T) {
		gotBody := policyCapturedBodyBackend()
		reg, backend := policyRegistryWithCapture(t, "anthropic", gotBody)
		defer backend.Close()
		policy := stubPolicyEvaluator{
			decorate: func(context.Context, PolicyDecorateRequest) (*PolicyDecorateResult, error) {
				return &PolicyDecorateResult{SystemPatch: "policy context"}, nil
			},
		}
		h := NewHandler(reg, policyContextLoader("agent", "agent:tok", nil, nil), logging.New(io.Discard), WithPolicyEvaluator(policy))
		req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{"model":"anthropic/claude-sonnet-4","system":"base","messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("Authorization", "Bearer agent:tok")
		w := httptest.NewRecorder()

		h.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
		}
		var payload map[string]any
		if err := json.Unmarshal(*gotBody, &payload); err != nil {
			t.Fatalf("unmarshal backend body: %v", err)
		}
		if payload["system"] != "base\n\npolicy context" {
			t.Fatalf("expected appended system patch, got %+v", payload["system"])
		}
	})
}

func TestPolicyToolFilterRestrictsManagedTools(t *testing.T) {
	gotBody := policyCapturedBodyBackend()
	reg, backend := policyRegistryWithCapture(t, "openai", gotBody)
	defer backend.Close()
	policy := stubPolicyEvaluator{
		gateRequest: func(context.Context, PolicyGateRequest) (*PolicyGateRequestResult, error) {
			return &PolicyGateRequestResult{
				Verdict: policyVerdictAllow,
				ToolFilter: &PolicyToolFilter{
					Mode:  policyToolFilterAllowList,
					Tools: []string{"svc.allowed"},
				},
			}, nil
		},
	}
	h := NewHandler(reg, policyContextLoader("agent", "agent:tok", policyTwoToolManifest(), nil), logging.New(io.Discard), WithPolicyEvaluator(policy))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer agent:tok")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	var payload map[string]any
	if err := json.Unmarshal(*gotBody, &payload); err != nil {
		t.Fatalf("unmarshal backend body: %v", err)
	}
	tools := payload["tools"].([]any)
	if len(tools) != 1 {
		t.Fatalf("expected only one managed tool after filter, got %+v", tools)
	}
	raw, _ := json.Marshal(tools[0])
	if !strings.Contains(string(raw), managedToolHashlessAliasForCanonical("svc.allowed")) || strings.Contains(string(raw), managedToolHashlessAliasForCanonical("svc.denied")) {
		t.Fatalf("unexpected filtered tools: %s", raw)
	}
}

func TestPolicyResponseGateAmendsBufferedResponseAndScores(t *testing.T) {
	scoreCh := make(chan PolicyScoreRequest, 1)
	gotBody := policyCapturedBodyBackend()
	reg, backend := policyRegistryWithCapture(t, "openai", gotBody)
	defer backend.Close()
	policy := stubPolicyEvaluator{
		gateResponse: func(context.Context, PolicyGateResponseRequest) (*PolicyGateResponseResult, error) {
			return &PolicyGateResponseResult{
				Verdict:     policyVerdictAmend,
				AmendedBody: json.RawMessage(`{"choices":[{"message":{"content":"amended"}}]}`),
			}, nil
		},
		score: func(_ context.Context, req PolicyScoreRequest) error {
			scoreCh <- req
			return nil
		},
	}
	h := NewHandler(reg, policyContextLoader("agent", "agent:tok", nil, nil), logging.New(io.Discard), WithPolicyEvaluator(policy))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer agent:tok")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "amended") || strings.Contains(w.Body.String(), "backend") {
		t.Fatalf("expected amended response, got %s", w.Body.String())
	}
	select {
	case score := <-scoreCh:
		if !strings.Contains(score.ResponseBody, "amended") {
			t.Fatalf("expected score to receive amended body, got %+v", score)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for policy score")
	}
}

func TestPolicyResponseGateDeniesStreamBeforeFirstByte(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[]}\n\n"))
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer"})
	policy := stubPolicyEvaluator{
		gateResponse: func(context.Context, PolicyGateResponseRequest) (*PolicyGateResponseResult, error) {
			return &PolicyGateResponseResult{Verdict: policyVerdictDeny, Reason: "stream blocked"}, nil
		},
	}
	h := NewHandler(reg, policyContextLoader("agent", "agent:tok", nil, nil), logging.New(io.Discard), WithPolicyEvaluator(policy))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"openai/gpt-4o","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer agent:tok")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusForbidden {
		t.Fatalf("expected 403, got %d: %s", w.Code, w.Body.String())
	}
	if strings.Contains(w.Body.String(), "data:") {
		t.Fatalf("expected denial before stream bytes, got %s", w.Body.String())
	}
}

func TestPolicyFailOpenAndPolicyExemptSkipGateFailures(t *testing.T) {
	t.Run("fail open", func(t *testing.T) {
		t.Setenv(EnvPolicyFailMode, policyFailModeOpen)
		gotBody := policyCapturedBodyBackend()
		reg, backend := policyRegistryWithCapture(t, "openai", gotBody)
		defer backend.Close()
		policy := stubPolicyEvaluator{
			gateRequest: func(context.Context, PolicyGateRequest) (*PolicyGateRequestResult, error) {
				return nil, errors.New("sidecar down")
			},
		}
		h := NewHandler(reg, policyContextLoader("agent", "agent:tok", nil, nil), logging.New(io.Discard), WithPolicyEvaluator(policy))
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("Authorization", "Bearer agent:tok")
		w := httptest.NewRecorder()

		h.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("expected fail-open 200, got %d: %s", w.Code, w.Body.String())
		}
		if len(*gotBody) == 0 {
			t.Fatal("expected backend to receive request")
		}
	})

	t.Run("policy exempt", func(t *testing.T) {
		var policyCalls int
		gotBody := policyCapturedBodyBackend()
		reg, backend := policyRegistryWithCapture(t, "openai", gotBody)
		defer backend.Close()
		policy := stubPolicyEvaluator{
			gateRequest: func(context.Context, PolicyGateRequest) (*PolicyGateRequestResult, error) {
				policyCalls++
				return &PolicyGateRequestResult{Verdict: policyVerdictDeny}, nil
			},
			decorate: func(context.Context, PolicyDecorateRequest) (*PolicyDecorateResult, error) {
				policyCalls++
				return nil, nil
			},
			gateResponse: func(context.Context, PolicyGateResponseRequest) (*PolicyGateResponseResult, error) {
				policyCalls++
				return &PolicyGateResponseResult{Verdict: policyVerdictDeny}, nil
			},
		}
		h := NewHandler(reg, policyContextLoader("agent", "agent:tok", nil, map[string]any{"policy_exempt": true}), logging.New(io.Discard), WithPolicyEvaluator(policy))
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("Authorization", "Bearer agent:tok")
		w := httptest.NewRecorder()

		h.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("expected exempt request through, got %d: %s", w.Code, w.Body.String())
		}
		if policyCalls != 0 {
			t.Fatalf("policy_exempt context should skip hooks, saw %d call(s)", policyCalls)
		}
	})
}

type policyPassthroughResult struct {
	status      int
	proxyBody   string
	backendBody string
}

func runPolicyPassthroughCase(t *testing.T, format string, managed bool, stream bool, opts ...HandlerOption) policyPassthroughResult {
	t.Helper()
	var gotBody []byte
	reg, backend := policyRegistryWithCapture(t, format, &gotBody)
	defer backend.Close()
	var tools *agentctx.ToolManifest
	if managed {
		tools = managedToolManifest()
	}
	h := NewHandler(reg, policyContextLoader("agent", "agent:tok", tools, nil), logging.New(io.Discard), opts...)

	body := policyRequestBody(format, stream)
	path := "/v1/chat/completions"
	if format == "anthropic" {
		path = "/v1/messages"
	}
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewBufferString(body))
	req.Header.Set("Authorization", "Bearer agent:tok")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	return policyPassthroughResult{
		status:      w.Code,
		proxyBody:   w.Body.String(),
		backendBody: string(gotBody),
	}
}

func policyCapturedBodyBackend() *[]byte {
	var gotBody []byte
	return &gotBody
}

func policyRegistryWithCapture(t *testing.T, format string, gotBody *[]byte) (*provider.Registry, *httptest.Server) {
	t.Helper()
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read backend body: %v", err)
		}
		*gotBody = body
		w.Header().Set("Content-Type", "application/json")
		if format == "anthropic" {
			_, _ = w.Write([]byte(`{"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"backend"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`))
			return
		}
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"role":"assistant","content":"backend"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	reg := provider.NewRegistry("")
	switch format {
	case "anthropic":
		reg.Set("anthropic", &provider.Provider{Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant", Auth: "x-api-key", APIFormat: "anthropic"})
	default:
		reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer"})
	}
	return reg, backend
}

func policyRequestBody(format string, stream bool) string {
	streamField := ""
	if stream {
		streamField = `,"stream":true`
	}
	if format == "anthropic" {
		return `{"model":"anthropic/claude-sonnet-4"` + streamField + `,"messages":[{"role":"user","content":"hi"}]}`
	}
	return `{"model":"openai/gpt-4o"` + streamField + `,"messages":[{"role":"user","content":"hi"}]}`
}

func policyContextLoader(agentID, token string, tools *agentctx.ToolManifest, extraMetadata map[string]any) ContextLoader {
	return func(id string) (*agentctx.AgentContext, error) {
		if id != agentID {
			return nil, io.EOF
		}
		metadata := map[string]any{"token": token}
		for k, v := range extraMetadata {
			metadata[k] = v
		}
		return &agentctx.AgentContext{
			AgentID:     id,
			ContextDir:  "/claw/context/" + id,
			AgentsMD:    []byte("# Contract"),
			ClawdapusMD: []byte("# Infra"),
			Metadata:    metadata,
			Tools:       tools,
		}, nil
	}
}

func policyTwoToolManifest() *agentctx.ToolManifest {
	return &agentctx.ToolManifest{
		Version: 1,
		Tools: []agentctx.ToolManifestEntry{
			{
				Name:        "svc.allowed",
				Description: "Allowed tool",
				InputSchema: map[string]any{"type": "object"},
			},
			{
				Name:        "svc.denied",
				Description: "Denied tool",
				InputSchema: map[string]any{"type": "object"},
			},
		},
		Policy: agentctx.ToolPolicy{
			MaxRounds:        8,
			TimeoutPerToolMS: 30000,
			TotalTimeoutMS:   120000,
		},
	}
}
