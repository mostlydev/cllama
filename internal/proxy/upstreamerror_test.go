package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

type observedErrorBody struct {
	io.Reader
	closed bool
}

func TestHandlerUpstreamErrorAuditPreservesResponse(t *testing.T) {
	const rejection = `{"error":{"message":"sensitive upstream explanation","type":"invalid_request_error","code":"unsupported_parameter","param":"max_tokens"}}`
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(400)
		io.WriteString(w, rejection)
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "test-key", Auth: "bearer"})
	var logs bytes.Buffer
	h := NewHandler(reg, stubContextLoaderWithToken("worker", "worker:test"), logging.New(&logs))
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(`{"model":"openai/gpt-4o","messages":[{"role":"user","content":"synthetic"}]}`))
	req.Header.Set("Authorization", "Bearer worker:test")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != 400 || w.Body.String() != rejection {
		t.Fatalf("response changed: %d %s", w.Code, w.Body.String())
	}
	if !strings.Contains(logs.String(), `"upstream_error_param":"max_tokens"`) {
		t.Fatalf("missing diagnostic: %s", logs.String())
	}
	if strings.Contains(logs.String(), "sensitive upstream explanation") {
		t.Fatal("upstream message logged")
	}
}

func (b *observedErrorBody) Close() error { b.closed = true; return nil }

func TestUpstreamErrorDiagnostics(t *testing.T) {
	for _, tc := range []struct{ name, body, code, kind, param string }{
		{"known", `{"error":{"message":"secret prompt sk-secret","code":"unsupported_parameter","type":"invalid_request_error","param":"temperature"}}`, "unsupported_parameter", "invalid_request_error", "temperature"},
		{"identifier_secrets", `{"error":{"code":"sk_secret","type":"customer_prompt","param":"secret_token"}}`, "other", "other", "other"},
		{"invalid", `secret upstream HTML`, "", "", ""},
		{"oversized", `{"error":{"code":"unsupported_parameter","message":"` + strings.Repeat("x", upstreamErrorPeekLimit) + `"}}`, "", "", ""},
		{"null", `{"error":{"code":null,"type":123,"param":null}}`, "", "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var logs bytes.Buffer
			h := &Handler{logger: logging.New(&logs)}
			body := &observedErrorBody{Reader: strings.NewReader(tc.body)}
			resp := &http.Response{StatusCode: 400, Body: body}
			h.logUpstreamError("worker", "openai/example", "openai", resp)
			if body.closed {
				t.Fatal("peek closed original body")
			}
			got, err := io.ReadAll(resp.Body)
			if err != nil || string(got) != tc.body {
				t.Fatalf("body changed: %v", err)
			}
			resp.Body.Close()
			if !body.closed {
				t.Fatal("original body not closed")
			}
			var event map[string]any
			if err := json.Unmarshal(logs.Bytes(), &event); err != nil {
				t.Fatal(err)
			}
			for key, want := range map[string]string{"upstream_error_code": tc.code, "upstream_error_type": tc.kind, "upstream_error_param": tc.param} {
				got, _ := event[key].(string)
				if got != want {
					t.Errorf("%s=%q want %q", key, got, want)
				}
			}
			for _, forbidden := range []string{"sk_secret", "customer_prompt", "secret_token", "secret prompt", "sk-secret", "message"} {
				if strings.Contains(logs.String(), forbidden) {
					t.Fatalf("sensitive content in log: %s", forbidden)
				}
			}
			if event["type"] != "upstream_error" || event["status_code"] != float64(400) {
				t.Fatalf("bad event: %v", event)
			}
		})
	}
}

func TestUpstreamErrorDiagnosticsIgnoresSuccess(t *testing.T) {
	var logs bytes.Buffer
	h := &Handler{logger: logging.New(&logs)}
	body := &observedErrorBody{Reader: strings.NewReader("ok")}
	h.logUpstreamError("worker", "example", "provider", &http.Response{StatusCode: 200, Body: body})
	if logs.Len() != 0 || body.closed {
		t.Fatal("success inspected")
	}
	got, _ := io.ReadAll(body)
	if string(got) != "ok" {
		t.Fatal("success body consumed")
	}
}
