package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/cost"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

func TestDualServerIntegrationSmoke(t *testing.T) {
	var mu sync.Mutex
	var gotAuth string
	var gotModel string
	var gotPath string

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		gotAuth = r.Header.Get("Authorization")
		gotPath = r.URL.Path
		var payload map[string]any
		_ = json.NewDecoder(r.Body).Decode(&payload)
		if m, ok := payload["model"].(string); ok {
			gotModel = m
		}
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-1","choices":[{"message":{"content":"ok"}}]}`)
	}))
	defer backend.Close()

	contextRoot := t.TempDir()
	agentDir := filepath.Join(contextRoot, "tiverton")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "AGENTS.md"), []byte("# contract"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "CLAWDAPUS.md"), []byte("# infra"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "metadata.json"), []byte(`{"token":"tiverton:dummy123"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	authDir := t.TempDir()
	providersJSON := `{
		"providers": {
			"openai": {
				"base_url": "` + backend.URL + `/v1",
				"api_key": "sk-real",
				"auth": "bearer"
			}
		}
	}`
	if err := os.WriteFile(filepath.Join(authDir, "providers.json"), []byte(providersJSON), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := provider.NewRegistry(authDir)
	if err := reg.LoadFromFile(); err != nil {
		t.Fatal(err)
	}
	pricing := cost.DefaultPricing()
	acc := cost.NewAccumulator()
	apiHandler := newAPIHandler(contextRoot, reg, logging.New(io.Discard), acc, pricing, "test-pod")
	uiHandler := newUIHandler(reg, acc, contextRoot, "")

	apiServer := &http.Server{Handler: apiHandler}
	uiServer := &http.Server{Handler: uiHandler}

	apiLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer apiLn.Close()
	uiLn, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer uiLn.Close()

	go func() { _ = apiServer.Serve(apiLn) }()
	go func() { _ = uiServer.Serve(uiLn) }()
	defer func() {
		_ = apiServer.Close()
		_ = uiServer.Close()
	}()

	time.Sleep(50 * time.Millisecond)

	apiURL := "http://" + apiLn.Addr().String() + "/v1/chat/completions"
	body := `{"model":"openai/gpt-4o","messages":[{"role":"user","content":"hello"}]}`
	req, err := http.NewRequest(http.MethodPost, apiURL, strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("api call failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		respBody, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200, got %d body=%s", resp.StatusCode, string(respBody))
	}

	mu.Lock()
	if gotAuth != "Bearer sk-real" {
		t.Fatalf("expected backend auth Bearer sk-real, got %q", gotAuth)
	}
	if gotModel != "gpt-4o" {
		t.Fatalf("expected stripped model gpt-4o, got %q", gotModel)
	}
	if gotPath != "/v1/chat/completions" && gotPath != "/chat/completions" {
		t.Fatalf("unexpected backend path: %q", gotPath)
	}
	mu.Unlock()

	uiResp, err := http.Get("http://" + uiLn.Addr().String() + "/")
	if err != nil {
		t.Fatalf("ui call failed: %v", err)
	}
	defer uiResp.Body.Close()
	if uiResp.StatusCode != http.StatusOK {
		t.Fatalf("expected ui status 200, got %d", uiResp.StatusCode)
	}
	uiBody, _ := io.ReadAll(uiResp.Body)
	if !bytes.Contains(uiBody, []byte("openai")) {
		t.Fatalf("expected provider list in UI body: %s", string(uiBody))
	}
}

func TestHealthcheckURL(t *testing.T) {
	cases := []struct {
		addr string
		want string
	}{
		{addr: ":8080", want: "http://127.0.0.1:8080/health"},
		{addr: "0.0.0.0:9000", want: "http://127.0.0.1:9000/health"},
		{addr: "127.0.0.1:9001", want: "http://127.0.0.1:9001/health"},
	}
	for _, tc := range cases {
		if got := healthcheckURL(tc.addr); got != tc.want {
			t.Fatalf("addr=%s got=%s want=%s", tc.addr, got, tc.want)
		}
	}
}
