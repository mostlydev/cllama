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
	"github.com/mostlydev/cllama/internal/sessionhistory"
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
	apiHandler := newAPIHandler(contextRoot, reg, logging.New(io.Discard), acc, pricing, "test-pod", nil, "")
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

func TestConfigFromEnvSessionHistoryDir(t *testing.T) {
	t.Setenv("CLAW_SESSION_HISTORY_DIR", "/claw/session-history")
	cfg := configFromEnv()
	if cfg.SessionHistoryDir != "/claw/session-history" {
		t.Errorf("SessionHistoryDir = %q; want /claw/session-history", cfg.SessionHistoryDir)
	}
}

func TestAPIHistoryEndpointAllowsAgentAndDedicatedReplayAuth(t *testing.T) {
	contextRoot := t.TempDir()
	agentDir := filepath.Join(contextRoot, "tiverton")
	if err := os.MkdirAll(filepath.Join(agentDir, "service-auth"), 0o700); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{
		"AGENTS.md":     "# contract",
		"CLAWDAPUS.md":  "# infra",
		"metadata.json": `{"token":"tiverton:dummy123"}`,
	} {
		if err := os.WriteFile(filepath.Join(agentDir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := os.WriteFile(filepath.Join(agentDir, "service-auth", "cllama-history.json"), []byte(`{"service":"cllama-history","auth_type":"bearer","token":"history-token","principal":"team-memory"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "service-auth", "team-memory.json"), []byte(`{"service":"team-memory","auth_type":"bearer","token":"memory-token"}`), 0o600); err != nil {
		t.Fatal(err)
	}

	histDir := t.TempDir()
	recorder := sessionhistory.New(histDir)
	base := time.Date(2026, 3, 31, 12, 0, 0, 0, time.UTC)
	for i := 0; i < 2; i++ {
		if err := recorder.Record("tiverton", sessionhistory.Entry{
			Version: 1,
			TS:      base.Add(time.Duration(i) * time.Minute).Format(time.RFC3339),
			ClawID:  "tiverton",
			Response: sessionhistory.Payload{
				Format: "json",
				JSON:   json.RawMessage(`{}`),
			},
		}); err != nil {
			t.Fatal(err)
		}
	}

	apiHandler := newAPIHandler(contextRoot, provider.NewRegistry(""), logging.New(io.Discard), cost.NewAccumulator(), cost.DefaultPricing(), "", recorder, "")

	req := httptest.NewRequest(http.MethodGet, "/history/tiverton?limit=1", nil)
	req.Header.Set("Authorization", "Bearer history-token")
	rec := httptest.NewRecorder()
	apiHandler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for replay auth, got %d body=%s", rec.Code, rec.Body.String())
	}
	lines := strings.Split(strings.TrimSpace(rec.Body.String()), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected one ndjson line, got %q", rec.Body.String())
	}

	req = httptest.NewRequest(http.MethodGet, "/history/tiverton?limit=1", nil)
	req.Header.Set("Authorization", "Bearer memory-token")
	rec = httptest.NewRecorder()
	apiHandler.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("expected 403 for non-replay service auth, got %d body=%s", rec.Code, rec.Body.String())
	}

	req = httptest.NewRequest(http.MethodGet, "/history/tiverton?after=2026-03-31T12:00:00Z", nil)
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	rec = httptest.NewRecorder()
	apiHandler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for agent auth, got %d body=%s", rec.Code, rec.Body.String())
	}
	lines = strings.Split(strings.TrimSpace(rec.Body.String()), "\n")
	if len(lines) != 1 {
		t.Fatalf("expected one filtered ndjson line, got %q", rec.Body.String())
	}
}

func TestAPIHistoryEndpointAllowsAdminTokenAndRejectsWrongBearer(t *testing.T) {
	contextRoot := t.TempDir()
	agentDir := filepath.Join(contextRoot, "tiverton")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{
		"AGENTS.md":     "# contract",
		"CLAWDAPUS.md":  "# infra",
		"metadata.json": `{"token":"tiverton:dummy123"}`,
	} {
		if err := os.WriteFile(filepath.Join(agentDir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	histDir := t.TempDir()
	recorder := sessionhistory.New(histDir)
	if err := recorder.Record("tiverton", sessionhistory.Entry{
		Version: 1,
		TS:      time.Date(2026, 3, 31, 12, 0, 0, 0, time.UTC).Format(time.RFC3339),
		ClawID:  "tiverton",
		Response: sessionhistory.Payload{
			Format: "json",
			JSON:   json.RawMessage(`{}`),
		},
	}); err != nil {
		t.Fatal(err)
	}

	apiHandler := newAPIHandler(contextRoot, provider.NewRegistry(""), logging.New(io.Discard), cost.NewAccumulator(), cost.DefaultPricing(), "", recorder, "ui-secret")

	req := httptest.NewRequest(http.MethodGet, "/history/tiverton", nil)
	req.Header.Set("Authorization", "Bearer wrong-token")
	rec := httptest.NewRecorder()
	apiHandler.ServeHTTP(rec, req)
	if rec.Code != http.StatusForbidden {
		t.Fatalf("expected 403 for wrong token, got %d body=%s", rec.Code, rec.Body.String())
	}

	req = httptest.NewRequest(http.MethodGet, "/history/tiverton", nil)
	req.Header.Set("Authorization", "Bearer ui-secret")
	rec = httptest.NewRecorder()
	apiHandler.ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("expected 200 for admin token, got %d body=%s", rec.Code, rec.Body.String())
	}
}

func TestConfigFromEnvSessionHistoryDirDefault(t *testing.T) {
	t.Setenv("CLAW_SESSION_HISTORY_DIR", "")
	cfg := configFromEnv()
	if cfg.SessionHistoryDir != "" {
		t.Errorf("SessionHistoryDir should be empty when unset; got %q", cfg.SessionHistoryDir)
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
