//go:build dashboard

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/cost"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

// TestLiveDashboard stands up a mock LLM backend, configures three
// providers, creates agent contexts for a mock pod, boots the
// real proxy, fires a burst of requests as each agent, and then blocks
// so you can browse the live dashboard.
//
// Run: go test -tags dashboard -v -run TestLiveDashboard ./cmd/cllama/...
//
// Dashboard: http://127.0.0.1:9081/
//   - Providers  → /
//   - Pod        → /pod
//   - Costs      → /costs
//   - Costs JSON → /costs/api
func TestLiveDashboard(t *testing.T) {
	// ── Mock LLM backend ─────────────────────────────────────────────────
	var reqCount atomic.Int64
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := reqCount.Add(1)

		var payload map[string]any
		body, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(body, &payload)
		model, _ := payload["model"].(string)

		// Vary token counts per request to make the dashboard interesting.
		promptTokens := 200 + rand.Intn(1800)
		completionTokens := 50 + rand.Intn(950)

		// Simulate some latency to make the burst feel real.
		time.Sleep(time.Duration(20+rand.Intn(80)) * time.Millisecond)

		w.Header().Set("Content-Type", "application/json")
		resp := map[string]any{
			"id":      fmt.Sprintf("chatcmpl-demo-%d", n),
			"object":  "chat.completion",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]any{
				{
					"index":         0,
					"message":       map[string]string{"role": "assistant", "content": fmt.Sprintf("[demo response %d from %s]", n, model)},
					"finish_reason": "stop",
				},
			},
			"usage": map[string]int{
				"prompt_tokens":     promptTokens,
				"completion_tokens": completionTokens,
				"total_tokens":      promptTokens + completionTokens,
			},
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer backend.Close()
	t.Logf("mock backend: %s", backend.URL)

	// ── Context directory (agents) ───────────────────────────────────────
	contextRoot := t.TempDir()
	agents := []struct {
		id      string
		token   string
		pod     string
		service string
		typ     string
	}{
		{"tiverton", "tiverton:aabbccdd00112233445566778899aabbccdd00112233445566", "trading-desk", "tiverton", "openclaw"},
		{"westin", "westin:11223344556677889900aabbccddeeff11223344556677889900", "trading-desk", "westin", "openclaw"},
		{"allen", "allen:ffeeddccbbaa99887766554433221100ffeeddccbbaa99887766", "trading-desk", "allen", "nanoclaw"},
	}
	for _, a := range agents {
		dir := filepath.Join(contextRoot, a.id)
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		meta := fmt.Sprintf(`{"token":%q,"pod":%q,"service":%q,"type":%q}`,
			a.token, a.pod, a.service, a.typ)
		if err := os.WriteFile(filepath.Join(dir, "metadata.json"), []byte(meta), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "AGENTS.md"), []byte("# "+a.id+" agent contract"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "CLAWDAPUS.md"), []byte("# "+a.id+" infra"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	// ── Providers config (all pointing at mock backend) ──────────────────
	authDir := t.TempDir()
	providersJSON := fmt.Sprintf(`{
		"providers": {
			"openrouter": {
				"base_url": "%s/v1",
				"api_key": "sk-or-demo-key",
				"auth": "bearer"
			},
			"anthropic": {
				"base_url": "%s/v1",
				"api_key": "sk-ant-demo-key",
				"auth": "x-api-key",
				"api_format": "openai"
			},
			"ollama": {
				"base_url": "%s/v1",
				"auth": "none"
			}
		}
	}`, backend.URL, backend.URL, backend.URL)
	if err := os.WriteFile(filepath.Join(authDir, "providers.json"), []byte(providersJSON), 0o600); err != nil {
		t.Fatal(err)
	}

	reg := provider.NewRegistry(authDir)
	if err := reg.LoadFromFile(); err != nil {
		t.Fatal(err)
	}

	pricing := cost.DefaultPricing()
	acc := cost.NewAccumulator()
	logger := logging.New(os.Stdout)

	apiHandler := newAPIHandler(contextRoot, reg, logger, acc, pricing, "", nil, "", nil)
	uiHandler := newUIHandler(reg, acc, contextRoot, "", nil)

	// ── Listen on fixed ports ────────────────────────────────────────────
	apiLn, err := net.Listen("tcp", "127.0.0.1:9080")
	if err != nil {
		t.Fatalf("listen api :9080: %v (is another instance running?)", err)
	}
	defer apiLn.Close()
	uiLn, err := net.Listen("tcp", "127.0.0.1:9081")
	if err != nil {
		t.Fatalf("listen ui :9081: %v", err)
	}
	defer uiLn.Close()

	apiServer := &http.Server{Handler: apiHandler}
	uiServer := &http.Server{Handler: uiHandler}
	go func() { _ = apiServer.Serve(apiLn) }()
	go func() { _ = uiServer.Serve(uiLn) }()
	defer func() {
		_ = apiServer.Close()
		_ = uiServer.Close()
	}()

	time.Sleep(50 * time.Millisecond)
	t.Log("")
	t.Log("═══════════════════════════════════════════════════════════")
	t.Log("  cllama dashboard test is live")
	t.Log("")
	t.Logf("  API:       http://127.0.0.1:9080/v1/chat/completions")
	t.Logf("  Dashboard: http://127.0.0.1:9081/")
	t.Logf("  Pod:       http://127.0.0.1:9081/pod")
	t.Logf("  Costs:     http://127.0.0.1:9081/costs")
	t.Logf("  JSON API:  http://127.0.0.1:9081/costs/api")
	t.Log("")
	t.Log("  Firing requests...")
	t.Log("═══════════════════════════════════════════════════════════")
	t.Log("")

	// ── Fire a burst of requests ─────────────────────────────────────────
	type request struct {
		agentID string
		token   string
		model   string
	}
	requests := []request{
		// Tiverton — coordinator, uses multiple providers
		{"tiverton", agents[0].token, "openrouter/anthropic/claude-sonnet-4-20250514"},
		{"tiverton", agents[0].token, "openrouter/anthropic/claude-sonnet-4-20250514"},
		{"tiverton", agents[0].token, "openrouter/google/gemini-2.5-pro"},
		{"tiverton", agents[0].token, "anthropic/claude-sonnet-4-20250514"},
		{"tiverton", agents[0].token, "openrouter/anthropic/claude-sonnet-4-20250514"},
		{"tiverton", agents[0].token, "anthropic/claude-haiku-3-5-20241022"},

		// Westin — momentum trader, heavy OpenRouter user
		{"westin", agents[1].token, "openrouter/anthropic/claude-sonnet-4-20250514"},
		{"westin", agents[1].token, "openrouter/anthropic/claude-sonnet-4-20250514"},
		{"westin", agents[1].token, "openrouter/anthropic/claude-sonnet-4-20250514"},
		{"westin", agents[1].token, "openrouter/google/gemini-2.5-flash"},
		{"westin", agents[1].token, "openrouter/google/gemini-2.5-flash"},

		// Allen — systems monitor, cheap local models + occasional Anthropic
		{"allen", agents[2].token, "ollama/llama3.2:8b"},
		{"allen", agents[2].token, "ollama/llama3.2:8b"},
		{"allen", agents[2].token, "ollama/llama3.2:8b"},
		{"allen", agents[2].token, "ollama/llama3.2:8b"},
		{"allen", agents[2].token, "anthropic/claude-haiku-3-5-20241022"},
	}

	client := &http.Client{Timeout: 10 * time.Second}
	for i, req := range requests {
		body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"demo request %d from %s"}]}`,
			req.model, i+1, req.agentID)
		httpReq, err := http.NewRequest(http.MethodPost, "http://127.0.0.1:9080/v1/chat/completions",
			strings.NewReader(body))
		if err != nil {
			t.Fatalf("req %d: %v", i+1, err)
		}
		httpReq.Header.Set("Authorization", "Bearer "+req.token)
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(httpReq)
		if err != nil {
			t.Fatalf("req %d (%s → %s): %v", i+1, req.agentID, req.model, err)
		}
		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("req %d (%s → %s): status %d body=%s", i+1, req.agentID, req.model, resp.StatusCode, string(respBody))
		}
		t.Logf("  ✓ %s → %s [%d]", req.agentID, req.model, resp.StatusCode)
	}

	// ── Print summary ────────────────────────────────────────────────────
	t.Log("")

	costsResp, err := client.Get("http://127.0.0.1:9081/costs/api")
	if err != nil {
		t.Fatalf("costs api: %v", err)
	}
	costsBody, _ := io.ReadAll(costsResp.Body)
	costsResp.Body.Close()

	var costsData map[string]any
	_ = json.Unmarshal(costsBody, &costsData)
	pretty, _ := json.MarshalIndent(costsData, "  ", "  ")
	t.Logf("  Cost summary:\n  %s", string(pretty))

	t.Log("")
	t.Log("═══════════════════════════════════════════════════════════")
	t.Log("  Dashboard is live — open http://127.0.0.1:9081/")
	t.Log("  Press Ctrl-C to stop.")
	t.Log("═══════════════════════════════════════════════════════════")

	// ── Block until Ctrl-C ───────────────────────────────────────────────
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	t.Log("\nshutting down.")
}
