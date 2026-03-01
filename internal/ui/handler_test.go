package ui

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/cost"
	"github.com/mostlydev/cllama/internal/provider"
)

func TestMaskKey(t *testing.T) {
	if got := maskKey(""); got != "" {
		t.Fatalf("expected empty mask, got %q", got)
	}
	if got := maskKey("abcd"); got != "****" {
		t.Fatalf("expected short key mask, got %q", got)
	}
	if got := maskKey("sk-example-1234"); got != "sk-e...1234" {
		t.Fatalf("unexpected mask: %q", got)
	}
}

func TestNotFound(t *testing.T) {
	h := NewHandler(provider.NewRegistry(t.TempDir()))
	req := httptest.NewRequest(http.MethodGet, "/missing", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusNotFound {
		b, _ := io.ReadAll(w.Result().Body)
		t.Fatalf("expected 404, got %d body=%s", w.Code, string(b))
	}
}

func TestUICostsAPIReturnsJSON(t *testing.T) {
	reg := provider.NewRegistry(t.TempDir())
	acc := cost.NewAccumulator()
	acc.Record("tiverton", "anthropic", "claude-sonnet-4", 1000, 500, 0.0105)

	h := NewHandler(reg, WithAccumulator(acc))
	req := httptest.NewRequest("GET", "/costs/api", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "application/json") {
		t.Errorf("expected JSON content type, got %q", ct)
	}

	var result map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if _, ok := result["total_cost_usd"]; !ok {
		t.Error("expected total_cost_usd field")
	}
	if _, ok := result["agents"]; !ok {
		t.Error("expected agents field")
	}

	// Verify structure more deeply
	totalCost, ok := result["total_cost_usd"].(float64)
	if !ok {
		t.Fatal("total_cost_usd is not a number")
	}
	if totalCost < 0.01 {
		t.Errorf("expected total_cost_usd >= 0.01, got %f", totalCost)
	}

	agents, ok := result["agents"].(map[string]interface{})
	if !ok {
		t.Fatal("agents is not an object")
	}
	if _, ok := agents["tiverton"]; !ok {
		t.Error("expected 'tiverton' in agents")
	}
}

func TestUICostsAPIEmptyAccumulator(t *testing.T) {
	reg := provider.NewRegistry(t.TempDir())
	h := NewHandler(reg) // no accumulator

	req := httptest.NewRequest("GET", "/costs/api", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var result costsAPIResponse
	if err := json.Unmarshal(w.Body.Bytes(), &result); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if result.TotalCostUSD != 0 {
		t.Errorf("expected 0 total cost, got %f", result.TotalCostUSD)
	}
	if len(result.Agents) != 0 {
		t.Errorf("expected empty agents map, got %d entries", len(result.Agents))
	}
}

func TestDashboardRendersAllSections(t *testing.T) {
	reg := provider.NewRegistry(t.TempDir())
	reg.Set("anthropic", &provider.Provider{Name: "anthropic", BaseURL: "https://api.anthropic.com/v1", APIKey: "sk-test-key-1234", Auth: "bearer"})
	acc := cost.NewAccumulator()
	acc.Record("tiverton", "anthropic", "claude-sonnet-4", 1000, 500, 0.0105)

	h := NewHandler(reg, WithAccumulator(acc))
	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != 200 {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	body := w.Body.String()

	// Should contain provider info (read-only)
	if !strings.Contains(body, "anthropic") {
		t.Error("expected provider name in dashboard")
	}
	if !strings.Contains(body, "sk-t...1234") {
		t.Error("expected masked API key in dashboard")
	}
	// Should NOT contain provider form
	if strings.Contains(body, "method=\"post\"") {
		t.Error("dashboard should not contain provider management form")
	}
	// Should contain SSE connection script
	if !strings.Contains(body, "EventSource") {
		t.Error("expected EventSource script for live updates")
	}
}

func TestSSEEndpointStreamsEvents(t *testing.T) {
	reg := provider.NewRegistry(t.TempDir())
	reg.Set("anthropic", &provider.Provider{Name: "anthropic", BaseURL: "https://api.anthropic.com/v1", APIKey: "sk-test", Auth: "bearer"})
	acc := cost.NewAccumulator()
	acc.Record("tiverton", "anthropic", "claude-sonnet-4", 1000, 500, 0.0105)

	h := NewHandler(reg, WithAccumulator(acc))

	req := httptest.NewRequest("GET", "/events", nil)
	w := httptest.NewRecorder()

	// Run handler in goroutine since SSE blocks; cancel via context
	ctx, cancel := context.WithTimeout(req.Context(), 200*time.Millisecond)
	defer cancel()
	req = req.WithContext(ctx)

	h.ServeHTTP(w, req)

	body := w.Body.String()
	if !strings.Contains(body, "data:") {
		t.Fatal("expected SSE data line")
	}
	if !strings.Contains(body, "anthropic") {
		t.Error("expected provider name in SSE payload")
	}
	if !strings.Contains(body, "tiverton") {
		t.Error("expected agent name in SSE payload")
	}

	// Verify it's valid JSON inside the data line
	lines := strings.Split(body, "\n")
	for _, line := range lines {
		if strings.HasPrefix(line, "data:") {
			jsonStr := strings.TrimPrefix(line, "data:")
			var state map[string]interface{}
			if err := json.Unmarshal([]byte(jsonStr), &state); err != nil {
				t.Fatalf("SSE data is not valid JSON: %v", err)
			}
			if _, ok := state["providers"]; !ok {
				t.Error("expected 'providers' key in state")
			}
			if _, ok := state["agents"]; !ok {
				t.Error("expected 'agents' key in state")
			}
			if _, ok := state["totalCostUSD"]; !ok {
				t.Error("expected 'totalCostUSD' key in state")
			}
			break
		}
	}
}
