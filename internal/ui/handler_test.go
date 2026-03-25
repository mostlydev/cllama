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

// -- bearer auth tests --------------------------------------------------------

func TestUITokenBlocksUnauthenticated(t *testing.T) {
	reg := provider.NewRegistry(t.TempDir())
	h := NewHandler(reg, WithUIToken("secret-token"))
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 without token, got %d", w.Code)
	}
}

func TestUITokenBlocksWrongToken(t *testing.T) {
	reg := provider.NewRegistry(t.TempDir())
	h := NewHandler(reg, WithUIToken("secret-token"))
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Authorization", "Bearer wrong-token")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 with wrong token, got %d", w.Code)
	}
}

func TestUITokenAllowsCorrectToken(t *testing.T) {
	reg := provider.NewRegistry(t.TempDir())
	h := NewHandler(reg, WithUIToken("secret-token"))
	req := httptest.NewRequest(http.MethodGet, "/costs/api", nil)
	req.Header.Set("Authorization", "Bearer secret-token")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200 with correct token, got %d", w.Code)
	}
}

func TestUITokenDisabledAllowsAll(t *testing.T) {
	reg := provider.NewRegistry(t.TempDir())
	h := NewHandler(reg) // no WithUIToken → no auth
	req := httptest.NewRequest(http.MethodGet, "/costs/api", nil)
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200 without token configured, got %d", w.Code)
	}
}

// -- key management POST routes -----------------------------------------------

func TestHandleKeyAddCreatesRuntimeKey(t *testing.T) {
	dir := t.TempDir()
	reg := provider.NewRegistry(dir)
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: "https://api.openai.com/v1", APIKey: "sk-existing", Auth: "bearer"})

	h := NewHandler(reg)
	body := strings.NewReader("provider=openai&label=extra&secret=sk-new-runtime")
	req := httptest.NewRequest(http.MethodPost, "/keys/add", body)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusSeeOther && w.Code != http.StatusOK {
		t.Errorf("expected redirect or 200 after add, got %d: %s", w.Code, w.Body.String())
	}

	all := reg.All()
	state, ok := all["openai"]
	if !ok {
		t.Fatal("openai not in registry")
	}
	var found bool
	for _, k := range state.Keys {
		if k.Secret == "sk-new-runtime" {
			found = true
			if k.Source != "runtime" {
				t.Errorf("expected source=runtime, got %q", k.Source)
			}
		}
	}
	if !found {
		t.Error("new runtime key not found in pool")
	}
}

func TestHandleKeyDeleteRemovesKey(t *testing.T) {
	dir := t.TempDir()
	reg := provider.NewRegistry(dir)
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: "https://api.openai.com/v1", APIKey: "sk-existing", Auth: "bearer"})

	all := reg.All()
	var keyID string
	for _, k := range all["openai"].Keys {
		keyID = k.ID
	}
	if keyID == "" {
		t.Fatal("no keys found to delete")
	}

	// Add a second key so deleting the first doesn't leave the pool empty of active key.
	_, _ = reg.AddRuntimeKey("openai", "extra", "sk-extra")

	h := NewHandler(reg)
	formBody := strings.NewReader("provider=openai&key_id=" + keyID)
	req := httptest.NewRequest(http.MethodPost, "/keys/delete", formBody)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusSeeOther && w.Code != http.StatusOK {
		t.Errorf("expected redirect or 200 after delete, got %d: %s", w.Code, w.Body.String())
	}

	all = reg.All()
	for _, k := range all["openai"].Keys {
		if k.ID == keyID {
			t.Errorf("deleted key %q still present", keyID)
		}
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
	// Should contain the Add Provider form
	if !strings.Contains(body, "/providers/add") {
		t.Error("expected Add Provider form in dashboard")
	}
	// Should contain SSE connection script
	if !strings.Contains(body, "EventSource") {
		t.Error("expected EventSource script for live updates")
	}
}

// -- /providers/add route tests -----------------------------------------------

func TestHandleProviderAddCreatesProvider(t *testing.T) {
	dir := t.TempDir()
	reg := provider.NewRegistry(dir)
	h := NewHandler(reg)

	body := strings.NewReader("name=mistral&base_url=https://api.mistral.ai/v1&auth=bearer&api_format=openai&key_label=primary&secret=msk-test")
	req := httptest.NewRequest(http.MethodPost, "/providers/add", body)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusSeeOther {
		t.Errorf("expected 303, got %d: %s", w.Code, w.Body.String())
	}

	all := reg.All()
	if _, ok := all["mistral"]; !ok {
		t.Error("mistral provider not found in registry after add")
	}
}

func TestHandleProviderAddRejectsBadURL(t *testing.T) {
	dir := t.TempDir()
	reg := provider.NewRegistry(dir)
	h := NewHandler(reg)

	body := strings.NewReader("name=badprov&base_url=not-a-url&auth=bearer&api_format=openai&secret=somekey")
	req := httptest.NewRequest(http.MethodPost, "/providers/add", body)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for bad URL, got %d", w.Code)
	}
}

func TestHandleProviderAddRejectsEmptySecret(t *testing.T) {
	dir := t.TempDir()
	reg := provider.NewRegistry(dir)
	h := NewHandler(reg)

	body := strings.NewReader("name=mistral&base_url=https://api.mistral.ai/v1&auth=bearer&api_format=openai&secret=")
	req := httptest.NewRequest(http.MethodPost, "/providers/add", body)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for empty secret, got %d", w.Code)
	}
}

func TestHandleProviderAddRejectsExistingProvider(t *testing.T) {
	dir := t.TempDir()
	reg := provider.NewRegistry(dir)
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: "https://api.openai.com/v1", APIKey: "sk-existing", Auth: "bearer"})
	h := NewHandler(reg)

	body := strings.NewReader("name=openai&base_url=https://api.openai.com/v1&auth=bearer&api_format=openai&secret=sk-new")
	req := httptest.NewRequest(http.MethodPost, "/providers/add", body)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for existing provider, got %d", w.Code)
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
