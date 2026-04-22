package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
)

func TestHandlerCapturesOpenAIContextSnapshot(t *testing.T) {
	feedSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("Wallet: $5,000 cash | $20,000 invested"))
	}))
	defer feedSrv.Close()

	memorySrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"memories":[{"text":"Remember to hedge NVDA if implied vol spikes.","kind":"policy"}]}`))
	}))
	defer memorySrv.Close()

	ctxRoot := t.TempDir()
	agentDir := filepath.Join(ctxRoot, "weston")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	for name, content := range map[string]string{
		"AGENTS.md":     "# Contract",
		"CLAWDAPUS.md":  "# Infra",
		"metadata.json": `{"token":"weston:secret","pod":"test-pod","timezone":"America/New_York"}`,
		"feeds.json":    `[{"name":"market-context","source":"trading-api","path":"/api/v1/market_context/weston","ttl":300,"url":"` + feedSrv.URL + `"}]`,
		"memory.json":   `{"version":1,"service":"team-memory","base_url":"` + memorySrv.URL + `","recall":{"path":"/recall","timeout_ms":1000}}`,
	} {
		if err := os.WriteFile(filepath.Join(agentDir, name), []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[{"message":{"content":"ok"}}]}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("openrouter", &provider.Provider{
		Name: "openrouter", BaseURL: backend.URL + "/v1", APIKey: "sk-real", Auth: "bearer",
	})

	store := NewContextSnapshotStore()
	h := NewHandler(reg, func(agentID string) (*agentctx.AgentContext, error) {
		return agentctx.Load(ctxRoot, agentID)
	}, logging.New(io.Discard), WithFeeds("test-pod"), WithSnapshotStore(store))

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{
		"model":"openrouter/anthropic/claude-sonnet-4",
		"messages":[{"role":"user","content":"hi"}],
		"tools":[{"type":"function","function":{"name":"runner_local","parameters":{"type":"object"}}}]
	}`))
	req.Header.Set("Authorization", "Bearer weston:secret")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	snapshot, ok := store.Get("weston")
	if !ok {
		t.Fatal("expected snapshot for weston")
	}
	if snapshot.Format != "openai" {
		t.Fatalf("Format = %q; want openai", snapshot.Format)
	}
	if snapshot.RequestedModel != "openrouter/anthropic/claude-sonnet-4" {
		t.Fatalf("RequestedModel = %q", snapshot.RequestedModel)
	}
	if snapshot.ChosenRef != "openrouter/anthropic/claude-sonnet-4" {
		t.Fatalf("ChosenRef = %q", snapshot.ChosenRef)
	}
	if len(snapshot.Candidates) != 1 || snapshot.Candidates[0].Provider != "openrouter" || snapshot.Candidates[0].UpstreamModel != "anthropic/claude-sonnet-4" {
		t.Fatalf("unexpected Candidates: %+v", snapshot.Candidates)
	}
	if snapshot.ManagedTool {
		t.Fatalf("expected ManagedTool=false, got %+v", snapshot)
	}
	if snapshot.TurnCount != 1 {
		t.Fatalf("TurnCount = %d; want 1", snapshot.TurnCount)
	}
	if len(snapshot.FeedBlocks) != 1 || !strings.Contains(snapshot.FeedBlocks[0], "Wallet: $5,000") {
		t.Fatalf("unexpected FeedBlocks: %+v", snapshot.FeedBlocks)
	}
	if !strings.Contains(snapshot.MemoryRecall, "Remember to hedge NVDA") {
		t.Fatalf("unexpected MemoryRecall: %q", snapshot.MemoryRecall)
	}
	if !strings.Contains(snapshot.TimeContext, "Current time:") {
		t.Fatalf("unexpected TimeContext: %q", snapshot.TimeContext)
	}

	system, ok := snapshot.System.(string)
	if !ok {
		t.Fatalf("expected string system content, got %T", snapshot.System)
	}
	if !strings.Contains(system, "BEGIN MEMORY") || !strings.Contains(system, "BEGIN FEED: market-context") || !strings.Contains(system, "Current time:") {
		t.Fatalf("unexpected system content: %q", system)
	}
	if memoryIdx := strings.Index(system, "BEGIN MEMORY"); memoryIdx == -1 {
		t.Fatalf("memory block missing from system: %q", system)
	} else if feedIdx := strings.Index(system, "BEGIN FEED:"); feedIdx < memoryIdx {
		t.Fatalf("expected memory before feeds, got system=%q", system)
	} else if timeIdx := strings.Index(system, "Current time:"); timeIdx < feedIdx {
		t.Fatalf("expected time after feeds, got system=%q", system)
	}
	if len(snapshot.Placements) != 3 {
		t.Fatalf("expected memory/feed/time placements, got %+v", snapshot.Placements)
	}
	for i, want := range []string{"memory", "feed", "time"} {
		placement := snapshot.Placements[i]
		if placement.Order != i+1 || placement.Kind != want {
			t.Fatalf("unexpected placement %d: %+v", i, placement)
		}
		if placement.Carrier != "openai.messages[0].content" || placement.Role != "system" || placement.MessageIndex != 0 {
			t.Fatalf("unexpected placement carrier: %+v", placement)
		}
		if placement.Visibility != "provider_visible" || placement.Persistence != "per_request" || placement.Relation != "system_context_before_conversation" {
			t.Fatalf("unexpected placement semantics: %+v", placement)
		}
		if placement.StartChar < 0 || placement.EndChar <= placement.StartChar || placement.Occurrences != 1 {
			t.Fatalf("unexpected placement position: %+v", placement)
		}
	}
	if !(snapshot.Placements[0].StartChar < snapshot.Placements[1].StartChar && snapshot.Placements[1].StartChar < snapshot.Placements[2].StartChar) {
		t.Fatalf("expected placement offsets in assembly order, got %+v", snapshot.Placements)
	}
	if len(snapshot.RecentCaptures) != 1 {
		t.Fatalf("expected one recent capture, got %+v", snapshot.RecentCaptures)
	}
	recent := snapshot.RecentCaptures[0]
	if recent.DynamicInputs != 3 || recent.FeedBlocks != 1 || !recent.MemoryRecall || !recent.TimeContext || recent.PlacementCount != 3 {
		t.Fatalf("unexpected recent capture summary: %+v", recent)
	}

	if len(snapshot.Tools) != 1 {
		t.Fatalf("expected 1 tool in snapshot, got %+v", snapshot.Tools)
	}
	firstTool, _ := snapshot.Tools[0].(map[string]any)
	function, _ := firstTool["function"].(map[string]any)
	if function["name"] != "runner_local" {
		t.Fatalf("unexpected snapshot tool: %+v", snapshot.Tools[0])
	}
}

func TestHandlerCapturesAnthropicContextSnapshotWithBlockSystem(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"msg_01","type":"message","content":[{"type":"text","text":"ok"}]}`))
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("anthropic", &provider.Provider{
		Name: "anthropic", BaseURL: backend.URL + "/v1", APIKey: "sk-ant-real", Auth: "x-api-key", APIFormat: "anthropic",
	})

	store := NewContextSnapshotStore()
	h := NewHandler(reg, func(agentID string) (*agentctx.AgentContext, error) {
		if agentID != "nano-bot" {
			return nil, io.EOF
		}
		return &agentctx.AgentContext{
			AgentID:     agentID,
			ContextDir:  "/claw/context/" + agentID,
			AgentsMD:    []byte("# Contract"),
			ClawdapusMD: []byte("# Infra"),
			Metadata: map[string]any{
				"token":    "nano-bot:secret456",
				"timezone": "America/New_York",
			},
		}, nil
	}, logging.New(io.Discard), WithSnapshotStore(store))

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewBufferString(`{
		"model":"claude-sonnet-4-20250514",
		"system":[{"type":"text","text":"You are a trader."}],
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Authorization", "Bearer nano-bot:secret456")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Anthropic-Version", "2023-06-01")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	snapshot, ok := store.Get("nano-bot")
	if !ok {
		t.Fatal("expected snapshot for nano-bot")
	}
	if snapshot.Format != "anthropic" {
		t.Fatalf("Format = %q; want anthropic", snapshot.Format)
	}
	if snapshot.RequestedModel != "claude-sonnet-4-20250514" || snapshot.ChosenRef != "claude-sonnet-4-20250514" {
		t.Fatalf("unexpected model fields: %+v", snapshot)
	}
	if len(snapshot.Candidates) != 1 || snapshot.Candidates[0].Provider != "anthropic" || snapshot.Candidates[0].UpstreamModel != "claude-sonnet-4-20250514" {
		t.Fatalf("unexpected Candidates: %+v", snapshot.Candidates)
	}
	if snapshot.TurnCount != 1 {
		t.Fatalf("TurnCount = %d; want 1", snapshot.TurnCount)
	}

	blocks, ok := snapshot.System.([]any)
	if !ok || len(blocks) != 2 {
		t.Fatalf("expected 2 anthropic system blocks, got %#v", snapshot.System)
	}
	first, _ := blocks[0].(map[string]any)
	if first["text"] != "You are a trader." {
		t.Fatalf("unexpected first system block: %+v", first)
	}
	second, _ := blocks[1].(map[string]any)
	if second["type"] != "text" || !strings.Contains(second["text"].(string), "Current time:") {
		t.Fatalf("unexpected appended system block: %+v", second)
	}
	if len(snapshot.Placements) != 1 {
		t.Fatalf("expected time placement, got %+v", snapshot.Placements)
	}
	placement := snapshot.Placements[0]
	if placement.Kind != "time" || placement.Carrier != "anthropic.system[1].text" || placement.BlockIndex != 1 || placement.StartChar != 0 {
		t.Fatalf("unexpected anthropic placement: %+v", placement)
	}
}

func TestHandlerManagedSnapshotTracksCompletedTurnCount(t *testing.T) {
	presentedName := managedToolPresentedNameForCanonical("trading-api.get_market_context")
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"balance":5000}`))
	}))
	defer toolSrv.Close()

	xaiRounds := 0
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		xaiRounds++
		w.Header().Set("Content-Type", "application/json")
		switch xaiRounds {
		case 1:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-1",
				"choices":[{
					"finish_reason":"tool_calls",
					"message":{
						"role":"assistant",
						"tool_calls":[{"id":"call_1","type":"function","function":{"name":"` + presentedName + `","arguments":"{}"}}]
					}
				}]
			}`))
		case 2:
			_, _ = w.Write([]byte(`{
				"id":"chatcmpl-2",
				"choices":[{"message":{"role":"assistant","content":"market context loaded"}}]
			}`))
		default:
			t.Fatalf("unexpected xai round %d", xaiRounds)
		}
	}))
	defer backend.Close()

	reg := provider.NewRegistry("")
	reg.Set("xai", &provider.Provider{
		Name: "xai", BaseURL: backend.URL + "/v1", APIKey: "xai-real", Auth: "bearer",
	})

	store := NewContextSnapshotStore()
	h := NewHandler(reg, stubContextLoaderWithTools("tiverton", "tiverton:dummy123", managedToolManifestForURL(toolSrv.URL, http.MethodGet, "/api/v1/market_context/{claw_id}", "")), logging.New(io.Discard), WithSnapshotStore(store))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewBufferString(`{
		"model":"xai/grok-4.1-fast",
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Authorization", "Bearer tiverton:dummy123")
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	h.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "market context loaded") {
		t.Fatalf("unexpected response body: %s", w.Body.String())
	}

	snapshot, ok := store.Get("tiverton")
	if !ok {
		t.Fatal("expected snapshot for tiverton")
	}
	if !snapshot.ManagedTool {
		t.Fatalf("expected ManagedTool=true, got %+v", snapshot)
	}
	if snapshot.TurnCount != 2 {
		t.Fatalf("TurnCount = %d; want 2", snapshot.TurnCount)
	}
	if len(snapshot.Tools) != 1 {
		t.Fatalf("expected managed tool schema captured, got %+v", snapshot.Tools)
	}
}

func TestContextSnapshotStoreAgentIDsSorted(t *testing.T) {
	store := NewContextSnapshotStore()
	store.Put(ContextSnapshot{AgentID: "bravo"})
	store.Put(ContextSnapshot{AgentID: "alpha"})

	raw, err := json.Marshal(struct {
		Agents []string `json:"agents"`
	}{
		Agents: store.AgentIDs(),
	})
	if err != nil {
		t.Fatalf("marshal store ids: %v", err)
	}
	if string(raw) != `{"agents":["alpha","bravo"]}` {
		t.Fatalf("unexpected agent id order: %s", raw)
	}
}

func TestContextSnapshotStoreKeepsRecentCaptureSummaries(t *testing.T) {
	store := NewContextSnapshotStore()
	store.Put(ContextSnapshot{
		AgentID:        "alpha",
		CapturedAt:     time.Date(2026, 4, 17, 20, 0, 0, 0, time.UTC),
		Format:         "openai",
		RequestedModel: "openrouter/model-a",
		FeedBlocks:     []string{"feed-a"},
		TimeContext:    "Current time: 2026-04-17T20:00:00Z",
		Placements: []ContextPlacement{{
			Order: 1,
			Kind:  "feed",
		}},
		TurnCount: 1,
	})
	store.Put(ContextSnapshot{
		AgentID:        "alpha",
		CapturedAt:     time.Date(2026, 4, 17, 20, 5, 0, 0, time.UTC),
		Format:         "openai",
		RequestedModel: "openrouter/model-b",
		MemoryRecall:   "memory",
		TimeContext:    "Current time: 2026-04-17T20:05:00Z",
		Placements: []ContextPlacement{{
			Order: 1,
			Kind:  "memory",
		}, {
			Order: 2,
			Kind:  "time",
		}},
		TurnCount: 1,
	})
	store.UpdateTurnCount("alpha", 2)

	snapshot, ok := store.Get("alpha")
	if !ok {
		t.Fatal("expected alpha snapshot")
	}
	if len(snapshot.RecentCaptures) != 2 {
		t.Fatalf("expected two capture summaries, got %+v", snapshot.RecentCaptures)
	}
	first, second := snapshot.RecentCaptures[0], snapshot.RecentCaptures[1]
	if first.Sequence == 0 || second.Sequence <= first.Sequence {
		t.Fatalf("expected monotonic capture sequence, got %+v", snapshot.RecentCaptures)
	}
	if first.DynamicInputs != 2 || first.FeedBlocks != 1 || first.MemoryRecall {
		t.Fatalf("unexpected first summary: %+v", first)
	}
	if second.DynamicInputs != 2 || !second.MemoryRecall || !second.TimeContext || second.PlacementCount != 2 || second.TurnCount != 2 {
		t.Fatalf("unexpected second summary: %+v", second)
	}
}
