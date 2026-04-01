package sessionhistory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestReadEntriesAppliesAfterAndLimit(t *testing.T) {
	dir := t.TempDir()
	r := New(dir)

	base := time.Date(2026, 3, 31, 12, 0, 0, 0, time.UTC)
	for i := 0; i < 3; i++ {
		entry := Entry{
			Version: 1,
			TS:      base.Add(time.Duration(i) * time.Minute).Format(time.RFC3339),
			ClawID:  "agent-1",
			Response: Payload{
				Format: "json",
				JSON:   json.RawMessage(`{}`),
			},
		}
		if err := r.Record("agent-1", entry); err != nil {
			t.Fatalf("Record(%d): %v", i, err)
		}
	}

	after := base
	entries, err := ReadEntries(dir, "agent-1", &after, 1)
	if err != nil {
		t.Fatalf("ReadEntries: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry after filtering+limit, got %d", len(entries))
	}
	if entries[0].TS != base.Add(time.Minute).Format(time.RFC3339) {
		t.Fatalf("unexpected first filtered timestamp: %+v", entries[0])
	}
	if entries[0].ID == "" {
		t.Fatalf("expected read entries to include stable ID, got %+v", entries[0])
	}
}

func TestReadEntriesMissingFileReturnsEmpty(t *testing.T) {
	entries, err := ReadEntries(t.TempDir(), "missing-agent", nil, 100)
	if err != nil {
		t.Fatalf("ReadEntries: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected no entries, got %+v", entries)
	}
}

func TestReadEntriesHydratesLegacyIDsFromRawJSON(t *testing.T) {
	dir := t.TempDir()
	agentDir := filepath.Join(dir, "agent-legacy")
	if err := os.MkdirAll(agentDir, 0o755); err != nil {
		t.Fatal(err)
	}
	raw := `{"version":1,"ts":"2026-04-01T00:00:00Z","claw_id":"agent-legacy","response":{"format":"json","json":{}}}` + "\n"
	if err := os.WriteFile(filepath.Join(agentDir, "history.jsonl"), []byte(raw), 0o644); err != nil {
		t.Fatal(err)
	}

	entries, err := ReadEntries(dir, "agent-legacy", nil, 1)
	if err != nil {
		t.Fatalf("ReadEntries: %v", err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}
	if entries[0].ID == "" {
		t.Fatalf("expected legacy entry ID to be hydrated, got %+v", entries[0])
	}
}
