package sessionhistory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
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

func TestReadEntriesCreatesHistoryIndex(t *testing.T) {
	dir := t.TempDir()
	r := New(dir)
	entry := Entry{
		Version: 1,
		TS:      time.Date(2026, 3, 31, 12, 0, 0, 0, time.UTC).Format(time.RFC3339),
		ClawID:  "agent-1",
		Response: Payload{
			Format: "json",
			JSON:   json.RawMessage(`{}`),
		},
	}
	if err := r.Record("agent-1", entry); err != nil {
		t.Fatalf("Record: %v", err)
	}

	if _, err := ReadEntries(dir, "agent-1", &time.Time{}, 1); err != nil {
		t.Fatalf("ReadEntries: %v", err)
	}
	if _, err := os.Stat(filepath.Join(dir, "agent-1", "history.index.json")); err != nil {
		t.Fatalf("expected history index file to exist: %v", err)
	}
}

func TestReadStartOffsetUsesCheckpointIndex(t *testing.T) {
	dir := t.TempDir()
	agentDir := filepath.Join(dir, "agent-1")
	if err := os.MkdirAll(agentDir, 0o755); err != nil {
		t.Fatal(err)
	}

	var raw strings.Builder
	base := time.Date(2026, 3, 31, 12, 0, 0, 0, time.UTC)
	for i := 0; i < historyIndexCheckpointEvery*3; i++ {
		raw.WriteString(`{"version":1,"ts":"`)
		raw.WriteString(base.Add(time.Duration(i) * time.Minute).Format(time.RFC3339))
		raw.WriteString(`","claw_id":"agent-1","response":{"format":"json","json":{}}}` + "\n")
	}
	historyPath := filepath.Join(agentDir, "history.jsonl")
	if err := os.WriteFile(historyPath, []byte(raw.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	after := base.Add(time.Duration(historyIndexCheckpointEvery*2) * time.Minute)
	offset, err := readStartOffset(historyPath, &after)
	if err != nil {
		t.Fatalf("readStartOffset: %v", err)
	}
	if offset <= 0 {
		t.Fatalf("expected indexed start offset > 0, got %d", offset)
	}
}
