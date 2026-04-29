package proxy

import (
	"os"
	"path/filepath"
	"testing"
)

func TestChannelCursorStoreRoundTripsAndAdvancesMonotonically(t *testing.T) {
	store := newChannelCursorStore(t.TempDir())
	if err := store.Commit("agent-1", map[string]channelCursor{
		"chan-1": {LastMessageID: "101"},
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	if err := store.Commit("agent-1", map[string]channelCursor{
		"chan-1": {LastMessageID: "100"},
		"chan-2": {LastMessageID: "200"},
	}); err != nil {
		t.Fatalf("second commit: %v", err)
	}
	got, err := store.Load("agent-1")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got["chan-1"].LastMessageID != "101" {
		t.Fatalf("cursor regressed: %+v", got)
	}
	if got["chan-2"].LastMessageID != "200" {
		t.Fatalf("cursor missing chan-2: %+v", got)
	}
	if _, err := os.Stat(filepath.Join(store.dir, "agent-1", "cursor.json")); err != nil {
		t.Fatalf("expected cursor file: %v", err)
	}
}

func TestChannelCursorStoreMemoryFallback(t *testing.T) {
	store := newChannelCursorStore("")
	if err := store.Commit("agent-1", map[string]channelCursor{
		"chan-1": {LastMessageID: "101"},
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	got, err := store.Load("agent-1")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if got["chan-1"].LastMessageID != "101" {
		t.Fatalf("unexpected memory cursor: %+v", got)
	}
}
