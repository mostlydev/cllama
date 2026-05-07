package proxy

import (
	"os"
	"path/filepath"
	"testing"
)

func TestChannelCursorStoreRoundTripsAndAdvancesMonotonically(t *testing.T) {
	store := newChannelCursorStore(t.TempDir())
	if err := store.Commit("agent-1", ledgerCommitInput{CursorUpdates: map[string]channelCursor{
		"chan-1": {LastMessageID: "101"},
	}}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	if err := store.Commit("agent-1", ledgerCommitInput{CursorUpdates: map[string]channelCursor{
		"chan-1": {LastMessageID: "100"},
		"chan-2": {LastMessageID: "200"},
	}}); err != nil {
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
	if err := store.Commit("agent-1", ledgerCommitInput{CursorUpdates: map[string]channelCursor{
		"chan-1": {LastMessageID: "101"},
	}}); err != nil {
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

func TestChannelCursorStoreEpochRoundTrip(t *testing.T) {
	store := newChannelCursorStore(t.TempDir())
	if err := store.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr(""),
		NewEpoch:              "epoch-1",
		CursorUpdates: map[string]channelCursor{
			"chan-1": {LastMessageID: "101"},
		},
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	snapshot, err := store.LoadSnapshot("agent-1")
	if err != nil {
		t.Fatalf("load snapshot: %v", err)
	}
	if snapshot.Epoch != "epoch-1" {
		t.Fatalf("unexpected epoch: %q", snapshot.Epoch)
	}
	if snapshot.Cursors["chan-1"].LastMessageID != "101" {
		t.Fatalf("unexpected cursor: %+v", snapshot.Cursors)
	}
}

func TestChannelCursorStoreEpochCASApplies(t *testing.T) {
	store := newChannelCursorStore(t.TempDir())
	if err := store.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr(""),
		NewEpoch:              "epoch-1",
	}); err != nil {
		t.Fatalf("seed epoch: %v", err)
	}
	if err := store.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr("epoch-1"),
		NewEpoch:              "epoch-2",
		CursorUpdates: map[string]channelCursor{
			"chan-1": {LastMessageID: "101"},
		},
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	snapshot, err := store.LoadSnapshot("agent-1")
	if err != nil {
		t.Fatalf("load snapshot: %v", err)
	}
	if snapshot.Epoch != "epoch-2" {
		t.Fatalf("expected epoch-2, got %q", snapshot.Epoch)
	}
	if snapshot.Cursors["chan-1"].LastMessageID != "101" {
		t.Fatalf("unexpected cursor: %+v", snapshot.Cursors)
	}
}

func TestChannelCursorStoreEpochCASRejects(t *testing.T) {
	store := newChannelCursorStore(t.TempDir())
	if err := store.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr(""),
		NewEpoch:              "epoch-1",
	}); err != nil {
		t.Fatalf("seed epoch: %v", err)
	}
	if err := store.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr("stale"),
		NewEpoch:              "epoch-2",
		CursorUpdates: map[string]channelCursor{
			"chan-2": {LastMessageID: "202"},
		},
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	snapshot, err := store.LoadSnapshot("agent-1")
	if err != nil {
		t.Fatalf("load snapshot: %v", err)
	}
	if snapshot.Epoch != "epoch-1" {
		t.Fatalf("stale CAS changed epoch to %q", snapshot.Epoch)
	}
	if snapshot.Cursors["chan-2"].LastMessageID != "202" {
		t.Fatalf("cursor updates should still merge: %+v", snapshot.Cursors)
	}
}

func TestChannelCursorStoreEpochCASFromEmptyApplies(t *testing.T) {
	store := newChannelCursorStore(t.TempDir())
	if err := store.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr(""),
		NewEpoch:              "epoch-1",
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	snapshot, err := store.LoadSnapshot("agent-1")
	if err != nil {
		t.Fatalf("load snapshot: %v", err)
	}
	if snapshot.Epoch != "epoch-1" {
		t.Fatalf("expected epoch-1, got %q", snapshot.Epoch)
	}
}

func TestChannelCursorStoreEpochCASFromEmptyRejectsAfterRace(t *testing.T) {
	store := newChannelCursorStore(t.TempDir())
	if err := store.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr(""),
		NewEpoch:              "epoch-1",
		CursorUpdates: map[string]channelCursor{
			"chan-1": {LastMessageID: "101"},
		},
	}); err != nil {
		t.Fatalf("first commit: %v", err)
	}
	if err := store.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr(""),
		NewEpoch:              "epoch-2",
		CursorUpdates: map[string]channelCursor{
			"chan-1": {LastMessageID: "102"},
		},
	}); err != nil {
		t.Fatalf("second commit: %v", err)
	}
	snapshot, err := store.LoadSnapshot("agent-1")
	if err != nil {
		t.Fatalf("load snapshot: %v", err)
	}
	if snapshot.Epoch != "epoch-1" {
		t.Fatalf("second empty-CAS commit should not win epoch, got %q", snapshot.Epoch)
	}
	if snapshot.Cursors["chan-1"].LastMessageID != "102" {
		t.Fatalf("cursor should still advance monotonically: %+v", snapshot.Cursors)
	}
}

func TestChannelCursorStoreEpochUnchangedWhenNewEpochEmpty(t *testing.T) {
	store := newChannelCursorStore(t.TempDir())
	if err := store.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr(""),
		NewEpoch:              "epoch-1",
	}); err != nil {
		t.Fatalf("seed epoch: %v", err)
	}
	if err := store.Commit("agent-1", ledgerCommitInput{
		CursorUpdates: map[string]channelCursor{
			"chan-1": {LastMessageID: "101"},
		},
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	snapshot, err := store.LoadSnapshot("agent-1")
	if err != nil {
		t.Fatalf("load snapshot: %v", err)
	}
	if snapshot.Epoch != "epoch-1" {
		t.Fatalf("epoch changed: %q", snapshot.Epoch)
	}
}

func TestChannelCursorStoreEpochOnlyCommit(t *testing.T) {
	store := newChannelCursorStore(t.TempDir())
	if err := store.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr(""),
		NewEpoch:              "epoch-1",
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	snapshot, err := store.LoadSnapshot("agent-1")
	if err != nil {
		t.Fatalf("load snapshot: %v", err)
	}
	if snapshot.Epoch != "epoch-1" {
		t.Fatalf("unexpected epoch: %q", snapshot.Epoch)
	}
}

func TestChannelCursorStoreCursorOnlyCommitNoCAS(t *testing.T) {
	store := newChannelCursorStore(t.TempDir())
	if err := store.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr(""),
		NewEpoch:              "epoch-1",
	}); err != nil {
		t.Fatalf("seed epoch: %v", err)
	}
	if err := store.Commit("agent-1", ledgerCommitInput{
		CursorUpdates: map[string]channelCursor{
			"chan-1": {LastMessageID: "101"},
		},
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}
	snapshot, err := store.LoadSnapshot("agent-1")
	if err != nil {
		t.Fatalf("load snapshot: %v", err)
	}
	if snapshot.Epoch != "epoch-1" {
		t.Fatalf("cursor-only commit changed epoch: %q", snapshot.Epoch)
	}
	if snapshot.Cursors["chan-1"].LastMessageID != "101" {
		t.Fatalf("unexpected cursor: %+v", snapshot.Cursors)
	}
}

func stringPtr(value string) *string {
	return &value
}
