package proxy

import (
	"strings"
	"testing"

	"github.com/mostlydev/cllama/internal/feeds"
)

func TestPrepareChannelContextFeedAddsAfterCursor(t *testing.T) {
	h := NewHandler(nil, nil, nil, WithChannelCursorLedger(""))
	if err := h.channelCursors.Commit("agent-1", ledgerCommitInput{CursorUpdates: map[string]channelCursor{
		"chan-a": {LastMessageID: "101"},
		"chan-b": {LastMessageID: "201"},
	}}); err != nil {
		t.Fatalf("commit: %v", err)
	}

	entry, decision, err := h.prepareChannelContextFeed("agent-1", feeds.FeedEntry{
		Name: "channel-context",
		URL:  "http://claw-wall:8080/channel-context?channels=chan-b,chan-a&mode=tail&since=24h",
	}, "")
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if !decision.AppliedAfter {
		t.Fatal("expected after cursor to be applied")
	}
	if decision.Bootstrapped {
		t.Fatalf("did not expect bootstrap decision: %+v", decision)
	}
	if !strings.Contains(entry.URL, "after=chan-b%3A201%2Cchan-a%3A101") {
		t.Fatalf("unexpected URL: %s", entry.URL)
	}
	if !strings.Contains(entry.URL, "context_kind=delta_tail") {
		t.Fatalf("expected delta_tail context kind, got %s", entry.URL)
	}
}

func TestPrepareChannelContextFeedBootstrapsWhenEpochMissing(t *testing.T) {
	h := NewHandler(nil, nil, nil, WithChannelCursorLedger(""))
	if err := h.channelCursors.Commit("agent-1", ledgerCommitInput{CursorUpdates: map[string]channelCursor{
		"chan-a": {LastMessageID: "101"},
	}}); err != nil {
		t.Fatalf("commit: %v", err)
	}

	entry, decision, err := h.prepareChannelContextFeed("agent-1", feeds.FeedEntry{
		Name: "channel-context",
		URL:  "http://claw-wall:8080/channel-context?channels=chan-a&mode=tail&since=24h",
	}, "epoch-1")
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if strings.Contains(entry.URL, "after=") {
		t.Fatalf("bootstrap should not include after cursor: %s", entry.URL)
	}
	if !strings.Contains(entry.URL, "context_kind=bootstrap_tail") {
		t.Fatalf("expected bootstrap_tail context kind, got %s", entry.URL)
	}
	if !decision.Bootstrapped || decision.AppliedAfter || decision.PriorEpoch != "" || decision.IncomingEpoch != "epoch-1" {
		t.Fatalf("unexpected decision: %+v", decision)
	}
}

func TestPrepareChannelContextFeedAppliesAfterWhenEpochMatches(t *testing.T) {
	h := NewHandler(nil, nil, nil, WithChannelCursorLedger(""))
	if err := h.channelCursors.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr(""),
		NewEpoch:              "epoch-1",
		CursorUpdates: map[string]channelCursor{
			"chan-a": {LastMessageID: "101"},
		},
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}

	entry, decision, err := h.prepareChannelContextFeed("agent-1", feeds.FeedEntry{
		Name: "channel-context",
		URL:  "http://claw-wall:8080/channel-context?channels=chan-a&mode=tail&since=24h",
	}, "epoch-1")
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if !decision.AppliedAfter || decision.Bootstrapped {
		t.Fatalf("unexpected decision: %+v", decision)
	}
	if !strings.Contains(entry.URL, "after=chan-a%3A101") {
		t.Fatalf("expected after cursor, got %s", entry.URL)
	}
	if !strings.Contains(entry.URL, "context_kind=delta_tail") {
		t.Fatalf("expected delta_tail context kind, got %s", entry.URL)
	}
}

func TestPrepareChannelContextFeedBootstrapsWhenEpochDiffers(t *testing.T) {
	h := NewHandler(nil, nil, nil, WithChannelCursorLedger(""))
	if err := h.channelCursors.Commit("agent-1", ledgerCommitInput{
		ExpectedPreviousEpoch: stringPtr(""),
		NewEpoch:              "epoch-1",
		CursorUpdates: map[string]channelCursor{
			"chan-a": {LastMessageID: "101"},
		},
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}

	entry, decision, err := h.prepareChannelContextFeed("agent-1", feeds.FeedEntry{
		Name: "channel-context",
		URL:  "http://claw-wall:8080/channel-context?channels=chan-a&mode=tail&since=24h",
	}, "epoch-2")
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if strings.Contains(entry.URL, "after=") {
		t.Fatalf("bootstrap should not include after cursor: %s", entry.URL)
	}
	if !strings.Contains(entry.URL, "context_kind=bootstrap_tail") {
		t.Fatalf("expected bootstrap_tail context kind, got %s", entry.URL)
	}
	if !decision.Bootstrapped || decision.AppliedAfter || decision.PriorEpoch != "epoch-1" || decision.IncomingEpoch != "epoch-2" {
		t.Fatalf("unexpected decision: %+v", decision)
	}
}

func TestParseChannelContextMetadata(t *testing.T) {
	meta := parseChannelContextMetadata("[channel-context delta] kind=delta_tail mode=tail messages=2 available=5 omitted=3 after=chan-a:100 cursor=chan-a:105 range=2026-04-29T05:00Z..2026-04-29T05:05Z\nbody")
	if meta.Available != 5 || meta.Omitted != 3 {
		t.Fatalf("unexpected counts: %+v", meta)
	}
	if meta.Kind != "delta_tail" || meta.Returned != 2 || meta.Retained != 5 {
		t.Fatalf("unexpected metadata fields: %+v", meta)
	}
	if meta.Cursor["chan-a"].LastMessageID != "105" {
		t.Fatalf("unexpected cursor: %+v", meta.Cursor)
	}
	if meta.RangeEnd != "2026-04-29T05:05Z" {
		t.Fatalf("unexpected range end: %q", meta.RangeEnd)
	}
}

func TestParseChannelAwarenessMetadata(t *testing.T) {
	meta := parseChannelContextMetadata("[channel-awareness] kind=raw_window since=24h channels=chan-a,chan-b messages=40 available=60 omitted=20 retained=60/since-24h digest=unavailable\nbody")
	if meta.Kind != "raw_window" || meta.Returned != 40 || meta.Retained != 60 || meta.Omitted != 20 || meta.Status != "ok" {
		t.Fatalf("unexpected awareness metadata: %+v", meta)
	}
	if len(meta.Channels) != 2 || meta.Channels[0] != "chan-a" || meta.Channels[1] != "chan-b" {
		t.Fatalf("unexpected channels: %+v", meta.Channels)
	}
}
