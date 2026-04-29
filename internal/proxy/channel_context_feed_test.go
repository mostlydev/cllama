package proxy

import (
	"strings"
	"testing"

	"github.com/mostlydev/cllama/internal/feeds"
)

func TestPrepareChannelContextFeedAddsAfterCursor(t *testing.T) {
	h := NewHandler(nil, nil, nil, WithChannelCursorLedger(""))
	if err := h.channelCursors.Commit("agent-1", map[string]channelCursor{
		"chan-a": {LastMessageID: "101"},
		"chan-b": {LastMessageID: "201"},
	}); err != nil {
		t.Fatalf("commit: %v", err)
	}

	entry, applied, err := h.prepareChannelContextFeed("agent-1", feeds.FeedEntry{
		Name: "channel-context",
		URL:  "http://claw-wall:8080/channel-context?channels=chan-b,chan-a&mode=tail&since=24h",
	})
	if err != nil {
		t.Fatalf("prepare: %v", err)
	}
	if !applied {
		t.Fatal("expected after cursor to be applied")
	}
	if !strings.Contains(entry.URL, "after=chan-b%3A201%2Cchan-a%3A101") {
		t.Fatalf("unexpected URL: %s", entry.URL)
	}
}

func TestParseChannelContextMetadata(t *testing.T) {
	meta := parseChannelContextMetadata("[channel-context] mode=tail messages=2 available=5 omitted=3 after=chan-a:100 cursor=chan-a:105 range=2026-04-29T05:00Z..2026-04-29T05:05Z\nbody")
	if meta.Available != 5 || meta.Omitted != 3 {
		t.Fatalf("unexpected counts: %+v", meta)
	}
	if meta.Cursor["chan-a"].LastMessageID != "105" {
		t.Fatalf("unexpected cursor: %+v", meta.Cursor)
	}
	if meta.RangeEnd != "2026-04-29T05:05Z" {
		t.Fatalf("unexpected range end: %q", meta.RangeEnd)
	}
}
