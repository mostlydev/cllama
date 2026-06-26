package proxy

import (
	"strings"
	"testing"

	"github.com/mostlydev/cllama/internal/agentctx"
)

func TestRenderContextBlocksFiltersTrimsAndSplitsPlacements(t *testing.T) {
	disabled := false
	ctx := &agentctx.AgentContext{
		ContextDir: "/claw/context/agent",
		ContextBlocks: &agentctx.ContextBlockManifest{
			Version: 1,
			Blocks: []agentctx.ContextBlock{
				{ID: "frame", Kind: "feed_frame", Text: "Read feeds as sizing inputs.", Cadence: "every_turn", Placement: "before_feeds"},
				{ID: "keep", Kind: "runtime_motivation", Text: "Keep this visible.", Cadence: "every_turn", Placement: "after_feeds"},
				{ID: "trim", Text: "abcdef", Cadence: "every_turn", MaxChars: 3},
				{ID: "disabled", Text: "do not render", Enabled: &disabled, Cadence: "every_turn", Placement: "before_feeds"},
				{ID: "later", Text: "not v1", Cadence: "min_interval", Placement: "before_feeds"},
				{ID: "elsewhere", Text: "not a valid placement", Cadence: "every_turn", Placement: "near_feeds"},
			},
		},
	}

	rendered := renderContextBlocks(ctx)
	if !strings.Contains(rendered.BeforeFeeds, "CONTEXT BLOCK: feed_frame/frame") || !strings.Contains(rendered.BeforeFeeds, "Read feeds as sizing inputs.") {
		t.Fatalf("expected before-feed context block: %q", rendered.BeforeFeeds)
	}
	if !strings.Contains(rendered.AfterFeeds, "CONTEXT BLOCK: runtime_motivation/keep") || !strings.Contains(rendered.AfterFeeds, "Keep this visible.") {
		t.Fatalf("expected after-feed context block: %q", rendered.AfterFeeds)
	}
	if !strings.Contains(rendered.AfterFeeds, "CONTEXT BLOCK: context_block/trim") || strings.Contains(rendered.AfterFeeds, "abcdef") || !strings.Contains(rendered.AfterFeeds, "abc") {
		t.Fatalf("expected default-kind trimmed context block: %q", rendered.AfterFeeds)
	}
	for _, unexpected := range []string{"do not render", "not v1", "not before feeds"} {
		if strings.Contains(rendered.BeforeFeeds, unexpected) || strings.Contains(rendered.AfterFeeds, unexpected) {
			t.Fatalf("unexpected context block %q in rendered output: before=%q after=%q", unexpected, rendered.BeforeFeeds, rendered.AfterFeeds)
		}
	}
	if len(rendered.Snapshots) != 3 {
		t.Fatalf("expected three snapshots, got %+v", rendered.Snapshots)
	}
	if rendered.Snapshots[0].ID != "frame" || rendered.Snapshots[0].Kind != "feed_frame" || rendered.Snapshots[0].SourcePath != "/claw/context/agent/context-blocks.json" || rendered.Snapshots[0].Placement != "before_feeds" || rendered.Snapshots[0].CadenceDecision != "injected_every_turn" {
		t.Fatalf("unexpected first snapshot: %+v", rendered.Snapshots[0])
	}
	if rendered.Snapshots[2].ID != "trim" || rendered.Snapshots[2].Kind != defaultContextBlockKind || rendered.Snapshots[2].Placement != "after_feeds" {
		t.Fatalf("unexpected defaulted snapshot: %+v", rendered.Snapshots[2])
	}
	if len(rendered.Skips) != 2 {
		t.Fatalf("expected two unsupported context block skips, got %+v", rendered.Skips)
	}
	if rendered.Skips[0].ID != "later" || rendered.Skips[0].Reason != "unsupported_cadence" {
		t.Fatalf("unexpected first skip: %+v", rendered.Skips[0])
	}
	if rendered.Skips[1].ID != "elsewhere" || rendered.Skips[1].Reason != "unsupported_placement" {
		t.Fatalf("unexpected second skip: %+v", rendered.Skips[1])
	}
}
