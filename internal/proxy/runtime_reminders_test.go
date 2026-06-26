package proxy

import (
	"strings"
	"testing"

	"github.com/mostlydev/cllama/internal/agentctx"
)

func TestRenderRuntimeRemindersFiltersAndTrims(t *testing.T) {
	disabled := false
	ctx := &agentctx.AgentContext{
		ContextDir: "/claw/context/agent",
		RuntimeReminders: &agentctx.RuntimeReminderManifest{
			Version: 1,
			Reminders: []agentctx.RuntimeReminder{
				{ID: "keep", Text: "Keep this visible.", Cadence: "every_turn", Placement: "before_feeds"},
				{ID: "trim", Text: "abcdef", Cadence: "every_turn", Placement: "before_feeds", MaxChars: 3},
				{ID: "disabled", Text: "do not render", Enabled: &disabled, Cadence: "every_turn", Placement: "before_feeds"},
				{ID: "later", Text: "not v1", Cadence: "min_interval", Placement: "before_feeds"},
				{ID: "elsewhere", Text: "not before feeds", Cadence: "every_turn", Placement: "after_feeds"},
			},
		},
	}

	block, snapshots, skips := renderRuntimeReminders(ctx)
	if !strings.Contains(block, "RUNTIME REMINDER: keep") || !strings.Contains(block, "Keep this visible.") {
		t.Fatalf("expected keep reminder in block: %q", block)
	}
	if !strings.Contains(block, "RUNTIME REMINDER: trim") || strings.Contains(block, "abcdef") || !strings.Contains(block, "abc") {
		t.Fatalf("expected trimmed reminder in block: %q", block)
	}
	for _, unexpected := range []string{"do not render", "not v1", "not before feeds"} {
		if strings.Contains(block, unexpected) {
			t.Fatalf("unexpected reminder %q in block: %q", unexpected, block)
		}
	}
	if len(snapshots) != 2 {
		t.Fatalf("expected two snapshots, got %+v", snapshots)
	}
	if snapshots[0].ID != "keep" || snapshots[0].SourcePath != "/claw/context/agent/runtime-reminders.json" || snapshots[0].CadenceDecision != "injected_every_turn" {
		t.Fatalf("unexpected first snapshot: %+v", snapshots[0])
	}
	if len(skips) != 2 {
		t.Fatalf("expected two unsupported reminder skips, got %+v", skips)
	}
	if skips[0].ID != "later" || skips[0].Reason != "unsupported_cadence" {
		t.Fatalf("unexpected first skip: %+v", skips[0])
	}
	if skips[1].ID != "elsewhere" || skips[1].Reason != "unsupported_placement" {
		t.Fatalf("unexpected second skip: %+v", skips[1])
	}
}
