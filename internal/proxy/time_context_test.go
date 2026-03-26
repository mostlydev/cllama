package proxy

import (
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
)

func TestCurrentTimeLineUsesMetadataTimezone(t *testing.T) {
	ctx := &agentctx.AgentContext{
		Metadata: map[string]any{
			"timezone": "America/New_York",
		},
	}

	got := currentTimeLine(ctx, time.Date(2026, 3, 25, 14, 15, 0, 0, time.UTC))
	want := "Current time: 2026-03-25 10:15 AM EDT (America/New_York)"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestCurrentTimeLineFallsBackToEnvTimezone(t *testing.T) {
	t.Setenv("TZ", "America/Chicago")

	ctx := &agentctx.AgentContext{
		Metadata: map[string]any{
			"timezone": "Mars/Olympus",
		},
	}

	got := currentTimeLine(ctx, time.Date(2026, 3, 25, 14, 15, 0, 0, time.UTC))
	want := "Current time: 2026-03-25 9:15 AM CDT (America/Chicago)"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}

func TestCurrentTimeLineFallsBackToUTC(t *testing.T) {
	t.Setenv("TZ", "")

	got := currentTimeLine(&agentctx.AgentContext{}, time.Date(2026, 3, 25, 14, 15, 0, 0, time.UTC))
	want := "Current time: 2026-03-25 2:15 PM UTC"
	if got != want {
		t.Fatalf("expected %q, got %q", want, got)
	}
}
