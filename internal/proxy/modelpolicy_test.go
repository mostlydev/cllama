package proxy

import (
	"testing"

	"github.com/mostlydev/cllama/internal/agentctx"
)

func TestCandidateRefsFromPolicyAppendsFailoverTailForNonFailoverSlot(t *testing.T) {
	policy := &agentctx.ModelPolicy{
		Mode: "clamp",
		Allowed: []agentctx.AllowedModel{
			{Slot: "primary", Ref: "openai/gpt-4o"},
			{Slot: "fallback", Ref: "anthropic/claude-haiku-4-5"},
			{Slot: "cheap", Ref: "openai/gpt-4o-mini"},
		},
	}

	got := candidateRefsFromPolicy(policy, "openai/gpt-4o-mini")
	want := []string{
		"openai/gpt-4o-mini",
		"openai/gpt-4o",
		"anthropic/claude-haiku-4-5",
	}
	if len(got) != len(want) {
		t.Fatalf("expected %d candidate refs, got %d: %#v", len(want), len(got), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("candidate[%d] = %q, want %q (all=%#v)", i, got[i], want[i], got)
		}
	}
}
