package proxy

import (
	"testing"

	"github.com/mostlydev/cllama/internal/agentctx"
)

func TestResolveManagedToolAcceptsUniqueHashlessAlias(t *testing.T) {
	agentCtx := &agentctx.AgentContext{
		Tools: &agentctx.ToolManifest{
			Tools: []agentctx.ToolManifestEntry{
				{Name: "trading-api.get_quote"},
				{Name: "claw-wall.search_channel_context"},
			},
		},
	}

	resolved, ok := resolveManagedTool(agentCtx, "claw-wall_search_channel_context")
	if !ok {
		t.Fatal("expected unique hashless alias to resolve")
	}
	if resolved.CanonicalName != "claw-wall.search_channel_context" {
		t.Fatalf("canonical name = %q, want claw-wall.search_channel_context", resolved.CanonicalName)
	}
	if resolved.PresentedName != "claw-wall_search_channel_context" {
		t.Fatalf("presented name = %q, want hash-free claw-wall_search_channel_context", resolved.PresentedName)
	}
}

func TestManagedToolPresentedNamesHashFreeWhenUnique(t *testing.T) {
	tools := []agentctx.ToolManifestEntry{
		{Name: "trading-api.propose_trade"},
		{Name: "claw-wall.search_channel_context"},
		{Name: "already_safe"},
	}
	names := managedToolPresentedNames(tools)
	if names["trading-api.propose_trade"] != "trading-api_propose_trade" {
		t.Fatalf("propose_trade presented = %q, want trading-api_propose_trade", names["trading-api.propose_trade"])
	}
	if names["claw-wall.search_channel_context"] != "claw-wall_search_channel_context" {
		t.Fatalf("search presented = %q, want claw-wall_search_channel_context", names["claw-wall.search_channel_context"])
	}
	if names["already_safe"] != "already_safe" {
		t.Fatalf("provider-safe name must pass through, got %q", names["already_safe"])
	}
}

func TestManagedToolPresentedNamesCollisionKeepsHash(t *testing.T) {
	tools := []agentctx.ToolManifestEntry{
		{Name: "svc.one"},
		{Name: "svc/one"},
	}
	names := managedToolPresentedNames(tools)
	if names["svc.one"] != managedToolHashedNameForCanonical("svc.one") {
		t.Fatalf("colliding name must keep hash, got %q", names["svc.one"])
	}
	if names["svc/one"] != managedToolHashedNameForCanonical("svc/one") {
		t.Fatalf("colliding name must keep hash, got %q", names["svc/one"])
	}
	if names["svc.one"] == names["svc/one"] {
		t.Fatal("hashed forms must remain distinct")
	}
}

func TestManagedToolPresentedNamesCanonicalCollisionKeepsHash(t *testing.T) {
	tools := []agentctx.ToolManifestEntry{
		{Name: "svc_tool"},
		{Name: "svc.tool"},
	}
	names := managedToolPresentedNames(tools)
	if names["svc_tool"] != "svc_tool" {
		t.Fatalf("provider-safe canonical must pass through, got %q", names["svc_tool"])
	}
	if names["svc.tool"] != managedToolHashedNameForCanonical("svc.tool") {
		t.Fatalf("hashless candidate colliding with a canonical must keep hash, got %q", names["svc.tool"])
	}
}

func TestResolveManagedToolAcceptsLegacyHashedName(t *testing.T) {
	agentCtx := &agentctx.AgentContext{
		Tools: &agentctx.ToolManifest{
			Tools: []agentctx.ToolManifestEntry{
				{Name: "claw-wall.search_channel_context"},
			},
		},
	}
	legacy := managedToolHashedNameForCanonical("claw-wall.search_channel_context")
	resolved, ok := resolveManagedTool(agentCtx, legacy)
	if !ok {
		t.Fatalf("legacy hashed name %q must keep resolving for session-history replays", legacy)
	}
	if resolved.CanonicalName != "claw-wall.search_channel_context" {
		t.Fatalf("canonical = %q", resolved.CanonicalName)
	}
}

func TestResolveManagedToolKeepsAmbiguousHashlessAliasUnresolved(t *testing.T) {
	agentCtx := &agentctx.AgentContext{
		Tools: &agentctx.ToolManifest{
			Tools: []agentctx.ToolManifestEntry{
				{Name: "svc.one"},
				{Name: "svc/one"},
			},
		},
	}

	if resolved, ok := resolveManagedTool(agentCtx, "svc_one"); ok {
		t.Fatalf("ambiguous hashless alias resolved unexpectedly: %+v", resolved)
	}
}
