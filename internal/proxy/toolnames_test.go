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

func TestResolveManagedToolCanonicalBeatsLegacyHashedShadow(t *testing.T) {
	// A tool canonically named like another tool's legacy hashed form must win
	// on canonical match; the legacy alias must not shadow it.
	legacy := managedToolHashedNameForCanonical("svc.foo")
	agentCtx := &agentctx.AgentContext{
		Tools: &agentctx.ToolManifest{
			Tools: []agentctx.ToolManifestEntry{
				{Name: "svc.foo"},
				{Name: legacy},
			},
		},
	}
	resolved, ok := resolveManagedTool(agentCtx, legacy)
	if !ok {
		t.Fatalf("expected canonical match for %q", legacy)
	}
	if resolved.CanonicalName != legacy {
		t.Fatalf("canonical match must beat legacy hashed alias: got %q, want %q", resolved.CanonicalName, legacy)
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

func TestRewriteManagedToolChoiceUsesPresentedNames(t *testing.T) {
	agentCtx := &agentctx.AgentContext{
		Tools: &agentctx.ToolManifest{
			Tools: []agentctx.ToolManifestEntry{
				{Name: "trading-api.propose_trade"},
			},
		},
	}
	legacy := managedToolHashedNameForCanonical("trading-api.propose_trade")

	for name, emitted := range map[string]string{
		"canonical":     "trading-api.propose_trade",
		"legacy hashed": legacy,
		"hash-free":     "trading-api_propose_trade",
	} {
		openai := rewriteManagedOpenAIToolChoice(map[string]any{
			"type":     "function",
			"function": map[string]any{"name": emitted},
		}, agentCtx)
		function, _ := openai.(map[string]any)["function"].(map[string]any)
		if function["name"] != "trading-api_propose_trade" {
			t.Fatalf("%s: openai tool_choice rewrote to %q, want hash-free presented", name, function["name"])
		}
		anthropic := rewriteManagedAnthropicToolChoice(map[string]any{
			"type": "tool",
			"name": emitted,
		}, agentCtx)
		if anthropic.(map[string]any)["name"] != "trading-api_propose_trade" {
			t.Fatalf("%s: anthropic tool_choice rewrote to %q, want hash-free presented", name, anthropic.(map[string]any)["name"])
		}
	}
}

func TestRewriteManagedToolChoiceKeepsHashOnCollision(t *testing.T) {
	agentCtx := &agentctx.AgentContext{
		Tools: &agentctx.ToolManifest{
			Tools: []agentctx.ToolManifestEntry{
				{Name: "svc.one"},
				{Name: "svc/one"},
			},
		},
	}
	rewritten := rewriteManagedOpenAIToolChoice(map[string]any{
		"type":     "function",
		"function": map[string]any{"name": "svc.one"},
	}, agentCtx)
	function, _ := rewritten.(map[string]any)["function"].(map[string]any)
	if function["name"] != managedToolHashedNameForCanonical("svc.one") {
		t.Fatalf("collision tool_choice rewrote to %q, want hashed form", function["name"])
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
