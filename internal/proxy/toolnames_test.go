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
	if resolved.PresentedName != managedToolPresentedNameForCanonical("claw-wall.search_channel_context") {
		t.Fatalf("unexpected presented name %q", resolved.PresentedName)
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
