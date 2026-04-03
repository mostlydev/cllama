package proxy

import (
	"crypto/sha1"
	"encoding/hex"
	"strings"

	"github.com/mostlydev/cllama/internal/agentctx"
)

const (
	maxManagedToolPresentedNameLen = 128
	managedToolAliasHashBytes      = 4
)

type resolvedManagedTool struct {
	Manifest      agentctx.ToolManifestEntry
	CanonicalName string
	PresentedName string
}

func managedToolPresentedName(tool agentctx.ToolManifestEntry) string {
	return managedToolPresentedNameForCanonical(tool.Name)
}

func managedToolPresentedNameForCanonical(name string) string {
	name = strings.TrimSpace(name)
	if name == "" {
		return "managed_tool"
	}
	if isProviderSafeToolName(name) {
		return name
	}

	safe := sanitizeManagedToolName(name)
	if safe == "" {
		safe = "managed_tool"
	}

	sum := sha1.Sum([]byte(name))
	suffix := "_" + hex.EncodeToString(sum[:managedToolAliasHashBytes])
	maxBaseLen := maxManagedToolPresentedNameLen - len(suffix)
	if maxBaseLen < 1 {
		maxBaseLen = 1
	}
	if len(safe) > maxBaseLen {
		safe = safe[:maxBaseLen]
	}
	return safe + suffix
}

func managedToolDisplayName(agentCtx *agentctx.AgentContext, name string) string {
	resolved, ok := resolveManagedTool(agentCtx, name)
	if ok && strings.TrimSpace(resolved.CanonicalName) != "" {
		return resolved.CanonicalName
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return "tool"
	}
	return name
}

func resolveManagedTool(agentCtx *agentctx.AgentContext, name string) (resolvedManagedTool, bool) {
	if agentCtx == nil || agentCtx.Tools == nil {
		return resolvedManagedTool{}, false
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return resolvedManagedTool{}, false
	}
	for _, tool := range agentCtx.Tools.Tools {
		canonical := strings.TrimSpace(tool.Name)
		presented := managedToolPresentedName(tool)
		if name == canonical || name == presented {
			return resolvedManagedTool{
				Manifest:      tool,
				CanonicalName: canonical,
				PresentedName: presented,
			}, true
		}
	}
	return resolvedManagedTool{}, false
}

func isProviderSafeToolName(name string) bool {
	if len(name) < 1 || len(name) > maxManagedToolPresentedNameLen {
		return false
	}
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z':
		case r >= 'A' && r <= 'Z':
		case r >= '0' && r <= '9':
		case r == '_' || r == '-':
		default:
			return false
		}
	}
	return true
}

func sanitizeManagedToolName(name string) string {
	var b strings.Builder
	b.Grow(len(name))
	for _, r := range name {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= 'A' && r <= 'Z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '_' || r == '-':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return strings.Trim(b.String(), "_")
}
