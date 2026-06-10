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

// managedToolPresentedNames computes the model-facing name for every tool in
// a manifest. Provider-safe canonical names pass through. Non-safe names are
// presented hash-free (sanitized) when that form is unique across the
// manifest's sanitized forms and collides with no canonical name; otherwise
// they keep the legacy hash suffix. Hash-free presentation eliminates the
// suffix-dropping failure class for typical pods: some models consistently
// drop trailing hex hashes when emitting tool calls.
func managedToolPresentedNames(tools []agentctx.ToolManifestEntry) map[string]string {
	canonicals := make(map[string]struct{}, len(tools))
	hashlessCounts := make(map[string]int, len(tools))
	for _, tool := range tools {
		canonical := strings.TrimSpace(tool.Name)
		canonicals[canonical] = struct{}{}
		if isProviderSafeToolName(canonical) {
			continue
		}
		hashlessCounts[managedToolHashlessAliasForCanonical(canonical)]++
	}

	names := make(map[string]string, len(tools))
	for _, tool := range tools {
		canonical := strings.TrimSpace(tool.Name)
		if isProviderSafeToolName(canonical) {
			names[canonical] = canonical
			continue
		}
		hashless := managedToolHashlessAliasForCanonical(canonical)
		if hashlessCounts[hashless] == 1 {
			if _, taken := canonicals[hashless]; !taken {
				names[canonical] = hashless
				continue
			}
		}
		names[canonical] = managedToolHashedNameForCanonical(canonical)
	}
	return names
}

// managedToolHashedNameForCanonical is the legacy presented form: sanitized
// name plus an 8-hex hash suffix. It is no longer the default presentation,
// but resolution keeps accepting it because session histories and cross-turn
// continuity replays recorded under older proxies contain these names.
func managedToolHashedNameForCanonical(name string) string {
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

func managedToolHashlessAliasForCanonical(name string) string {
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

	suffixLen := 1 + managedToolAliasHashBytes*2
	maxBaseLen := maxManagedToolPresentedNameLen - suffixLen
	if maxBaseLen < 1 {
		maxBaseLen = 1
	}
	if len(safe) > maxBaseLen {
		safe = safe[:maxBaseLen]
	}
	return safe
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

// resolveManagedTool maps a model-emitted (or replayed) tool name onto a
// manifest entry. Accepted forms per tool: the canonical name, the current
// presented name, and the legacy hashed form. A name matching the hashless
// form of two or more colliding tools resolves to nothing — ambiguity stays
// a hard miss.
func resolveManagedTool(agentCtx *agentctx.AgentContext, name string) (resolvedManagedTool, bool) {
	if agentCtx == nil || agentCtx.Tools == nil {
		return resolvedManagedTool{}, false
	}
	name = strings.TrimSpace(name)
	if name == "" {
		return resolvedManagedTool{}, false
	}
	presented := managedToolPresentedNames(agentCtx.Tools.Tools)
	for _, tool := range agentCtx.Tools.Tools {
		canonical := strings.TrimSpace(tool.Name)
		if name == canonical || name == presented[canonical] || name == managedToolHashedNameForCanonical(canonical) {
			return resolvedManagedTool{
				Manifest:      tool,
				CanonicalName: canonical,
				PresentedName: presented[canonical],
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
