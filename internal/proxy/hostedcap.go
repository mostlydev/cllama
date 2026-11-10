package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"github.com/mostlydev/cllama/internal/agentctx"
)

// A provider-hosted capability is something the vendor does on the agent's
// behalf when a request asks for it: searching the web, calling a remote
// server, running code. That is authority, not data handling, so it is
// refused unless the agent's compiled context grants it by name. The check
// reads the final wire body and nothing the request can set about itself.

type hostedGrantKey struct{}

// hostedUnlisted names a hosted feature this build does not recognise. It
// cannot be granted.
const hostedUnlisted = "unlisted"

type hostedCapabilityError struct{ capability string }

func (e *hostedCapabilityError) Error() string {
	return "hosted capability not granted: " + e.capability
}

func requestHostedGrants(ctx context.Context) map[string]bool {
	grants, _ := ctx.Value(hostedGrantKey{}).(map[string]bool)
	return grants
}

// hostedToolCapability names the hosted capability a tool entry asks for,
// or "" when the tool is the agent's own: a function or custom tool, a
// namespace of those, or a tool the client itself executes.
func hostedToolCapability(tool map[string]any) string {
	rawType, hasType := tool["type"]
	typ, isString := rawType.(string)
	if hasType && !isString {
		return hostedUnlisted
	}
	// An entry with no type is the agent's own function definition, unless
	// it carries the marks of a remote server.
	if carriesRemoteServer(tool) {
		return "mcp"
	}
	switch {
	case typ == "" || typ == "function" || typ == "custom":
		return ""
	case typ == "namespace":
		members, ok := tool["tools"].([]any)
		if !ok {
			return hostedUnlisted
		}
		for _, raw := range members {
			member, ok := raw.(map[string]any)
			if !ok {
				return hostedUnlisted
			}
			if mt, _ := member["type"].(string); (mt != "function" && mt != "custom") || carriesRemoteServer(member) {
				return hostedUnlisted
			}
		}
		return ""
	case typ == "web_search" || strings.HasPrefix(typ, "web_search_"):
		return agentctx.HostedWebSearch
	case typ == "code_interpreter" || strings.HasPrefix(typ, "code_execution_"):
		return agentctx.HostedCodeInterpreter
	case typ == "file_search":
		return agentctx.HostedFileSearch
	case typ == "image_generation":
		return agentctx.HostedImageGeneration
	case typ == "mcp":
		return "mcp"
	// Anthropic's client tools: the model asks, the agent's own runtime
	// acts. Only the dated schema names are known; a name that merely
	// starts the same way is not.
	case clientToolRe.MatchString(typ), typ == "computer_use_preview":
		// OpenAI's computer tool is also acted on by the client's runtime.
		return ""
	}
	return hostedUnlisted
}

// clientToolRe matches Anthropic's dated client-tool schema names.
var clientToolRe = regexp.MustCompile(`^(bash|text_editor|computer|memory)_20[0-9]{6}$`)

// carriesRemoteServer reports whether a tool entry names a remote server
// for the vendor to call.
func carriesRemoteServer(tool map[string]any) bool {
	for _, marker := range []string{"server_url", "server_label", "connector_id", "mcp_server_name"} {
		if _, present := tool[marker]; present {
			return true
		}
	}
	return false
}

// hostedCapabilities returns the hosted capabilities a wire body asks for,
// sorted and without repeats.
func hostedCapabilities(body map[string]any) []string {
	found := map[string]bool{}
	if model, _ := body["model"].(string); onlineModelVariant(model) {
		found[agentctx.HostedWebSearch] = true
	}
	if models, ok := body["models"].([]any); ok {
		for _, m := range models {
			if name, _ := m.(string); onlineModelVariant(name) {
				found[agentctx.HostedWebSearch] = true
			}
		}
	}
	present := func(key string) bool {
		v, ok := body[key]
		if !ok || v == nil {
			return false
		}
		if list, isList := v.([]any); isList {
			return len(list) > 0
		}
		return true
	}
	if present("web_search_options") {
		found[agentctx.HostedWebSearch] = true
	}
	if present("mcp_servers") {
		found["mcp"] = true
	}
	if present("plugins") {
		found["plugins"] = true
	}
	if rawTools, present := body["tools"]; present && rawTools != nil {
		tools, ok := rawTools.([]any)
		if !ok {
			found[hostedUnlisted] = true
		}
		for _, raw := range tools {
			tool, ok := raw.(map[string]any)
			if !ok {
				found[hostedUnlisted] = true
				continue
			}
			if name := hostedToolCapability(tool); name != "" {
				found[name] = true
			}
		}
	}
	out := make([]string, 0, len(found))
	for name := range found {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// enforceHostedCapabilities refuses a wire body that asks for a hosted
// capability the context does not grant.
func enforceHostedCapabilities(grants map[string]bool, wire []byte) error {
	var body map[string]any
	if err := json.Unmarshal(wire, &body); err != nil || body == nil {
		return fmt.Errorf("hosted capability check: request is not a JSON object")
	}
	for _, name := range hostedCapabilities(body) {
		if !agentctx.HostedCapabilityGrantable(name) || !grants[name] {
			return &hostedCapabilityError{capability: name}
		}
	}
	return nil
}
