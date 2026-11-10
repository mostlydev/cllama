package agentctx

import "fmt"

// Provider-hosted capabilities an agent's compiled context may grant. A
// capability outside this list cannot be granted: a remote MCP server would
// be a grant of any destination, and anything this build does not know by
// name has no meaning to grant.
const (
	HostedWebSearch       = "web_search"
	HostedCodeInterpreter = "code_interpreter"
	HostedFileSearch      = "file_search"
	HostedImageGeneration = "image_generation"
)

// HostedCapabilityGrantable reports whether a capability name may be granted.
func HostedCapabilityGrantable(name string) bool {
	switch name {
	case HostedWebSearch, HostedCodeInterpreter, HostedFileSearch, HostedImageGeneration:
		return true
	}
	return false
}

// HostedCapabilities returns the hosted capabilities the compiled context
// grants, from metadata "hosted_capabilities". Absent means none. A value
// that is not a list of grantable names is an invalid context, never a
// partial grant.
func (a *AgentContext) HostedCapabilities() (map[string]bool, error) {
	out := map[string]bool{}
	if a == nil {
		return out, nil
	}
	raw, ok := a.Metadata["hosted_capabilities"]
	if !ok {
		return out, nil
	}
	list, isList := raw.([]any)
	if !isList {
		return nil, fmt.Errorf("hosted_capabilities must be a list")
	}
	for _, item := range list {
		name, isString := item.(string)
		if !isString || !HostedCapabilityGrantable(name) {
			return nil, fmt.Errorf("invalid hosted_capabilities entry")
		}
		out[name] = true
	}
	return out, nil
}
