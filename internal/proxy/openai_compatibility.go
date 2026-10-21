package proxy

import (
	"strings"
	"time"
)

// These exact model families and dated snapshots are documented by OpenAI.
// Avoid guessing capabilities for an unknown sibling or an OpenAI-compatible
// gateway merely because its model name happens to contain "gpt-6".
func openAIReasoningFamily(model string) string {
	for _, family := range []string{"gpt-6.1-sol", "gpt-6-astra", "gpt-6-sol", "gpt-6-luna"} {
		if model == family {
			return family
		}
		if suffix, ok := strings.CutPrefix(model, family+"-"); ok {
			if _, err := time.Parse("2006-01-02", suffix); err == nil {
				return family
			}
		}
	}
	return ""
}

func openAIModelAcceptsSampling(family, effort string) bool {
	return (family == "gpt-6-sol" || family == "gpt-6-luna") && effort == "none"
}

// Current reasoning models reject the sampling defaults emitted by older
// runtimes. Remove only those unsupported controls, and report the mutation
// for audit at both direct and managed dispatch boundaries. Never change a
// caller's explicit reasoning effort to make a request pass.
func normalizeOpenAIResponses(candidate dispatchCandidate, translated map[string]any) ([]string, error) {
	if !strings.EqualFold(candidate.ProviderName, "openai") {
		return nil, nil
	}
	family := openAIReasoningFamily(candidate.UpstreamModel)
	if family == "" {
		return nil, nil
	}
	reasoning, _ := translated["reasoning"].(map[string]any)
	effort, _ := reasoning["effort"].(string)
	if effort == "minimal" || (effort == "none" && (family == "gpt-6.1-sol" || family == "gpt-6-astra")) {
		return nil, &responsesAdapterRequestError{message: "reasoning_effort is not supported for the selected OpenAI model"}
	}
	if openAIModelAcceptsSampling(family, effort) {
		return nil, nil
	}
	dropped := false
	for _, field := range []string{"temperature", "top_p"} {
		if _, ok := translated[field]; ok {
			delete(translated, field)
			dropped = true
		}
	}
	if dropped {
		return []string{"responses_adapter_dropped_sampling"}, nil
	}
	return nil, nil
}
