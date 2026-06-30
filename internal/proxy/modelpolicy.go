package proxy

import (
	"fmt"
	"strings"

	"github.com/mostlydev/cllama/internal/agentctx"
)

type dispatchCandidate struct {
	Ref           string
	ProviderName  string
	UpstreamModel string
}

type modelResolution struct {
	ChosenRef    string
	Candidates   []dispatchCandidate
	Intervention string
}

func (h *Handler) resolveOpenAIExecution(agentCtx *agentctx.AgentContext, requestedModel string) (modelResolution, error) {
	requestedModel = strings.TrimSpace(requestedModel)
	if agentCtx == nil || !agentCtx.HasPolicy() {
		if requestedModel == "" {
			return modelResolution{}, fmt.Errorf("missing model")
		}
		candidate, err := h.openAICandidateFromRef(requestedModel)
		if err != nil {
			return modelResolution{}, err
		}
		return modelResolution{
			ChosenRef:  requestedModel,
			Candidates: []dispatchCandidate{candidate},
		}, nil
	}

	chosenRef, intervention := h.normalizeOpenAIRequestedModel(agentCtx.ModelPolicy, requestedModel)
	candidates, err := h.openAICandidates(agentCtx.ModelPolicy, chosenRef)
	if err != nil {
		return modelResolution{}, err
	}
	return modelResolution{
		ChosenRef:    chosenRef,
		Candidates:   candidates,
		Intervention: intervention,
	}, nil
}

func (h *Handler) resolveAnthropicExecution(agentCtx *agentctx.AgentContext, requestedModel string) (modelResolution, error) {
	requestedModel = strings.TrimSpace(requestedModel)
	if agentCtx == nil || !agentCtx.HasPolicy() {
		if requestedModel == "" {
			return modelResolution{}, fmt.Errorf("missing model")
		}
		upstreamModel := requestedModel
		if strings.Contains(requestedModel, "/") {
			providerName, strippedModel, err := splitModel(requestedModel)
			if err != nil {
				return modelResolution{}, err
			}
			if providerName != "anthropic" {
				return modelResolution{}, fmt.Errorf("model must target anthropic on /v1/messages")
			}
			upstreamModel = strippedModel
		}
		return modelResolution{
			ChosenRef: requestedModel,
			Candidates: []dispatchCandidate{{
				ProviderName:  "anthropic",
				UpstreamModel: upstreamModel,
			}},
		}, nil
	}

	chosenRef, intervention := normalizePolicyRequestedModel(agentCtx.ModelPolicy, requestedModel)
	candidates, err := anthropicCandidates(agentCtx.ModelPolicy, chosenRef)
	if err != nil {
		return modelResolution{}, err
	}
	return modelResolution{
		ChosenRef:    chosenRef,
		Candidates:   candidates,
		Intervention: intervention,
	}, nil
}

func (h *Handler) normalizeOpenAIRequestedModel(policy *agentctx.ModelPolicy, requestedModel string) (string, string) {
	requestedModel = strings.TrimSpace(requestedModel)
	if policy == nil || !policy.HasPolicy() {
		return requestedModel, ""
	}

	defaultRef := policy.DefaultModel()
	if defaultRef == "" {
		return requestedModel, ""
	}

	if requestedModel == "" {
		return defaultRef, "missing"
	}
	if exact, ok := exactAllowedRef(policy, requestedModel); ok {
		return exact, ""
	}
	if !strings.Contains(requestedModel, "/") {
		match, count := uniqueAllowedBareMatch(policy, requestedModel)
		switch {
		case count == 1:
			return match, "bare_model_normalized"
		case count > 1:
			return defaultRef, "ambiguous_clamped"
		default:
			return defaultRef, "disallowed_clamped"
		}
	}
	if match, count := h.uniqueAllowedNormalizedOpenAIMatch(policy, requestedModel); count == 1 {
		if match == requestedModel {
			return match, ""
		}
		return match, "provider_normalized"
	} else if count > 1 {
		return defaultRef, "ambiguous_clamped"
	}
	return defaultRef, "disallowed_clamped"
}

func normalizePolicyRequestedModel(policy *agentctx.ModelPolicy, requestedModel string) (string, string) {
	requestedModel = strings.TrimSpace(requestedModel)
	if policy == nil || !policy.HasPolicy() {
		return requestedModel, ""
	}

	defaultRef := policy.DefaultModel()
	if defaultRef == "" {
		return requestedModel, ""
	}
	if requestedModel == "" {
		return defaultRef, "missing"
	}
	if exact, ok := exactAllowedRef(policy, requestedModel); ok {
		return exact, ""
	}

	match, count := uniqueAllowedBareMatch(policy, requestedModel)
	switch {
	case count == 1:
		return match, "bare_model_normalized"
	case count > 1:
		return defaultRef, "ambiguous_clamped"
	default:
		return defaultRef, "disallowed_clamped"
	}
}

func (h *Handler) openAICandidates(policy *agentctx.ModelPolicy, chosenRef string) ([]dispatchCandidate, error) {
	refs := candidateRefsFromPolicy(policy, chosenRef)
	out := make([]dispatchCandidate, 0, len(refs))
	for _, ref := range refs {
		candidate, err := h.openAICandidateFromRef(ref)
		if err != nil {
			return nil, err
		}
		out = append(out, candidate)
	}
	return out, nil
}

func anthropicCandidates(policy *agentctx.ModelPolicy, chosenRef string) ([]dispatchCandidate, error) {
	refs := candidateRefsFromPolicy(policy, chosenRef)
	out := make([]dispatchCandidate, 0, len(refs))
	for _, ref := range refs {
		candidate, err := anthropicCandidateFromRef(ref)
		if err != nil {
			return nil, err
		}
		out = append(out, candidate)
	}
	return out, nil
}

func (h *Handler) openAICandidateFromRef(ref string) (dispatchCandidate, error) {
	providerName, upstreamModel, err := splitModel(ref)
	if err != nil {
		return dispatchCandidate{}, err
	}
	resolvedProvider, resolvedUpstream, err := h.resolveOpenAIProvider(providerName, upstreamModel)
	if err != nil {
		return dispatchCandidate{}, err
	}
	return dispatchCandidate{
		Ref:           ref,
		ProviderName:  resolvedProvider,
		UpstreamModel: resolvedUpstream,
	}, nil
}

func anthropicCandidateFromRef(ref string) (dispatchCandidate, error) {
	providerName, upstreamModel, err := splitModel(ref)
	if err != nil {
		return dispatchCandidate{}, err
	}
	if providerName != "anthropic" {
		return dispatchCandidate{}, fmt.Errorf("model policy requires anthropic-compatible provider on /v1/messages")
	}
	return dispatchCandidate{
		Ref:           ref,
		ProviderName:  "anthropic",
		UpstreamModel: upstreamModel,
	}, nil
}

func (h *Handler) uniqueAllowedNormalizedOpenAIMatch(policy *agentctx.ModelPolicy, requestedModel string) (string, int) {
	requestedKey, ok := h.normalizedOpenAIRef(requestedModel)
	if !ok {
		return "", 0
	}
	matches := make([]string, 0, 1)
	for _, ref := range policy.AllowedModelRefs() {
		allowedKey, ok := h.normalizedOpenAIRef(ref)
		if !ok || allowedKey != requestedKey {
			continue
		}
		matches = append(matches, ref)
	}
	if len(matches) == 1 {
		return matches[0], 1
	}
	return "", len(matches)
}

func (h *Handler) normalizedOpenAIRef(ref string) (string, bool) {
	providerName, upstreamModel, err := splitModel(ref)
	if err != nil {
		return "", false
	}
	resolvedProvider, resolvedUpstream, err := h.resolveOpenAIProvider(providerName, upstreamModel)
	if err != nil {
		return "", false
	}
	return resolvedProvider + "/" + resolvedUpstream, true
}

func exactAllowedRef(policy *agentctx.ModelPolicy, requestedModel string) (string, bool) {
	for _, ref := range policy.AllowedModelRefs() {
		if ref == requestedModel {
			return ref, true
		}
	}
	return "", false
}

func uniqueAllowedBareMatch(policy *agentctx.ModelPolicy, requestedModel string) (string, int) {
	requestedModel = strings.TrimSpace(requestedModel)
	if requestedModel == "" {
		return "", 0
	}
	matches := make([]string, 0, 1)
	for _, ref := range policy.AllowedModelRefs() {
		if !matchesBareAlias(ref, requestedModel) {
			continue
		}
		matches = append(matches, ref)
	}
	if len(matches) == 1 {
		return matches[0], 1
	}
	return "", len(matches)
}

func matchesBareAlias(ref, requestedModel string) bool {
	for _, alias := range bareModelAliases(ref) {
		if alias == requestedModel {
			return true
		}
	}
	return false
}

func bareModelAliases(ref string) []string {
	_, model, err := splitModel(ref)
	if err != nil {
		return nil
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return nil
	}
	out := []string{model}
	if idx := strings.LastIndex(model, "/"); idx >= 0 && idx+1 < len(model) {
		last := model[idx+1:]
		if last != model {
			out = append(out, last)
		}
	}
	return out
}

func candidateRefsFromPolicy(policy *agentctx.ModelPolicy, chosenRef string) []string {
	if strings.TrimSpace(chosenRef) == "" {
		return nil
	}
	if policy == nil || !policy.HasPolicy() {
		return []string{chosenRef}
	}
	failover := policy.FailoverRefs()
	if len(failover) == 0 {
		return []string{chosenRef}
	}
	for i, ref := range failover {
		if ref == chosenRef {
			out := make([]string, 0, len(failover)-i)
			out = append(out, failover[i:]...)
			return out
		}
	}
	out := []string{chosenRef}
	for _, ref := range failover {
		if ref == chosenRef {
			continue
		}
		out = append(out, ref)
	}
	return out
}
