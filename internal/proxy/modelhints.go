package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
	"sync"
)

// A model's request shape is learned from the vendor's own rejections, so a
// model released after this build needs no code change. Two facts are learned
// per provider and model, both from a structured 400 naming one parameter:
//
//   - chat completions refused max_tokens: send the model through the
//     Responses adapter, which already translates that field;
//   - the Responses API refused temperature or top_p: send without them.
//
// The compiled-in lists remain as a first-request shortcut for models known
// when this build was made.

const (
	// maxModelHints bounds the learned set. Model names reach it only after
	// model policy has admitted them, but a bound costs nothing.
	maxModelHints = 256

	unsupportedParameterCode = "unsupported_parameter"
)

type modelHint struct {
	viaResponses bool
	dropSampling bool
}

type modelHints struct {
	mu sync.Mutex
	m  map[string]modelHint
}

// modelHintKey binds a hint to the endpoint it was learned from as well as
// the provider and model, so a provider pointed at another endpoint does not
// inherit it.
func modelHintKey(baseURL string, candidate dispatchCandidate) string {
	return strings.ToLower(strings.TrimSpace(candidate.ProviderName)) + "\x00" + strings.TrimSpace(baseURL) + "\x00" + strings.TrimSpace(candidate.UpstreamModel)
}

// apply returns the candidate carrying what earlier successes taught. The
// operator's switch that turns the Responses adapter off also turns this off.
func (s *modelHints) apply(baseURL string, candidate dispatchCandidate) dispatchCandidate {
	if boolEnv(EnvResponsesAPIDisabled) {
		return candidate
	}
	s.mu.Lock()
	hint := s.m[modelHintKey(baseURL, candidate)]
	s.mu.Unlock()
	candidate.ViaResponses = candidate.ViaResponses || hint.viaResponses
	candidate.DropSampling = candidate.DropSampling || hint.dropSampling
	return candidate
}

// learn records the shape a candidate was just served in. It is called only
// after a 2xx reply, so a shape that was tried and failed is never carried
// into later requests. A full set stops learning: each request then pays for
// its own rejection, which is slower, not wrong.
func (s *modelHints) learn(baseURL string, candidate dispatchCandidate) {
	if !candidate.ViaResponses && !candidate.DropSampling {
		return
	}
	key := modelHintKey(baseURL, candidate)
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.m == nil {
		s.m = map[string]modelHint{}
	}
	if _, known := s.m[key]; !known && len(s.m) >= maxModelHints {
		return
	}
	hint := s.m[key]
	hint.viaResponses = hint.viaResponses || candidate.ViaResponses
	hint.dropSampling = hint.dropSampling || candidate.DropSampling
	s.m[key] = hint
}

// unsupportedParameter returns the parameter a 400 reply rejects with the
// code unsupported_parameter, or "". Only the structured code and param are
// read, never the message. The body is restored for whoever forwards it.
func unsupportedParameter(resp *http.Response) string {
	if resp == nil || resp.Body == nil || resp.StatusCode != http.StatusBadRequest {
		return ""
	}
	original := resp.Body
	prefix, err := io.ReadAll(io.LimitReader(original, upstreamErrorPeekLimit))
	resp.Body = struct {
		io.Reader
		io.Closer
	}{Reader: io.MultiReader(bytes.NewReader(prefix), original), Closer: original}
	if err != nil {
		return ""
	}
	var shaped struct {
		Error struct {
			Code  any `json:"code"`
			Param any `json:"param"`
		} `json:"error"`
	}
	if json.Unmarshal(prefix, &shaped) != nil {
		return ""
	}
	if code, _ := shaped.Error.Code.(string); code != unsupportedParameterCode {
		return ""
	}
	param, _ := shaped.Error.Param.(string)
	return param
}

// chatRejectsMaxTokens reports whether OpenAI's chat completions endpoint
// refused max_tokens for this model. That does not prove the model is
// Responses-only: some chat models only want max_completion_tokens. Moving
// the request to the Responses API on the same provider, key and model is
// still correct for OpenAI, where the adapter translates the field and
// current families need the Responses API for tool calls anyway. Other
// providers are left alone: an OpenAI-compatible endpoint need not offer it.
func chatRejectsMaxTokens(candidate dispatchCandidate, resp *http.Response) bool {
	if boolEnv(EnvResponsesAPIDisabled) || !strings.EqualFold(strings.TrimSpace(candidate.ProviderName), "openai") {
		return false
	}
	return unsupportedParameter(resp) == "max_tokens"
}

// responsesRejectsSampling reports whether a Responses API attempt was
// refused for carrying temperature or top_p. A plan route never sends them.
func responsesRejectsSampling(candidate dispatchCandidate, upstream upstreamEncoding, resp *http.Response) bool {
	if !upstream.Adapter || upstream.Kind != adapterResponses || upstream.Plan || candidate.DropSampling {
		return false
	}
	switch unsupportedParameter(resp) {
	case "temperature", "top_p":
		return true
	}
	return false
}

// dropSampling removes the sampling controls from a translated request.
func dropSampling(translated map[string]any) bool {
	dropped := false
	for _, field := range []string{"temperature", "top_p"} {
		if _, ok := translated[field]; ok {
			delete(translated, field)
			dropped = true
		}
	}
	return dropped
}

// relearnShape decides whether a rejected attempt should be sent again in
// another shape, and returns the candidate carrying the new fact. It is the
// only place a reply changes a request's shape. Each fact is learned once per
// candidate, so a request is re-sent at most twice: once to move to the
// Responses API and once to drop sampling controls.
func (h *Handler) relearnShape(candidate dispatchCandidate, upstream upstreamEncoding, requestPath string, resp *http.Response) (dispatchCandidate, bool) {
	if !responsesAdapterEligible(requestPath) {
		return candidate, false
	}
	if !upstream.Adapter {
		// The vendor names the Responses endpoint, or refuses max_tokens.
		if responsesOnlyUpstreamSignal(resp) || chatRejectsMaxTokens(candidate, resp) {
			candidate.ViaResponses = true
			return candidate, true
		}
		return candidate, false
	}
	if responsesRejectsSampling(candidate, upstream, resp) {
		candidate.DropSampling = true
		return candidate, true
	}
	return candidate, false
}

// hinted applies learned shape to a candidate, keyed by the provider's
// configured endpoint.
func (h *Handler) hinted(candidate dispatchCandidate) dispatchCandidate {
	prov, err := h.registry.Get(candidate.ProviderName)
	if err != nil || prov == nil {
		return candidate
	}
	return h.hints.apply(prov.BaseURL, candidate)
}
