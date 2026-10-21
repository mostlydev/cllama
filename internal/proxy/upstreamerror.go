package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"
)

const upstreamErrorPeekLimit = 16 << 10

type replayResponseBody struct {
	io.Reader
	io.Closer
}

type responseReadError struct{ err error }

func (r responseReadError) Read([]byte) (int, error) { return 0, r.err }

// A regex is insufficient here: even an identifier-shaped provider field can
// contain a credential or echoed input. Only known schema vocabulary may be
// emitted. Unknown values retain their presence as "other".
func errorVocabulary(value any, vocabulary string) string {
	s, ok := value.(string)
	if !ok || s == "" {
		return ""
	}
	for _, allowed := range strings.Fields(vocabulary) {
		if s == allowed {
			return s
		}
	}
	return "other"
}

func (h *Handler) logUpstreamError(agentID, model, provider string, resp *http.Response) {
	if resp.StatusCode < 400 || h.logger == nil {
		return
	}
	var code, kind, param string
	if resp.Body != nil {
		original := resp.Body
		prefix, err := io.ReadAll(io.LimitReader(original, upstreamErrorPeekLimit+1))
		// Preserve the exact response and original Close, including for huge,
		// invalid or partially read errors. Diagnostics never consume its tail.
		var replay io.Reader = io.MultiReader(bytes.NewReader(prefix), original)
		if err != nil {
			replay = io.MultiReader(bytes.NewReader(prefix), responseReadError{err})
		}
		resp.Body = &replayResponseBody{Reader: replay, Closer: original}
		if err == nil && len(prefix) <= upstreamErrorPeekLimit {
			var envelope struct {
				Error map[string]any `json:"error"`
			}
			if json.Unmarshal(prefix, &envelope) == nil {
				code = errorVocabulary(envelope.Error["code"], "unsupported_parameter unsupported_value invalid_value invalid_request_error model_not_found context_length_exceeded insufficient_quota rate_limit_exceeded invalid_api_key content_policy_violation")
				kind = errorVocabulary(envelope.Error["type"], "invalid_request_error authentication_error permission_error not_found_error rate_limit_error api_error overloaded_error server_error tokens")
				param = errorVocabulary(envelope.Error["param"], "model messages input tools tool_choice temperature top_p top_logprobs logprobs max_tokens max_completion_tokens max_output_tokens reasoning_effort reasoning.effort stream stream_options response_format text.format parallel_tool_calls n stop seed presence_penalty frequency_penalty")
			}
		}
	}
	h.logger.LogUpstreamError(agentID, model, provider, resp.StatusCode, code, kind, param)
}
