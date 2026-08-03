package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// Outbound adapter for OpenAI's Responses API.
//
// cllama's agent-facing contract stays /v1/chat/completions. Some OpenAI models
// (the gpt-5.6 family, gpt-5-pro) refuse function tools on chat/completions and
// are only reachable through POST /v1/responses. Rather than adding a second
// inbound pipeline — which would move audit, budget, history, and tool
// mediation off the shape they are built around — this adapter translates at
// the provider boundary only, so every governance surface keeps observing the
// unchanged chat/completions shape.

const (
	// EnvResponsesAPIModels adds provider-scoped model prefixes that must be
	// dispatched through /v1/responses, as a comma-separated list of
	// "<provider>/<model-prefix>" entries. Entries add to the built-in set.
	EnvResponsesAPIModels = "CLLAMA_RESPONSES_API_MODELS"
	// EnvResponsesAPIDisabled turns the adapter off entirely. It is the escape
	// hatch for the day OpenAI makes these models work on chat/completions and
	// the built-in list becomes wrong.
	EnvResponsesAPIDisabled = "CLLAMA_RESPONSES_API_DISABLED"
	// EnvResponsesDefaultReasoningEffort supplies a Responses API reasoning
	// effort only when the inbound Chat Completions request omits one. Explicit
	// caller values always win; leaving this unset preserves existing behavior.
	EnvResponsesDefaultReasoningEffort = "CLLAMA_RESPONSES_DEFAULT_REASONING_EFFORT"
	// EnvResponsesRequiredToolChoiceAsAuto relaxes only a caller's required
	// tool choice at the Responses adapter boundary. It is an opt-in escape
	// hatch for runtimes that apply chat-only tool policy to scheduled turns.
	EnvResponsesRequiredToolChoiceAsAuto = "CLLAMA_RESPONSES_REQUIRED_TOOL_CHOICE_AS_AUTO"

	// responsesAPIPath is the upstream path the adapter dispatches to. It is
	// expressed as an inbound-style path because buildUpstreamURL strips the
	// /v1 prefix before joining it onto the provider base URL.
	responsesAPIPath = "/v1/responses"
)

// responsesAPIModelPrefixes lists provider-scoped model prefixes measured to
// reject function tools on chat/completions. Kept deliberately narrow: rerouting
// a model that already works on chat/completions would be a regression, so
// entries are added only with evidence.
var responsesAPIModelPrefixes = map[string][]string{
	"openai": {"gpt-5.6", "gpt-5-pro"},
}

// responsesAPIRequired reports whether a declared provider/model pair must be
// dispatched through the Responses API rather than chat/completions.
func responsesAPIRequired(providerName, upstreamModel string) bool {
	if boolEnv(EnvResponsesAPIDisabled) {
		return false
	}
	provider := strings.ToLower(strings.TrimSpace(providerName))
	model := strings.TrimSpace(upstreamModel)
	if provider == "" || model == "" {
		return false
	}

	for _, prefix := range responsesAPIModelPrefixes[provider] {
		if strings.HasPrefix(model, prefix) {
			return true
		}
	}
	for _, entry := range strings.Split(os.Getenv(EnvResponsesAPIModels), ",") {
		entryProvider, entryPrefix, ok := strings.Cut(strings.TrimSpace(entry), "/")
		if !ok || entryPrefix == "" {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(entryProvider), provider) && strings.HasPrefix(model, strings.TrimSpace(entryPrefix)) {
			return true
		}
	}
	return false
}

func boolEnv(name string) bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(name))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

// chatPassthroughFields are accepted unchanged by the Responses API. Anything
// not listed here and not translated explicitly below is dropped, because
// forwarding an unknown field to /v1/responses turns into a 400 that the agent
// sees as an opaque upstream failure.
var chatPassthroughFields = map[string]bool{
	"model":                  true,
	"temperature":            true,
	"top_p":                  true,
	"parallel_tool_calls":    true,
	"metadata":               true,
	"user":                   true,
	"safety_identifier":      true,
	"prompt_cache_key":       true,
	"prompt_cache_retention": true,
	"service_tier":           true,
}

type responsesAdapterRequestError struct {
	message string
}

func (e *responsesAdapterRequestError) Error() string {
	return e.message
}

func responsesAdapterClientError(err error) (string, bool) {
	var requestErr *responsesAdapterRequestError
	if !errors.As(err, &requestErr) {
		return "", false
	}
	return requestErr.message, true
}

// chatToResponsesRequest translates a chat/completions request payload into a
// Responses API request payload. The input payload is not mutated.
func chatToResponsesRequest(payload map[string]any) (map[string]any, error) {
	translated, _, err := chatToResponsesRequestWithReasoning(payload, nil)
	return translated, err
}

// validResponsesReasoningEfforts is the accepted value set for
// CLLAMA_RESPONSES_DEFAULT_REASONING_EFFORT. An invalid configured value skips
// injection (preserving pre-knob behavior) instead of poisoning every adapted
// request with an effort upstream will reject.
var validResponsesReasoningEfforts = map[string]struct{}{
	"none":    {},
	"minimal": {},
	"low":     {},
	"medium":  {},
	"high":    {},
}

// chatToResponsesRequestWithReasoning translates a chat/completions payload
// for the Responses API. The second return lists env-knob request mutations
// (tool-choice relaxation, effort defaulting, invalid effort skipped) so
// dispatch sites can log them as interventions — mutations must never be
// silent in a governance proxy.
func chatToResponsesRequestWithReasoning(payload map[string]any, replay []responsesReasoningReplay) (map[string]any, []string, error) {
	if n, ok := payload["n"]; ok && !isSingleChoice(n) {
		return nil, nil, &responsesAdapterRequestError{
			message: "n must be 1 for models routed through the Responses API",
		}
	}
	if field, ok := unsupportedSemanticChatField(payload); ok {
		return nil, nil, &responsesAdapterRequestError{
			message: fmt.Sprintf("%s is not supported for models routed through the Responses API", field),
		}
	}

	var mutations []string

	out := make(map[string]any, len(payload))

	for key, value := range payload {
		if chatPassthroughFields[key] {
			out[key] = value
		}
	}

	if messages, ok := payload["messages"].([]any); ok {
		out["input"] = chatMessagesToResponsesInputWithReasoning(messages, replay)
	}
	if tools, ok := payload["tools"].([]any); ok {
		out["tools"] = chatToolsToResponsesTools(tools)
	}
	if choice, ok := payload["tool_choice"]; ok {
		if choice == "required" && boolEnv(EnvResponsesRequiredToolChoiceAsAuto) {
			choice = "auto"
			mutations = append(mutations, "responses_required_tool_choice_relaxed")
		}
		out["tool_choice"] = chatToolChoiceToResponses(choice)
	}
	effort, hasEffort := payload["reasoning_effort"]
	if !hasEffort {
		if configured := strings.ToLower(strings.TrimSpace(os.Getenv(EnvResponsesDefaultReasoningEffort))); configured != "" {
			if _, valid := validResponsesReasoningEfforts[configured]; valid {
				effort = configured
				hasEffort = true
				mutations = append(mutations, "responses_reasoning_effort_defaulted:"+configured)
			} else {
				mutations = append(mutations, "responses_reasoning_effort_invalid:"+configured)
			}
		}
	}
	if hasEffort {
		out["reasoning"] = map[string]any{"effort": effort}
	}
	if limit, ok := firstPresent(payload, "max_completion_tokens", "max_tokens"); ok {
		out["max_output_tokens"] = limit
	}
	if format, ok := chatResponseFormatToResponsesText(payload["response_format"]); ok {
		out["text"] = format
	}

	// chat/completions never persists a turn upstream. Keep that property so
	// routing through the adapter does not quietly start retaining agent
	// traffic on the provider side.
	out["store"] = false
	// Stateless Responses requests return encrypted reasoning by default on
	// current models. The legacy include value remains accepted and keeps the
	// adapter compatible with older reasoning models that required the opt-in.
	out["include"] = []any{"reasoning.encrypted_content"}

	return out, mutations, nil
}

func unsupportedSemanticChatField(payload map[string]any) (string, bool) {
	for _, field := range []string{"frequency_penalty", "presence_penalty"} {
		if value, ok := payload[field]; ok && !isZeroOrNil(value) {
			return field, true
		}
	}
	if value, ok := payload["logprobs"]; ok {
		disabled, valid := value.(bool)
		if !valid || disabled {
			return "logprobs", true
		}
	}
	if value, ok := payload["logit_bias"]; ok && !isEmptyMapOrNil(value) {
		return "logit_bias", true
	}
	if value, ok := payload["stop"]; ok && !isEmptyStop(value) {
		return "stop", true
	}
	if value, ok := payload["seed"]; ok && value != nil {
		return "seed", true
	}
	if value, ok := payload["store"]; ok {
		disabled, valid := value.(bool)
		if !valid || disabled {
			return "store", true
		}
	}
	return "", false
}

func isZeroOrNil(raw any) bool {
	if raw == nil {
		return true
	}
	switch value := raw.(type) {
	case float64:
		return value == 0
	case float32:
		return value == 0
	case int:
		return value == 0
	case int64:
		return value == 0
	case json.Number:
		parsed, err := value.Float64()
		return err == nil && parsed == 0
	default:
		return false
	}
}

func isEmptyMapOrNil(raw any) bool {
	if raw == nil {
		return true
	}
	value, ok := raw.(map[string]any)
	return ok && len(value) == 0
}

func isEmptyStop(raw any) bool {
	switch value := raw.(type) {
	case nil:
		return true
	case string:
		return value == ""
	case []any:
		return len(value) == 0
	default:
		return false
	}
}

func isSingleChoice(raw any) bool {
	switch value := raw.(type) {
	case float64:
		return value == 1
	case float32:
		return value == 1
	case int:
		return value == 1
	case int64:
		return value == 1
	case json.Number:
		parsed, err := value.Float64()
		return err == nil && parsed == 1
	default:
		return false
	}
}

// chatMessagesToResponsesInput converts chat messages into Responses input
// items. Assistant tool_calls and role:"tool" results become top-level
// function_call / function_call_output items rather than message fields, which
// is what keeps multi-round managed tool mediation working through the adapter.
func chatMessagesToResponsesInput(messages []any) []any {
	return chatMessagesToResponsesInputWithReasoning(messages, nil)
}

func chatMessagesToResponsesInputWithReasoning(messages []any, replay []responsesReasoningReplay) []any {
	input := make([]any, 0, len(messages))
	usedReplay := make([]bool, len(replay))
	for _, raw := range messages {
		msg, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		role, _ := msg["role"].(string)

		if role == "tool" || role == "function" {
			input = append(input, map[string]any{
				"type":    "function_call_output",
				"call_id": stringField(msg, "tool_call_id"),
				"output":  chatToolResultOutput(msg["content"]),
			})
			continue
		}

		toolCalls, _ := msg["tool_calls"].([]any)
		if items := reasoningItemsForAssistant(toolCalls, replay, usedReplay); len(items) > 0 {
			input = append(input, items...)
		}
		if content, ok := nonEmptyContent(msg["content"]); ok || len(toolCalls) == 0 {
			item := map[string]any{"role": role}
			if ok {
				item["content"] = chatContentToResponsesContent(content, role)
			} else {
				item["content"] = msg["content"]
			}
			if name, exists := msg["name"]; exists {
				item["name"] = name
			}
			input = append(input, item)
		}

		for _, rawCall := range toolCalls {
			call, ok := rawCall.(map[string]any)
			if !ok {
				continue
			}
			fn, _ := call["function"].(map[string]any)
			input = append(input, map[string]any{
				"type":      "function_call",
				"call_id":   stringField(call, "id"),
				"name":      stringField(fn, "name"),
				"arguments": stringField(fn, "arguments"),
			})
		}
	}
	return input
}

type responsesReasoningReplay struct {
	ProviderName  string
	UpstreamModel string
	ToolCallIDs   []string
	Items         []json.RawMessage
}

func reasoningItemsForAssistant(toolCalls []any, replay []responsesReasoningReplay, used []bool) []any {
	callIDs := chatToolCallIDs(toolCalls)
	if len(callIDs) == 0 {
		return nil
	}
	for i, segment := range replay {
		if i < len(used) && used[i] {
			continue
		}
		if !equalStrings(callIDs, segment.ToolCallIDs) {
			continue
		}
		items := make([]any, 0, len(segment.Items))
		for _, raw := range segment.Items {
			if !json.Valid(raw) {
				continue
			}
			items = append(items, append(json.RawMessage(nil), raw...))
		}
		if i < len(used) {
			used[i] = true
		}
		return items
	}
	return nil
}

func chatToolCallIDs(toolCalls []any) []string {
	ids := make([]string, 0, len(toolCalls))
	for _, raw := range toolCalls {
		call, _ := raw.(map[string]any)
		id := strings.TrimSpace(stringField(call, "id"))
		if id == "" {
			return nil
		}
		ids = append(ids, id)
	}
	return ids
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// chatContentToResponsesContent rewrites content-part types. Responses
// distinguishes input from output parts, so "text" becomes input_text on
// inbound roles and output_text on assistant turns.
func chatContentToResponsesContent(content any, role string) any {
	parts, ok := content.([]any)
	if !ok {
		return content
	}
	textType := "input_text"
	if role == "assistant" {
		textType = "output_text"
	}
	out := make([]any, 0, len(parts))
	for _, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			out = append(out, raw)
			continue
		}
		converted := make(map[string]any, len(part))
		for key, value := range part {
			converted[key] = value
		}
		switch part["type"] {
		case "text":
			converted["type"] = textType
		case "image_url":
			converted["type"] = "input_image"
			if img, ok := part["image_url"].(map[string]any); ok {
				converted["image_url"] = img["url"]
				if detail, exists := img["detail"]; exists {
					converted["detail"] = detail
				}
			}
		case "input_audio":
			converted["type"] = "input_audio"
		}
		out = append(out, converted)
	}
	return out
}

// chatToolResultOutput normalizes a tool result to the string Responses
// expects. Content-part arrays are flattened to their concatenated text so a
// structured result is not dropped.
func chatToolResultOutput(content any) string {
	switch typed := content.(type) {
	case string:
		return typed
	case nil:
		return ""
	case []any:
		var joined string
		for _, raw := range typed {
			part, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			if text, ok := part["text"].(string); ok {
				joined += text
			}
		}
		return joined
	default:
		encoded, err := json.Marshal(typed)
		if err != nil {
			return ""
		}
		return string(encoded)
	}
}

func nonEmptyContent(content any) (any, bool) {
	switch typed := content.(type) {
	case nil:
		return nil, false
	case string:
		if typed == "" {
			return nil, false
		}
		return typed, true
	case []any:
		if len(typed) == 0 {
			return nil, false
		}
		return typed, true
	default:
		return typed, true
	}
}

func stringField(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	value, _ := m[key].(string)
	return value
}

// upstreamEncoding is the wire form of one candidate's request. It differs from
// the agent-facing request only when the Responses adapter is active.
type upstreamEncoding struct {
	Path    string
	Body    []byte
	Adapter bool
	// Mutations lists env-knob request rewrites applied during translation.
	// Dispatch sites log each as an intervention alongside the adapter event.
	Mutations []string
}

// encodeUpstreamRequest decides whether a candidate must be dispatched through
// the Responses API and, if so, produces the translated path and body.
func encodeUpstreamRequest(candidate dispatchCandidate, requestPath string, payload map[string]any, chatBody []byte) (upstreamEncoding, error) {
	encoded, _, err := encodeUpstreamRequestWithReasoning(candidate, requestPath, payload, chatBody, nil)
	return encoded, err
}

func encodeUpstreamRequestWithReasoning(candidate dispatchCandidate, requestPath string, payload map[string]any, chatBody []byte, replay []responsesReasoningReplay) (upstreamEncoding, bool, error) {
	if !responsesAdapterEligible(requestPath) || !responsesAPIRequired(candidate.ProviderName, candidate.UpstreamModel) {
		return upstreamEncoding{Path: requestPath, Body: chatBody}, false, nil
	}
	matchingReplay, droppedReplay := responsesReasoningForCandidate(replay, candidate)
	translated, mutations, err := chatToResponsesRequestWithReasoning(payload, matchingReplay)
	if err != nil {
		return upstreamEncoding{}, droppedReplay, err
	}
	body, err := json.Marshal(translated)
	if err != nil {
		return upstreamEncoding{}, droppedReplay, err
	}
	return upstreamEncoding{Path: responsesAPIPath, Body: body, Adapter: true, Mutations: mutations}, droppedReplay, nil
}

func responsesReasoningForCandidate(replay []responsesReasoningReplay, candidate dispatchCandidate) ([]responsesReasoningReplay, bool) {
	matching := make([]responsesReasoningReplay, 0, len(replay))
	dropped := false
	for _, segment := range replay {
		if strings.EqualFold(strings.TrimSpace(segment.ProviderName), strings.TrimSpace(candidate.ProviderName)) &&
			strings.TrimSpace(segment.UpstreamModel) == strings.TrimSpace(candidate.UpstreamModel) {
			matching = append(matching, segment)
			continue
		}
		if len(segment.Items) > 0 {
			dropped = true
		}
	}
	return matching, dropped
}

// responsesAdapterEligible reports whether an inbound path may be translated to
// the Responses API. The adapter is an OpenAI chat/completions concept; the
// Anthropic path never crosses it, whatever an upstream rejection happens to
// say.
func responsesAdapterEligible(requestPath string) bool {
	return strings.HasPrefix(requestPath, "/v1/chat/completions")
}

// adaptChatBodyToResponses re-encodes an already-marshalled chat/completions
// body for the Responses API. It is the recovery path for models upstream
// declares responses-only that the built-in list does not yet know about.
func adaptChatBodyToResponses(chatBody []byte) (upstreamEncoding, error) {
	return adaptChatBodyToResponsesWithReasoning(chatBody, nil)
}

func adaptChatBodyToResponsesWithReasoning(chatBody []byte, replay []responsesReasoningReplay) (upstreamEncoding, error) {
	var payload map[string]any
	if err := json.Unmarshal(chatBody, &payload); err != nil {
		return upstreamEncoding{}, err
	}
	translated, mutations, err := chatToResponsesRequestWithReasoning(payload, replay)
	if err != nil {
		return upstreamEncoding{}, err
	}
	body, err := json.Marshal(translated)
	if err != nil {
		return upstreamEncoding{}, err
	}
	return upstreamEncoding{Path: responsesAPIPath, Body: body, Adapter: true, Mutations: mutations}, nil
}

// responsesOnlyUpstreamSignal reports whether an upstream rejection says the
// model is reachable only through /v1/responses. OpenAI returns this as a 400
// for "function tools with reasoning" and as a 404 for responses-only models.
//
// The body prefix is inspected and the entire stream is restored, so a caller
// that decides not to retry can still forward the original rejection.
func responsesOnlyUpstreamSignal(resp *http.Response) bool {
	if resp.Body == nil {
		return false
	}
	if resp.StatusCode != http.StatusBadRequest && resp.StatusCode != http.StatusNotFound {
		return false
	}
	original := resp.Body
	body, err := io.ReadAll(io.LimitReader(original, 4096))
	resp.Body = struct {
		io.Reader
		io.Closer
	}{
		Reader: io.MultiReader(bytes.NewReader(body), original),
		Closer: original,
	}
	if err != nil {
		return false
	}
	return bodyNamesResponsesAPI(body)
}

func bodyNamesResponsesAPI(body []byte) bool {
	lower := strings.ToLower(string(body))
	return strings.Contains(lower, "use /v1/responses") ||
		strings.Contains(lower, "only supported in v1/responses") ||
		strings.Contains(lower, "only supported in /v1/responses")
}

// adaptResponsesResponse rewrites an upstream Responses reply in place so every
// downstream consumer — managed tool mediation, policy, session history, budget
// accounting — keeps seeing the chat/completions shape.
//
// When the agent asked for a stream, the buffered completion is re-emitted as
// synthetic chat SSE, because the Responses event taxonomy is not the chat one.
func adaptResponsesResponse(resp *http.Response, downstreamStream, downstreamIncludeUsage bool) error {
	body, err := io.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return err
	}

	converted, err := responsesToChatCompletion(body)
	if err != nil {
		// A recognized Responses object in a failure state must surface as a
		// failure — passing it through would hand the agent a 200 whose body
		// is a raw failed Responses object (or, streamed, a clean empty stop).
		var failure *responsesFailureError
		if errors.As(err, &failure) {
			return err
		}
		// Not a Responses object (an upstream error page, say). Pass it through
		// rather than masking the failure with a synthesized empty reply.
		converted = body
	}

	contentType := "application/json"
	if downstreamStream && resp.StatusCode >= 200 && resp.StatusCode < 300 {
		sse, err := chatCompletionToSSE(converted, downstreamIncludeUsage)
		if err != nil {
			return err
		}
		converted = sse
		contentType = "text/event-stream"
	}

	resp.Body = io.NopCloser(bytes.NewReader(converted))
	resp.ContentLength = int64(len(converted))
	resp.Header.Set("Content-Type", contentType)
	resp.Header.Del("Content-Length")
	resp.Header.Del("Content-Encoding")
	return nil
}

// responsesToChatCompletion translates a Responses API response body back into
// the chat/completions shape. Every governance surface downstream — audit,
// session history, budget accounting, managed tool mediation — reads that
// shape, so this is what keeps them working unchanged.
//
// Bodies that are not Responses objects (upstream errors, in particular) pass
// through untouched; rewriting them would convert a visible failure into a
// silently empty completion.
func responsesToChatCompletion(body []byte) ([]byte, error) {
	var resp map[string]any
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	if err := validateResponsesStatus(resp); err != nil {
		return nil, err
	}
	output, ok := resp["output"].([]any)
	if !ok {
		if resp["object"] == "response" || stringField(resp, "status") != "" {
			return nil, &responsesFailureError{message: "responses API reply is missing an output array"}
		}
		return body, nil
	}

	var content any
	var refusal string
	var toolCalls []any
	for _, raw := range output {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		switch item["type"] {
		case "message":
			if text, ok := responsesOutputText(item); ok {
				content = text
			}
			if explanation, ok := responsesOutputRefusal(item); ok {
				refusal = explanation
			}
		case "function_call":
			toolCalls = append(toolCalls, map[string]any{
				"index":    float64(len(toolCalls)),
				"id":       stringField(item, "call_id"),
				"type":     "function",
				"function": map[string]any{"name": stringField(item, "name"), "arguments": stringField(item, "arguments")},
			})
		}
	}

	message := map[string]any{"role": "assistant", "content": content}
	if refusal != "" {
		message["refusal"] = refusal
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}

	choice := map[string]any{
		"index":         0,
		"message":       message,
		"finish_reason": responsesFinishReason(resp, len(toolCalls) > 0),
	}

	chat := map[string]any{
		"object":  "chat.completion",
		"choices": []any{choice},
	}
	if id, ok := resp["id"]; ok {
		chat["id"] = id
	}
	if created, ok := resp["created_at"]; ok {
		chat["created"] = created
	}
	if model, ok := resp["model"]; ok {
		chat["model"] = model
	}
	if usage, ok := responsesUsageToChatUsage(resp["usage"]); ok {
		chat["usage"] = usage
	}

	return json.Marshal(chat)
}

func responsesToChatCompletionWithReasoning(body []byte) ([]byte, []json.RawMessage, error) {
	converted, err := responsesToChatCompletion(body)
	if err != nil {
		return nil, nil, err
	}
	return converted, encryptedResponsesReasoningItems(body), nil
}

// encryptedResponsesReasoningItems retains the opaque output objects exactly
// as received. They are never added to the chat-shaped response or request;
// managed mediation may replay them only at the outbound Responses boundary.
func encryptedResponsesReasoningItems(body []byte) []json.RawMessage {
	var response struct {
		Output []json.RawMessage `json:"output"`
	}
	if err := json.Unmarshal(body, &response); err != nil {
		return nil
	}
	items := make([]json.RawMessage, 0, len(response.Output))
	for _, raw := range response.Output {
		var item map[string]any
		if err := json.Unmarshal(raw, &item); err != nil {
			continue
		}
		if stringField(item, "type") != "reasoning" || strings.TrimSpace(stringField(item, "encrypted_content")) == "" {
			continue
		}
		items = append(items, append(json.RawMessage(nil), raw...))
	}
	return items
}

// responsesFailureError marks a body that IS a recognized Responses object but
// cannot be presented as a successful chat completion. It is distinguished from
// ordinary translation errors so callers pass through only unrecognized bodies.
type responsesFailureError struct {
	message string
}

func (e *responsesFailureError) Error() string {
	return e.message
}

func validateResponsesStatus(resp map[string]any) error {
	status := stringField(resp, "status")
	switch status {
	case "failed", "cancelled", "in_progress", "queued":
		message := ""
		if failure, ok := resp["error"].(map[string]any); ok {
			message = stringField(failure, "message")
		}
		if message == "" {
			return &responsesFailureError{message: fmt.Sprintf("responses API returned status %q", status)}
		}
		return &responsesFailureError{message: fmt.Sprintf("responses API returned status %q: %s", status, message)}
	default:
		return nil
	}
}

// chatCompletionToSSE re-emits a buffered chat completion as a chat SSE stream.
// The adapter always dispatches non-streaming upstream, so this is what an agent
// that asked for a stream receives.
func chatCompletionToSSE(body []byte, includeUsage bool) ([]byte, error) {
	var completion map[string]any
	if err := json.Unmarshal(body, &completion); err != nil {
		return nil, err
	}

	id, _ := completion["id"].(string)
	if strings.TrimSpace(id) == "" {
		id = "chatcmpl-responses-adapter"
	}
	model, _ := completion["model"].(string)
	created := time.Now().Unix()
	if raw, ok := completion["created"].(float64); ok {
		created = int64(raw)
	}

	envelope := func(choices any) map[string]any {
		return map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": created,
			"model":   model,
			"choices": choices,
		}
	}
	deltaChunk := func(delta map[string]any) map[string]any {
		return envelope([]map[string]any{{"index": 0, "delta": delta, "finish_reason": nil}})
	}

	var message map[string]any
	finishReason := "stop"
	if choices, ok := completion["choices"].([]any); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]any); ok {
			message, _ = choice["message"].(map[string]any)
			if reason, ok := choice["finish_reason"].(string); ok && reason != "" {
				finishReason = reason
			}
		}
	}

	var stream bytes.Buffer
	writeSSEChunk(&stream, deltaChunk(map[string]any{"role": "assistant"}))

	if content, ok := message["content"].(string); ok && content != "" {
		writeSSEChunk(&stream, deltaChunk(map[string]any{"content": content}))
	}
	if refusal, ok := message["refusal"].(string); ok && refusal != "" {
		writeSSEChunk(&stream, deltaChunk(map[string]any{"refusal": refusal}))
	}
	if calls, ok := message["tool_calls"].([]any); ok {
		for i, raw := range calls {
			call, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			fn, _ := call["function"].(map[string]any)
			arguments := stringField(fn, "arguments")
			if strings.TrimSpace(arguments) == "" {
				arguments = "{}"
			}
			writeSSEChunk(&stream, deltaChunk(map[string]any{
				"tool_calls": []map[string]any{{
					"index":    i,
					"id":       stringField(call, "id"),
					"type":     "function",
					"function": map[string]any{"name": stringField(fn, "name"), "arguments": arguments},
				}},
			}))
		}
	}

	writeSSEChunk(&stream, envelope([]map[string]any{{
		"index":         0,
		"delta":         map[string]any{},
		"finish_reason": finishReason,
	}}))

	if usage, ok := completion["usage"]; ok && includeUsage {
		final := envelope([]any{})
		final["usage"] = usage
		writeSSEChunk(&stream, final)
	}

	stream.WriteString("data: [DONE]\n\n")
	return stream.Bytes(), nil
}

func responsesOutputText(item map[string]any) (string, bool) {
	parts, ok := item["content"].([]any)
	if !ok {
		return "", false
	}
	var joined string
	found := false
	for _, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if part["type"] != "output_text" && part["type"] != "text" {
			continue
		}
		if text, ok := part["text"].(string); ok {
			joined += text
			found = true
		}
	}
	return joined, found
}

func responsesOutputRefusal(item map[string]any) (string, bool) {
	parts, ok := item["content"].([]any)
	if !ok {
		return "", false
	}
	var joined string
	found := false
	for _, raw := range parts {
		part, ok := raw.(map[string]any)
		if !ok || part["type"] != "refusal" {
			continue
		}
		if explanation, ok := part["refusal"].(string); ok {
			joined += explanation
			found = true
		}
	}
	return joined, found
}

func responsesFinishReason(resp map[string]any, hasToolCalls bool) string {
	if details, ok := resp["incomplete_details"].(map[string]any); ok {
		switch stringField(details, "reason") {
		case "max_output_tokens":
			return "length"
		case "content_filter":
			return "content_filter"
		}
	}
	if hasToolCalls {
		return "tool_calls"
	}
	return "stop"
}

func responsesUsageToChatUsage(raw any) (map[string]any, bool) {
	usage, ok := raw.(map[string]any)
	if !ok {
		return nil, false
	}
	out := map[string]any{}
	for chatKey, responsesKey := range map[string]string{
		"prompt_tokens":     "input_tokens",
		"completion_tokens": "output_tokens",
		"total_tokens":      "total_tokens",
	} {
		if value, ok := usage[responsesKey]; ok {
			out[chatKey] = value
		}
	}
	if details, ok := usage["input_tokens_details"]; ok {
		out["prompt_tokens_details"] = details
	}
	if details, ok := usage["output_tokens_details"]; ok {
		out["completion_tokens_details"] = details
	}
	if reported, ok := usage["cost"]; ok {
		out["cost"] = reported
	}
	return out, true
}

func chatToolsToResponsesTools(tools []any) []any {
	out := make([]any, 0, len(tools))
	for _, raw := range tools {
		tool, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		fn, ok := tool["function"].(map[string]any)
		if !ok {
			// Non-function tools already use the flat Responses shape.
			out = append(out, tool)
			continue
		}
		flat := map[string]any{"type": "function"}
		for key, value := range fn {
			flat[key] = value
		}
		if _, ok := flat["strict"]; !ok {
			flat["strict"] = false
		}
		out = append(out, flat)
	}
	return out
}

func chatToolChoiceToResponses(choice any) any {
	named, ok := choice.(map[string]any)
	if !ok {
		return choice
	}
	fn, ok := named["function"].(map[string]any)
	if !ok {
		return named
	}
	flat := map[string]any{}
	for key, value := range named {
		if key == "function" {
			continue
		}
		flat[key] = value
	}
	for key, value := range fn {
		flat[key] = value
	}
	return flat
}

func chatResponseFormatToResponsesText(raw any) (map[string]any, bool) {
	format, ok := raw.(map[string]any)
	if !ok {
		return nil, false
	}
	flat := map[string]any{}
	for key, value := range format {
		if key == "json_schema" {
			continue
		}
		flat[key] = value
	}
	if schema, ok := format["json_schema"].(map[string]any); ok {
		for key, value := range schema {
			flat[key] = value
		}
	}
	return map[string]any{"format": flat}, true
}

func firstPresent(payload map[string]any, keys ...string) (any, bool) {
	for _, key := range keys {
		if value, ok := payload[key]; ok {
			return value, true
		}
	}
	return nil, false
}
