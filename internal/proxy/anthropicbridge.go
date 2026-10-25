package proxy

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"
)

// The Anthropic Messages adapter serves OpenAI chat/completions requests from
// runners that only speak that dialect (PicoClaw, for one) against a provider
// whose wire format is Anthropic's. Like the Responses adapter it translates at
// the provider boundary only: audit, budget, session history, policy and
// managed tool mediation keep observing the agent-facing chat/completions
// shape. The upstream call is always buffered; a streamed downstream request
// is re-emitted as synthetic chat SSE from the translated completion.

const (
	// anthropicMessagesPath is the upstream path the adapter dispatches to,
	// expressed inbound-style because buildUpstreamURL strips the /v1 prefix.
	anthropicMessagesPath = "/v1/messages"
	// anthropicAPIVersion is sent on every adapted request; the Messages API
	// rejects requests without it.
	anthropicAPIVersion = "2023-06-01"
	// anthropicDefaultMaxTokens fills max_tokens, which Anthropic requires and
	// chat/completions leaves optional. The substitution is logged as the
	// anthropicDefaultMaxTokensIntervention mutation.
	anthropicDefaultMaxTokens             = 8192
	anthropicDefaultMaxTokensIntervention = "anthropic_adapter_default_max_tokens"
	anthropicTopPDroppedIntervention      = "anthropic_adapter_dropped_top_p"
	anthropicSamplingDroppedIntervention  = "anthropic_adapter_dropped_sampling"
	anthropicMessagesAdapterIntervention  = "anthropic_messages_adapter"
)

// anthropicIgnoredChatFields are chat/completions transport or bookkeeping
// fields with no Anthropic counterpart and no effect on the model's answer.
var anthropicIgnoredChatFields = map[string]bool{
	"model":            true,
	"messages":         true,
	"stream":           true,
	"stream_options":   true,
	"metadata":         true,
	"prompt_cache_key": true,
}

// anthropicModelAcceptsSampling reports whether a model still accepts
// temperature and top_p. Claude generations after 4.6 (Opus 4.7 and 4.8, and
// every 5.x, Fable and Mythos model) reject non-default sampling values, so
// sampling is forwarded only to Claude 3 and Claude 4.0-4.6 model IDs. An
// unrecognized ID is treated as current and gets no sampling fields.
func anthropicModelAcceptsSampling(model string) bool {
	rest, ok := strings.CutPrefix(strings.ToLower(strings.TrimSpace(model)), "claude-")
	if !ok {
		return false
	}
	if strings.HasPrefix(rest, "3-") || strings.HasPrefix(rest, "3.") {
		return true
	}
	parts := strings.Split(rest, "-")
	if len(parts) < 2 {
		return false
	}
	switch parts[0] {
	case "opus", "sonnet", "haiku":
	default:
		return false
	}
	major, err := strconv.Atoi(parts[1])
	if err != nil {
		return false
	}
	if major < 4 {
		return true
	}
	if major > 4 {
		return false
	}
	// claude-<family>-4, -4-<minor>, or -4-<yyyymmdd> (minor 0).
	if len(parts) == 2 || len(parts[2]) == 8 {
		return true
	}
	minor, err := strconv.Atoi(parts[2])
	return err == nil && minor <= 6
}

// anthropicAdapterError builds a client error. Callers pass only fixed text,
// array indices and field names from the adapter's own vocabulary: the
// message is returned to the agent and logged, so request-supplied keys,
// roles or types must never be interpolated into it.
func anthropicAdapterError(format string, args ...any) error {
	return &responsesAdapterRequestError{message: fmt.Sprintf(format, args...)}
}

// chatToAnthropicRequest translates a chat/completions payload into an
// Anthropic Messages payload. Semantics Anthropic cannot honor are rejected
// with a client error rather than silently dropped. The input is not mutated.
func chatToAnthropicRequest(payload map[string]any) (map[string]any, []string, error) {
	out := map[string]any{"model": payload["model"]}
	var mutations []string

	keys := make([]string, 0, len(payload))
	for key := range payload {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		value := payload[key]
		switch key {
		case "max_tokens", "max_completion_tokens", "temperature", "top_p", "stop",
			"tools", "tool_choice", "parallel_tool_calls", "user":
			// Translated below.
		case "n":
			if !isSingleChoice(value) {
				return nil, nil, anthropicAdapterError("anthropic adapter: n must be 1")
			}
		case "response_format":
			if format, ok := value.(map[string]any); value != nil && (!ok || stringField(format, "type") != "text") {
				return nil, nil, anthropicAdapterError("anthropic adapter: response_format is not supported")
			}
		case "frequency_penalty", "presence_penalty":
			if !isZeroOrNil(value) {
				return nil, nil, anthropicAdapterError("anthropic adapter: %s is not supported", key)
			}
		case "logprobs":
			if enabled, ok := value.(bool); value != nil && (!ok || enabled) {
				return nil, nil, anthropicAdapterError("anthropic adapter: logprobs is not supported")
			}
		case "reasoning_effort":
			// Anthropic's effort has no none or minimal level; reject them
			// rather than run the caller at a different effort.
			if value != nil {
				effort, _ := value.(string)
				if effort != "low" && effort != "medium" && effort != "high" {
					return nil, nil, anthropicAdapterError("anthropic adapter: reasoning_effort must be low, medium or high")
				}
				out["output_config"] = map[string]any{"effort": effort}
			}
		case "top_logprobs", "seed", "modalities", "audio", "prediction", "web_search_options", "functions", "function_call":
			if value != nil {
				return nil, nil, anthropicAdapterError("anthropic adapter: %s is not supported", key)
			}
		case "store":
			if stored, ok := value.(bool); value != nil && (!ok || stored) {
				return nil, nil, anthropicAdapterError("anthropic adapter: store is not supported")
			}
		case "service_tier":
			if tier, ok := value.(string); value != nil && (!ok || (tier != "auto" && tier != "default")) {
				return nil, nil, anthropicAdapterError("anthropic adapter: service_tier is not supported")
			}
		case "logit_bias":
			if !isEmptyMapOrNil(value) {
				return nil, nil, anthropicAdapterError("anthropic adapter: logit_bias is not supported")
			}
		default:
			if !anthropicIgnoredChatFields[key] {
				return nil, nil, anthropicAdapterError("anthropic adapter: unsupported request field")
			}
		}
	}

	maxTokens, ok := firstPresent(payload, "max_completion_tokens", "max_tokens")
	if !ok || maxTokens == nil {
		maxTokens = anthropicDefaultMaxTokens
		mutations = append(mutations, anthropicDefaultMaxTokensIntervention)
	}
	out["max_tokens"] = maxTokens

	temperature, hasTemperature := payload["temperature"]
	hasTemperature = hasTemperature && temperature != nil
	topP, hasTopP := payload["top_p"]
	hasTopP = hasTopP && topP != nil
	model, _ := payload["model"].(string)
	switch {
	case !anthropicModelAcceptsSampling(model):
		// Runners such as PicoClaw send a sampling default on every turn,
		// which these models reject outright.
		if hasTemperature || hasTopP {
			mutations = append(mutations, anthropicSamplingDroppedIntervention)
		}
	case hasTemperature:
		// Anthropic rejects temperature and top_p together.
		out["temperature"] = temperature
		if hasTopP {
			mutations = append(mutations, anthropicTopPDroppedIntervention)
		}
	case hasTopP:
		out["top_p"] = topP
	}

	if stop, ok := payload["stop"]; ok && !isEmptyStop(stop) {
		switch typed := stop.(type) {
		case string:
			out["stop_sequences"] = []any{typed}
		case []any:
			out["stop_sequences"] = typed
		default:
			return nil, nil, anthropicAdapterError("anthropic adapter: stop must be a string or an array of strings")
		}
	}

	if user, ok := payload["user"].(string); ok && strings.TrimSpace(user) != "" {
		out["metadata"] = map[string]any{"user_id": user}
	}

	messages, _ := payload["messages"].([]any)
	system, converted, err := chatMessagesToAnthropic(messages)
	if err != nil {
		return nil, nil, err
	}
	if len(system) > 0 {
		out["system"] = system
	}
	out["messages"] = converted

	if err := chatToolsToAnthropic(payload, out); err != nil {
		return nil, nil, err
	}
	return out, mutations, nil
}

// chatMessagesToAnthropic splits system/developer messages into Anthropic's
// top-level system blocks and converts the rest into content-block messages.
// Consecutive messages of the same role are merged, which keeps a run of tool
// results in one user turn in their original order.
func chatMessagesToAnthropic(messages []any) ([]any, []any, error) {
	var system []any
	var out []any
	appendBlocks := func(role string, blocks []any) {
		if len(blocks) == 0 {
			return
		}
		if n := len(out); n > 0 {
			if last, _ := out[n-1].(map[string]any); last["role"] == role {
				last["content"] = append(last["content"].([]any), blocks...)
				return
			}
		}
		out = append(out, map[string]any{"role": role, "content": blocks})
	}

	for i, raw := range messages {
		message, ok := raw.(map[string]any)
		if !ok {
			return nil, nil, anthropicAdapterError("anthropic adapter: messages[%d] is not an object", i)
		}
		role := stringField(message, "role")
		switch role {
		case "system", "developer":
			blocks, err := chatContentToAnthropicBlocks(message["content"], false)
			if err != nil {
				return nil, nil, err
			}
			system = append(system, blocks...)
		case "user":
			blocks, err := chatContentToAnthropicBlocks(message["content"], true)
			if err != nil {
				return nil, nil, err
			}
			appendBlocks("user", blocks)
		case "assistant":
			blocks, err := chatContentToAnthropicBlocks(message["content"], false)
			if err != nil {
				return nil, nil, err
			}
			calls, _ := message["tool_calls"].([]any)
			for j, rawCall := range calls {
				call, ok := rawCall.(map[string]any)
				if !ok {
					return nil, nil, anthropicAdapterError("anthropic adapter: messages[%d].tool_calls[%d] is not an object", i, j)
				}
				fn, _ := call["function"].(map[string]any)
				input := map[string]any{}
				if args := strings.TrimSpace(stringField(fn, "arguments")); args != "" {
					// "null" decodes into a nil map without error.
					if err := json.Unmarshal([]byte(args), &input); err != nil || input == nil {
						return nil, nil, anthropicAdapterError("anthropic adapter: messages[%d].tool_calls[%d] arguments are not a JSON object", i, j)
					}
				}
				blocks = append(blocks, map[string]any{
					"type":  "tool_use",
					"id":    stringField(call, "id"),
					"name":  stringField(fn, "name"),
					"input": input,
				})
			}
			appendBlocks("assistant", blocks)
		case "tool":
			content, err := chatToolResultToAnthropic(message["content"])
			if err != nil {
				return nil, nil, err
			}
			result := map[string]any{
				"type":        "tool_result",
				"tool_use_id": stringField(message, "tool_call_id"),
				"content":     content,
			}
			appendBlocks("user", []any{result})
		default:
			return nil, nil, anthropicAdapterError("anthropic adapter: messages[%d] role is not supported", i)
		}
	}
	return system, out, nil
}

// chatToolResultToAnthropic keeps string results as strings and converts part
// arrays with the same fail-closed rules as user content, so an attached image
// is forwarded and an unsupported part is rejected instead of dropped.
func chatToolResultToAnthropic(content any) (any, error) {
	switch typed := content.(type) {
	case nil:
		return "", nil
	case string:
		return typed, nil
	}
	blocks, err := chatContentToAnthropicBlocks(content, true)
	if err != nil {
		return nil, err
	}
	if len(blocks) == 0 {
		return "", nil
	}
	return blocks, nil
}

// chatContentToAnthropicBlocks converts string or part-array content. Only
// text parts are accepted, plus images on user turns; anything else is a
// client error rather than a silently dropped attachment.
func chatContentToAnthropicBlocks(content any, allowImages bool) ([]any, error) {
	switch typed := content.(type) {
	case nil:
		return nil, nil
	case string:
		if typed == "" {
			return nil, nil
		}
		return []any{map[string]any{"type": "text", "text": typed}}, nil
	case []any:
		blocks := make([]any, 0, len(typed))
		for _, raw := range typed {
			part, ok := raw.(map[string]any)
			if !ok {
				return nil, anthropicAdapterError("anthropic adapter: content part is not an object")
			}
			switch stringField(part, "type") {
			case "text":
				if text := stringField(part, "text"); text != "" {
					blocks = append(blocks, map[string]any{"type": "text", "text": text})
				}
			case "image_url":
				if !allowImages {
					return nil, anthropicAdapterError("anthropic adapter: image content is only supported on user messages")
				}
				block, err := chatImageToAnthropic(part)
				if err != nil {
					return nil, err
				}
				blocks = append(blocks, block)
			default:
				return nil, anthropicAdapterError("anthropic adapter: content part type is not supported")
			}
		}
		return blocks, nil
	default:
		return nil, anthropicAdapterError("anthropic adapter: message content must be a string or an array of parts")
	}
}

func chatImageToAnthropic(part map[string]any) (map[string]any, error) {
	var url string
	switch image := part["image_url"].(type) {
	case string:
		url = image
	case map[string]any:
		url = stringField(image, "url")
	}
	url = strings.TrimSpace(url)
	if rest, ok := strings.CutPrefix(url, "data:"); ok {
		meta, data, found := strings.Cut(rest, ",")
		mediaType, isBase64 := strings.CutSuffix(meta, ";base64")
		if !found || !isBase64 || mediaType == "" {
			return nil, anthropicAdapterError("anthropic adapter: image data URLs must be base64")
		}
		return map[string]any{
			"type":   "image",
			"source": map[string]any{"type": "base64", "media_type": mediaType, "data": data},
		}, nil
	}
	if strings.HasPrefix(url, "https://") || strings.HasPrefix(url, "http://") {
		return map[string]any{
			"type":   "image",
			"source": map[string]any{"type": "url", "url": url},
		}, nil
	}
	return nil, anthropicAdapterError("anthropic adapter: image_url must be an http(s) or base64 data URL")
}

func chatToolsToAnthropic(payload, out map[string]any) error {
	rawTools, _ := payload["tools"].([]any)
	tools := make([]any, 0, len(rawTools))
	for i, raw := range rawTools {
		tool, ok := raw.(map[string]any)
		if !ok || stringField(tool, "type") != "function" {
			return anthropicAdapterError("anthropic adapter: tools[%d] must be a function tool", i)
		}
		fn, _ := tool["function"].(map[string]any)
		name := stringField(fn, "name")
		if name == "" {
			return anthropicAdapterError("anthropic adapter: tools[%d] has no name", i)
		}
		schema, _ := fn["parameters"].(map[string]any)
		if schema == nil {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		converted := map[string]any{"name": name, "input_schema": schema}
		if description := stringField(fn, "description"); description != "" {
			converted["description"] = description
		}
		switch strict := fn["strict"].(type) {
		case nil:
		case bool:
			if strict {
				converted["strict"] = true
			}
		default:
			return anthropicAdapterError("anthropic adapter: tools[%d] strict must be a boolean", i)
		}
		tools = append(tools, converted)
	}
	if len(tools) > 0 {
		out["tools"] = tools
	}

	var choice map[string]any
	switch typed := payload["tool_choice"].(type) {
	case nil:
	case string:
		switch typed {
		case "auto":
			choice = map[string]any{"type": "auto"}
		case "none":
			choice = map[string]any{"type": "none"}
		case "required":
			choice = map[string]any{"type": "any"}
		default:
			return anthropicAdapterError("anthropic adapter: tool_choice is not supported")
		}
	case map[string]any:
		fn, _ := typed["function"].(map[string]any)
		name := stringField(fn, "name")
		if stringField(typed, "type") != "function" || name == "" {
			return anthropicAdapterError("anthropic adapter: tool_choice must name a function")
		}
		choice = map[string]any{"type": "tool", "name": name}
	default:
		return anthropicAdapterError("anthropic adapter: tool_choice is not supported")
	}
	if parallel, ok := payload["parallel_tool_calls"].(bool); ok && !parallel && len(tools) > 0 {
		if choice == nil {
			choice = map[string]any{"type": "auto"}
		}
		if choice["type"] != "none" {
			choice["disable_parallel_tool_use"] = true
		}
	}
	if choice != nil && (len(tools) > 0 || choice["type"] == "none") {
		out["tool_choice"] = choice
	}
	return nil
}

// anthropicToChatCompletion translates an Anthropic Messages response into the
// chat/completions shape. A body that is not a Messages object (an upstream
// error, for one) is reported as not converted so it passes through untouched.
func anthropicToChatCompletion(body []byte) ([]byte, bool, error) {
	var resp map[string]any
	if err := json.Unmarshal(body, &resp); err != nil || stringField(resp, "type") != "message" {
		return nil, false, nil
	}
	blocks, _ := resp["content"].([]any)
	var text strings.Builder
	var toolCalls []any
	for _, raw := range blocks {
		block, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		switch stringField(block, "type") {
		case "text":
			text.WriteString(stringField(block, "text"))
		case "tool_use":
			input := block["input"]
			if input == nil {
				input = map[string]any{}
			}
			arguments, err := json.Marshal(input)
			if err != nil {
				return nil, true, err
			}
			toolCalls = append(toolCalls, map[string]any{
				"id":   stringField(block, "id"),
				"type": "function",
				"function": map[string]any{
					"name":      stringField(block, "name"),
					"arguments": string(arguments),
				},
			})
		case "thinking", "redacted_thinking":
			// Not requested by the adapter and not representable in the chat
			// shape; never surfaced as answer text.
		default:
			return nil, true, fmt.Errorf("anthropic adapter: unsupported response block")
		}
	}

	message := map[string]any{"role": "assistant", "content": nil}
	if text.Len() > 0 || len(toolCalls) == 0 {
		message["content"] = text.String()
	}
	if len(toolCalls) > 0 {
		message["tool_calls"] = toolCalls
	}
	completion := map[string]any{
		"id":      stringField(resp, "id"),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   stringField(resp, "model"),
		"choices": []any{map[string]any{
			"index":         0,
			"message":       message,
			"finish_reason": anthropicFinishReason(stringField(resp, "stop_reason"), len(toolCalls) > 0),
		}},
	}
	if usage, ok := anthropicUsageToChatUsage(resp["usage"]); ok {
		completion["usage"] = usage
	}
	out, err := json.Marshal(completion)
	return out, true, err
}

func anthropicFinishReason(stopReason string, hasToolCalls bool) string {
	switch stopReason {
	case "max_tokens", "model_context_window_exceeded":
		return "length"
	case "tool_use":
		return "tool_calls"
	case "refusal":
		return "content_filter"
	}
	if hasToolCalls {
		return "tool_calls"
	}
	return "stop"
}

// anthropicUsageToChatUsage reports prompt_tokens in OpenAI terms (all input,
// cached or not) and keeps Anthropic's cache split, both in the chat details
// and in the native top-level fields the cost parser already understands.
func anthropicUsageToChatUsage(raw any) (map[string]any, bool) {
	usage, ok := raw.(map[string]any)
	if !ok {
		return nil, false
	}
	input := intFromJSON(usage["input_tokens"])
	cacheRead := intFromJSON(usage["cache_read_input_tokens"])
	cacheWrite := intFromJSON(usage["cache_creation_input_tokens"])
	output := intFromJSON(usage["output_tokens"])
	prompt := input + cacheRead + cacheWrite
	out := map[string]any{
		"prompt_tokens":     prompt,
		"completion_tokens": output,
		"total_tokens":      prompt + output,
		"prompt_tokens_details": map[string]any{
			"cached_tokens":      cacheRead,
			"cache_write_tokens": cacheWrite,
		},
		"cache_read_input_tokens":     cacheRead,
		"cache_creation_input_tokens": cacheWrite,
	}
	return out, true
}

func intFromJSON(raw any) int {
	if value, ok := raw.(float64); ok {
		return int(value)
	}
	return 0
}
