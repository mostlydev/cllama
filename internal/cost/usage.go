package cost

import (
	"bytes"
	"encoding/json"
	"strconv"
)

// Usage holds token counts from an OpenAI-compatible response.
type Usage struct {
	PromptTokens     int      `json:"prompt_tokens"`
	CompletionTokens int      `json:"completion_tokens"`
	TotalTokens      int      `json:"total_tokens"`
	ReportedCostUSD  *float64 `json:"-"`
	CachedTokens     *int     `json:"-"`
	CacheWriteTokens *int     `json:"-"`
}

// ExtractUsage parses usage from a non-streamed JSON response body.
func ExtractUsage(body []byte) (Usage, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return Usage{}, err
	}
	return extractUsageFromPayload(payload), nil
}

// ExtractUsageFromSSE scans SSE data lines for the last one containing a "usage" field.
// OpenAI streams include usage in the final data chunk before "data: [DONE]".
func ExtractUsageFromSSE(stream []byte) (Usage, error) {
	var observed Usage
	for _, line := range bytes.Split(stream, []byte("\n")) {
		line = bytes.TrimSpace(line)
		if !bytes.HasPrefix(line, []byte("data: ")) {
			continue
		}
		payload := bytes.TrimPrefix(line, []byte("data: "))
		if bytes.Equal(payload, []byte("[DONE]")) {
			continue
		}
		var chunk map[string]any
		if json.Unmarshal(payload, &chunk) == nil {
			observed = mergeUsage(observed, extractUsageFromPayload(chunk))
		}
	}
	return observed, nil
}

func extractUsageFromPayload(payload map[string]any) Usage {
	var usage Usage
	if raw, ok := payload["usage"]; ok {
		usage = mergeUsage(usage, parseUsageObject(raw))
	}
	if rawMessage, ok := payload["message"].(map[string]any); ok {
		if rawUsage, ok := rawMessage["usage"]; ok {
			usage = mergeUsage(usage, parseUsageObject(rawUsage))
		}
	}
	return usage
}

func parseUsageObject(raw any) Usage {
	obj, ok := raw.(map[string]any)
	if !ok {
		return Usage{}
	}

	prompt := firstInt(obj, "prompt_tokens", "input_tokens")
	completion := firstInt(obj, "completion_tokens", "output_tokens")
	total := intFromAny(obj["total_tokens"])
	if total == 0 && (prompt > 0 || completion > 0) {
		total = prompt + completion
	}

	usage := Usage{
		PromptTokens:     prompt,
		CompletionTokens: completion,
		TotalTokens:      total,
		ReportedCostUSD:  floatPtrFromAny(obj["cost"]),
	}
	if details, ok := obj["prompt_tokens_details"].(map[string]any); ok {
		if v, ok := intPtrFromAny(details["cached_tokens"]); ok {
			usage.CachedTokens = v
		}
		if v, ok := intPtrFromAny(details["cache_write_tokens"]); ok {
			usage.CacheWriteTokens = v
		}
	}
	if v, ok := intPtrFromAny(obj["cache_read_input_tokens"]); ok {
		usage.CachedTokens = v
	}
	if v, ok := intPtrFromAny(obj["cache_creation_input_tokens"]); ok {
		usage.CacheWriteTokens = v
	}
	return usage
}

func mergeUsage(base, next Usage) Usage {
	if next.PromptTokens > base.PromptTokens {
		base.PromptTokens = next.PromptTokens
	}
	if next.CompletionTokens > base.CompletionTokens {
		base.CompletionTokens = next.CompletionTokens
	}
	if next.TotalTokens > base.TotalTokens {
		base.TotalTokens = next.TotalTokens
	}
	if derivedTotal := base.PromptTokens + base.CompletionTokens; derivedTotal > base.TotalTokens {
		base.TotalTokens = derivedTotal
	}
	if next.ReportedCostUSD != nil {
		if base.ReportedCostUSD == nil || *next.ReportedCostUSD > *base.ReportedCostUSD {
			cost := *next.ReportedCostUSD
			base.ReportedCostUSD = &cost
		}
	}
	if next.CachedTokens != nil {
		if base.CachedTokens == nil || *next.CachedTokens > *base.CachedTokens {
			v := *next.CachedTokens
			base.CachedTokens = &v
		}
	}
	if next.CacheWriteTokens != nil {
		if base.CacheWriteTokens == nil || *next.CacheWriteTokens > *base.CacheWriteTokens {
			v := *next.CacheWriteTokens
			base.CacheWriteTokens = &v
		}
	}
	return base
}

func firstInt(obj map[string]any, keys ...string) int {
	for _, key := range keys {
		if v, ok := obj[key]; ok {
			if parsed := intFromAny(v); parsed > 0 {
				return parsed
			}
		}
	}
	return 0
}

func intFromAny(v any) int {
	switch t := v.(type) {
	case float64:
		return int(t)
	case float32:
		return int(t)
	case int:
		return t
	case int64:
		return int(t)
	case int32:
		return int(t)
	case json.Number:
		n, err := t.Int64()
		if err == nil {
			return int(n)
		}
	case string:
		n, err := strconv.Atoi(t)
		if err == nil {
			return n
		}
	}
	return 0
}

func intPtrFromAny(v any) (*int, bool) {
	switch t := v.(type) {
	case nil:
		return nil, false
	case float64:
		out := int(t)
		return &out, true
	case float32:
		out := int(t)
		return &out, true
	case int:
		out := t
		return &out, true
	case int64:
		out := int(t)
		return &out, true
	case int32:
		out := int(t)
		return &out, true
	case json.Number:
		n, err := t.Int64()
		if err != nil {
			return nil, false
		}
		out := int(n)
		return &out, true
	case string:
		n, err := strconv.Atoi(t)
		if err != nil {
			return nil, false
		}
		return &n, true
	default:
		return nil, false
	}
}

func floatPtrFromAny(v any) *float64 {
	switch t := v.(type) {
	case float64:
		out := t
		return &out
	case float32:
		out := float64(t)
		return &out
	case int:
		out := float64(t)
		return &out
	case int64:
		out := float64(t)
		return &out
	case json.Number:
		n, err := t.Float64()
		if err == nil {
			return &n
		}
	case string:
		n, err := strconv.ParseFloat(t, 64)
		if err == nil {
			return &n
		}
	}
	return nil
}
