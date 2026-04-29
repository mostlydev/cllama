package proxy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"

	"github.com/mostlydev/cllama/internal/logging"
)

func requestTelemetry(format string, payload map[string]any, dynamicContext string) *logging.RequestInfo {
	info := &logging.RequestInfo{
		DynamicContextHash: hashText(dynamicContext),
		ToolsHash:          hashJSONValue(payload["tools"]),
	}
	switch format {
	case "openai":
		info.StaticSystemHash = hashOpenAISystem(payload)
		info.FirstSystemHash = info.StaticSystemHash
		info.FirstNonSystemHash = hashOpenAIFirstNonSystem(payload)
	case "anthropic":
		info.StaticSystemHash = hashJSONValue(payload["system"])
		info.FirstSystemHash = info.StaticSystemHash
		info.FirstNonSystemHash = hashAnthropicFirstNonSystem(payload)
	}
	return info
}

func hashOpenAISystem(payload map[string]any) string {
	messages, _ := payload["messages"].([]any)
	if len(messages) == 0 {
		return ""
	}
	first, _ := messages[0].(map[string]any)
	if first == nil {
		return ""
	}
	if role, _ := first["role"].(string); role != "system" {
		return ""
	}
	return hashJSONValue(first["content"])
}

func hashOpenAIFirstNonSystem(payload map[string]any) string {
	messages, _ := payload["messages"].([]any)
	for _, raw := range messages {
		msg, _ := raw.(map[string]any)
		if msg == nil {
			continue
		}
		if role, _ := msg["role"].(string); role == "system" {
			continue
		}
		return hashJSONValue(msg)
	}
	return ""
}

func hashAnthropicFirstNonSystem(payload map[string]any) string {
	messages, _ := payload["messages"].([]any)
	for _, raw := range messages {
		msg, _ := raw.(map[string]any)
		if msg == nil {
			continue
		}
		return hashJSONValue(msg)
	}
	return ""
}

func hashText(s string) string {
	if s == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(s))
	return hex.EncodeToString(sum[:])
}

func hashJSONValue(v any) string {
	if v == nil {
		return ""
	}
	data, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}
