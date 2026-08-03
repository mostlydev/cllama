package proxy

import (
	"bytes"
	"encoding/json"
	"testing"
)

// A replay segment whose tool_call IDs do not match the serialized assistant
// round must NOT be injected — mis-association would replay ciphertext against
// the wrong round. This is the discriminating negative case: it fails if the
// ID comparison is weakened to match-anything.
func TestResponsesReasoningReplayIgnoresMismatchedCallIDs(t *testing.T) {
	const ciphertext = "stale-round-reasoning"
	priorAssistant := map[string]any{
		"role": "assistant",
		"tool_calls": []any{
			map[string]any{
				"id":       "call_prior",
				"type":     "function",
				"function": map[string]any{"name": "managed_tool", "arguments": "{}"},
			},
		},
	}
	replay := appendResponsesReasoningReplay(nil, &capturedResponse{
		ProviderName:       "openai",
		UpstreamModel:      "gpt-5.6-terra",
		ResponsesReasoning: []json.RawMessage{json.RawMessage(`{"type":"reasoning","id":"rs_1","encrypted_content":"` + ciphertext + `"}`)},
	}, priorAssistant)

	currentAssistant := map[string]any{
		"role": "assistant",
		"tool_calls": []any{
			map[string]any{
				"id":       "call_other",
				"type":     "function",
				"function": map[string]any{"name": "managed_tool", "arguments": "{}"},
			},
		},
	}
	payload := map[string]any{
		"model": "gpt-5.6-terra",
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			currentAssistant,
			map[string]any{"role": "tool", "tool_call_id": "call_other", "content": `{"ok":true}`},
		},
	}
	translated, _, err := chatToResponsesRequestWithReasoning(payload, replay)
	if err != nil {
		t.Fatalf("translate with reasoning replay: %v", err)
	}
	body, err := json.Marshal(translated)
	if err != nil {
		t.Fatalf("marshal translated request: %v", err)
	}
	if bytes.Contains(body, []byte(ciphertext)) {
		t.Fatalf("ciphertext from a mismatched round must not be replayed: %s", body)
	}
}
