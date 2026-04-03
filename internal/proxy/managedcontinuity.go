package proxy

import (
	"encoding/json"
	"strings"
	"sync"
)

const defaultManagedToolContinuityTurns = 8

type managedOpenAIContinuityStore struct {
	mu       sync.RWMutex
	maxTurns int
	agents   map[string][]managedOpenAIContinuityTurn
}

type managedOpenAIContinuityTurn struct {
	Anchor         string
	HiddenMessages []json.RawMessage
}

func newManagedOpenAIContinuityStore(maxTurns int) *managedOpenAIContinuityStore {
	if maxTurns <= 0 {
		maxTurns = defaultManagedToolContinuityTurns
	}
	return &managedOpenAIContinuityStore{
		maxTurns: maxTurns,
		agents:   make(map[string][]managedOpenAIContinuityTurn),
	}
}

func (s *managedOpenAIContinuityStore) Record(agentID string, finalAssistant map[string]any, hiddenMessages []json.RawMessage) {
	if s == nil || strings.TrimSpace(agentID) == "" || len(hiddenMessages) == 0 {
		return
	}
	anchor, ok := openAIVisibleAssistantAnchor(finalAssistant)
	if !ok {
		return
	}

	turn := managedOpenAIContinuityTurn{
		Anchor:         anchor,
		HiddenMessages: cloneRawMessages(hiddenMessages),
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	turns := append(s.agents[agentID], turn)
	if len(turns) > s.maxTurns {
		turns = turns[len(turns)-s.maxTurns:]
	}
	s.agents[agentID] = turns
}

func (s *managedOpenAIContinuityStore) Inject(agentID string, payload map[string]any) bool {
	if s == nil || strings.TrimSpace(agentID) == "" {
		return false
	}
	messages, _ := payload["messages"].([]any)
	if len(messages) == 0 {
		return false
	}

	s.mu.RLock()
	turns := cloneContinuityTurns(s.agents[agentID])
	s.mu.RUnlock()
	if len(turns) == 0 {
		return false
	}

	out := make([]any, 0, len(messages)+hiddenMessageCount(turns))
	turnIdx := 0
	inserted := false
	for _, rawMsg := range messages {
		msg, _ := rawMsg.(map[string]any)
		if turnIdx < len(turns) && openAIMessageMatchesAnchor(msg, turns[turnIdx].Anchor) {
			for _, hidden := range turns[turnIdx].HiddenMessages {
				var decoded any
				if err := json.Unmarshal(hidden, &decoded); err != nil {
					continue
				}
				out = append(out, decoded)
				inserted = true
			}
			turnIdx++
		}
		out = append(out, rawMsg)
	}
	if inserted {
		payload["messages"] = out
	}
	return inserted
}

func appendManagedOpenAIContinuityMessages(dst []json.RawMessage, assistantMessage map[string]any, toolMessages []any) []json.RawMessage {
	if raw, err := json.Marshal(assistantMessage); err == nil {
		dst = append(dst, append(json.RawMessage(nil), raw...))
	}
	for _, msg := range toolMessages {
		raw, err := json.Marshal(msg)
		if err != nil {
			continue
		}
		dst = append(dst, append(json.RawMessage(nil), raw...))
	}
	return dst
}

func openAIVisibleAssistantAnchor(message map[string]any) (string, bool) {
	if len(message) == 0 {
		return "", false
	}
	role, _ := message["role"].(string)
	if !strings.EqualFold(strings.TrimSpace(role), "assistant") {
		return "", false
	}
	if toolCalls, _ := message["tool_calls"].([]any); len(toolCalls) > 0 {
		return "", false
	}
	if functionCall, _ := message["function_call"].(map[string]any); len(functionCall) > 0 {
		return "", false
	}

	normalized := map[string]any{
		"role": "assistant",
	}
	if name, _ := message["name"].(string); strings.TrimSpace(name) != "" {
		normalized["name"] = strings.TrimSpace(name)
	}
	if content, ok := normalizeOpenAIMessageContent(message["content"]); ok {
		normalized["content"] = content
	}

	raw, err := json.Marshal(normalized)
	if err != nil {
		return "", false
	}
	return string(raw), true
}

func openAIMessageMatchesAnchor(message map[string]any, anchor string) bool {
	current, ok := openAIVisibleAssistantAnchor(message)
	if !ok {
		return false
	}
	return current == anchor
}

func normalizeOpenAIMessageContent(content any) (any, bool) {
	switch typed := content.(type) {
	case string:
		return strings.TrimSpace(typed), true
	case []any:
		parts := make([]any, 0, len(typed))
		for _, raw := range typed {
			part, _ := raw.(map[string]any)
			if part == nil {
				parts = append(parts, raw)
				continue
			}
			normalized := make(map[string]any, len(part))
			for key, value := range part {
				if key == "text" {
					if text, ok := value.(string); ok {
						normalized[key] = strings.TrimSpace(text)
						continue
					}
				}
				normalized[key] = value
			}
			parts = append(parts, normalized)
		}
		return parts, true
	case nil:
		return nil, true
	default:
		return typed, true
	}
}

func hiddenMessageCount(turns []managedOpenAIContinuityTurn) int {
	total := 0
	for _, turn := range turns {
		total += len(turn.HiddenMessages)
	}
	return total
}

func cloneContinuityTurns(in []managedOpenAIContinuityTurn) []managedOpenAIContinuityTurn {
	if len(in) == 0 {
		return nil
	}
	out := make([]managedOpenAIContinuityTurn, 0, len(in))
	for _, turn := range in {
		out = append(out, managedOpenAIContinuityTurn{
			Anchor:         turn.Anchor,
			HiddenMessages: cloneRawMessages(turn.HiddenMessages),
		})
	}
	return out
}

func cloneRawMessages(in []json.RawMessage) []json.RawMessage {
	if len(in) == 0 {
		return nil
	}
	out := make([]json.RawMessage, 0, len(in))
	for _, raw := range in {
		out = append(out, append(json.RawMessage(nil), raw...))
	}
	return out
}
