package proxy

import (
	"encoding/json"
	"strings"
	"sync"
)

type managedAnthropicContinuityStore struct {
	mu        sync.RWMutex
	maxTurns  int
	maxAgents int
	clock     uint64
	agents    map[string]*managedAnthropicContinuityState
}

type managedAnthropicContinuityState struct {
	Turns      []managedAnthropicContinuityTurn
	LastAccess uint64
}

type managedAnthropicContinuityTurn struct {
	Anchor                  string
	VisibleAssistantFromEnd int
	HiddenMessages          []json.RawMessage
}

func newManagedAnthropicContinuityStore(maxTurns int) *managedAnthropicContinuityStore {
	if maxTurns <= 0 {
		maxTurns = defaultManagedToolContinuityTurns
	}
	return &managedAnthropicContinuityStore{
		maxTurns:  maxTurns,
		maxAgents: defaultManagedToolContinuityAgents,
		agents:    make(map[string]*managedAnthropicContinuityState),
	}
}

func (s *managedAnthropicContinuityStore) ObserveTerminalAssistant(agentID string, finalAssistant map[string]any, hiddenMessages []json.RawMessage) bool {
	if s == nil || strings.TrimSpace(agentID) == "" {
		return false
	}
	anchor, ok := anthropicVisibleAssistantAnchor(finalAssistant)
	if !ok {
		return false
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	state, exists := s.agents[agentID]
	if exists {
		for i := range state.Turns {
			state.Turns[i].VisibleAssistantFromEnd++
		}
		state.Turns = filterAnthropicContinuityTurns(state.Turns, s.maxTurns)
		if len(state.Turns) == 0 && len(hiddenMessages) == 0 {
			delete(s.agents, agentID)
			return false
		}
	}

	if len(hiddenMessages) == 0 {
		if exists {
			s.touchLocked(agentID, state)
		}
		return false
	}

	if !exists {
		state = &managedAnthropicContinuityState{}
		s.agents[agentID] = state
	}
	state.Turns = append(state.Turns, managedAnthropicContinuityTurn{
		Anchor:                  anchor,
		VisibleAssistantFromEnd: 1,
		HiddenMessages:          cloneRawMessages(hiddenMessages),
	})
	state.Turns = filterAnthropicContinuityTurns(state.Turns, s.maxTurns)
	s.touchLocked(agentID, state)
	s.evictAgentsLocked()
	return true
}

func (s *managedAnthropicContinuityStore) Inject(agentID string, payload map[string]any) bool {
	if s == nil || strings.TrimSpace(agentID) == "" {
		return false
	}
	messages, _ := payload["messages"].([]any)
	if len(messages) == 0 {
		return false
	}

	s.mu.Lock()
	state := s.agents[agentID]
	if state == nil || len(state.Turns) == 0 {
		s.mu.Unlock()
		return false
	}
	turns := cloneAnthropicContinuityTurns(state.Turns)
	s.touchLocked(agentID, state)
	s.mu.Unlock()

	totalVisible := countVisibleAnthropicAssistantMessages(messages)
	if totalVisible == 0 {
		return false
	}
	turnsByOffset := make(map[int]managedAnthropicContinuityTurn, len(turns))
	for _, turn := range turns {
		if turn.VisibleAssistantFromEnd <= 0 {
			continue
		}
		turnsByOffset[turn.VisibleAssistantFromEnd] = turn
	}

	out := make([]any, 0, len(messages)+hiddenAnthropicMessageCount(turns))
	visibleSeen := 0
	inserted := false
	for _, rawMsg := range messages {
		msg, _ := rawMsg.(map[string]any)
		if isVisibleAnthropicAssistantMessage(msg) {
			visibleSeen++
			offsetFromEnd := totalVisible - visibleSeen + 1
			if turn, ok := turnsByOffset[offsetFromEnd]; ok && anthropicMessageMatchesAnchor(msg, turn.Anchor) {
				for _, hidden := range turn.HiddenMessages {
					var decoded any
					if err := json.Unmarshal(hidden, &decoded); err != nil {
						continue
					}
					out = append(out, decoded)
					inserted = true
				}
			}
		}
		out = append(out, rawMsg)
	}
	if inserted {
		payload["messages"] = out
	}
	return inserted
}

func (s *managedAnthropicContinuityStore) touchLocked(agentID string, state *managedAnthropicContinuityState) {
	if state == nil {
		return
	}
	s.clock++
	state.LastAccess = s.clock
	s.agents[agentID] = state
}

func (s *managedAnthropicContinuityStore) evictAgentsLocked() {
	if s.maxAgents <= 0 {
		return
	}
	for len(s.agents) > s.maxAgents {
		var (
			evictID   string
			evictSeen bool
			oldest    uint64
		)
		for agentID, state := range s.agents {
			if state == nil {
				evictID = agentID
				evictSeen = true
				break
			}
			if !evictSeen || state.LastAccess < oldest {
				evictID = agentID
				oldest = state.LastAccess
				evictSeen = true
			}
		}
		if !evictSeen {
			return
		}
		delete(s.agents, evictID)
	}
}

func appendManagedAnthropicContinuityMessages(dst []json.RawMessage, assistantMessage map[string]any, toolResultMessage map[string]any) []json.RawMessage {
	if raw, err := json.Marshal(assistantMessage); err == nil {
		dst = append(dst, append(json.RawMessage(nil), raw...))
	}
	if len(toolResultMessage) > 0 {
		if raw, err := json.Marshal(toolResultMessage); err == nil {
			dst = append(dst, append(json.RawMessage(nil), raw...))
		}
	}
	return dst
}

func anthropicVisibleAssistantAnchor(message map[string]any) (string, bool) {
	if !isVisibleAnthropicAssistantMessage(message) {
		return "", false
	}

	normalized := map[string]any{
		"role": "assistant",
	}
	if content, ok := normalizeAnthropicMessageContent(message["content"]); ok {
		normalized["content"] = content
	}
	raw, err := json.Marshal(normalized)
	if err != nil {
		return "", false
	}
	return string(raw), true
}

func anthropicMessageMatchesAnchor(message map[string]any, anchor string) bool {
	current, ok := anthropicVisibleAssistantAnchor(message)
	if !ok {
		return false
	}
	return current == anchor
}

func isVisibleAnthropicAssistantMessage(message map[string]any) bool {
	if len(message) == 0 {
		return false
	}
	role, _ := message["role"].(string)
	if !strings.EqualFold(strings.TrimSpace(role), "assistant") {
		return false
	}
	switch content := message["content"].(type) {
	case []any:
		for _, raw := range content {
			block, _ := raw.(map[string]any)
			if block == nil {
				continue
			}
			if blockType, _ := block["type"].(string); blockType == "tool_use" {
				return false
			}
		}
	}
	return true
}

func normalizeAnthropicMessageContent(content any) (any, bool) {
	switch typed := content.(type) {
	case string:
		return strings.TrimSpace(typed), true
	case []any:
		textParts := make([]string, 0, len(typed))
		allText := true
		parts := make([]any, 0, len(typed))
		for _, raw := range typed {
			part, _ := raw.(map[string]any)
			if part == nil {
				allText = false
				parts = append(parts, raw)
				continue
			}
			normalized := make(map[string]any, len(part))
			for key, value := range part {
				if key == "text" {
					if text, ok := value.(string); ok {
						trimmed := strings.TrimSpace(text)
						normalized[key] = trimmed
						textParts = append(textParts, trimmed)
						continue
					}
				}
				if key == "type" {
					if blockType, ok := value.(string); ok && blockType != "" && blockType != "text" {
						allText = false
					}
				}
				normalized[key] = value
			}
			parts = append(parts, normalized)
		}
		if allText {
			return strings.Join(textParts, "\n"), true
		}
		return parts, true
	case nil:
		return nil, true
	default:
		return typed, true
	}
}

func countVisibleAnthropicAssistantMessages(messages []any) int {
	total := 0
	for _, raw := range messages {
		msg, _ := raw.(map[string]any)
		if isVisibleAnthropicAssistantMessage(msg) {
			total++
		}
	}
	return total
}

func filterAnthropicContinuityTurns(turns []managedAnthropicContinuityTurn, maxTurns int) []managedAnthropicContinuityTurn {
	if len(turns) == 0 {
		return nil
	}
	out := make([]managedAnthropicContinuityTurn, 0, len(turns))
	for _, turn := range turns {
		if turn.VisibleAssistantFromEnd > 0 && (maxTurns <= 0 || turn.VisibleAssistantFromEnd <= maxTurns) {
			out = append(out, turn)
		}
	}
	return out
}

func hiddenAnthropicMessageCount(turns []managedAnthropicContinuityTurn) int {
	total := 0
	for _, turn := range turns {
		total += len(turn.HiddenMessages)
	}
	return total
}

func cloneAnthropicContinuityTurns(in []managedAnthropicContinuityTurn) []managedAnthropicContinuityTurn {
	if len(in) == 0 {
		return nil
	}
	out := make([]managedAnthropicContinuityTurn, 0, len(in))
	for _, turn := range in {
		out = append(out, managedAnthropicContinuityTurn{
			Anchor:                  turn.Anchor,
			VisibleAssistantFromEnd: turn.VisibleAssistantFromEnd,
			HiddenMessages:          cloneRawMessages(turn.HiddenMessages),
		})
	}
	return out
}
