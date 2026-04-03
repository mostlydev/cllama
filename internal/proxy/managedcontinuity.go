package proxy

import (
	"encoding/json"
	"strings"
	"sync"
)

const (
	defaultManagedToolContinuityTurns  = 8
	defaultManagedToolContinuityAgents = 256
)

type managedOpenAIContinuityStore struct {
	mu        sync.RWMutex
	maxTurns  int
	maxAgents int
	clock     uint64
	agents    map[string]*managedOpenAIContinuityState
}

type managedOpenAIContinuityState struct {
	Turns      []managedOpenAIContinuityTurn
	LastAccess uint64
}

type managedOpenAIContinuityTurn struct {
	Anchor                  string
	VisibleAssistantFromEnd int
	HiddenMessages          []json.RawMessage
}

func newManagedOpenAIContinuityStore(maxTurns int) *managedOpenAIContinuityStore {
	if maxTurns <= 0 {
		maxTurns = defaultManagedToolContinuityTurns
	}
	return &managedOpenAIContinuityStore{
		maxTurns:  maxTurns,
		maxAgents: defaultManagedToolContinuityAgents,
		agents:    make(map[string]*managedOpenAIContinuityState),
	}
}

func (s *managedOpenAIContinuityStore) ObserveTerminalAssistant(agentID string, finalAssistant map[string]any, hiddenMessages []json.RawMessage) bool {
	if s == nil || strings.TrimSpace(agentID) == "" {
		return false
	}
	anchor, ok := openAIVisibleAssistantAnchor(finalAssistant)
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
		state.Turns = filterContinuityTurns(state.Turns, s.maxTurns)
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
		state = &managedOpenAIContinuityState{}
		s.agents[agentID] = state
	}
	state.Turns = append(state.Turns, managedOpenAIContinuityTurn{
		Anchor:                  anchor,
		VisibleAssistantFromEnd: 1,
		HiddenMessages:          cloneRawMessages(hiddenMessages),
	})
	state.Turns = filterContinuityTurns(state.Turns, s.maxTurns)
	s.touchLocked(agentID, state)
	s.evictAgentsLocked()
	return true
}

func (s *managedOpenAIContinuityStore) Inject(agentID string, payload map[string]any) bool {
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
	turns := cloneContinuityTurns(state.Turns)
	s.touchLocked(agentID, state)
	s.mu.Unlock()

	totalVisible := countVisibleOpenAIAssistantMessages(messages)
	if totalVisible == 0 {
		return false
	}
	turnsByOffset := make(map[int]managedOpenAIContinuityTurn, len(turns))
	for _, turn := range turns {
		if turn.VisibleAssistantFromEnd <= 0 {
			continue
		}
		turnsByOffset[turn.VisibleAssistantFromEnd] = turn
	}

	out := make([]any, 0, len(messages)+hiddenMessageCount(turns))
	visibleSeen := 0
	inserted := false
	for _, rawMsg := range messages {
		msg, _ := rawMsg.(map[string]any)
		if isVisibleOpenAIAssistantMessage(msg) {
			visibleSeen++
			offsetFromEnd := totalVisible - visibleSeen + 1
			if turn, ok := turnsByOffset[offsetFromEnd]; ok && openAIMessageMatchesAnchor(msg, turn.Anchor) {
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

func (s *managedOpenAIContinuityStore) touchLocked(agentID string, state *managedOpenAIContinuityState) {
	if state == nil {
		return
	}
	s.clock++
	state.LastAccess = s.clock
	s.agents[agentID] = state
}

func (s *managedOpenAIContinuityStore) evictAgentsLocked() {
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

func openAIVisibleAssistantAnchor(message map[string]any) (string, bool) {
	if !isVisibleOpenAIAssistantMessage(message) {
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

func isVisibleOpenAIAssistantMessage(message map[string]any) bool {
	if len(message) == 0 {
		return false
	}
	role, _ := message["role"].(string)
	if !strings.EqualFold(strings.TrimSpace(role), "assistant") {
		return false
	}
	if toolCalls, _ := message["tool_calls"].([]any); len(toolCalls) > 0 {
		return false
	}
	if functionCall, _ := message["function_call"].(map[string]any); len(functionCall) > 0 {
		return false
	}
	return true
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

func countVisibleOpenAIAssistantMessages(messages []any) int {
	total := 0
	for _, raw := range messages {
		msg, _ := raw.(map[string]any)
		if isVisibleOpenAIAssistantMessage(msg) {
			total++
		}
	}
	return total
}

func filterContinuityTurns(turns []managedOpenAIContinuityTurn, maxTurns int) []managedOpenAIContinuityTurn {
	if len(turns) == 0 {
		return nil
	}
	out := make([]managedOpenAIContinuityTurn, 0, len(turns))
	for _, turn := range turns {
		if turn.VisibleAssistantFromEnd > 0 && (maxTurns <= 0 || turn.VisibleAssistantFromEnd <= maxTurns) {
			out = append(out, turn)
		}
	}
	return out
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
			Anchor:                  turn.Anchor,
			VisibleAssistantFromEnd: turn.VisibleAssistantFromEnd,
			HiddenMessages:          cloneRawMessages(turn.HiddenMessages),
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
