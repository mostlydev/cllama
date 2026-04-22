package proxy

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

const defaultContextSnapshotHistoryLimit = 20

type CandidateSnapshot struct {
	Provider      string `json:"provider"`
	UpstreamModel string `json:"upstream_model"`
}

type ContextPlacement struct {
	Order        int    `json:"order"`
	Kind         string `json:"kind"`
	Label        string `json:"label,omitempty"`
	Carrier      string `json:"carrier"`
	Role         string `json:"role,omitempty"`
	MessageIndex int    `json:"message_index"`
	BlockIndex   int    `json:"block_index"`
	StartChar    int    `json:"start_char"`
	EndChar      int    `json:"end_char"`
	Occurrences  int    `json:"occurrences"`
	Visibility   string `json:"visibility"`
	Phase        string `json:"phase"`
	Persistence  string `json:"persistence"`
	Relation     string `json:"relation"`
}

type ContextCaptureSummary struct {
	Sequence       int64     `json:"sequence"`
	CapturedAt     time.Time `json:"captured_at"`
	Format         string    `json:"format"`
	RequestedModel string    `json:"requested_model,omitempty"`
	ChosenRef      string    `json:"chosen_ref,omitempty"`
	DynamicInputs  int       `json:"dynamic_inputs"`
	FeedBlocks     int       `json:"feed_blocks"`
	MemoryRecall   bool      `json:"memory_recall"`
	TimeContext    bool      `json:"time_context"`
	PlacementCount int       `json:"placement_count"`
	ManagedTool    bool      `json:"managed_tool"`
	TurnCount      int       `json:"turn_count"`
}

type ContextSnapshot struct {
	AgentID        string                  `json:"agent_id"`
	CapturedAt     time.Time               `json:"captured_at"`
	Format         string                  `json:"format"`
	System         any                     `json:"system"`
	Tools          []any                   `json:"tools"`
	RequestedModel string                  `json:"requested_model"`
	ChosenRef      string                  `json:"chosen_ref"`
	Candidates     []CandidateSnapshot     `json:"candidates"`
	FeedBlocks     []string                `json:"feed_blocks"`
	MemoryRecall   string                  `json:"memory_recall"`
	TimeContext    string                  `json:"time_context"`
	Placements     []ContextPlacement      `json:"placements"`
	RecentCaptures []ContextCaptureSummary `json:"recent_captures,omitempty"`
	Intervention   string                  `json:"intervention"`
	ManagedTool    bool                    `json:"managed_tool"`
	TurnCount      int                     `json:"turn_count"`
}

type ContextSnapshotStore struct {
	snapshots sync.Map
	historyMu sync.Mutex
	history   map[string][]ContextCaptureSummary
	nextSeq   int64
}

func NewContextSnapshotStore() *ContextSnapshotStore {
	return &ContextSnapshotStore{history: make(map[string][]ContextCaptureSummary)}
}

func (s *ContextSnapshotStore) Put(snapshot ContextSnapshot) {
	if s == nil || snapshot.AgentID == "" {
		return
	}
	cloned := cloneContextSnapshot(snapshot)
	cloned.RecentCaptures = nil
	s.snapshots.Store(snapshot.AgentID, cloned)
	s.appendHistory(cloned)
}

func (s *ContextSnapshotStore) Get(agentID string) (ContextSnapshot, bool) {
	if s == nil {
		return ContextSnapshot{}, false
	}
	raw, ok := s.snapshots.Load(agentID)
	if !ok {
		return ContextSnapshot{}, false
	}
	snapshot, _ := raw.(ContextSnapshot)
	snapshot.RecentCaptures = s.History(agentID, defaultContextSnapshotHistoryLimit)
	return cloneContextSnapshot(snapshot), true
}

func (s *ContextSnapshotStore) History(agentID string, limit int) []ContextCaptureSummary {
	if s == nil || agentID == "" {
		return nil
	}
	s.historyMu.Lock()
	defer s.historyMu.Unlock()
	history := s.history[agentID]
	if limit <= 0 || limit > len(history) {
		limit = len(history)
	}
	start := len(history) - limit
	if start < 0 {
		start = 0
	}
	return append([]ContextCaptureSummary(nil), history[start:]...)
}

func (s *ContextSnapshotStore) AgentIDs() []string {
	if s == nil {
		return nil
	}
	var ids []string
	s.snapshots.Range(func(key, _ any) bool {
		id, _ := key.(string)
		if id != "" {
			ids = append(ids, id)
		}
		return true
	})
	sort.Strings(ids)
	return ids
}

func (s *ContextSnapshotStore) UpdateTurnCount(agentID string, turnCount int) {
	if s == nil || agentID == "" || turnCount <= 0 {
		return
	}
	raw, ok := s.snapshots.Load(agentID)
	if !ok {
		return
	}
	snapshot, _ := raw.(ContextSnapshot)
	snapshot.TurnCount = turnCount
	s.snapshots.Store(agentID, cloneContextSnapshot(snapshot))
	s.historyMu.Lock()
	if history := s.history[agentID]; len(history) > 0 {
		history[len(history)-1].TurnCount = turnCount
		s.history[agentID] = history
	}
	s.historyMu.Unlock()
}

func (h *Handler) captureContextSnapshot(agentID, format, requestedModel string, payload map[string]any, resolution modelResolution, feedBlocks []string, memoryRecall, timeContext string, managedTool bool, turnCount int) {
	if h == nil || h.snapshots == nil {
		return
	}
	system := snapshotSystemForFormat(format, payload)
	h.snapshots.Put(ContextSnapshot{
		AgentID:        agentID,
		CapturedAt:     time.Now().UTC(),
		Format:         format,
		System:         system,
		Tools:          snapshotTools(payload),
		RequestedModel: requestedModel,
		ChosenRef:      resolution.ChosenRef,
		Candidates:     candidateSnapshots(resolution.Candidates),
		FeedBlocks:     append([]string(nil), feedBlocks...),
		MemoryRecall:   memoryRecall,
		TimeContext:    timeContext,
		Placements:     contextPlacements(format, payload, system, feedBlocks, memoryRecall, timeContext),
		Intervention:   resolution.Intervention,
		ManagedTool:    managedTool,
		TurnCount:      turnCount,
	})
}

func (h *Handler) updateContextSnapshotTurnCount(agentID string, turnCount int) {
	if h == nil || h.snapshots == nil {
		return
	}
	h.snapshots.UpdateTurnCount(agentID, turnCount)
}

func candidateSnapshots(candidates []dispatchCandidate) []CandidateSnapshot {
	if len(candidates) == 0 {
		return nil
	}
	out := make([]CandidateSnapshot, 0, len(candidates))
	for _, candidate := range candidates {
		out = append(out, CandidateSnapshot{
			Provider:      candidate.ProviderName,
			UpstreamModel: candidate.UpstreamModel,
		})
	}
	return out
}

func (s *ContextSnapshotStore) appendHistory(snapshot ContextSnapshot) {
	if s == nil || snapshot.AgentID == "" {
		return
	}
	s.historyMu.Lock()
	defer s.historyMu.Unlock()
	s.nextSeq++
	summary := summarizeContextSnapshot(snapshot)
	summary.Sequence = s.nextSeq
	history := append(s.history[snapshot.AgentID], summary)
	if len(history) > defaultContextSnapshotHistoryLimit {
		history = append([]ContextCaptureSummary(nil), history[len(history)-defaultContextSnapshotHistoryLimit:]...)
	}
	s.history[snapshot.AgentID] = history
}

func summarizeContextSnapshot(snapshot ContextSnapshot) ContextCaptureSummary {
	dynamicInputs := len(snapshot.FeedBlocks)
	if strings.TrimSpace(snapshot.MemoryRecall) != "" {
		dynamicInputs++
	}
	if strings.TrimSpace(snapshot.TimeContext) != "" {
		dynamicInputs++
	}
	return ContextCaptureSummary{
		CapturedAt:     snapshot.CapturedAt,
		Format:         snapshot.Format,
		RequestedModel: snapshot.RequestedModel,
		ChosenRef:      snapshot.ChosenRef,
		DynamicInputs:  dynamicInputs,
		FeedBlocks:     len(snapshot.FeedBlocks),
		MemoryRecall:   strings.TrimSpace(snapshot.MemoryRecall) != "",
		TimeContext:    strings.TrimSpace(snapshot.TimeContext) != "",
		PlacementCount: len(snapshot.Placements),
		ManagedTool:    snapshot.ManagedTool,
		TurnCount:      snapshot.TurnCount,
	}
}

func contextPlacements(format string, payload map[string]any, system any, feedBlocks []string, memoryRecall, timeContext string) []ContextPlacement {
	segments := contextPlacementSegments(feedBlocks, memoryRecall, timeContext)
	if len(segments) == 0 {
		return nil
	}

	switch format {
	case "openai":
		text, _ := system.(string)
		return textContextPlacements(segments, text, "openai.messages[0].content", "system", openAISystemMessageIndex(payload), -1)
	case "anthropic":
		if text, ok := system.(string); ok {
			return textContextPlacements(segments, text, "anthropic.system", "system", -1, -1)
		}
		if blocks, ok := system.([]any); ok {
			return blockContextPlacements(segments, blocks)
		}
	}
	return nil
}

type contextPlacementSegment struct {
	Kind  string
	Label string
	Text  string
}

func contextPlacementSegments(feedBlocks []string, memoryRecall, timeContext string) []contextPlacementSegment {
	var segments []contextPlacementSegment
	if strings.TrimSpace(memoryRecall) != "" {
		segments = append(segments, contextPlacementSegment{Kind: "memory", Label: "Memory recall", Text: memoryRecall})
	}
	for _, block := range feedBlocks {
		if strings.TrimSpace(block) == "" {
			continue
		}
		segments = append(segments, contextPlacementSegment{Kind: "feed", Label: feedBlockLabel(block), Text: block})
	}
	if strings.TrimSpace(timeContext) != "" {
		segments = append(segments, contextPlacementSegment{Kind: "time", Label: "Current time", Text: timeContext})
	}
	return segments
}

func textContextPlacements(segments []contextPlacementSegment, text, carrier, role string, messageIndex, blockIndex int) []ContextPlacement {
	placements := make([]ContextPlacement, 0, len(segments))
	cursor := 0
	for _, segment := range segments {
		start := strings.Index(text[cursor:], segment.Text)
		if start >= 0 {
			start += cursor
		} else {
			start = strings.Index(text, segment.Text)
		}
		end := -1
		if start >= 0 {
			end = start + len(segment.Text)
			cursor = end
		}
		placements = append(placements, newContextPlacement(len(placements)+1, segment, carrier, role, messageIndex, blockIndex, start, end, countOccurrences(text, segment.Text)))
	}
	return placements
}

func blockContextPlacements(segments []contextPlacementSegment, blocks []any) []ContextPlacement {
	placements := make([]ContextPlacement, 0, len(segments))
	nextBlock := 0
	for _, segment := range segments {
		blockIndex, text, start := findSegmentInAnthropicBlocks(blocks, segment.Text, nextBlock)
		end := -1
		if start >= 0 {
			end = start + len(segment.Text)
			nextBlock = blockIndex
		}
		carrier := "anthropic.system"
		if blockIndex >= 0 {
			carrier = "anthropic.system[" + strconv.Itoa(blockIndex) + "].text"
		}
		placements = append(placements, newContextPlacement(len(placements)+1, segment, carrier, "system", -1, blockIndex, start, end, countOccurrences(text, segment.Text)))
	}
	return placements
}

func findSegmentInAnthropicBlocks(blocks []any, segment string, startBlock int) (int, string, int) {
	for i := startBlock; i < len(blocks); i++ {
		text := anthropicTextBlock(blocks[i])
		if idx := strings.Index(text, segment); idx >= 0 {
			return i, text, idx
		}
	}
	for i := 0; i < startBlock && i < len(blocks); i++ {
		text := anthropicTextBlock(blocks[i])
		if idx := strings.Index(text, segment); idx >= 0 {
			return i, text, idx
		}
	}
	return -1, "", -1
}

func anthropicTextBlock(block any) string {
	m, _ := block.(map[string]any)
	if m == nil {
		return ""
	}
	text, _ := m["text"].(string)
	return text
}

func newContextPlacement(order int, segment contextPlacementSegment, carrier, role string, messageIndex, blockIndex, start, end, occurrences int) ContextPlacement {
	return ContextPlacement{
		Order:        order,
		Kind:         segment.Kind,
		Label:        segment.Label,
		Carrier:      carrier,
		Role:         role,
		MessageIndex: messageIndex,
		BlockIndex:   blockIndex,
		StartChar:    start,
		EndChar:      end,
		Occurrences:  occurrences,
		Visibility:   "provider_visible",
		Phase:        "pre_dispatch_context_assembly",
		Persistence:  "per_request",
		Relation:     "system_context_before_conversation",
	}
}

func openAISystemMessageIndex(payload map[string]any) int {
	messages, _ := payload["messages"].([]any)
	for i, raw := range messages {
		msg, _ := raw.(map[string]any)
		if msg == nil {
			continue
		}
		role, _ := msg["role"].(string)
		if role == "system" {
			return i
		}
	}
	return -1
}

func feedBlockLabel(block string) string {
	firstLine, _, _ := strings.Cut(block, "\n")
	firstLine = strings.TrimSpace(strings.TrimPrefix(firstLine, "--- BEGIN FEED:"))
	firstLine = strings.TrimSuffix(firstLine, "---")
	if before, _, ok := strings.Cut(firstLine, "("); ok {
		firstLine = before
	}
	firstLine = strings.TrimSpace(firstLine)
	if firstLine == "" {
		return "Feed"
	}
	return firstLine
}

func countOccurrences(text, segment string) int {
	if text == "" || segment == "" {
		return 0
	}
	return strings.Count(text, segment)
}

func snapshotSystemForFormat(format string, payload map[string]any) any {
	switch format {
	case "openai":
		return snapshotOpenAISystem(payload)
	case "anthropic":
		return cloneJSONValue(payload["system"])
	default:
		return nil
	}
}

func snapshotOpenAISystem(payload map[string]any) any {
	messages, _ := payload["messages"].([]any)
	for _, raw := range messages {
		msg, _ := raw.(map[string]any)
		if msg == nil {
			continue
		}
		role, _ := msg["role"].(string)
		if role == "system" {
			return cloneJSONValue(msg["content"])
		}
	}
	return nil
}

func snapshotTools(payload map[string]any) []any {
	tools, _ := payload["tools"].([]any)
	if len(tools) == 0 {
		return nil
	}
	cloned := cloneJSONValue(tools)
	out, _ := cloned.([]any)
	return out
}

func cloneContextSnapshot(snapshot ContextSnapshot) ContextSnapshot {
	cloned := snapshot
	cloned.System = cloneJSONValue(snapshot.System)
	cloned.Tools = snapshotTools(map[string]any{"tools": snapshot.Tools})
	cloned.Candidates = append([]CandidateSnapshot(nil), snapshot.Candidates...)
	cloned.FeedBlocks = append([]string(nil), snapshot.FeedBlocks...)
	cloned.Placements = append([]ContextPlacement(nil), snapshot.Placements...)
	cloned.RecentCaptures = append([]ContextCaptureSummary(nil), snapshot.RecentCaptures...)
	return cloned
}

func cloneJSONValue(v any) any {
	if v == nil {
		return nil
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return v
	}
	var cloned any
	if err := json.Unmarshal(raw, &cloned); err != nil {
		return v
	}
	return cloned
}
