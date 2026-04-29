package proxy

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
)

type channelCursor struct {
	LastMessageID string `json:"last_message_id"`
	LastTimestamp string `json:"last_timestamp,omitempty"`
}

type channelCursorLedger struct {
	Version  int                      `json:"version"`
	Channels map[string]channelCursor `json:"channels"`
}

type channelCursorStore struct {
	mu     sync.Mutex
	dir    string
	memory map[string]channelCursorLedger
}

func newChannelCursorStore(dir string) *channelCursorStore {
	return &channelCursorStore{
		dir:    strings.TrimSpace(dir),
		memory: make(map[string]channelCursorLedger),
	}
}

func (s *channelCursorStore) Load(agentID string) (map[string]channelCursor, error) {
	if s == nil || strings.TrimSpace(agentID) == "" {
		return nil, nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	ledger, err := s.loadLocked(agentID)
	if err != nil {
		return nil, err
	}
	out := make(map[string]channelCursor, len(ledger.Channels))
	for channelID, cursor := range ledger.Channels {
		out[channelID] = cursor
	}
	return out, nil
}

func (s *channelCursorStore) Commit(agentID string, updates map[string]channelCursor) error {
	if s == nil || strings.TrimSpace(agentID) == "" || len(updates) == 0 {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	ledger, err := s.loadLocked(agentID)
	if err != nil {
		return err
	}
	if ledger.Version == 0 {
		ledger.Version = 1
	}
	if ledger.Channels == nil {
		ledger.Channels = make(map[string]channelCursor)
	}
	for channelID, next := range updates {
		channelID = strings.TrimSpace(channelID)
		next.LastMessageID = strings.TrimSpace(next.LastMessageID)
		if channelID == "" || next.LastMessageID == "" {
			continue
		}
		current := ledger.Channels[channelID]
		if compareMessageID(next.LastMessageID, current.LastMessageID) > 0 {
			ledger.Channels[channelID] = next
		}
	}
	return s.saveLocked(agentID, ledger)
}

func (s *channelCursorStore) loadLocked(agentID string) (channelCursorLedger, error) {
	if s.dir == "" {
		ledger := s.memory[agentID]
		if ledger.Version == 0 {
			ledger.Version = 1
		}
		if ledger.Channels == nil {
			ledger.Channels = make(map[string]channelCursor)
		}
		return ledger, nil
	}
	data, err := os.ReadFile(s.cursorPath(agentID))
	if err != nil {
		if os.IsNotExist(err) {
			return channelCursorLedger{Version: 1, Channels: make(map[string]channelCursor)}, nil
		}
		return channelCursorLedger{}, err
	}
	var ledger channelCursorLedger
	if err := json.Unmarshal(data, &ledger); err != nil {
		return channelCursorLedger{}, fmt.Errorf("parse channel cursor ledger: %w", err)
	}
	if ledger.Version == 0 {
		ledger.Version = 1
	}
	if ledger.Channels == nil {
		ledger.Channels = make(map[string]channelCursor)
	}
	return ledger, nil
}

func (s *channelCursorStore) saveLocked(agentID string, ledger channelCursorLedger) error {
	if s.dir == "" {
		s.memory[agentID] = ledger
		return nil
	}
	agentDir := filepath.Join(s.dir, agentID)
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(ledger, "", "  ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	tmp, err := os.CreateTemp(agentDir, "cursor-*.json")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	if _, err := tmp.Write(data); err != nil {
		_ = tmp.Close()
		_ = os.Remove(tmpPath)
		return err
	}
	if err := tmp.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return err
	}
	return os.Rename(tmpPath, s.cursorPath(agentID))
}

func (s *channelCursorStore) cursorPath(agentID string) string {
	return filepath.Join(s.dir, agentID, "cursor.json")
}

type pendingChannelCursorCommit struct {
	updates map[string]channelCursor
}

func (p *pendingChannelCursorCommit) Merge(updates map[string]channelCursor) {
	if p == nil || len(updates) == 0 {
		return
	}
	if p.updates == nil {
		p.updates = make(map[string]channelCursor)
	}
	for channelID, next := range updates {
		current := p.updates[channelID]
		if compareMessageID(next.LastMessageID, current.LastMessageID) > 0 {
			p.updates[channelID] = next
		}
	}
}

func (p *pendingChannelCursorCommit) Commit(h *Handler, agentID, model string) {
	if p == nil || len(p.updates) == 0 || h == nil || h.channelCursors == nil {
		return
	}
	if err := h.channelCursors.Commit(agentID, p.updates); err != nil {
		h.logger.LogError(agentID, model, 0, 0, fmt.Errorf("channel context cursor commit: %w", err))
	}
}

func encodeAfterCursors(channelIDs []string, cursors map[string]channelCursor) string {
	if len(channelIDs) == 0 || len(cursors) == 0 {
		return ""
	}
	pairs := make([]string, 0, len(channelIDs))
	for _, channelID := range channelIDs {
		cursor := cursors[channelID]
		if strings.TrimSpace(cursor.LastMessageID) == "" {
			continue
		}
		pairs = append(pairs, channelID+":"+cursor.LastMessageID)
	}
	return strings.Join(pairs, ",")
}

func compareMessageID(a, b string) int {
	a = strings.TrimSpace(a)
	b = strings.TrimSpace(b)
	if a == b {
		return 0
	}
	if a == "" {
		return -1
	}
	if b == "" {
		return 1
	}
	ai, aerr := strconv.ParseUint(a, 10, 64)
	bi, berr := strconv.ParseUint(b, 10, 64)
	if aerr == nil && berr == nil {
		switch {
		case ai < bi:
			return -1
		case ai > bi:
			return 1
		default:
			return 0
		}
	}
	return strings.Compare(a, b)
}
