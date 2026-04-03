package sessionhistory

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	DefaultReadLimit = 100
	MaxReadLimit     = 1000
)

// ReadEntries returns up to limit entries for agentID, strictly after the
// provided timestamp when after is non-nil. Missing history files return an
// empty slice and no error.
func ReadEntries(baseDir, agentID string, after *time.Time, limit int) ([]Entry, error) {
	if baseDir == "" {
		return nil, nil
	}
	if limit <= 0 {
		limit = DefaultReadLimit
	}
	if limit > MaxReadLimit {
		limit = MaxReadLimit
	}

	histPath := filepath.Join(baseDir, agentID, "history.jsonl")
	f, err := os.Open(histPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	startOffset, err := readStartOffset(histPath, after)
	if err != nil {
		return nil, err
	}
	if startOffset > 0 {
		if _, err := f.Seek(startOffset, io.SeekStart); err != nil {
			return nil, err
		}
	}

	entries := make([]Entry, 0, limit)
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		if len(entries) >= limit {
			break
		}
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}
		var entry Entry
		if err := json.Unmarshal(line, &entry); err != nil {
			return nil, fmt.Errorf("parse history entry: %w", err)
		}
		if strings.TrimSpace(entry.ID) == "" {
			entry.ID = IDFromJSON(line)
		}
		if after != nil {
			ts, err := time.Parse(time.RFC3339, entry.TS)
			if err != nil {
				return nil, fmt.Errorf("parse history timestamp %q: %w", entry.TS, err)
			}
			if !ts.After(*after) {
				continue
			}
		}
		entries = append(entries, entry)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return entries, nil
}
