package sessionhistory

import (
	"bufio"
	"bytes"
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

type WindowSummary struct {
	Requests        int
	ReportedCostUSD float64
	UnknownCost     int
}

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
	reader := bufio.NewReader(f)
	for len(entries) < limit {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			line = bytes.TrimRight(line, "\n")
			if len(line) > 0 {
				var entry Entry
				if perr := json.Unmarshal(line, &entry); perr != nil {
					return nil, fmt.Errorf("parse history entry: %w", perr)
				}
				if strings.TrimSpace(entry.ID) == "" {
					entry.ID = IDFromJSON(line)
				}
				if after != nil {
					ts, perr := time.Parse(time.RFC3339, entry.TS)
					if perr != nil {
						return nil, fmt.Errorf("parse history timestamp %q: %w", entry.TS, perr)
					}
					if !ts.After(*after) {
						if err == io.EOF {
							break
						}
						continue
					}
				}
				entries = append(entries, entry)
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return nil, err
		}
	}
	return entries, nil
}

// SummarizeWindow scans all session-history entries for agentID strictly after
// since. Missing history files return an empty summary.
func SummarizeWindow(baseDir, agentID string, since time.Time) (WindowSummary, error) {
	var summary WindowSummary
	if baseDir == "" {
		return summary, nil
	}

	histPath := filepath.Join(baseDir, agentID, "history.jsonl")
	f, err := os.Open(histPath)
	if err != nil {
		if os.IsNotExist(err) {
			return summary, nil
		}
		return summary, err
	}
	defer f.Close()

	startOffset, err := readStartOffset(histPath, &since)
	if err != nil {
		return summary, err
	}
	if startOffset > 0 {
		if _, err := f.Seek(startOffset, io.SeekStart); err != nil {
			return summary, err
		}
	}

	reader := bufio.NewReader(f)
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			line = bytes.TrimRight(line, "\n")
			if len(line) > 0 {
				var entry Entry
				if perr := json.Unmarshal(line, &entry); perr != nil {
					return summary, fmt.Errorf("parse history entry: %w", perr)
				}
				ts, perr := time.Parse(time.RFC3339, entry.TS)
				if perr != nil {
					return summary, fmt.Errorf("parse history timestamp %q: %w", entry.TS, perr)
				}
				if ts.After(since) {
					summary.Requests++
					if entry.Usage.ReportedCostUSD != nil {
						summary.ReportedCostUSD += *entry.Usage.ReportedCostUSD
					} else {
						summary.UnknownCost++
					}
				}
			}
		}
		if err != nil {
			if err == io.EOF {
				break
			}
			return summary, err
		}
	}
	return summary, nil
}
