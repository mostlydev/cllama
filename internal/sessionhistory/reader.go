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

	paths, err := historyReadPaths(baseDir, agentID)
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, nil
	}

	currentPath := filepath.Join(baseDir, agentID, "history.jsonl")
	entries := make([]Entry, 0, limit)
	for _, path := range paths {
		more, err := readEntriesFromPath(path, after, limit-len(entries), path == currentPath)
		if err != nil {
			return nil, err
		}
		entries = append(entries, more...)
		if len(entries) >= limit {
			break
		}
	}
	return entries, nil
}

func readEntriesFromPath(histPath string, after *time.Time, limit int, useIndex bool) ([]Entry, error) {
	if limit <= 0 {
		return nil, nil
	}
	f, err := os.Open(histPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	defer f.Close()

	if useIndex {
		startOffset, err := readStartOffset(histPath, after)
		if err != nil {
			return nil, err
		}
		if startOffset > 0 {
			if _, err := f.Seek(startOffset, io.SeekStart); err != nil {
				return nil, err
			}
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

func historyReadPaths(baseDir, agentID string) ([]string, error) {
	agentDir := filepath.Join(baseDir, agentID)
	paths := []string{
		filepath.Join(agentDir, "history.jsonl.1"),
		filepath.Join(agentDir, "history.jsonl"),
	}
	out := make([]string, 0, len(paths))
	for _, path := range paths {
		if _, err := os.Stat(path); err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		out = append(out, path)
	}
	return out, nil
}

// SummarizeWindow scans all session-history entries for agentID strictly after
// since. Missing history files return an empty summary.
func SummarizeWindow(baseDir, agentID string, since time.Time) (WindowSummary, error) {
	var summary WindowSummary
	if baseDir == "" {
		return summary, nil
	}

	paths, err := historyReadPaths(baseDir, agentID)
	if err != nil {
		return summary, err
	}
	currentPath := filepath.Join(baseDir, agentID, "history.jsonl")
	for _, path := range paths {
		if err := summarizeWindowFromPath(path, since, path == currentPath, &summary); err != nil {
			return summary, err
		}
	}
	return summary, nil
}

func summarizeWindowFromPath(histPath string, since time.Time, useIndex bool, summary *WindowSummary) error {
	f, err := os.Open(histPath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()

	if useIndex {
		startOffset, err := readStartOffset(histPath, &since)
		if err != nil {
			return err
		}
		if startOffset > 0 {
			if _, err := f.Seek(startOffset, io.SeekStart); err != nil {
				return err
			}
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
					return fmt.Errorf("parse history entry: %w", perr)
				}
				ts, perr := time.Parse(time.RFC3339, entry.TS)
				if perr != nil {
					return fmt.Errorf("parse history timestamp %q: %w", entry.TS, perr)
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
			return err
		}
	}
	return nil
}
