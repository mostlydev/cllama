package feeds

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	MaxFeedResponseBytes    = 32 * 1024
	MaxTotalFeedBytes       = 64 * 1024
	DefaultTTLSeconds       = 60
	FetchTimeout            = 3 * time.Second
	EnvMaxFeedResponseBytes = "CLLAMA_FEED_MAX_RESPONSE_BYTES"
	EnvMaxTotalFeedBytes    = "CLLAMA_FEED_MAX_TOTAL_BYTES"
)

// Budget controls the fetch-time per-feed byte cap and the aggregate injected
// feed block cap. Zero values are normalized to the bounded defaults.
type Budget struct {
	MaxFeedResponseBytes int
	MaxTotalFeedBytes    int
}

func DefaultBudget() Budget {
	return Budget{
		MaxFeedResponseBytes: MaxFeedResponseBytes,
		MaxTotalFeedBytes:    MaxTotalFeedBytes,
	}
}

func (b Budget) Normalize() Budget {
	defaults := DefaultBudget()
	if b.MaxFeedResponseBytes <= 0 {
		b.MaxFeedResponseBytes = defaults.MaxFeedResponseBytes
	}
	if b.MaxTotalFeedBytes <= 0 {
		b.MaxTotalFeedBytes = defaults.MaxTotalFeedBytes
	}
	return b
}

// BudgetFromEnv reads optional byte caps from process env. Invalid values fall
// back to defaults so a bad knob cannot accidentally unbound feed injection.
func BudgetFromEnv() Budget {
	budget := DefaultBudget()
	if v, ok := positiveIntEnv(EnvMaxFeedResponseBytes); ok {
		budget.MaxFeedResponseBytes = v
	}
	if v, ok := positiveIntEnv(EnvMaxTotalFeedBytes); ok {
		budget.MaxTotalFeedBytes = v
	}
	return budget
}

func positiveIntEnv(key string) (int, bool) {
	raw := strings.TrimSpace(os.Getenv(key))
	if raw == "" {
		return 0, false
	}
	value, err := strconv.Atoi(raw)
	if err != nil || value <= 0 {
		return 0, false
	}
	return value, true
}

// FeedEntry matches the manifest shape written by claw up.
type FeedEntry struct {
	Name   string `json:"name"`
	Source string `json:"source"`
	Path   string `json:"path"`
	TTL    int    `json:"ttl"`
	URL    string `json:"url"`
	Auth   string `json:"auth,omitempty"` // bearer token for authenticated feeds
	// NoCache is set internally for invocation-time streams such as live channel
	// context. It is not part of the on-disk manifest contract.
	NoCache bool `json:"-"`
}

// LoadManifest reads feeds.json from contextDir. Returns nil, nil if the file
// does not exist.
func LoadManifest(contextDir string) ([]FeedEntry, error) {
	data, err := os.ReadFile(filepath.Join(contextDir, "feeds.json"))
	if err != nil {
		if errors.Is(err, os.ErrNotExist) {
			return nil, nil
		}
		return nil, fmt.Errorf("read feeds.json: %w", err)
	}

	var entries []FeedEntry
	if err := json.Unmarshal(data, &entries); err != nil {
		return nil, fmt.Errorf("parse feeds.json: %w", err)
	}
	return entries, nil
}
