package feeds

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

const (
	MaxFeedResponseBytes = 32 * 1024
	MaxTotalFeedBytes    = 64 * 1024
	DefaultTTLSeconds    = 60
	FetchTimeout         = 3 * time.Second
)

// FeedEntry matches the manifest shape written by claw up.
type FeedEntry struct {
	Name   string `json:"name"`
	Source string `json:"source"`
	Path   string `json:"path"`
	TTL    int    `json:"ttl"`
	URL    string `json:"url"`
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
