package feeds

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/mostlydev/cllama/internal/logging"
)

// FeedResult is the outcome of a single feed fetch.
type FeedResult struct {
	Name        string
	Source      string
	Content     string
	FetchedAt   time.Time
	Stale       bool
	Truncated   bool
	Unavailable bool
}

type cacheEntry struct {
	content   string
	fetchedAt time.Time
	expiresAt time.Time
	truncated bool
}

// Fetcher fetches feed content with TTL caching.
type Fetcher struct {
	podName string
	client  *http.Client
	logger  *logging.Logger
	mu      sync.RWMutex
	cache   map[string]*cacheEntry
	nowFunc func() time.Time
}

// NewFetcher creates a feed fetcher. Pass nil for the default http.Client.
func NewFetcher(podName string, client *http.Client, logger *logging.Logger) *Fetcher {
	if client == nil {
		client = &http.Client{Timeout: FetchTimeout}
	}
	return &Fetcher{
		podName: podName,
		client:  client,
		logger:  logger,
		cache:   make(map[string]*cacheEntry),
		nowFunc: time.Now,
	}
}

func (f *Fetcher) cacheKey(agentID, feedURL string) string {
	return agentID + "|" + feedURL
}

func (f *Fetcher) now() time.Time {
	return f.nowFunc()
}

// Fetch returns feed content, using cache when fresh. On fetch failure, returns
// stale cached content if available, otherwise an unavailable marker.
func (f *Fetcher) Fetch(ctx context.Context, agentID string, entry FeedEntry) (FeedResult, error) {
	key := f.cacheKey(agentID, entry.URL)
	now := f.now()

	f.mu.RLock()
	cached, hasCached := f.cache[key]
	f.mu.RUnlock()

	if hasCached && now.Before(cached.expiresAt) {
		return FeedResult{
			Name:      entry.Name,
			Source:    entry.Source,
			Content:   cached.content,
			FetchedAt: cached.fetchedAt,
			Truncated: cached.truncated,
		}, nil
	}

	content, truncated, err := f.doFetch(ctx, agentID, entry)
	if err != nil {
		if hasCached {
			return FeedResult{
				Name:      entry.Name,
				Source:    entry.Source,
				Content:   cached.content,
				FetchedAt: cached.fetchedAt,
				Stale:     true,
				Truncated: cached.truncated,
			}, nil
		}
		return FeedResult{
			Name:        entry.Name,
			Source:      entry.Source,
			Unavailable: true,
		}, nil
	}

	ttl := entry.TTL
	if ttl <= 0 {
		ttl = DefaultTTLSeconds
	}

	newEntry := &cacheEntry{
		content:   content,
		fetchedAt: now,
		expiresAt: now.Add(time.Duration(ttl) * time.Second),
		truncated: truncated,
	}
	f.mu.Lock()
	f.cache[key] = newEntry
	f.mu.Unlock()

	return FeedResult{
		Name:      entry.Name,
		Source:    entry.Source,
		Content:   content,
		FetchedAt: now,
		Truncated: truncated,
	}, nil
}

func (f *Fetcher) doFetch(ctx context.Context, agentID string, entry FeedEntry) (content string, truncated bool, err error) {
	start := time.Now()
	statusCode := 0
	defer func() {
		if f.logger != nil {
			f.logger.LogFeedFetch(agentID, entry.Name, entry.URL, statusCode, time.Since(start).Milliseconds(), err)
		}
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, entry.URL, nil)
	if err != nil {
		return "", false, fmt.Errorf("build feed request: %w", err)
	}
	req.Header.Set("X-Claw-ID", agentID)
	req.Header.Set("X-Claw-Pod", f.podName)
	req.Header.Set("Accept", "text/plain, text/markdown, application/json")
	req.Header.Set("X-Forwarded-Proto", "https")

	resp, err := f.client.Do(req)
	if err != nil {
		return "", false, fmt.Errorf("fetch feed %q: %w", entry.Name, err)
	}
	defer resp.Body.Close()
	statusCode = resp.StatusCode

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", false, fmt.Errorf("feed %q returned status %d", entry.Name, resp.StatusCode)
	}

	limited := io.LimitReader(resp.Body, int64(MaxFeedResponseBytes+1))
	body, err := io.ReadAll(limited)
	if err != nil {
		return "", false, fmt.Errorf("read feed %q body: %w", entry.Name, err)
	}

	if len(body) > MaxFeedResponseBytes {
		body = body[:MaxFeedResponseBytes]
		truncated = true
	}

	return formatFeedContent(body, resp.Header.Get("Content-Type")), truncated, nil
}

func formatFeedContent(body []byte, contentType string) string {
	ct := strings.ToLower(strings.TrimSpace(contentType))
	if strings.Contains(ct, "application/json") {
		var b strings.Builder
		b.WriteString("```json\n")
		b.Write(body)
		if len(body) == 0 || body[len(body)-1] != '\n' {
			b.WriteByte('\n')
		}
		b.WriteString("```")
		return b.String()
	}
	return string(body)
}
