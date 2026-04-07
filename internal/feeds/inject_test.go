package feeds

import (
	"strings"
	"testing"
	"time"
)

func TestFormatFeedBlock(t *testing.T) {
	result := FeedResult{
		Name:      "market-context",
		Source:    "trading-api",
		Content:   "Wallet: $5,000 cash | $20,000 invested",
		FetchedAt: time.Date(2026, 3, 22, 14, 30, 0, 0, time.UTC),
	}
	block := FormatFeedBlock(result)
	if !strings.Contains(block, "--- BEGIN FEED: market-context") {
		t.Error("missing begin delimiter")
	}
	if !strings.Contains(block, "--- END FEED: market-context ---") {
		t.Error("missing end delimiter")
	}
	if !strings.Contains(block, "Wallet: $5,000") {
		t.Error("missing content")
	}
	if !strings.Contains(block, "trading-api") {
		t.Error("missing source attribution")
	}
}

func TestFormatFeedBlockStale(t *testing.T) {
	result := FeedResult{
		Name:      "alerts",
		Source:    "claw-api",
		Content:   "All clear",
		FetchedAt: time.Date(2026, 3, 22, 14, 0, 0, 0, time.UTC),
		Stale:     true,
	}
	block := FormatFeedBlock(result)
	if !strings.Contains(block, "STALE") {
		t.Error("stale feed should be marked")
	}
}

func TestFormatFeedBlockUnavailable(t *testing.T) {
	result := FeedResult{
		Name:        "down-service",
		Source:      "broken-api",
		Unavailable: true,
	}
	block := FormatFeedBlock(result)
	if !strings.Contains(block, "unavailable") {
		t.Error("unavailable feed should say so")
	}
}

func TestFormatFeedBlockEmptyContent(t *testing.T) {
	result := FeedResult{
		Name:      "quiet-feed",
		Source:    "claw-wall",
		FetchedAt: time.Now(),
	}
	if block := FormatFeedBlock(result); block != "" {
		t.Fatalf("expected empty block for empty feed content, got %q", block)
	}
}

func TestFormatFeedBlockTruncated(t *testing.T) {
	result := FeedResult{
		Name:      "big-feed",
		Source:    "api",
		Content:   "partial...",
		FetchedAt: time.Now(),
		Truncated: true,
	}
	block := FormatFeedBlock(result)
	if !strings.Contains(block, "truncated") {
		t.Error("truncated feed should be marked")
	}
}

func TestFormatAllFeeds(t *testing.T) {
	results := []FeedResult{
		{Name: "feed-a", Source: "svc-a", Content: "data-a", FetchedAt: time.Now()},
		{Name: "feed-b", Source: "svc-b", Content: "data-b", FetchedAt: time.Now()},
	}
	combined := FormatAllFeeds(results)
	if !strings.Contains(combined, "feed-a") || !strings.Contains(combined, "feed-b") {
		t.Error("combined output should contain both feeds")
	}
	if strings.Count(combined, "BEGIN FEED") != 2 {
		t.Errorf("expected 2 feed blocks, got %d", strings.Count(combined, "BEGIN FEED"))
	}
}

func TestFormatAllFeedsSkipsEmptyBlocks(t *testing.T) {
	results := []FeedResult{
		{Name: "quiet-feed", Source: "claw-wall", FetchedAt: time.Now()},
		{Name: "feed-b", Source: "svc-b", Content: "data-b", FetchedAt: time.Now()},
	}
	combined := FormatAllFeeds(results)
	if strings.HasPrefix(combined, "\n") {
		t.Fatalf("expected no leading newline when skipping empty feed block, got %q", combined)
	}
	if strings.Count(combined, "BEGIN FEED") != 1 {
		t.Fatalf("expected only one non-empty feed block, got %q", combined)
	}
}

func TestFormatAllFeedsTotalSizeCap(t *testing.T) {
	halfContent := strings.Repeat("x", MaxTotalFeedBytes/2+1000)
	results := []FeedResult{
		{Name: "feed-a", Source: "svc", Content: halfContent, FetchedAt: time.Now()},
		{Name: "feed-b", Source: "svc", Content: halfContent, FetchedAt: time.Now()},
	}
	combined := FormatAllFeeds(results)
	if !strings.Contains(combined, "BEGIN FEED: feed-a") {
		t.Error("first feed should be included")
	}
	if strings.Contains(combined, "BEGIN FEED: feed-b") {
		t.Error("second feed should be skipped (total cap)")
	}
	if !strings.Contains(combined, "feed-b skipped") {
		t.Error("skip notice should appear for second feed")
	}
}

func TestInjectOpenAINoExistingSystem(t *testing.T) {
	payload := map[string]any{
		"model": "openai/gpt-4o",
		"messages": []any{
			map[string]any{"role": "user", "content": "hello"},
		},
	}
	ok := InjectOpenAI(payload, "--- BEGIN FEED: test ---\ndata\n--- END FEED: test ---")
	if !ok {
		t.Fatal("expected injection to succeed")
	}
	messages := payload["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}
	first := messages[0].(map[string]any)
	if first["role"] != "system" {
		t.Errorf("expected system role, got %q", first["role"])
	}
}

func TestInjectOpenAIMergesIntoExistingSystem(t *testing.T) {
	payload := map[string]any{
		"model": "openai/gpt-4o",
		"messages": []any{
			map[string]any{"role": "system", "content": "You are a trader."},
			map[string]any{"role": "user", "content": "hello"},
		},
	}
	ok := InjectOpenAI(payload, "feed data")
	if !ok {
		t.Fatal("expected injection to succeed")
	}
	messages := payload["messages"].([]any)
	if len(messages) != 2 {
		t.Fatalf("expected 2 messages (merged, not stacked), got %d", len(messages))
	}
	first := messages[0].(map[string]any)
	content := first["content"].(string)
	if content != "You are a trader.\n\nfeed data" {
		t.Errorf("expected system prompt before feed (cache-friendly order), got %q", content)
	}
}

func TestInjectOpenAIEmptyBlock(t *testing.T) {
	payload := map[string]any{"messages": []any{}}
	ok := InjectOpenAI(payload, "")
	if ok {
		t.Error("empty block should not modify payload")
	}
}

func TestInjectAnthropicNoExistingSystem(t *testing.T) {
	payload := map[string]any{
		"model":    "claude-sonnet-4-20250514",
		"messages": []any{map[string]any{"role": "user", "content": "hi"}},
	}
	ok := InjectAnthropic(payload, "feed data")
	if !ok {
		t.Fatal("expected injection to succeed")
	}
	if payload["system"] != "feed data" {
		t.Errorf("expected system=feed data, got %q", payload["system"])
	}
}

func TestInjectAnthropicExistingStringSystem(t *testing.T) {
	payload := map[string]any{
		"system":   "You are a trader.",
		"messages": []any{},
	}
	ok := InjectAnthropic(payload, "feed data")
	if !ok {
		t.Fatal("expected injection to succeed")
	}
	sys := payload["system"].(string)
	if sys != "You are a trader.\n\nfeed data" {
		t.Errorf("expected system prompt before feed (cache-friendly order), got %q", sys)
	}
}

func TestInjectAnthropicExistingBlockSystem(t *testing.T) {
	payload := map[string]any{
		"system": []any{
			map[string]any{"type": "text", "text": "existing"},
		},
		"messages": []any{},
	}
	ok := InjectAnthropic(payload, "feed data")
	if !ok {
		t.Fatal("expected injection to succeed")
	}
	blocks := payload["system"].([]any)
	if len(blocks) != 2 {
		t.Fatalf("expected 2 blocks, got %d", len(blocks))
	}
	first := blocks[0].(map[string]any)
	if first["text"] != "existing" {
		t.Errorf("expected existing system block first (cache-friendly order), got %q", first["text"])
	}
	second := blocks[1].(map[string]any)
	if second["text"] != "feed data" {
		t.Errorf("expected feed data second, got %q", second["text"])
	}
}
