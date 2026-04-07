package feeds

import (
	"fmt"
	"strings"
)

// FormatFeedBlock formats a single feed result as a delimited context block.
func FormatFeedBlock(r FeedResult) string {
	if !r.Unavailable && r.Content == "" {
		return ""
	}

	var b strings.Builder

	if r.Unavailable {
		fmt.Fprintf(&b, "--- BEGIN FEED: %s (from %s) ---\n", r.Name, r.Source)
		fmt.Fprintf(&b, "[Feed unavailable]\n")
		fmt.Fprintf(&b, "--- END FEED: %s ---", r.Name)
		return b.String()
	}

	staleTag := ""
	if r.Stale {
		staleTag = ", STALE"
	}
	fmt.Fprintf(&b, "--- BEGIN FEED: %s (from %s, refreshed %s%s) ---\n",
		r.Name, r.Source, r.FetchedAt.UTC().Format("2006-01-02T15:04:05Z"), staleTag)

	b.WriteString(r.Content)
	if !strings.HasSuffix(r.Content, "\n") {
		b.WriteByte('\n')
	}

	if r.Truncated {
		b.WriteString("[Content truncated]\n")
	}

	fmt.Fprintf(&b, "--- END FEED: %s ---", r.Name)
	return b.String()
}

// FormatAllFeeds concatenates formatted feed blocks in manifest order while
// enforcing a total size cap.
func FormatAllFeeds(results []FeedResult) string {
	if len(results) == 0 {
		return ""
	}

	var b strings.Builder
	totalBytes := 0

	for i, r := range results {
		block := FormatFeedBlock(r)
		if block == "" {
			continue
		}
		if totalBytes+len(block) > MaxTotalFeedBytes {
			fmt.Fprintf(&b, "--- FEED: %s skipped (total feed size cap reached) ---\n", r.Name)
			continue
		}
		if b.Len() > 0 && i > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(block)
		totalBytes += len(block)
	}

	return b.String()
}

// InjectOpenAI appends feed context to the system message in an OpenAI-compatible
// messages array. Appending (rather than prepending) keeps the static system
// prompt as a stable prefix, which is critical for Anthropic-compatible prompt
// caching (prefix-matched, 5-min TTL).
func InjectOpenAI(payload map[string]any, feedBlock string) bool {
	if feedBlock == "" {
		return false
	}

	messages, ok := payload["messages"].([]any)
	if !ok {
		return false
	}

	if len(messages) > 0 {
		if first, ok := messages[0].(map[string]any); ok {
			if role, _ := first["role"].(string); role == "system" {
				if existing, ok := first["content"].(string); ok {
					first["content"] = existing + "\n\n" + feedBlock
					return true
				}
			}
		}
	}

	feedMessage := map[string]any{
		"role":    "system",
		"content": feedBlock,
	}
	// Insert system message at the front — no existing system to append to.
	newMessages := make([]any, 0, len(messages)+1)
	newMessages = append(newMessages, feedMessage)
	newMessages = append(newMessages, messages...)
	payload["messages"] = newMessages
	return true
}

// InjectAnthropic appends feed context to an Anthropic /v1/messages payload.
// Appending keeps the static system prompt as a stable prefix for prompt
// caching (prefix-matched, 5-min TTL).
func InjectAnthropic(payload map[string]any, feedBlock string) bool {
	if feedBlock == "" {
		return false
	}

	existing, hasSystem := payload["system"]
	if !hasSystem || existing == nil {
		payload["system"] = feedBlock
		return true
	}

	if s, ok := existing.(string); ok {
		payload["system"] = s + "\n\n" + feedBlock
		return true
	}

	if blocks, ok := existing.([]any); ok {
		feedContentBlock := map[string]any{
			"type": "text",
			"text": feedBlock,
		}
		newBlocks := append(blocks, feedContentBlock)
		payload["system"] = newBlocks
		return true
	}

	return false
}
