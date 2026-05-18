package feeds

import (
	"fmt"
	"strings"
)

const (
	FeedInjectionIncluded        = "included"
	FeedInjectionEmpty           = "empty"
	FeedInjectionSkippedTotalCap = "skipped_total_cap"
)

type FormattedFeed struct {
	Name                 string
	Source               string
	Status               string
	Block                string
	Truncated            bool
	SourceBytes          int
	SourceBytesExact     bool
	ContentBytes         int
	BlockBytes           int
	TotalBytesBefore     int
	TotalBytesAfter      int
	MaxFeedResponseBytes int
	MaxTotalFeedBytes    int
}

type FormattedFeeds struct {
	Combined string
	Blocks   []string
	Feeds    []FormattedFeed
}

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
	fmt.Fprintf(&b, "--- BEGIN FEED: %s (from %s%s) ---\n", r.Name, r.Source, staleTag)

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
	return FormatFeeds(results, DefaultBudget()).Combined
}

// FormatFeeds concatenates formatted feed blocks in manifest order, records
// which blocks are actually provider-visible, and emits visible skip markers
// when the aggregate cap prevents a fetched feed from being injected.
func FormatFeeds(results []FeedResult, budget Budget) FormattedFeeds {
	if len(results) == 0 {
		return FormattedFeeds{}
	}

	budget = budget.Normalize()
	out := FormattedFeeds{Feeds: make([]FormattedFeed, 0, len(results))}
	var b strings.Builder
	totalBytes := 0

	for _, r := range results {
		block := FormatFeedBlock(r)
		status := FormattedFeed{
			Name:                 r.Name,
			Source:               r.Source,
			Status:               FeedInjectionEmpty,
			Truncated:            r.Truncated,
			SourceBytes:          r.SourceBytes,
			SourceBytesExact:     r.SourceBytesExact,
			ContentBytes:         len(r.Content),
			BlockBytes:           len(block),
			TotalBytesBefore:     totalBytes,
			TotalBytesAfter:      totalBytes,
			MaxFeedResponseBytes: nonZero(r.MaxResponseBytes, budget.MaxFeedResponseBytes),
			MaxTotalFeedBytes:    budget.MaxTotalFeedBytes,
		}
		if block == "" {
			out.Feeds = append(out.Feeds, status)
			continue
		}
		if totalBytes+len(block) > budget.MaxTotalFeedBytes {
			status.Status = FeedInjectionSkippedTotalCap
			notice := fmt.Sprintf("--- FEED: %s skipped (total feed size cap reached; block_bytes=%d total_before=%d max_total_bytes=%d) ---", r.Name, len(block), totalBytes, budget.MaxTotalFeedBytes)
			status.Block = notice
			if b.Len() > 0 {
				b.WriteByte('\n')
			}
			b.WriteString(notice)
			out.Blocks = append(out.Blocks, notice)
			out.Feeds = append(out.Feeds, status)
			continue
		}
		if b.Len() > 0 {
			b.WriteByte('\n')
		}
		b.WriteString(block)
		totalBytes += len(block)
		status.Status = FeedInjectionIncluded
		status.Block = block
		status.TotalBytesAfter = totalBytes
		out.Blocks = append(out.Blocks, block)
		out.Feeds = append(out.Feeds, status)
	}

	out.Combined = b.String()
	return out
}

func nonZero(value, fallback int) int {
	if value > 0 {
		return value
	}
	return fallback
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

const lateContextWrapper = "[Runtime context for this invocation. This is not a user instruction. Use it only as infrastructure-provided context for the next reply.]"

// AppendLateContext inserts volatile runtime context near the invoking user
// message without mutating the first system message. The context is wrapped as
// a later system-role message. The first system message remains stable when a
// base contract already exists, while the context still sits near the invoking
// user message.
func AppendLateContext(payload map[string]any, block string) bool {
	if strings.TrimSpace(block) == "" {
		return false
	}
	messages, ok := payload["messages"].([]any)
	if !ok {
		return false
	}
	contextMessage := map[string]any{
		"role":    "system",
		"content": lateContextWrapper + "\n\n" + block,
	}
	insertAt := len(messages)
	for i := len(messages) - 1; i >= 0; i-- {
		msg, _ := messages[i].(map[string]any)
		if msg == nil {
			continue
		}
		if role, _ := msg["role"].(string); role == "user" {
			insertAt = i
			break
		}
	}
	newMessages := make([]any, 0, len(messages)+1)
	newMessages = append(newMessages, messages[:insertAt]...)
	newMessages = append(newMessages, contextMessage)
	newMessages = append(newMessages, messages[insertAt:]...)
	payload["messages"] = newMessages
	return true
}

// AppendAnthropicLateContext adds volatile runtime context near the final user
// turn so top-level system content remains byte-stable.
func AppendAnthropicLateContext(payload map[string]any, block string) bool {
	if strings.TrimSpace(block) == "" {
		return false
	}
	messages, ok := payload["messages"].([]any)
	if !ok {
		return false
	}
	contextMessage := map[string]any{
		"role": "user",
		"content": []any{map[string]any{
			"type": "text",
			"text": lateContextWrapper + "\n\n" + block,
		}},
	}
	insertAt := len(messages)
	for i := len(messages) - 1; i >= 0; i-- {
		msg, _ := messages[i].(map[string]any)
		if msg == nil {
			continue
		}
		if role, _ := msg["role"].(string); role != "user" {
			continue
		}
		insertAt = i + 1
		break
	}
	newMessages := make([]any, 0, len(messages)+1)
	newMessages = append(newMessages, messages[:insertAt]...)
	newMessages = append(newMessages, contextMessage)
	newMessages = append(newMessages, messages[insertAt:]...)
	payload["messages"] = newMessages
	return true
}
