package proxy

import (
	"fmt"
	"net/url"
	"strconv"
	"strings"

	"github.com/mostlydev/cllama/internal/feeds"
)

type channelContextMetadata struct {
	Available int
	Omitted   int
	Cursor    map[string]channelCursor
	RangeEnd  string
}

type channelContextPrepareDecision struct {
	AppliedAfter  bool
	Bootstrapped  bool
	PriorEpoch    string
	IncomingEpoch string
}

func isChannelContextFeed(entry feeds.FeedEntry) bool {
	if entry.Name == "channel-context" {
		return true
	}
	u, err := url.Parse(entry.URL)
	if err != nil {
		return strings.HasPrefix(entry.Path, "/channel-context")
	}
	return strings.HasSuffix(u.Path, "/channel-context") || u.Path == "/channel-context"
}

func (h *Handler) prepareChannelContextFeed(agentID string, entry feeds.FeedEntry, incomingEpoch string) (feeds.FeedEntry, channelContextPrepareDecision, error) {
	var decision channelContextPrepareDecision
	incomingEpoch = strings.TrimSpace(incomingEpoch)
	if h == nil || h.channelCursors == nil {
		return entry, decision, nil
	}
	u, err := url.Parse(entry.URL)
	if err != nil {
		return entry, decision, err
	}
	q := u.Query()
	channels := splitCSV(q.Get("channels"))
	if len(channels) == 0 {
		return entry, decision, nil
	}
	snapshot, err := h.channelCursors.LoadSnapshot(agentID)
	if err != nil {
		return entry, decision, fmt.Errorf("load channel context cursor: %w", err)
	}
	decision.PriorEpoch = snapshot.Epoch
	if incomingEpoch != "" && incomingEpoch != snapshot.Epoch {
		decision.Bootstrapped = true
		decision.IncomingEpoch = incomingEpoch
		return entry, decision, nil
	}
	after := encodeAfterCursors(channels, snapshot.Cursors)
	if after == "" {
		return entry, decision, nil
	}
	q.Set("after", after)
	u.RawQuery = q.Encode()
	entry.URL = u.String()
	decision.AppliedAfter = true
	return entry, decision, nil
}

func parseChannelContextMetadata(content string) channelContextMetadata {
	var meta channelContextMetadata
	firstLine, _, _ := strings.Cut(content, "\n")
	firstLine = strings.TrimSpace(firstLine)
	if !strings.HasPrefix(firstLine, "[channel-context]") {
		return meta
	}
	for _, field := range strings.Fields(firstLine) {
		key, value, ok := strings.Cut(field, "=")
		if !ok {
			continue
		}
		switch key {
		case "available":
			meta.Available, _ = strconv.Atoi(value)
		case "omitted":
			meta.Omitted, _ = strconv.Atoi(value)
		case "cursor":
			meta.Cursor = parseCursorPairs(value)
		case "range":
			_, end, ok := strings.Cut(value, "..")
			if ok {
				meta.RangeEnd = end
			}
		}
	}
	return meta
}

func appendChannelContextPartialAnnotation(content string, meta channelContextMetadata) string {
	if strings.TrimSpace(content) == "" || meta.Omitted <= 0 {
		return content
	}
	newest := meta.RangeEnd
	if newest == "" {
		newest = "unknown"
	}
	annotation := fmt.Sprintf("[channel-context delta] coverage_partial=true omitted_after_cursor=%d newest_returned=%s", meta.Omitted, newest)
	if strings.Contains(content, annotation) {
		return content
	}
	if strings.HasSuffix(content, "\n") {
		return content + annotation + "\n"
	}
	return content + "\n" + annotation
}

func parseCursorPairs(raw string) map[string]channelCursor {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	out := make(map[string]channelCursor)
	for _, part := range strings.Split(raw, ",") {
		channelID, messageID, ok := strings.Cut(strings.TrimSpace(part), ":")
		channelID = strings.TrimSpace(channelID)
		messageID = strings.TrimSpace(messageID)
		if !ok || channelID == "" || messageID == "" {
			continue
		}
		out[channelID] = channelCursor{LastMessageID: messageID}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}

func splitCSV(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	seen := make(map[string]struct{})
	var out []string
	for _, part := range strings.Split(raw, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		if _, ok := seen[part]; ok {
			continue
		}
		seen[part] = struct{}{}
		out = append(out, part)
	}
	return out
}
