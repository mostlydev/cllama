// Package alert delivers best-effort pool-event notifications to Discord webhooks.
// Webhook failures never block request failover — all errors are logged to stderr only.
package alert

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// PoolEvent describes a provider key state transition.
type PoolEvent struct {
	Provider      string
	KeyID         string
	Action        string // "cooldown", "dead", "activated", "added", "deleted"
	Reason        string
	CooldownUntil string // RFC3339 or ""
}

// Notifier delivers pool events to configured Discord webhooks.
type Notifier struct {
	webhooks []string
	mentions []string
	client   *http.Client
	stderr   io.Writer
}

// NewNotifier reads CLLAMA_ALERT_WEBHOOKS and CLLAMA_ALERT_MENTIONS from the
// environment and returns a Notifier.  An empty notifier silently discards events.
func NewNotifier(stderr io.Writer) *Notifier {
	if stderr == nil {
		stderr = io.Discard
	}
	webhooks := splitEnv("CLLAMA_ALERT_WEBHOOKS")
	mentions := splitEnv("CLLAMA_ALERT_MENTIONS")
	return &Notifier{
		webhooks: webhooks,
		mentions: mentions,
		client:   &http.Client{Timeout: 5 * time.Second},
		stderr:   stderr,
	}
}

func splitEnv(key string) []string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return nil
	}
	var out []string
	for _, part := range strings.Split(v, ",") {
		p := strings.TrimSpace(part)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// Notify sends a pool event to all configured webhooks in a background goroutine.
// It returns immediately — failures never block the caller.
func (n *Notifier) Notify(ev PoolEvent) {
	if len(n.webhooks) == 0 {
		return
	}
	content := n.formatMessage(ev)
	for _, url := range n.webhooks {
		go n.post(url, content)
	}
}

func (n *Notifier) formatMessage(ev PoolEvent) string {
	var sb strings.Builder
	// Mentions first so Discord highlights them.
	if len(n.mentions) > 0 {
		sb.WriteString(strings.Join(n.mentions, " "))
		sb.WriteString(" ")
	}

	switch ev.Action {
	case "dead":
		fmt.Fprintf(&sb, "**[cllama]** Provider `%s` key `%s` marked **dead** — reason: %s",
			ev.Provider, ev.KeyID, ev.Reason)
	case "cooldown":
		fmt.Fprintf(&sb, "**[cllama]** Provider `%s` key `%s` cooling down until %s — reason: %s",
			ev.Provider, ev.KeyID, ev.CooldownUntil, ev.Reason)
	case "activated":
		fmt.Fprintf(&sb, "**[cllama]** Provider `%s` key `%s` activated",
			ev.Provider, ev.KeyID)
	case "added":
		fmt.Fprintf(&sb, "**[cllama]** Provider `%s` new key `%s` added",
			ev.Provider, ev.KeyID)
	case "deleted":
		fmt.Fprintf(&sb, "**[cllama]** Provider `%s` key `%s` deleted",
			ev.Provider, ev.KeyID)
	default:
		fmt.Fprintf(&sb, "**[cllama]** Provider `%s` key `%s` — %s",
			ev.Provider, ev.KeyID, ev.Action)
	}
	return sb.String()
}

// discordPayload is the minimal Discord webhook POST body.
type discordPayload struct {
	Content string `json:"content"`
}

func (n *Notifier) post(webhookURL, content string) {
	body, err := json.Marshal(discordPayload{Content: content})
	if err != nil {
		fmt.Fprintf(n.stderr, "alert webhook marshal: %v\n", err)
		return
	}
	resp, err := n.client.Post(webhookURL, "application/json", bytes.NewReader(body))
	if err != nil {
		fmt.Fprintf(n.stderr, "alert webhook post: %v\n", err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		fmt.Fprintf(n.stderr, "alert webhook response: %s\n", resp.Status)
	}
}
