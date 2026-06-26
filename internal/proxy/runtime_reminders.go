package proxy

import (
	"fmt"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/logging"
)

const (
	runtimeRemindersFile            = "runtime-reminders.json"
	defaultRuntimeReminderMaxChars  = 800
	defaultRuntimeReminderCadence   = "every_turn"
	defaultRuntimeReminderPlacement = "before_feeds"
)

type RuntimeReminderSnapshot struct {
	ID              string `json:"id"`
	Text            string `json:"text"`
	SourcePath      string `json:"source_path"`
	Cadence         string `json:"cadence"`
	CadenceDecision string `json:"cadence_decision"`
	Placement       string `json:"placement"`
	Bytes           int    `json:"bytes"`
}

type runtimeReminderSkip struct {
	ID        string
	Cadence   string
	Placement string
	Reason    string
}

func renderRuntimeReminders(agentCtx *agentctx.AgentContext) (string, []RuntimeReminderSnapshot, []runtimeReminderSkip) {
	if agentCtx == nil || agentCtx.RuntimeReminders == nil || len(agentCtx.RuntimeReminders.Reminders) == 0 {
		return "", nil, nil
	}

	sourcePath := ""
	if strings.TrimSpace(agentCtx.ContextDir) != "" {
		sourcePath = filepath.Join(agentCtx.ContextDir, runtimeRemindersFile)
	}

	var blocks []string
	var snapshots []RuntimeReminderSnapshot
	var skips []runtimeReminderSkip
	for _, reminder := range agentCtx.RuntimeReminders.Reminders {
		id := strings.TrimSpace(reminder.ID)
		if id == "" {
			id = "runtime-reminder"
		}
		if reminder.Enabled != nil && !*reminder.Enabled {
			continue
		}
		text := strings.TrimSpace(reminder.Text)
		if text == "" {
			continue
		}
		cadence := normalizeRuntimeReminderCadence(reminder.Cadence)
		if cadence != defaultRuntimeReminderCadence {
			skips = append(skips, runtimeReminderSkip{ID: id, Cadence: cadence, Placement: normalizeRuntimeReminderPlacement(reminder.Placement), Reason: "unsupported_cadence"})
			continue
		}
		placement := normalizeRuntimeReminderPlacement(reminder.Placement)
		if placement != defaultRuntimeReminderPlacement {
			skips = append(skips, runtimeReminderSkip{ID: id, Cadence: cadence, Placement: placement, Reason: "unsupported_placement"})
			continue
		}

		text = trimRuntimeReminderText(text, reminder.MaxChars)
		block := fmt.Sprintf("--- RUNTIME REMINDER: %s ---\n%s", id, text)
		blocks = append(blocks, block)
		snapshots = append(snapshots, RuntimeReminderSnapshot{
			ID:              id,
			Text:            block,
			SourcePath:      sourcePath,
			Cadence:         cadence,
			CadenceDecision: "injected_every_turn",
			Placement:       placement,
			Bytes:           len([]byte(block)),
		})
	}

	if len(blocks) == 0 {
		return "", nil, skips
	}
	return "--- BEGIN RUNTIME REMINDERS ---\n" + strings.Join(blocks, "\n\n") + "\n--- END RUNTIME REMINDERS ---", snapshots, skips
}

func (h *Handler) logRuntimeReminderSkips(agentID, requestedModel string, skips []runtimeReminderSkip) {
	if h == nil || h.logger == nil {
		return
	}
	for _, skip := range skips {
		h.logger.LogRuntimeReminder(agentID, requestedModel, logging.RuntimeReminderInfo{
			ID:        skip.ID,
			Status:    "skipped",
			Cadence:   skip.Cadence,
			Placement: skip.Placement,
			Reason:    skip.Reason,
		})
	}
}

func normalizeRuntimeReminderCadence(cadence string) string {
	cadence = strings.TrimSpace(cadence)
	if cadence == "" {
		return defaultRuntimeReminderCadence
	}
	return cadence
}

func normalizeRuntimeReminderPlacement(placement string) string {
	placement = strings.TrimSpace(placement)
	if placement == "" {
		return defaultRuntimeReminderPlacement
	}
	return placement
}

func trimRuntimeReminderText(text string, maxChars int) string {
	if maxChars <= 0 {
		maxChars = defaultRuntimeReminderMaxChars
	}
	if utf8.RuneCountInString(text) <= maxChars {
		return text
	}
	runes := []rune(text)
	return strings.TrimSpace(string(runes[:maxChars]))
}
