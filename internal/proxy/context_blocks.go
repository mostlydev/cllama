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
	contextBlocksFile            = "context-blocks.json"
	defaultContextBlockKind      = "context_block"
	defaultContextBlockMaxChars  = 800
	defaultContextBlockCadence   = "every_turn"
	defaultContextBlockPlacement = "after_feeds"
	contextBlockPlacementBefore  = "before_feeds"
	contextBlockPlacementAfter   = "after_feeds"
)

type ContextBlockSnapshot struct {
	ID              string `json:"id"`
	Kind            string `json:"kind"`
	Text            string `json:"text"`
	SourcePath      string `json:"source_path"`
	Cadence         string `json:"cadence"`
	CadenceDecision string `json:"cadence_decision"`
	Placement       string `json:"placement"`
	Bytes           int    `json:"bytes"`
}

type contextBlockSkip struct {
	ID        string
	Kind      string
	Cadence   string
	Placement string
	Reason    string
}

type renderedContextBlocks struct {
	BeforeFeeds string
	AfterFeeds  string
	Snapshots   []ContextBlockSnapshot
	Skips       []contextBlockSkip
}

func renderContextBlocks(agentCtx *agentctx.AgentContext) renderedContextBlocks {
	if agentCtx == nil || agentCtx.ContextBlocks == nil || len(agentCtx.ContextBlocks.Blocks) == 0 {
		return renderedContextBlocks{}
	}

	sourcePath := ""
	if strings.TrimSpace(agentCtx.ContextDir) != "" {
		sourcePath = filepath.Join(agentCtx.ContextDir, contextBlocksFile)
	}

	var beforeFeeds []string
	var afterFeeds []string
	var snapshots []ContextBlockSnapshot
	var skips []contextBlockSkip
	for _, entry := range agentCtx.ContextBlocks.Blocks {
		id := strings.TrimSpace(entry.ID)
		if id == "" {
			id = "context-block"
		}
		kind := normalizeContextBlockKind(entry.Kind)
		if entry.Enabled != nil && !*entry.Enabled {
			continue
		}
		text := strings.TrimSpace(entry.Text)
		if text == "" {
			continue
		}
		cadence := normalizeContextBlockCadence(entry.Cadence)
		placement := normalizeContextBlockPlacement(entry.Placement)
		if cadence != defaultContextBlockCadence {
			skips = append(skips, contextBlockSkip{ID: id, Kind: kind, Cadence: cadence, Placement: placement, Reason: "unsupported_cadence"})
			continue
		}
		if placement != contextBlockPlacementBefore && placement != contextBlockPlacementAfter {
			skips = append(skips, contextBlockSkip{ID: id, Kind: kind, Cadence: cadence, Placement: placement, Reason: "unsupported_placement"})
			continue
		}

		text = trimContextBlockText(text, entry.MaxChars)
		block := fmt.Sprintf("--- CONTEXT BLOCK: %s/%s ---\n%s", kind, id, text)
		switch placement {
		case contextBlockPlacementBefore:
			beforeFeeds = append(beforeFeeds, block)
		case contextBlockPlacementAfter:
			afterFeeds = append(afterFeeds, block)
		}
		snapshots = append(snapshots, ContextBlockSnapshot{
			ID:              id,
			Kind:            kind,
			Text:            block,
			SourcePath:      sourcePath,
			Cadence:         cadence,
			CadenceDecision: "injected_every_turn",
			Placement:       placement,
			Bytes:           len([]byte(block)),
		})
	}

	return renderedContextBlocks{
		BeforeFeeds: joinContextBlocks(contextBlockPlacementBefore, beforeFeeds),
		AfterFeeds:  joinContextBlocks(contextBlockPlacementAfter, afterFeeds),
		Snapshots:   snapshots,
		Skips:       skips,
	}
}

func joinContextBlocks(placement string, blocks []string) string {
	if len(blocks) == 0 {
		return ""
	}
	return fmt.Sprintf("--- BEGIN CONTEXT BLOCKS placement=%s ---\n%s\n--- END CONTEXT BLOCKS ---", placement, strings.Join(blocks, "\n\n"))
}

func (h *Handler) logContextBlockSkips(agentID, requestedModel string, skips []contextBlockSkip) {
	if h == nil || h.logger == nil {
		return
	}
	for _, skip := range skips {
		h.logger.LogContextBlock(agentID, requestedModel, logging.ContextBlockInfo{
			ID:        skip.ID,
			Kind:      skip.Kind,
			Status:    "skipped",
			Cadence:   skip.Cadence,
			Placement: skip.Placement,
			Reason:    skip.Reason,
		})
	}
}

func normalizeContextBlockKind(kind string) string {
	kind = strings.TrimSpace(kind)
	if kind == "" {
		return defaultContextBlockKind
	}
	return kind
}

func normalizeContextBlockCadence(cadence string) string {
	cadence = strings.TrimSpace(cadence)
	if cadence == "" {
		return defaultContextBlockCadence
	}
	return cadence
}

func normalizeContextBlockPlacement(placement string) string {
	placement = strings.TrimSpace(placement)
	if placement == "" {
		return defaultContextBlockPlacement
	}
	return placement
}

func trimContextBlockText(text string, maxChars int) string {
	if maxChars <= 0 {
		maxChars = defaultContextBlockMaxChars
	}
	if utf8.RuneCountInString(text) <= maxChars {
		return text
	}
	runes := []rune(text)
	return strings.TrimSpace(string(runes[:maxChars]))
}
