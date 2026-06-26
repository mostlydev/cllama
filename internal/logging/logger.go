package logging

import (
	"encoding/json"
	"io"
	"sync"
	"time"
)

// Logger writes structured JSON logs suitable for claw audit ingestion.
type Logger struct {
	mu  sync.Mutex
	enc *json.Encoder
}

type entry struct {
	TS                 string   `json:"ts"`
	ClawID             string   `json:"claw_id,omitempty"`
	Type               string   `json:"type"`
	Model              string   `json:"model,omitempty"`
	StaticSystemHash   string   `json:"static_system_hash,omitempty"`
	FirstSystemHash    string   `json:"first_system_hash,omitempty"`
	FirstNonSystemHash string   `json:"first_non_system_hash,omitempty"`
	DynamicContextHash string   `json:"dynamic_context_hash,omitempty"`
	ToolsHash          string   `json:"tools_hash,omitempty"`
	ManifestPresent    *bool    `json:"manifest_present,omitempty"`
	ToolsCount         *int     `json:"tools_count,omitempty"`
	FeedName           string   `json:"feed_name,omitempty"`
	FeedURL            string   `json:"feed_url,omitempty"`
	FeedFetchedAt      string   `json:"feed_fetched_at,omitempty"`
	FeedCached         *bool    `json:"feed_cached,omitempty"`
	FeedStatus         string   `json:"feed_status,omitempty"`
	FeedTruncated      *bool    `json:"feed_truncated,omitempty"`
	FeedSourceBytes    *int     `json:"feed_source_bytes,omitempty"`
	FeedSourceExact    *bool    `json:"feed_source_exact,omitempty"`
	FeedContentBytes   *int     `json:"feed_content_bytes,omitempty"`
	FeedBlockBytes     *int     `json:"feed_block_bytes,omitempty"`
	FeedTotalBefore    *int     `json:"feed_total_before,omitempty"`
	FeedTotalAfter     *int     `json:"feed_total_after,omitempty"`
	FeedMaxResponse    *int     `json:"feed_max_response_bytes,omitempty"`
	FeedMaxTotal       *int     `json:"feed_max_total_bytes,omitempty"`
	FeedRawBytes       *int     `json:"feed_raw_bytes,omitempty"`
	FeedDigestBytes    *int     `json:"feed_digest_bytes,omitempty"`
	LatencyMS          *int64   `json:"latency_ms,omitempty"`
	StatusCode         *int     `json:"status_code,omitempty"`
	TokensIn           *int     `json:"tokens_in,omitempty"`
	TokensOut          *int     `json:"tokens_out,omitempty"`
	CachedTokens       *int     `json:"cached_tokens,omitempty"`
	CacheWriteTokens   *int     `json:"cache_write_tokens,omitempty"`
	CostUSD            *float64 `json:"cost_usd,omitempty"`
	Intervention       *string  `json:"intervention"`
	Error              string   `json:"error,omitempty"`
	// memory_op event fields
	MemoryService *string `json:"memory_service,omitempty"`
	MemoryOp      *string `json:"memory_op,omitempty"`
	MemoryStatus  *string `json:"memory_status,omitempty"`
	MemoryBlocks  *int    `json:"memory_blocks,omitempty"`
	MemoryBytes   *int    `json:"memory_bytes,omitempty"`
	MemoryRemoved *int    `json:"memory_removed,omitempty"`
	// context_block event fields
	ContextBlockID        *string `json:"context_block_id,omitempty"`
	ContextBlockKind      *string `json:"context_block_kind,omitempty"`
	ContextBlockStatus    *string `json:"context_block_status,omitempty"`
	ContextBlockCadence   *string `json:"context_block_cadence,omitempty"`
	ContextBlockPlacement *string `json:"context_block_placement,omitempty"`
	ContextBlockReason    *string `json:"context_block_reason,omitempty"`
	// channel_context_op event fields
	ChannelKind       *string  `json:"kind,omitempty"`
	Channels          []string `json:"channels,omitempty"`
	Retained          *int     `json:"retained,omitempty"`
	Returned          *int     `json:"returned,omitempty"`
	Omitted           *int     `json:"omitted,omitempty"`
	RawBytes          *int     `json:"raw_bytes,omitempty"`
	DigestBytes       *int     `json:"digest_bytes,omitempty"`
	DigestBlocks      *int     `json:"digest_blocks,omitempty"`
	CoverageGaps      *int     `json:"coverage_gaps,omitempty"`
	DeterministicOnly *bool    `json:"deterministic_only,omitempty"`
	Source            *string  `json:"source,omitempty"`
	Status            *string  `json:"status,omitempty"`
	ToolName          *string  `json:"tool_name,omitempty"`
	// provider_pool event fields
	Provider      string `json:"provider,omitempty"`
	KeyID         string `json:"key_id,omitempty"`
	Action        string `json:"action,omitempty"`
	Reason        string `json:"reason,omitempty"`
	CooldownUntil string `json:"cooldown_until,omitempty"`
}

// CostInfo holds token counts and estimated cost for a single LLM request.
type CostInfo struct {
	InputTokens      int
	OutputTokens     int
	CachedTokens     *int
	CacheWriteTokens *int
	CostUSD          *float64
}

type RequestInfo struct {
	StaticSystemHash   string
	FirstSystemHash    string
	FirstNonSystemHash string
	DynamicContextHash string
	ToolsHash          string
}

// FeedInjectionInfo records whether a fetched feed block actually reached the
// provider-visible runtime context after per-feed and aggregate byte caps.
type FeedInjectionInfo struct {
	Name                 string
	Source               string
	Status               string
	Truncated            bool
	SourceBytes          int
	SourceBytesExact     bool
	ContentBytes         int
	BlockBytes           int
	TotalBytesBefore     int
	TotalBytesAfter      int
	MaxFeedResponseBytes int
	MaxTotalFeedBytes    int
	RawBytes             int
	DigestBytes          int
}

// MemoryOpInfo holds structured telemetry for memory recall/retain hooks.
type MemoryOpInfo struct {
	Service       string
	Operation     string
	Status        string
	StatusCode    int
	LatencyMS     int64
	Blocks        *int
	InjectedBytes *int
	PolicyRemoved *int
	Error         error
}

type ContextBlockInfo struct {
	ID        string
	Kind      string
	Status    string
	Cadence   string
	Placement string
	Reason    string
}

// ChannelContextOpInfo holds structured telemetry for channel awareness,
// channel-context, and channel retrieval tool operations.
type ChannelContextOpInfo struct {
	Kind              string
	Channels          []string
	Retained          int
	Returned          int
	Omitted           int
	RawBytes          int
	DigestBytes       int
	DigestBlocks      int
	CoverageGaps      int
	DeterministicOnly *bool
	Source            string
	Status            string
	ToolName          string
	StatusCode        int
	LatencyMS         int64
	Error             error
}

func New(w io.Writer) *Logger {
	if w == nil {
		w = io.Discard
	}
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return &Logger{enc: enc}
}

func (l *Logger) LogRequest(clawID, model string) {
	l.LogRequestWithInfo(clawID, model, nil)
}

func (l *Logger) LogRequestWithInfo(clawID, model string, info *RequestInfo) {
	e := entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "request",
		Model:        model,
		Intervention: nil,
	}
	if info != nil {
		e.StaticSystemHash = info.StaticSystemHash
		e.FirstSystemHash = info.FirstSystemHash
		e.FirstNonSystemHash = info.FirstNonSystemHash
		e.DynamicContextHash = info.DynamicContextHash
		e.ToolsHash = info.ToolsHash
	}
	l.log(e)
}

func (l *Logger) LogResponse(clawID, model string, statusCode int, latencyMS int64) {
	l.log(entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "response",
		Model:        model,
		LatencyMS:    ptrI64(latencyMS),
		StatusCode:   ptrInt(statusCode),
		Intervention: nil,
	})
}

func (l *Logger) LogError(clawID, model string, statusCode int, latencyMS int64, err error) {
	errText := ""
	if err != nil {
		errText = err.Error()
	}
	l.log(entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "error",
		Model:        model,
		LatencyMS:    ptrI64(latencyMS),
		StatusCode:   ptrInt(statusCode),
		Intervention: nil,
		Error:        errText,
	})
}

func (l *Logger) LogResponseWithCost(clawID, model string, statusCode int, latencyMS int64, ci *CostInfo) {
	e := entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "response",
		Model:        model,
		LatencyMS:    ptrI64(latencyMS),
		StatusCode:   ptrInt(statusCode),
		Intervention: nil,
	}
	if ci != nil {
		e.TokensIn = ptrInt(ci.InputTokens)
		e.TokensOut = ptrInt(ci.OutputTokens)
		e.CachedTokens = ci.CachedTokens
		e.CacheWriteTokens = ci.CacheWriteTokens
		if ci.CostUSD != nil {
			e.CostUSD = ci.CostUSD
		}
	}
	l.log(e)
}

func (l *Logger) LogIntervention(clawID, model, reason string) {
	reasonCopy := reason
	l.log(entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "intervention",
		Model:        model,
		Intervention: &reasonCopy,
	})
}

func (l *Logger) LogFeedFetch(clawID, feedName, feedURL string, statusCode int, latencyMS int64, err error) {
	l.LogFeedFetchWithInfo(clawID, feedName, feedURL, statusCode, latencyMS, nil, "", err)
}

func (l *Logger) LogFeedFetchWithInfo(clawID, feedName, feedURL string, statusCode int, latencyMS int64, cached *bool, fetchedAt string, err error) {
	e := entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "feed_fetch",
		FeedName:     feedName,
		FeedURL:      feedURL,
		LatencyMS:    ptrI64(latencyMS),
		Intervention: nil,
	}
	e.FeedCached = cached
	e.FeedFetchedAt = fetchedAt
	if statusCode > 0 {
		e.StatusCode = ptrInt(statusCode)
	}
	if err != nil {
		e.Error = err.Error()
	}
	l.log(e)
}

func (l *Logger) LogFeedInjection(clawID, model string, info FeedInjectionInfo) {
	e := entry{
		TS:               time.Now().UTC().Format(time.RFC3339),
		ClawID:           clawID,
		Type:             "feed_injection",
		Model:            model,
		FeedName:         info.Name,
		Source:           ptrString(info.Source),
		FeedStatus:       info.Status,
		FeedTruncated:    ptrBool(info.Truncated),
		FeedSourceBytes:  ptrInt(info.SourceBytes),
		FeedSourceExact:  ptrBool(info.SourceBytesExact),
		FeedContentBytes: ptrInt(info.ContentBytes),
		FeedBlockBytes:   ptrInt(info.BlockBytes),
		FeedTotalBefore:  ptrInt(info.TotalBytesBefore),
		FeedTotalAfter:   ptrInt(info.TotalBytesAfter),
		FeedMaxResponse:  ptrInt(info.MaxFeedResponseBytes),
		FeedMaxTotal:     ptrInt(info.MaxTotalFeedBytes),
		FeedRawBytes:     ptrInt(info.RawBytes),
		FeedDigestBytes:  ptrInt(info.DigestBytes),
		Intervention:     nil,
	}
	l.log(e)
}

func (l *Logger) LogToolManifest(clawID, model string, manifestPresent bool, toolsCount int) {
	l.log(entry{
		TS:              time.Now().UTC().Format(time.RFC3339),
		ClawID:          clawID,
		Type:            "tool_manifest_loaded",
		Model:           model,
		ManifestPresent: ptrBool(manifestPresent),
		ToolsCount:      ptrInt(toolsCount),
		Intervention:    nil,
	})
}

func (l *Logger) LogMemoryOp(clawID, model string, info MemoryOpInfo) {
	e := entry{
		TS:            time.Now().UTC().Format(time.RFC3339),
		ClawID:        clawID,
		Type:          "memory_op",
		Model:         model,
		Intervention:  nil,
		MemoryService: ptrString(info.Service),
		MemoryOp:      ptrString(info.Operation),
		MemoryStatus:  ptrString(info.Status),
		MemoryBlocks:  info.Blocks,
		MemoryBytes:   info.InjectedBytes,
		MemoryRemoved: info.PolicyRemoved,
	}
	if info.LatencyMS > 0 {
		e.LatencyMS = ptrI64(info.LatencyMS)
	}
	if info.StatusCode > 0 {
		e.StatusCode = ptrInt(info.StatusCode)
	}
	if info.Error != nil {
		e.Error = info.Error.Error()
	}
	l.log(e)
}

func (l *Logger) LogContextBlock(clawID, model string, info ContextBlockInfo) {
	l.log(entry{
		TS:                    time.Now().UTC().Format(time.RFC3339),
		ClawID:                clawID,
		Type:                  "context_block",
		Model:                 model,
		Intervention:          nil,
		ContextBlockID:        ptrString(info.ID),
		ContextBlockKind:      ptrString(info.Kind),
		ContextBlockStatus:    ptrString(info.Status),
		ContextBlockCadence:   ptrString(info.Cadence),
		ContextBlockPlacement: ptrString(info.Placement),
		ContextBlockReason:    ptrString(info.Reason),
	})
}

func (l *Logger) LogChannelContextOp(clawID, model string, info ChannelContextOpInfo) {
	e := entry{
		TS:                time.Now().UTC().Format(time.RFC3339),
		ClawID:            clawID,
		Type:              "channel_context_op",
		Model:             model,
		Intervention:      nil,
		ChannelKind:       ptrString(info.Kind),
		Channels:          append([]string(nil), info.Channels...),
		Retained:          ptrInt(info.Retained),
		Returned:          ptrInt(info.Returned),
		Omitted:           ptrInt(info.Omitted),
		RawBytes:          ptrInt(info.RawBytes),
		DigestBytes:       ptrInt(info.DigestBytes),
		DigestBlocks:      ptrInt(info.DigestBlocks),
		CoverageGaps:      ptrInt(info.CoverageGaps),
		DeterministicOnly: info.DeterministicOnly,
		Source:            ptrString(info.Source),
		Status:            ptrString(info.Status),
		ToolName:          ptrString(info.ToolName),
	}
	if info.LatencyMS > 0 {
		e.LatencyMS = ptrI64(info.LatencyMS)
	}
	if info.StatusCode > 0 {
		e.StatusCode = ptrInt(info.StatusCode)
	}
	if info.Error != nil {
		e.Error = info.Error.Error()
	}
	l.log(e)
}

// LogProviderPool emits a structured provider_pool transition event.
// action is one of: "cooldown", "dead", "activated", "added", "deleted".
func (l *Logger) LogProviderPool(provider, keyID, action, reason, cooldownUntil string) {
	e := entry{
		TS:            time.Now().UTC().Format(time.RFC3339),
		Type:          "provider_pool",
		Provider:      provider,
		KeyID:         keyID,
		Action:        action,
		Reason:        reason,
		CooldownUntil: cooldownUntil,
		Intervention:  nil,
	}
	l.log(e)
}

func (l *Logger) log(e entry) {
	if l == nil || l.enc == nil {
		return
	}
	l.mu.Lock()
	_ = l.enc.Encode(e)
	l.mu.Unlock()
}

func ptrInt(v int) *int {
	return &v
}

func ptrI64(v int64) *int64 {
	return &v
}

func ptrF64(v float64) *float64 {
	return &v
}

func ptrString(v string) *string {
	if v == "" {
		return nil
	}
	return &v
}

func ptrBool(v bool) *bool {
	return &v
}
