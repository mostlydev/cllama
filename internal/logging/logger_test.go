package logging

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestLogRequestEmitsJSON(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	l.LogRequest("tiverton", "openai/gpt-4o")

	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v\nraw: %s", err, buf.String())
	}
	if entry["claw_id"] != "tiverton" {
		t.Errorf("expected claw_id=tiverton, got %v", entry["claw_id"])
	}
	if entry["type"] != "request" {
		t.Errorf("expected type=request, got %v", entry["type"])
	}
	if entry["model"] != "openai/gpt-4o" {
		t.Errorf("expected model, got %v", entry["model"])
	}
	if _, ok := entry["intervention"]; !ok {
		t.Errorf("expected intervention field to be present")
	}
}

func TestLogRequestIncludesPromptHashes(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	l.LogRequestWithInfo("tiverton", "openai/gpt-4o", &RequestInfo{
		StaticSystemHash:   "static",
		FirstSystemHash:    "first-system",
		FirstNonSystemHash: "first-user",
		DynamicContextHash: "dynamic",
		ToolsHash:          "tools",
	})

	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if entry["static_system_hash"] != "static" || entry["dynamic_context_hash"] != "dynamic" || entry["tools_hash"] != "tools" {
		t.Fatalf("missing prompt hashes: %+v", entry)
	}
}

func TestLogContextBlockIncludesSkipFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	l.LogContextBlock("agent", "openai/gpt-4o", ContextBlockInfo{
		ID:        "focus",
		Kind:      "runtime_motivation",
		Status:    "skipped",
		Cadence:   "min_interval",
		Placement: "before_feeds",
		Reason:    "unsupported_cadence",
	})

	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if entry["type"] != "context_block" || entry["context_block_id"] != "focus" || entry["context_block_kind"] != "runtime_motivation" || entry["context_block_status"] != "skipped" {
		t.Fatalf("unexpected context block entry: %+v", entry)
	}
	if entry["context_block_cadence"] != "min_interval" || entry["context_block_placement"] != "before_feeds" || entry["context_block_reason"] != "unsupported_cadence" {
		t.Fatalf("missing context block skip fields: %+v", entry)
	}
}

func TestLogResponseIncludesLatency(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	l.LogResponse("tiverton", "openai/gpt-4o", 200, 1250)

	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if entry["type"] != "response" {
		t.Errorf("expected type=response")
	}
	if entry["latency_ms"].(float64) != 1250 {
		t.Errorf("expected latency_ms=1250, got %v", entry["latency_ms"])
	}
	if entry["status_code"].(float64) != 200 {
		t.Errorf("expected status_code=200, got %v", entry["status_code"])
	}
}

func TestLogResponseIncludesCostFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	costUSD := 0.0105
	l.LogResponseWithCost("tiverton", "anthropic/claude-sonnet-4", 200, 1250,
		&CostInfo{InputTokens: 100, OutputTokens: 50, CostUSD: &costUSD})

	var entry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if entry["tokens_in"].(float64) != 100 {
		t.Errorf("expected tokens_in=100, got %v", entry["tokens_in"])
	}
	if entry["tokens_out"].(float64) != 50 {
		t.Errorf("expected tokens_out=50, got %v", entry["tokens_out"])
	}
	if entry["cost_usd"].(float64) < 0.01 || entry["cost_usd"].(float64) > 0.02 {
		t.Errorf("expected cost_usd ~0.0105, got %v", entry["cost_usd"])
	}
}

func TestLogResponseIncludesCacheTokenFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	cached := 100
	writes := 20
	l.LogResponseWithCost("tiverton", "openrouter/model", 200, 1250,
		&CostInfo{InputTokens: 100, OutputTokens: 50, CachedTokens: &cached, CacheWriteTokens: &writes})

	var entry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if entry["cached_tokens"].(float64) != 100 {
		t.Fatalf("expected cached_tokens=100, got %+v", entry)
	}
	if entry["cache_write_tokens"].(float64) != 20 {
		t.Fatalf("expected cache_write_tokens=20, got %+v", entry)
	}
}

func TestLogResponseWithoutCost(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	l.LogResponseWithCost("tiverton", "anthropic/claude-sonnet-4", 200, 500, nil)

	var entry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if _, ok := entry["tokens_in"]; ok {
		t.Error("expected no tokens_in when CostInfo is nil")
	}
}

func TestLogResponseWithUsageButUnknownCost(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	l.LogResponseWithCost("tiverton", "anthropic/claude-sonnet-4", 200, 500,
		&CostInfo{InputTokens: 321, OutputTokens: 89})

	var entry map[string]interface{}
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if entry["tokens_in"].(float64) != 321 {
		t.Errorf("expected tokens_in=321, got %v", entry["tokens_in"])
	}
	if _, ok := entry["cost_usd"]; ok {
		t.Error("expected cost_usd to be omitted when unknown")
	}
}

func TestLogFailoverIncludesStructuredFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	l.LogFailover("weston", "openai/gpt-4o", "openai", "gpt-4o", "openrouter", "anthropic/claude-haiku-4-5", "http_500", 1, 42)

	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v\nraw: %s", err, buf.String())
	}
	if entry["type"] != "failover" || entry["claw_id"] != "weston" || entry["model"] != "openai/gpt-4o" {
		t.Fatalf("unexpected failover identity fields: %+v", entry)
	}
	if entry["from_provider"] != "openai" || entry["from_model"] != "gpt-4o" || entry["to_provider"] != "openrouter" || entry["to_model"] != "anthropic/claude-haiku-4-5" {
		t.Fatalf("unexpected failover route fields: %+v", entry)
	}
	if entry["reason"] != "http_500" || entry["slot_index"].(float64) != 1 || entry["latency_ms"].(float64) != 42 {
		t.Fatalf("unexpected failover reason/timing fields: %+v", entry)
	}
}

func TestLogFeedFetchIncludesFeedFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	l.LogFeedFetch("weston", "market-context", "http://trading-api:4000/api/v1/market_context/weston", 200, 85, nil)

	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if entry["type"] != "feed_fetch" {
		t.Fatalf("expected type=feed_fetch, got %v", entry["type"])
	}
	if entry["feed_name"] != "market-context" {
		t.Errorf("expected feed_name, got %v", entry["feed_name"])
	}
	if entry["feed_url"] != "http://trading-api:4000/api/v1/market_context/weston" {
		t.Errorf("expected feed_url, got %v", entry["feed_url"])
	}
	if entry["status_code"].(float64) != 200 {
		t.Errorf("expected status_code=200, got %v", entry["status_code"])
	}
}

func TestLogFeedInjectionIncludesBudgetFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	l.LogFeedInjection("weston", "openrouter/model", FeedInjectionInfo{
		Name:                 "channel-awareness",
		Source:               "claw-wall",
		Status:               "skipped_total_cap",
		Truncated:            true,
		SourceBytes:          107000,
		SourceBytesExact:     true,
		ContentBytes:         32768,
		BlockBytes:           33000,
		TotalBytesBefore:     64000,
		TotalBytesAfter:      64000,
		MaxFeedResponseBytes: 32768,
		MaxTotalFeedBytes:    65536,
		RawBytes:             12000,
		DigestBytes:          800,
	})

	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if entry["type"] != "feed_injection" {
		t.Fatalf("expected type=feed_injection, got %v", entry["type"])
	}
	if entry["feed_name"] != "channel-awareness" || entry["source"] != "claw-wall" {
		t.Fatalf("unexpected feed identity fields: %+v", entry)
	}
	if entry["feed_status"] != "skipped_total_cap" || entry["feed_truncated"] != true {
		t.Fatalf("unexpected feed status fields: %+v", entry)
	}
	if entry["feed_source_bytes"].(float64) != 107000 || entry["feed_max_total_bytes"].(float64) != 65536 {
		t.Fatalf("unexpected feed byte fields: %+v", entry)
	}
	if entry["feed_raw_bytes"].(float64) != 12000 || entry["feed_digest_bytes"].(float64) != 800 {
		t.Fatalf("unexpected channel feed byte fields: %+v", entry)
	}
}

func TestLogToolManifestIncludesStructuredFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	l.LogToolManifest("weston", "openai/gpt-4o", true, 2)

	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if entry["type"] != "tool_manifest_loaded" {
		t.Fatalf("expected type=tool_manifest_loaded, got %v", entry["type"])
	}
	if entry["manifest_present"] != true {
		t.Fatalf("expected manifest_present=true, got %v", entry["manifest_present"])
	}
	if entry["tools_count"].(float64) != 2 {
		t.Fatalf("expected tools_count=2, got %v", entry["tools_count"])
	}
}

func TestLogMemoryOpIncludesStructuredFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	blocks := 2
	injectedBytes := 144
	l.LogMemoryOp("weston", "openai/gpt-4o", MemoryOpInfo{
		Service:       "team-memory",
		Operation:     "recall",
		Status:        "succeeded",
		StatusCode:    200,
		LatencyMS:     37,
		Blocks:        &blocks,
		InjectedBytes: &injectedBytes,
	})

	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if entry["type"] != "memory_op" {
		t.Fatalf("expected type=memory_op, got %v", entry["type"])
	}
	if entry["memory_service"] != "team-memory" {
		t.Fatalf("expected memory_service, got %v", entry["memory_service"])
	}
	if entry["memory_op"] != "recall" {
		t.Fatalf("expected memory_op=recall, got %v", entry["memory_op"])
	}
	if entry["memory_status"] != "succeeded" {
		t.Fatalf("expected memory_status=succeeded, got %v", entry["memory_status"])
	}
	if entry["memory_blocks"].(float64) != 2 {
		t.Fatalf("expected memory_blocks=2, got %v", entry["memory_blocks"])
	}
	if entry["memory_bytes"].(float64) != 144 {
		t.Fatalf("expected memory_bytes=144, got %v", entry["memory_bytes"])
	}
	if entry["latency_ms"].(float64) != 37 {
		t.Fatalf("expected latency_ms=37, got %v", entry["latency_ms"])
	}
}

func TestLogChannelContextOpIncludesStructuredFields(t *testing.T) {
	var buf bytes.Buffer
	l := New(&buf)
	deterministic := true
	l.LogChannelContextOp("weston", "openai/gpt-4o", ChannelContextOpInfo{
		Kind:              "raw_window+digest",
		Channels:          []string{"chan-1", "chan-2"},
		Retained:          60,
		Returned:          40,
		Omitted:           20,
		RawBytes:          2048,
		DigestBytes:       512,
		DigestBlocks:      3,
		CoverageGaps:      1,
		DeterministicOnly: &deterministic,
		Source:            "claw-wall",
		Status:            "coverage_gap",
		ToolName:          "search_channel_context",
		StatusCode:        200,
		LatencyMS:         17,
	})

	var entry map[string]any
	if err := json.Unmarshal(buf.Bytes(), &entry); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if entry["type"] != "channel_context_op" {
		t.Fatalf("expected type=channel_context_op, got %v", entry["type"])
	}
	if entry["kind"] != "raw_window+digest" || entry["source"] != "claw-wall" || entry["status"] != "coverage_gap" {
		t.Fatalf("unexpected channel context fields: %+v", entry)
	}
	if entry["retained"].(float64) != 60 || entry["returned"].(float64) != 40 || entry["omitted"].(float64) != 20 {
		t.Fatalf("unexpected counts: %+v", entry)
	}
	channels := entry["channels"].([]any)
	if len(channels) != 2 || channels[0] != "chan-1" || channels[1] != "chan-2" {
		t.Fatalf("unexpected channels: %+v", channels)
	}
	if entry["raw_bytes"].(float64) != 2048 || entry["digest_bytes"].(float64) != 512 || entry["digest_blocks"].(float64) != 3 || entry["coverage_gaps"].(float64) != 1 {
		t.Fatalf("unexpected digest telemetry fields: %+v", entry)
	}
	if entry["deterministic_only"] != true {
		t.Fatalf("expected deterministic_only=true, got %+v", entry)
	}
}
