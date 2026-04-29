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
