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
	l.LogResponseWithCost("tiverton", "anthropic/claude-sonnet-4", 200, 1250,
		&CostInfo{InputTokens: 100, OutputTokens: 50, CostUSD: 0.0105})

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
