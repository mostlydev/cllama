package proxy

import (
	"encoding/json"
	"io"
	"net/http"
	"testing"
)

func TestBuildManagedToolRequestPreservesFlatJSONBodyByDefault(t *testing.T) {
	targetURL, body, err := buildManagedToolRequest("http://trading-api:4000", http.MethodPost, "/api/v1/trades", "", map[string]any{
		"ticker":        "NVDA",
		"qty_requested": float64(50),
	})
	if err != nil {
		t.Fatalf("buildManagedToolRequest: %v", err)
	}
	if targetURL != "http://trading-api:4000/api/v1/trades" {
		t.Fatalf("unexpected target URL: %q", targetURL)
	}

	raw, err := io.ReadAll(body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if payload["ticker"] != "NVDA" || payload["qty_requested"] != float64(50) {
		t.Fatalf("unexpected flat payload: %#v", payload)
	}
}

func TestBuildManagedToolRequestWrapsJSONBodyWhenBodyKeyIsSet(t *testing.T) {
	targetURL, body, err := buildManagedToolRequest("http://trading-api:4000", http.MethodPost, "/api/v1/trades", "trade", map[string]any{
		"ticker":        "NVDA",
		"qty_requested": float64(50),
	})
	if err != nil {
		t.Fatalf("buildManagedToolRequest: %v", err)
	}
	if targetURL != "http://trading-api:4000/api/v1/trades" {
		t.Fatalf("unexpected target URL: %q", targetURL)
	}

	raw, err := io.ReadAll(body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}
	var payload map[string]map[string]any
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	trade := payload["trade"]
	if trade["ticker"] != "NVDA" || trade["qty_requested"] != float64(50) {
		t.Fatalf("unexpected wrapped payload: %#v", payload)
	}
}
