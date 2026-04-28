package proxy

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/mostlydev/cllama/internal/agentctx"
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

func TestDispatchManagedToolExecutesMCPTransport(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		method, _ := req["method"].(string)
		w.Header().Set("Content-Type", "application/json")
		switch method {
		case "initialize":
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"test","version":"1"}}}`))
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/call":
			params, _ := req["params"].(map[string]any)
			if params["name"] != "echo" {
				t.Fatalf("unexpected MCP tool call params: %+v", params)
			}
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"echoed"}],"structuredContent":{"ok":true}}}`))
		default:
			t.Fatalf("unexpected method %q", method)
		}
	}))
	defer server.Close()

	h := NewHandler(nil, nil, nil)
	raw, status, err := h.dispatchManagedTool(context.Background(), "agent", agentctx.ToolManifestEntry{
		Name: "mcp-svc.echo",
		Execution: agentctx.ToolExecution{
			Transport: "mcp",
			Service:   "mcp-svc",
			BaseURL:   server.URL,
			Path:      "/mcp",
			ToolName:  "echo",
		},
	}, map[string]any{"text": "hello"})
	if err != nil {
		t.Fatalf("dispatchManagedTool: %v", err)
	}
	if status != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d", status)
	}
	if !strings.Contains(string(raw), `"ok":true`) || !strings.Contains(string(raw), "echoed") {
		t.Fatalf("unexpected managed MCP payload: %s", raw)
	}
}
