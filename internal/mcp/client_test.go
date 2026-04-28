package mcp

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestClientCallUsesStreamableHTTPJSON(t *testing.T) {
	var calls []string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		method, _ := req["method"].(string)
		calls = append(calls, method)
		if r.Header.Get("Accept") != "application/json, text/event-stream" {
			t.Fatalf("unexpected Accept header: %q", r.Header.Get("Accept"))
		}
		if method == "tools/call" {
			if r.Header.Get("Authorization") != "Bearer mcp-token" {
				t.Fatalf("expected bearer auth, got %q", r.Header.Get("Authorization"))
			}
			if r.Header.Get("MCP-Session-Id") != "sess-1" {
				t.Fatalf("expected MCP session id, got %q", r.Header.Get("MCP-Session-Id"))
			}
			if r.Header.Get("MCP-Method") != "tools/call" || r.Header.Get("MCP-Name") != "search" {
				t.Fatalf("missing MCP request mirror headers: %v", r.Header)
			}
		}
		w.Header().Set("Content-Type", "application/json")
		switch method {
		case "initialize":
			w.Header().Set("MCP-Session-Id", "sess-1")
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"test","version":"1"}}}`))
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/call":
			params, _ := req["params"].(map[string]any)
			if params["name"] != "search" {
				t.Fatalf("unexpected tool name: %+v", params)
			}
			args, _ := params["arguments"].(map[string]any)
			if args["query"] != "clawdapus" {
				t.Fatalf("unexpected arguments: %+v", args)
			}
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"found"}],"structuredContent":{"ok":true}}}`))
		default:
			t.Fatalf("unexpected method %q", method)
		}
	}))
	defer server.Close()

	client := NewClient(server.Client(), 16*1024)
	result, status, err := client.Call(context.Background(), Target{
		BaseURL: server.URL,
		Path:    "/mcp",
		Auth:    &Auth{Type: "bearer", Token: "mcp-token"},
	}, "search", map[string]any{"query": "clawdapus"})
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if status != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d", status)
	}
	if !strings.Contains(string(result), `"text":"found"`) {
		t.Fatalf("unexpected result: %s", result)
	}
	if strings.Join(calls, ",") != "initialize,notifications/initialized,tools/call" {
		t.Fatalf("unexpected call sequence: %v", calls)
	}
}

func TestClientCallParsesSSEResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		method, _ := req["method"].(string)
		switch method {
		case "initialize":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"test","version":"1"}}}`))
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/call":
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"from sse\"}]}}\n\n"))
		default:
			t.Fatalf("unexpected method %q", method)
		}
	}))
	defer server.Close()

	client := NewClient(server.Client(), 16*1024)
	result, _, err := client.Call(context.Background(), Target{BaseURL: server.URL, Path: "/mcp"}, "echo", nil)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if !strings.Contains(string(result), "from sse") {
		t.Fatalf("unexpected result: %s", result)
	}
}

func TestClientRetriesAfterExpiredSession(t *testing.T) {
	var initCount int
	var callCount int
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		method, _ := req["method"].(string)
		switch method {
		case "initialize":
			initCount++
			w.Header().Set("MCP-Session-Id", "sess-new")
			if initCount == 1 {
				w.Header().Set("MCP-Session-Id", "sess-old")
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"test","version":"1"}}}`))
		case "notifications/initialized":
			w.WriteHeader(http.StatusAccepted)
		case "tools/call":
			callCount++
			if callCount == 1 {
				if r.Header.Get("MCP-Session-Id") != "sess-old" {
					t.Fatalf("expected first call to use old session, got %q", r.Header.Get("MCP-Session-Id"))
				}
				w.WriteHeader(http.StatusNotFound)
				_, _ = w.Write([]byte(`{"error":"expired"}`))
				return
			}
			if r.Header.Get("MCP-Session-Id") != "sess-new" {
				t.Fatalf("expected retry to use new session, got %q", r.Header.Get("MCP-Session-Id"))
			}
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"retried"}]}}`))
		default:
			t.Fatalf("unexpected method %q", method)
		}
	}))
	defer server.Close()

	client := NewClient(server.Client(), 16*1024)
	result, _, err := client.Call(context.Background(), Target{BaseURL: server.URL, Path: "/mcp"}, "echo", nil)
	if err != nil {
		t.Fatalf("Call: %v", err)
	}
	if initCount != 2 || callCount != 2 {
		t.Fatalf("expected retry after expired session, init=%d call=%d", initCount, callCount)
	}
	if !strings.Contains(string(result), "retried") {
		t.Fatalf("unexpected result: %s", result)
	}
}
