package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/logging"
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

func TestCallManagedHTTPToolForClawWallRequiresAllowedChannel(t *testing.T) {
	var hit bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		if got := r.Header.Get("X-Claw-ID"); got != "weston" {
			t.Fatalf("expected X-Claw-ID header, got %q", got)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer wall-token" {
			t.Fatalf("expected bearer auth, got %q", got)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer server.Close()

	h := NewHandler(nil, func(agentID string) (*agentctx.AgentContext, error) {
		if agentID != "weston" {
			t.Fatalf("unexpected agent id %q", agentID)
		}
		return &agentctx.AgentContext{
			ChannelAllowlist: map[string]struct{}{"chan-1": {}},
		}, nil
	}, nil)

	raw, status, err := h.callManagedHTTPTool(context.Background(), "weston", agentctx.ToolManifestEntry{
		Name: "claw-wall.search_channel_context",
		Execution: agentctx.ToolExecution{
			Transport: "http",
			Service:   "claw-wall",
			BaseURL:   server.URL,
			Method:    http.MethodPost,
			Path:      "/search_channel_context",
			Auth:      &agentctx.AuthEntry{Type: "bearer", Token: "wall-token"},
		},
	}, map[string]any{"channels": []any{"chan-1"}, "query": "alpha"})
	if err != nil {
		t.Fatalf("callManagedHTTPTool: %v", err)
	}
	if status != http.StatusOK || !strings.Contains(string(raw), `"ok":true`) {
		t.Fatalf("unexpected result status=%d raw=%s", status, raw)
	}
	if !hit {
		t.Fatal("expected claw-wall request to be forwarded")
	}
}

func TestCallManagedHTTPToolForClawWallRejectsDisallowedChannel(t *testing.T) {
	var hit bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	h := NewHandler(nil, func(agentID string) (*agentctx.AgentContext, error) {
		return &agentctx.AgentContext{
			ChannelAllowlist: map[string]struct{}{"chan-1": {}},
		}, nil
	}, nil)

	raw, status, err := h.callManagedHTTPTool(context.Background(), "weston", agentctx.ToolManifestEntry{
		Name: "claw-wall.get_channel_messages",
		Execution: agentctx.ToolExecution{
			Transport: "http",
			Service:   "claw-wall",
			BaseURL:   server.URL,
			Method:    http.MethodPost,
			Path:      "/get_channel_messages",
		},
	}, map[string]any{"channels": []any{"chan-2"}, "message_ids": []any{"200"}})
	if err != nil {
		t.Fatalf("callManagedHTTPTool: %v", err)
	}
	if status != http.StatusForbidden || !strings.Contains(string(raw), "channel_not_allowed") {
		t.Fatalf("expected channel_not_allowed 403, got status=%d raw=%s", status, raw)
	}
	if hit {
		t.Fatal("disallowed claw-wall request should not be forwarded")
	}
}

func TestExecuteManagedToolForClawWallPrependsHeaderAndRecordsStatus(t *testing.T) {
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"messages":[],"retained_coverage":{"buffer_size":500},"status":"not_in_buffer","hint":"evicted"}`))
	}))
	defer toolSrv.Close()

	var logs bytes.Buffer
	h := NewHandler(nil, func(agentID string) (*agentctx.AgentContext, error) {
		return &agentctx.AgentContext{
			ChannelAllowlist: map[string]struct{}{"chan-1": {}},
		}, nil
	}, logging.New(&logs))
	agentCtx := &agentctx.AgentContext{
		Tools: &agentctx.ToolManifest{
			Tools: []agentctx.ToolManifestEntry{{
				Name: "claw-wall.search_channel_context",
				Execution: agentctx.ToolExecution{
					Transport: "http",
					Service:   "claw-wall",
					BaseURL:   toolSrv.URL,
					Method:    http.MethodPost,
					Path:      "/search_channel_context",
				},
			}},
		},
	}
	args := map[string]any{"channels": []any{"chan-1"}, "query": "cmcsa"}
	argsRaw := json.RawMessage(`{"channels":["chan-1"],"query":"cmcsa"}`)

	outcome, err := h.executeManagedOpenAITool(context.Background(), "weston", "vercel/deepseek/deepseek-v4-pro", agentCtx, openAIToolCall{
		Name:         "claw-wall_search_channel_context",
		Arguments:    args,
		ArgumentsRaw: argsRaw,
	}, managedToolPolicy{PerToolTimeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("executeManagedOpenAITool: %v", err)
	}
	if !strings.HasPrefix(string(outcome.RawJSON), "[channel-tool] kind=tool_call name=claw-wall_search_channel_context status=not_in_buffer") {
		t.Fatalf("expected channel-tool header, got %s", outcome.RawJSON)
	}
	if outcome.Trace.Status != "not_in_buffer" {
		t.Fatalf("expected trace status not_in_buffer, got %+v", outcome.Trace)
	}
	if !json.Valid(outcome.Trace.Result) {
		t.Fatalf("trace result must remain valid JSON, got %s", outcome.Trace.Result)
	}
	entries := parseLogEntries(t, logs.Bytes())
	var channelEntry map[string]any
	for _, entry := range entries {
		if entry["type"] == "intervention" {
			t.Fatalf("hash-free presented name must not log an intervention, got %+v", entry)
		}
		if entry["type"] == "channel_context_op" {
			channelEntry = entry
		}
	}
	if channelEntry == nil || channelEntry["status"] != "not_in_buffer" || channelEntry["tool_name"] != "search_channel_context" {
		t.Fatalf("unexpected channel_context_op log: %+v", entries)
	}
}

func TestExecuteManagedAnthropicToolAcceptsHashlessAlias(t *testing.T) {
	var hit bool
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"messages":[],"retained_coverage":{"buffer_size":500},"status":"ok"}`))
	}))
	defer toolSrv.Close()

	var logs bytes.Buffer
	h := NewHandler(nil, func(agentID string) (*agentctx.AgentContext, error) {
		return &agentctx.AgentContext{
			ChannelAllowlist: map[string]struct{}{"chan-1": {}},
		}, nil
	}, logging.New(&logs))
	agentCtx := &agentctx.AgentContext{
		Tools: &agentctx.ToolManifest{
			Tools: []agentctx.ToolManifestEntry{{
				Name: "claw-wall.get_channel_messages",
				Execution: agentctx.ToolExecution{
					Transport: "http",
					Service:   "claw-wall",
					BaseURL:   toolSrv.URL,
					Method:    http.MethodPost,
					Path:      "/get_channel_messages",
				},
			}},
		},
	}
	args := map[string]any{"channels": []any{"chan-1"}, "author": "Wojtek"}
	argsRaw := json.RawMessage(`{"channels":["chan-1"],"author":"Wojtek"}`)

	outcome, err := h.executeManagedAnthropicTool(context.Background(), "weston", "anthropic/claude-sonnet-4", agentCtx, anthropicToolUse{
		Name:         "claw-wall_get_channel_messages",
		Arguments:    args,
		ArgumentsRaw: argsRaw,
	}, managedToolPolicy{PerToolTimeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("executeManagedAnthropicTool: %v", err)
	}
	if !hit {
		t.Fatal("expected hashless anthropic alias to execute managed HTTP tool")
	}
	if outcome.Trace.Name != "claw-wall.get_channel_messages" || outcome.Trace.Status != "ok" {
		t.Fatalf("unexpected outcome trace: %+v", outcome.Trace)
	}
	for _, entry := range parseLogEntries(t, logs.Bytes()) {
		if entry["type"] == "intervention" {
			t.Fatalf("hash-free presented name must not log an intervention, got %+v", entry)
		}
	}
	if !strings.HasPrefix(string(outcome.RawJSON), "[channel-tool] kind=tool_call name=claw-wall_get_channel_messages status=ok") {
		t.Fatalf("expected channel-tool header, got %s", outcome.RawJSON)
	}
}

func TestManagedToolDuplicateTrackerCanonicalizesArguments(t *testing.T) {
	agentCtx := &agentctx.AgentContext{Tools: managedToolManifest()}
	tracker := newManagedToolDuplicateTracker()

	first := openAIToolCall{
		Name:         "trading-api.get_market_context",
		Arguments:    map[string]any{"ticker": "NVDA", "window": "1d"},
		ArgumentsRaw: json.RawMessage(`{"ticker":"NVDA","window":"1d"}`),
	}
	if duplicate := tracker.ObserveOpenAI(agentCtx, first, 1); duplicate != nil {
		t.Fatalf("first call should not be duplicate: %+v", duplicate)
	}

	second := openAIToolCall{
		Name:         managedToolHashedNameForCanonical("trading-api.get_market_context"),
		Arguments:    map[string]any{"window": "1d", "ticker": "NVDA"},
		ArgumentsRaw: json.RawMessage(`{"window":"1d","ticker":"NVDA"}`),
	}
	duplicate := tracker.ObserveOpenAI(agentCtx, second, 2)
	if duplicate == nil {
		t.Fatal("expected duplicate for same canonical name and arguments")
	}
	if duplicate.CanonicalName != "trading-api.get_market_context" || duplicate.FirstRound != 1 || duplicate.Count != 2 {
		t.Fatalf("unexpected duplicate metadata: %+v", duplicate)
	}

	anthropicTracker := newManagedToolDuplicateTracker()
	use := anthropicToolUse{
		Name:         "trading-api.get_market_context",
		Arguments:    map[string]any{"ticker": "NVDA"},
		ArgumentsRaw: json.RawMessage(`{"ticker":"NVDA"}`),
	}
	if duplicate := anthropicTracker.ObserveAnthropic(agentCtx, use, 1); duplicate != nil {
		t.Fatalf("first anthropic call should not be duplicate: %+v", duplicate)
	}
	if duplicate := anthropicTracker.ObserveAnthropic(agentCtx, use, 2); duplicate == nil || duplicate.FirstRound != 1 {
		t.Fatalf("expected anthropic duplicate metadata, got %+v", duplicate)
	}
}
