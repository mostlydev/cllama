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
	"github.com/mostlydev/cllama/internal/sessionhistory"
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
	firstResult := []byte(`{"ok":true,"data":{"result":"fresh"},"status_code":200}`)
	tracker.StoreOpenAIResult(agentCtx, first, 1, managedToolOutcome{RawJSON: firstResult})

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
	if duplicate.Streak != 1 || !bytes.Equal(duplicate.CachedResult, firstResult) {
		t.Fatalf("expected cached result and first duplicate streak, got %+v", duplicate)
	}

	third := openAIToolCall{
		Name:         "trading-api.get_market_context",
		Arguments:    map[string]any{"ticker": "MSFT", "window": "1d"},
		ArgumentsRaw: json.RawMessage(`{"ticker":"MSFT","window":"1d"}`),
	}
	if duplicate := tracker.ObserveOpenAI(agentCtx, third, 3); duplicate != nil {
		t.Fatalf("different args should reset without duplicate: %+v", duplicate)
	}
	if duplicate := tracker.ObserveOpenAI(agentCtx, first, 4); duplicate == nil || duplicate.Streak != 1 || !bytes.Equal(duplicate.CachedResult, firstResult) {
		t.Fatalf("expected duplicate streak to restart after different args, got %+v", duplicate)
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
	anthropicTracker.StoreAnthropicResult(agentCtx, use, 1, managedToolOutcome{RawJSON: firstResult})
	if duplicate := anthropicTracker.ObserveAnthropic(agentCtx, use, 2); duplicate == nil || duplicate.FirstRound != 1 || duplicate.Streak != 1 || !bytes.Equal(duplicate.CachedResult, firstResult) {
		t.Fatalf("expected anthropic duplicate metadata, got %+v", duplicate)
	}
}

func TestManagedToolDuplicateTrackerInvalidatesReadsAfterMutation(t *testing.T) {
	agentCtx := &agentctx.AgentContext{Tools: &agentctx.ToolManifest{Tools: []agentctx.ToolManifestEntry{
		{
			Name:      "trading-api.get_trigger",
			Execution: agentctx.ToolExecution{Service: "trading-api", Method: http.MethodGet},
		},
		{
			Name:      "trading-api.update_trigger",
			Execution: agentctx.ToolExecution{Service: "trading-api", Method: http.MethodPatch},
		},
	}}}
	read := openAIToolCall{
		Name:         "trading-api.get_trigger",
		Arguments:    map[string]any{"trigger_id": float64(732)},
		ArgumentsRaw: json.RawMessage(`{"trigger_id":732}`),
	}
	write := openAIToolCall{
		Name:         "trading-api.update_trigger",
		Arguments:    map[string]any{"trigger_id": float64(732), "label": "repaired"},
		ArgumentsRaw: json.RawMessage(`{"trigger_id":732,"label":"repaired"}`),
	}
	tracker := newManagedToolDuplicateTracker()

	if duplicate := tracker.ObserveOpenAI(agentCtx, read, 1); duplicate != nil {
		t.Fatalf("first read should execute: %+v", duplicate)
	}
	oldResult := []byte(`{"id":732,"label":"original"}`)
	tracker.StoreOpenAIResult(agentCtx, read, 1, managedToolOutcome{
		RawJSON: oldResult,
		Trace:   sessionhistory.ToolCallTrace{Status: "ok", StatusCode: http.StatusOK},
	})
	if duplicate := tracker.ObserveOpenAI(agentCtx, write, 2); duplicate != nil {
		t.Fatalf("first mutation should execute: %+v", duplicate)
	}
	tracker.StoreOpenAIResult(agentCtx, write, 2, managedToolOutcome{
		RawJSON: []byte(`{"id":732,"label":"repaired"}`),
		Trace:   sessionhistory.ToolCallTrace{Status: "ok", StatusCode: http.StatusOK},
	})

	if duplicate := tracker.ObserveOpenAI(agentCtx, read, 3); duplicate != nil {
		t.Fatalf("read after mutation must execute instead of replaying stale state: %+v", duplicate)
	}
	newResult := []byte(`{"id":732,"label":"repaired"}`)
	tracker.StoreOpenAIResult(agentCtx, read, 3, managedToolOutcome{
		RawJSON: newResult,
		Trace:   sessionhistory.ToolCallTrace{Status: "ok", StatusCode: http.StatusOK},
	})
	duplicate := tracker.ObserveOpenAI(agentCtx, read, 4)
	if duplicate == nil || !bytes.Equal(duplicate.CachedResult, newResult) {
		t.Fatalf("subsequent duplicate should replay the post-mutation read: %+v", duplicate)
	}
}

func TestManagedToolDuplicateTrackerKeepsReadsAfterRejectedMutation(t *testing.T) {
	agentCtx := &agentctx.AgentContext{Tools: &agentctx.ToolManifest{Tools: []agentctx.ToolManifestEntry{
		{Name: "trading-api.get_trigger", Execution: agentctx.ToolExecution{Service: "trading-api", Method: http.MethodGet}},
		{Name: "trading-api.update_trigger", Execution: agentctx.ToolExecution{Service: "trading-api", Method: http.MethodPatch}},
	}}}
	read := openAIToolCall{Name: "trading-api.get_trigger", Arguments: map[string]any{"trigger_id": float64(732)}, ArgumentsRaw: json.RawMessage(`{"trigger_id":732}`)}
	write := openAIToolCall{Name: "trading-api.update_trigger", Arguments: map[string]any{"trigger_id": float64(732)}, ArgumentsRaw: json.RawMessage(`{"trigger_id":732}`)}
	tracker := newManagedToolDuplicateTracker()
	_ = tracker.ObserveOpenAI(agentCtx, read, 1)
	readResult := []byte(`{"id":732,"label":"original"}`)
	tracker.StoreOpenAIResult(agentCtx, read, 1, managedToolOutcome{RawJSON: readResult, Trace: sessionhistory.ToolCallTrace{Status: "ok", StatusCode: http.StatusOK}})
	_ = tracker.ObserveOpenAI(agentCtx, write, 2)
	rejected := toolErrorPayload("validation_error", "rejected", http.StatusUnprocessableEntity, nil)
	tracker.StoreOpenAIResult(agentCtx, write, 2, managedToolOutcome{RawJSON: rejected, Trace: sessionhistory.ToolCallTrace{Result: rejected, Status: "error", StatusCode: http.StatusUnprocessableEntity}})

	duplicate := tracker.ObserveOpenAI(agentCtx, read, 3)
	if duplicate == nil || !bytes.Equal(duplicate.CachedResult, readResult) {
		t.Fatalf("a cleanly rejected mutation should not invalidate prior reads: %+v", duplicate)
	}

	ambiguousTracker := newManagedToolDuplicateTracker()
	_ = ambiguousTracker.ObserveOpenAI(agentCtx, read, 1)
	ambiguousTracker.StoreOpenAIResult(agentCtx, read, 1, managedToolOutcome{RawJSON: readResult, Trace: sessionhistory.ToolCallTrace{Status: "ok", StatusCode: http.StatusOK}})
	_ = ambiguousTracker.ObserveOpenAI(agentCtx, write, 2)
	timeout := toolErrorPayload("timeout", "outcome unknown", http.StatusGatewayTimeout, nil)
	ambiguousTracker.StoreOpenAIResult(agentCtx, write, 2, managedToolOutcome{RawJSON: timeout, Trace: sessionhistory.ToolCallTrace{Result: timeout, Status: "error", StatusCode: http.StatusGatewayTimeout}})
	if duplicate := ambiguousTracker.ObserveOpenAI(agentCtx, read, 3); duplicate != nil {
		t.Fatalf("an ambiguous mutation may have committed, so the next read must execute: %+v", duplicate)
	}
}

func TestManagedToolDuplicateTrackerRetriesFailuresBeforeSuppressing(t *testing.T) {
	agentCtx := &agentctx.AgentContext{Tools: managedToolManifest()}
	tracker := newManagedToolDuplicateTracker()
	call := openAIToolCall{
		Name:         "trading-api.get_market_context",
		Arguments:    map[string]any{"ticker": "NVDA"},
		ArgumentsRaw: json.RawMessage(`{"ticker":"NVDA"}`),
	}
	failure := func(code string, statusCode int) managedToolOutcome {
		raw := toolErrorPayload(code, "failed", statusCode, nil)
		return managedToolOutcome{RawJSON: raw, Trace: sessionhistory.ToolCallTrace{
			Name: "trading-api.get_market_context", Result: raw, Status: "error", StatusCode: statusCode,
		}}
	}

	if duplicate := tracker.ObserveOpenAI(agentCtx, call, 1); duplicate != nil {
		t.Fatalf("first call should execute: %+v", duplicate)
	}
	tracker.StoreOpenAIResult(agentCtx, call, 1, failure("http_422", http.StatusUnprocessableEntity))

	second := tracker.ObserveOpenAI(agentCtx, call, 2)
	if second == nil || !second.Failed || second.FailureCount != 1 || second.FailureClass != "422:http_422" {
		t.Fatalf("second failed call should be eligible to re-execute: %+v", second)
	}
	if shouldSuppressManagedDuplicate(second, managedDuplicatePolicyReplay) {
		t.Fatal("second failed call must execute instead of replaying stale failure")
	}
	tracker.StoreOpenAIResult(agentCtx, call, 2, failure("http_422", http.StatusUnprocessableEntity))

	third := tracker.ObserveOpenAI(agentCtx, call, 3)
	if third == nil || !third.Failed || third.FailureCount != 2 {
		t.Fatalf("third same-class failure should be suppressible: %+v", third)
	}
	if !shouldSuppressManagedDuplicate(third, managedDuplicatePolicyReplay) {
		t.Fatal("third same-class failed retry should be suppressed")
	}
	outcome := duplicateManagedToolOutcome(third, managedDuplicatePolicyReplay)
	if outcome.Trace.StatusCode != http.StatusConflict || !bytes.Contains(outcome.RawJSON, []byte(`"code":"repeated_failed_tool_call"`)) {
		t.Fatalf("expected compact failed-retry advisory, got trace=%+v raw=%s", outcome.Trace, outcome.RawJSON)
	}

	classChange := newManagedToolDuplicateTracker()
	_ = classChange.ObserveOpenAI(agentCtx, call, 1)
	classChange.StoreOpenAIResult(agentCtx, call, 1, failure("http_422", http.StatusUnprocessableEntity))
	_ = classChange.ObserveOpenAI(agentCtx, call, 2)
	classChange.StoreOpenAIResult(agentCtx, call, 2, failure("http_409", http.StatusConflict))
	if duplicate := classChange.ObserveOpenAI(agentCtx, call, 3); duplicate == nil || duplicate.FailureCount != 1 || duplicate.FailureClass != "409:http_409" {
		t.Fatalf("different error class should restart the failure count: %+v", duplicate)
	}

	recovery := newManagedToolDuplicateTracker()
	_ = recovery.ObserveOpenAI(agentCtx, call, 1)
	recovery.StoreOpenAIResult(agentCtx, call, 1, failure("http_422", http.StatusUnprocessableEntity))
	_ = recovery.ObserveOpenAI(agentCtx, call, 2)
	successRaw := []byte(`{"ok":true,"status_code":200}`)
	recovery.StoreOpenAIResult(agentCtx, call, 2, managedToolOutcome{RawJSON: successRaw, Trace: sessionhistory.ToolCallTrace{
		Name: "trading-api.get_market_context", Result: successRaw, Status: "ok", StatusCode: http.StatusOK,
	}})
	if duplicate := recovery.ObserveOpenAI(agentCtx, call, 3); duplicate == nil || duplicate.Failed || duplicate.FailureCount != 0 || !bytes.Equal(duplicate.CachedResult, successRaw) {
		t.Fatalf("successful retry should replace the failed state: %+v", duplicate)
	}
}

func TestManagedToolDuplicateTrackerDoesNotBlindlyRetryAmbiguousMutation(t *testing.T) {
	if managedToolFailureMayHaveMutated(http.MethodPost, http.StatusUnprocessableEntity) {
		t.Fatal("clean 422 rejection must remain eligible for a corrective retry")
	}
	if !managedToolFailureMayHaveMutated(http.MethodPost, http.StatusGatewayTimeout) {
		t.Fatal("mutating timeout must be treated as possibly committed")
	}
	if managedToolFailureMayHaveMutated(http.MethodGet, http.StatusGatewayTimeout) {
		t.Fatal("read-only timeout cannot create a duplicate mutation")
	}

	duplicate := &managedToolDuplicate{
		CanonicalName:     "trading-api.create_trigger",
		Failed:            true,
		FailureClass:      "504:timeout",
		FailureCount:      1,
		AmbiguousMutation: true,
		HasCachedResult:   true,
		CachedResult:      toolErrorPayload("timeout", "timed out", http.StatusGatewayTimeout, nil),
		Status:            "error",
		StatusCode:        http.StatusGatewayTimeout,
	}
	if !shouldSuppressManagedDuplicate(duplicate, managedDuplicatePolicyReplay) {
		t.Fatal("ambiguous mutating failure should be suppressed before blind retry")
	}
	outcome := duplicateManagedToolOutcome(duplicate, managedDuplicatePolicyReplay)
	if !bytes.Contains(outcome.RawJSON, []byte(`"verify_state":true`)) || !bytes.Contains(outcome.RawJSON, []byte("may have completed")) {
		t.Fatalf("expected read-back guidance, got %s", outcome.RawJSON)
	}
}
