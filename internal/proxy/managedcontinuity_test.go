package proxy

import (
	"encoding/json"
	"testing"
)

func TestManagedOpenAIContinuityStoreMatchesDuplicateRepliesByTailOffset(t *testing.T) {
	store := newManagedOpenAIContinuityStore(8)

	if !store.ObserveTerminalAssistant("tiverton", map[string]any{
		"role":    "assistant",
		"content": "Done.",
	}, continuityHiddenMessages(
		map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"id": "call_1"}}},
		map[string]any{"role": "tool", "tool_call_id": "call_1", "content": `{"ok":true,"tag":"first"}`},
	)) {
		t.Fatal("expected first continuity turn to be recorded")
	}

	if store.ObserveTerminalAssistant("tiverton", map[string]any{
		"role":    "assistant",
		"content": "Still working.",
	}, nil) {
		t.Fatal("did not expect non-hidden turn to create a continuity entry")
	}

	if !store.ObserveTerminalAssistant("tiverton", map[string]any{
		"role":    "assistant",
		"content": "Done.",
	}, continuityHiddenMessages(
		map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"id": "call_2"}}},
		map[string]any{"role": "tool", "tool_call_id": "call_2", "content": `{"ok":true,"tag":"second"}`},
	)) {
		t.Fatal("expected second continuity turn to be recorded")
	}

	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "turn one"},
			map[string]any{"role": "assistant", "content": "Done."},
			map[string]any{"role": "user", "content": "turn two"},
			map[string]any{"role": "assistant", "content": "Still working."},
			map[string]any{"role": "user", "content": "turn three"},
			map[string]any{"role": "assistant", "content": "Done."},
			map[string]any{"role": "user", "content": "what next?"},
		},
	}

	if !store.Inject("tiverton", payload) {
		t.Fatal("expected continuity injection")
	}

	messages, _ := payload["messages"].([]any)
	if len(messages) != 11 {
		t.Fatalf("expected 11 messages after continuity injection, got %d", len(messages))
	}
	assertJSONToolTag(t, messages[2], "first")
	assertJSONToolTag(t, messages[8], "second")
}

func TestManagedOpenAIContinuityStoreEvictsLeastRecentlyUsedAgents(t *testing.T) {
	store := newManagedOpenAIContinuityStore(8)
	store.maxAgents = 2

	store.ObserveTerminalAssistant("agent-a", map[string]any{
		"role":    "assistant",
		"content": "A",
	}, continuityHiddenMessages(map[string]any{"role": "tool", "content": `{"ok":true,"agent":"a"}`}))
	store.ObserveTerminalAssistant("agent-b", map[string]any{
		"role":    "assistant",
		"content": "B",
	}, continuityHiddenMessages(map[string]any{"role": "tool", "content": `{"ok":true,"agent":"b"}`}))

	payloadA := map[string]any{
		"messages": []any{map[string]any{"role": "assistant", "content": "A"}},
	}
	if !store.Inject("agent-a", payloadA) {
		t.Fatal("expected agent-a continuity injection")
	}

	store.ObserveTerminalAssistant("agent-c", map[string]any{
		"role":    "assistant",
		"content": "C",
	}, continuityHiddenMessages(map[string]any{"role": "tool", "content": `{"ok":true,"agent":"c"}`}))

	if len(store.agents) != 2 {
		t.Fatalf("expected maxAgents cap to hold at 2, got %d", len(store.agents))
	}
	if _, ok := store.agents["agent-b"]; ok {
		t.Fatalf("expected least recently used agent-b to be evicted, store=%+v", store.agents)
	}
	if _, ok := store.agents["agent-a"]; !ok {
		t.Fatalf("expected recently touched agent-a to be retained, store=%+v", store.agents)
	}
	if _, ok := store.agents["agent-c"]; !ok {
		t.Fatalf("expected new agent-c to be retained, store=%+v", store.agents)
	}
}

func TestManagedOpenAIContinuityStoreInjectsPendingNativeToolCallOnce(t *testing.T) {
	store := newManagedOpenAIContinuityStore(8)

	if !store.ObserveNativeToolCallAssistant("tiverton", map[string]any{
		"role": "assistant",
		"tool_calls": []any{
			map[string]any{
				"id":   "call_native_1",
				"type": "function",
				"function": map[string]any{
					"name":      "runner_local",
					"arguments": "{}",
				},
			},
		},
	}, continuityHiddenMessages(
		map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"id": "call_managed_1"}}},
		map[string]any{"role": "tool", "tool_call_id": "call_managed_1", "content": `{"ok":true,"tag":"managed"}`},
	)) {
		t.Fatal("expected pending native handoff to be recorded")
	}

	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "tool_calls": []any{
				map[string]any{
					"id":   "call_native_1",
					"type": "function",
					"function": map[string]any{
						"name":      "runner_local",
						"arguments": "{}",
					},
				},
			}},
			map[string]any{"role": "tool", "tool_call_id": "call_native_1", "content": `{"native":true}`},
		},
	}

	if !store.Inject("tiverton", payload) {
		t.Fatal("expected pending native handoff injection")
	}

	messages, _ := payload["messages"].([]any)
	if len(messages) != 5 {
		t.Fatalf("expected 5 messages after pending native handoff injection, got %d", len(messages))
	}
	assertJSONToolTag(t, messages[2], "managed")
	if state := store.agents["tiverton"]; state != nil && state.PendingToolCallTurn != nil {
		t.Fatalf("expected pending native handoff to clear after injection, state=%+v", state)
	}

	secondPayload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "tool_calls": []any{
				map[string]any{
					"id":   "call_native_1",
					"type": "function",
					"function": map[string]any{
						"name":      "runner_local",
						"arguments": "{}",
					},
				},
			}},
			map[string]any{"role": "tool", "tool_call_id": "call_native_1", "content": `{"native":true}`},
		},
	}
	if store.Inject("tiverton", secondPayload) {
		t.Fatal("did not expect pending native handoff to inject twice")
	}
}

func TestManagedAnthropicContinuityStoreInjectsPendingNativeToolUseOnce(t *testing.T) {
	store := newManagedAnthropicContinuityStore(8)

	if !store.ObserveNativeToolUseAssistant("nano-bot", map[string]any{
		"role": "assistant",
		"content": []any{
			map[string]any{"type": "tool_use", "id": "toolu_native_1", "name": "runner_local", "input": map[string]any{}},
		},
	}, continuityHiddenMessages(
		map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "tool_use", "id": "toolu_managed_1", "name": "svc.tool", "input": map[string]any{}}}},
		map[string]any{"role": "user", "content": []any{map[string]any{"type": "tool_result", "tool_use_id": "toolu_managed_1", "content": `{"ok":true}`}}},
	)) {
		t.Fatal("expected pending native anthropic handoff to be recorded")
	}

	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "tool_use", "id": "toolu_native_1", "name": "runner_local", "input": map[string]any{}},
			}},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "tool_result", "tool_use_id": "toolu_native_1", "content": `{"native":true}`},
			}},
		},
	}

	if !store.Inject("nano-bot", payload) {
		t.Fatal("expected pending native anthropic handoff injection")
	}

	messages, _ := payload["messages"].([]any)
	if len(messages) != 5 {
		t.Fatalf("expected 5 messages after pending native anthropic handoff injection, got %d", len(messages))
	}
	if hiddenAssistant := messages[1].(map[string]any); hiddenAssistant["role"] != "assistant" {
		t.Fatalf("expected hidden assistant injected before native handoff, got %+v", hiddenAssistant)
	}
	if hiddenUser := messages[2].(map[string]any); hiddenUser["role"] != "user" {
		t.Fatalf("expected hidden user tool_result injected before native handoff, got %+v", hiddenUser)
	}
	if state := store.agents["nano-bot"]; state != nil && state.PendingToolUseTurn != nil {
		t.Fatalf("expected pending native anthropic handoff to clear after injection, state=%+v", state)
	}

	secondPayload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "tool_use", "id": "toolu_native_1", "name": "runner_local", "input": map[string]any{}},
			}},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "tool_result", "tool_use_id": "toolu_native_1", "content": `{"native":true}`},
			}},
		},
	}
	if store.Inject("nano-bot", secondPayload) {
		t.Fatal("did not expect pending native anthropic handoff to inject twice")
	}
}

func TestManagedOpenAIContinuityStorePreservesPendingNativeToolCallOnAnchorMiss(t *testing.T) {
	store := newManagedOpenAIContinuityStore(8)

	if !store.ObserveNativeToolCallAssistant("tiverton", map[string]any{
		"role": "assistant",
		"tool_calls": []any{
			map[string]any{
				"id":   "call_native_1",
				"type": "function",
				"function": map[string]any{
					"name":      "runner_local",
					"arguments": "{}",
				},
			},
		},
	}, continuityHiddenMessages(
		map[string]any{"role": "assistant", "tool_calls": []any{map[string]any{"id": "call_managed_1"}}},
		map[string]any{"role": "tool", "tool_call_id": "call_managed_1", "content": `{"ok":true,"tag":"managed"}`},
	)) {
		t.Fatal("expected pending native handoff to be recorded")
	}

	missPayload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "tool_calls": []any{
				map[string]any{
					"id":   "call_native_2",
					"type": "function",
					"function": map[string]any{
						"name":      "runner_local",
						"arguments": "{}",
					},
				},
			}},
			map[string]any{"role": "tool", "tool_call_id": "call_native_2", "content": `{"native":true}`},
		},
	}

	if store.Inject("tiverton", missPayload) {
		t.Fatal("did not expect anchor-miss payload to inject pending native handoff")
	}
	if state := store.agents["tiverton"]; state == nil || state.PendingToolCallTurn == nil {
		t.Fatalf("expected pending native handoff to remain after anchor miss, state=%+v", state)
	}

	matchPayload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "tool_calls": []any{
				map[string]any{
					"id":   "call_native_1",
					"type": "function",
					"function": map[string]any{
						"name":      "runner_local",
						"arguments": "{}",
					},
				},
			}},
			map[string]any{"role": "tool", "tool_call_id": "call_native_1", "content": `{"native":true}`},
		},
	}

	if !store.Inject("tiverton", matchPayload) {
		t.Fatal("expected pending native handoff to inject after later anchor match")
	}
}

func TestManagedAnthropicContinuityStorePreservesPendingNativeToolUseOnAnchorMiss(t *testing.T) {
	store := newManagedAnthropicContinuityStore(8)

	if !store.ObserveNativeToolUseAssistant("nano-bot", map[string]any{
		"role": "assistant",
		"content": []any{
			map[string]any{"type": "tool_use", "id": "toolu_native_1", "name": "runner_local", "input": map[string]any{}},
		},
	}, continuityHiddenMessages(
		map[string]any{"role": "assistant", "content": []any{map[string]any{"type": "tool_use", "id": "toolu_managed_1", "name": "svc.tool", "input": map[string]any{}}}},
		map[string]any{"role": "user", "content": []any{map[string]any{"type": "tool_result", "tool_use_id": "toolu_managed_1", "content": `{"ok":true}`}}},
	)) {
		t.Fatal("expected pending native anthropic handoff to be recorded")
	}

	missPayload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "tool_use", "id": "toolu_native_2", "name": "runner_local", "input": map[string]any{}},
			}},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "tool_result", "tool_use_id": "toolu_native_2", "content": `{"native":true}`},
			}},
		},
	}

	if store.Inject("nano-bot", missPayload) {
		t.Fatal("did not expect anchor-miss anthropic payload to inject pending native handoff")
	}
	if state := store.agents["nano-bot"]; state == nil || state.PendingToolUseTurn == nil {
		t.Fatalf("expected pending native anthropic handoff to remain after anchor miss, state=%+v", state)
	}

	matchPayload := map[string]any{
		"messages": []any{
			map[string]any{"role": "user", "content": "hi"},
			map[string]any{"role": "assistant", "content": []any{
				map[string]any{"type": "tool_use", "id": "toolu_native_1", "name": "runner_local", "input": map[string]any{}},
			}},
			map[string]any{"role": "user", "content": []any{
				map[string]any{"type": "tool_result", "tool_use_id": "toolu_native_1", "content": `{"native":true}`},
			}},
		},
	}

	if !store.Inject("nano-bot", matchPayload) {
		t.Fatal("expected pending native anthropic handoff to inject after later anchor match")
	}
}

func continuityHiddenMessages(messages ...map[string]any) []json.RawMessage {
	out := make([]json.RawMessage, 0, len(messages))
	for _, msg := range messages {
		raw, _ := json.Marshal(msg)
		out = append(out, raw)
	}
	return out
}

func assertJSONToolTag(t *testing.T, raw any, wantTag string) {
	t.Helper()
	msg, _ := raw.(map[string]any)
	if msg == nil {
		t.Fatalf("expected message object, got %#v", raw)
	}
	if msg["role"] != "tool" {
		t.Fatalf("expected tool message, got %+v", msg)
	}
	content, _ := msg["content"].(string)
	var payload map[string]any
	if err := json.Unmarshal([]byte(content), &payload); err != nil {
		t.Fatalf("unmarshal tool content: %v", err)
	}
	if payload["tag"] != wantTag {
		t.Fatalf("expected tool tag %q, got %+v", wantTag, payload)
	}
}
