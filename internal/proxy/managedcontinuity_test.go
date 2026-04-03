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
