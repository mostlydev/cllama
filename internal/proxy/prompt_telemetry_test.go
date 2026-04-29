package proxy

import "testing"

func TestRequestTelemetryStableSystemAndDynamicHashes(t *testing.T) {
	payload := map[string]any{
		"messages": []any{
			map[string]any{"role": "system", "content": "stable"},
			map[string]any{"role": "user", "content": "hello"},
		},
		"tools": []any{map[string]any{"name": "tool-a", "input": map[string]any{"b": 2, "a": 1}}},
	}
	first := requestTelemetry("openai", payload, "runtime-a")
	second := requestTelemetry("openai", payload, "runtime-b")
	if first.StaticSystemHash == "" || first.StaticSystemHash != second.StaticSystemHash {
		t.Fatalf("expected stable system hash, got %q and %q", first.StaticSystemHash, second.StaticSystemHash)
	}
	if first.DynamicContextHash == second.DynamicContextHash {
		t.Fatalf("expected dynamic context hash to change")
	}
	if first.ToolsHash == "" || first.ToolsHash != second.ToolsHash {
		t.Fatalf("expected stable tools hash, got %q and %q", first.ToolsHash, second.ToolsHash)
	}
}
