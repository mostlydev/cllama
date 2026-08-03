package proxy

import (
	"reflect"
	"testing"
)

func pruneSchema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"ticker":           map[string]any{"type": "string"},
			"qty_requested":    map[string]any{"type": "number", "minimum": 0.0001},
			"amount_requested": map[string]any{"type": "number", "minimum": 0.01},
			"stop_price":       map[string]any{"type": "number", "minimum": 0.0001},
			"note":             map[string]any{"type": "string"},
		},
		"required": []any{"ticker"},
	}
}

func TestPruneDropsOptionalArgsAtSchemaMinimum(t *testing.T) {
	args := map[string]any{
		"ticker":           "KLAC",
		"qty_requested":    float64(10),
		"amount_requested": 0.01,
		"stop_price":       0.0001,
	}
	pruned := pruneSentinelOptionalArgs(pruneSchema(), args)
	if !reflect.DeepEqual(pruned, []string{"amount_requested", "stop_price"}) {
		t.Fatalf("unexpected pruned set: %v", pruned)
	}
	if _, ok := args["amount_requested"]; ok {
		t.Fatal("amount_requested should have been dropped")
	}
	if args["qty_requested"] != float64(10) {
		t.Fatal("a real value must survive")
	}
	if args["ticker"] != "KLAC" {
		t.Fatal("required arg must survive")
	}
}

func TestPruneNeverDropsRequiredArgsEvenAtMinimum(t *testing.T) {
	schema := map[string]any{
		"type":       "object",
		"properties": map[string]any{"qty": map[string]any{"type": "number", "minimum": 0.0001}},
		"required":   []any{"qty"},
	}
	args := map[string]any{"qty": 0.0001}
	if pruned := pruneSentinelOptionalArgs(schema, args); len(pruned) != 0 {
		t.Fatalf("required arg must never be pruned, got %v", pruned)
	}
	if _, ok := args["qty"]; !ok {
		t.Fatal("required arg was dropped")
	}
}

func TestPruneLeavesValuesAboveMinimum(t *testing.T) {
	args := map[string]any{"ticker": "X", "amount_requested": float64(3600)}
	if pruned := pruneSentinelOptionalArgs(pruneSchema(), args); len(pruned) != 0 {
		t.Fatalf("a genuine value must survive, got %v", pruned)
	}
	if args["amount_requested"] != float64(3600) {
		t.Fatal("genuine value was dropped")
	}
}

func TestPruneIgnoresNonNumericAndUnknownProperties(t *testing.T) {
	args := map[string]any{"ticker": "X", "note": "hello", "unknown": 0.01}
	if pruned := pruneSentinelOptionalArgs(pruneSchema(), args); len(pruned) != 0 {
		t.Fatalf("nothing should be pruned, got %v", pruned)
	}
	if args["unknown"] != 0.01 {
		t.Fatal("unknown property must be left to the service")
	}
}

func TestPruneIsInertWithoutSchema(t *testing.T) {
	args := map[string]any{"amount_requested": 0.01}
	if pruned := pruneSentinelOptionalArgs(nil, args); pruned != nil {
		t.Fatalf("no schema means no pruning, got %v", pruned)
	}
	if len(args) != 1 {
		t.Fatal("args mutated without a schema")
	}
}

func TestPruneSelectionIsModelScoped(t *testing.T) {
	t.Setenv(EnvToolArgPruneSentinels, "vercel/openai/gpt-5.6")

	if !toolArgPruneSentinelsFor("vercel/openai/gpt-5.6-luna") {
		t.Fatal("selected model should prune")
	}
	// A different model sharing the same proxy must be unaffected: the
	// judgement was made about one model family, not the pod.
	if toolArgPruneSentinelsFor("vercel/minimax/minimax-m3") {
		t.Fatal("unselected model must not prune")
	}
	if toolArgPruneSentinelsFor("openai/gpt-5.6-luna") {
		t.Fatal("provider must match, not just the model prefix")
	}
}

func TestPruneIsOffWhenUnset(t *testing.T) {
	t.Setenv(EnvToolArgPruneSentinels, "")
	if toolArgPruneSentinelsFor("vercel/openai/gpt-5.6-luna") {
		t.Fatal("empty selector must disable pruning entirely")
	}
}

func TestPruneSelectorIgnoresMalformedEntries(t *testing.T) {
	t.Setenv(EnvToolArgPruneSentinels, "garbage,,vercel/,/x,vercel/openai/gpt-5.6")
	if !toolArgPruneSentinelsFor("vercel/openai/gpt-5.6-luna") {
		t.Fatal("a valid entry alongside malformed ones must still select")
	}
	if toolArgPruneSentinelsFor("vercel/anthropic/claude-haiku-4.5") {
		t.Fatal("malformed entries must not widen selection")
	}
}
