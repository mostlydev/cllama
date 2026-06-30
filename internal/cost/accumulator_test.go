package cost

import "testing"

func TestAccumulatorRecordAndQuery(t *testing.T) {
	a := NewAccumulator()
	a.Record("tiverton", "anthropic", "claude-sonnet-4", 1000, 500, 0.0105)
	a.Record("tiverton", "anthropic", "claude-sonnet-4", 2000, 1000, 0.021)

	summary := a.ByAgent("tiverton")
	if len(summary) != 1 {
		t.Fatalf("expected 1 model entry, got %d", len(summary))
	}
	entry := summary[0]
	if entry.TotalInputTokens != 3000 {
		t.Errorf("expected 3000 input tokens, got %d", entry.TotalInputTokens)
	}
	if entry.TotalOutputTokens != 1500 {
		t.Errorf("expected 1500 output tokens, got %d", entry.TotalOutputTokens)
	}
	if entry.TotalCostUSD < 0.031 || entry.TotalCostUSD > 0.032 {
		t.Errorf("expected ~0.0315 cost, got %f", entry.TotalCostUSD)
	}
	if entry.RequestCount != 2 {
		t.Errorf("expected 2 requests, got %d", entry.RequestCount)
	}
	if entry.PricedRequests != 2 || entry.UnpricedRequests != 0 {
		t.Errorf("expected priced=2 unpriced=0, got priced=%d unpriced=%d", entry.PricedRequests, entry.UnpricedRequests)
	}
}

func TestAccumulatorAllAgents(t *testing.T) {
	a := NewAccumulator()
	a.Record("tiverton", "anthropic", "claude-sonnet-4", 100, 50, 0.001)
	a.Record("westin", "openai", "gpt-4o", 200, 100, 0.002)

	all := a.All()
	if len(all) != 2 {
		t.Fatalf("expected 2 agents, got %d", len(all))
	}
}

func TestAccumulatorTotalCost(t *testing.T) {
	a := NewAccumulator()
	a.Record("tiverton", "anthropic", "claude-sonnet-4", 100, 50, 0.001)
	a.Record("westin", "openai", "gpt-4o", 200, 100, 0.002)
	total := a.TotalCost()
	if total < 0.002 || total > 0.004 {
		t.Errorf("expected ~0.003, got %f", total)
	}
}

func TestAccumulatorTracksUnpricedRequests(t *testing.T) {
	a := NewAccumulator()
	a.RecordWithStatus("tiverton", "openrouter", "unknown/model", 500, 100, 0, false)

	summary := a.ByAgent("tiverton")
	if len(summary) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(summary))
	}
	entry := summary[0]
	if entry.PricedRequests != 0 {
		t.Errorf("expected 0 priced requests, got %d", entry.PricedRequests)
	}
	if entry.UnpricedRequests != 1 {
		t.Errorf("expected 1 unpriced request, got %d", entry.UnpricedRequests)
	}
	if got := a.TotalUnpricedRequests(); got != 1 {
		t.Errorf("expected total unpriced requests=1, got %d", got)
	}
	if got := a.TotalRequests(); got != 1 {
		t.Errorf("expected total requests=1, got %d", got)
	}
}
