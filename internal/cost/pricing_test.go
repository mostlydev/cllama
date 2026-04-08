package cost

import "testing"

func TestLookupKnownModel(t *testing.T) {
	p := DefaultPricing()
	rate, ok := p.Lookup("anthropic", "claude-sonnet-4")
	if !ok {
		t.Fatal("expected to find claude-sonnet-4")
	}
	if rate.InputPerMTok <= 0 || rate.OutputPerMTok <= 0 {
		t.Errorf("expected positive rates, got in=%f out=%f", rate.InputPerMTok, rate.OutputPerMTok)
	}
}

func TestLookupUnknownModelReturnsFalse(t *testing.T) {
	p := DefaultPricing()
	_, ok := p.Lookup("anthropic", "nonexistent-model")
	if ok {
		t.Error("expected false for unknown model")
	}
}

func TestLookupOpenAIModel(t *testing.T) {
	p := DefaultPricing()
	rate, ok := p.Lookup("openai", "gpt-4o")
	if !ok {
		t.Fatal("expected to find gpt-4o")
	}
	if rate.InputPerMTok <= 0 {
		t.Error("expected positive input rate")
	}
}

func TestLookupGoogleGeminiFlashModel(t *testing.T) {
	p := DefaultPricing()
	rate, ok := p.Lookup("google", "gemini-2.5-flash")
	if !ok {
		t.Fatal("expected to find google/gemini-2.5-flash")
	}
	if rate.InputPerMTok != 0.30 || rate.OutputPerMTok != 2.50 {
		t.Fatalf("unexpected google/gemini-2.5-flash rates: %+v", rate)
	}
}

func TestLookupGoogleGeminiProModel(t *testing.T) {
	p := DefaultPricing()
	rate, ok := p.Lookup("google", "gemini-2.5-pro")
	if !ok {
		t.Fatal("expected to find google/gemini-2.5-pro")
	}
	if rate.InputPerMTok != 1.25 || rate.OutputPerMTok != 10.0 {
		t.Fatalf("unexpected google/gemini-2.5-pro rates: %+v", rate)
	}
}

func TestComputeCost(t *testing.T) {
	rate := Rate{InputPerMTok: 3.0, OutputPerMTok: 15.0}
	cost := rate.Compute(1000, 500)
	// 1000 input tokens = 1000/1_000_000 * 3.0 = 0.003
	// 500 output tokens = 500/1_000_000 * 15.0 = 0.0075
	expected := 0.003 + 0.0075
	if cost < expected-0.0001 || cost > expected+0.0001 {
		t.Errorf("expected ~%f, got %f", expected, cost)
	}
}
