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

func TestLookupCurrentClaudeAndGeminiModels(t *testing.T) {
	tests := []struct {
		provider   string
		model      string
		wantInput  float64
		wantOutput float64
	}{
		{provider: "anthropic", model: "claude-fable-5", wantInput: 10.0, wantOutput: 50.0},
		{provider: "anthropic", model: "claude-opus-5", wantInput: 5.0, wantOutput: 25.0},
		{provider: "anthropic", model: "claude-sonnet-5", wantInput: 3.0, wantOutput: 15.0},
		{provider: "anthropic", model: "claude-haiku-4-5", wantInput: 1.0, wantOutput: 5.0},
		{provider: "google", model: "gemini-3.6-flash", wantInput: 1.5, wantOutput: 7.5},
		{provider: "openrouter", model: "anthropic/claude-fable-5", wantInput: 10.0, wantOutput: 50.0},
		{provider: "openrouter", model: "anthropic/claude-opus-5", wantInput: 5.0, wantOutput: 25.0},
		{provider: "openrouter", model: "anthropic/claude-sonnet-5", wantInput: 3.0, wantOutput: 15.0},
		{provider: "openrouter", model: "anthropic/claude-haiku-4.5", wantInput: 1.0, wantOutput: 5.0},
		{provider: "openrouter", model: "google/gemini-3.6-flash", wantInput: 1.5, wantOutput: 7.5},
	}

	pricing := DefaultPricing()
	for _, tt := range tests {
		t.Run(tt.provider+"/"+tt.model, func(t *testing.T) {
			rate, ok := pricing.Lookup(tt.provider, tt.model)
			if !ok {
				t.Fatalf("missing pricing for %s/%s", tt.provider, tt.model)
			}
			if rate.InputPerMTok != tt.wantInput || rate.OutputPerMTok != tt.wantOutput {
				t.Fatalf("%s/%s pricing = %+v; want input=%v output=%v", tt.provider, tt.model, rate, tt.wantInput, tt.wantOutput)
			}
			wantCost := tt.wantInput + tt.wantOutput
			if got := rate.Compute(1_000_000, 1_000_000); got != wantCost {
				t.Fatalf("%s/%s cost = %v; want %v", tt.provider, tt.model, got, wantCost)
			}
		})
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

func TestLookupResponsesAdapterBuiltInPricing(t *testing.T) {
	tests := []struct {
		model      string
		wantInput  float64
		wantOutput float64
	}{
		{model: "gpt-5.6", wantInput: 5.0, wantOutput: 30.0},
		{model: "gpt-5.6-sol", wantInput: 5.0, wantOutput: 30.0},
		{model: "gpt-5.6-terra", wantInput: 2.0, wantOutput: 12.0},
		{model: "gpt-5.6-luna", wantInput: 0.20, wantOutput: 1.20},
		{model: "gpt-5.6-terra-2026-01-01", wantInput: 2.0, wantOutput: 12.0},
		{model: "gpt-5.6-future-tier", wantInput: 5.0, wantOutput: 30.0},
		{model: "gpt-5-pro", wantInput: 15.0, wantOutput: 120.0},
		{model: "gpt-5-pro-2025-10-06", wantInput: 15.0, wantOutput: 120.0},
	}

	pricing := DefaultPricing()
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			rate, ok := pricing.Lookup("openai", tt.model)
			if !ok {
				t.Fatalf("missing pricing for %s", tt.model)
			}
			if rate.InputPerMTok != tt.wantInput || rate.OutputPerMTok != tt.wantOutput {
				t.Fatalf("%s pricing = %+v; want input=%v output=%v", tt.model, rate, tt.wantInput, tt.wantOutput)
			}
		})
	}
}

func TestGPT56LongContextPricingBoundary(t *testing.T) {
	const outputTokens = 1000
	for _, tt := range []struct {
		model      string
		inputRate  float64
		outputRate float64
	}{
		{model: "gpt-5.6", inputRate: 5.0, outputRate: 30.0},
		{model: "gpt-5.6-sol", inputRate: 5.0, outputRate: 30.0},
		{model: "gpt-5.6-terra", inputRate: 2.0, outputRate: 12.0},
		{model: "gpt-5.6-luna", inputRate: 0.20, outputRate: 1.20},
	} {
		t.Run(tt.model, func(t *testing.T) {
			rate, ok := DefaultPricing().Lookup("openai", tt.model)
			if !ok {
				t.Fatalf("missing %s pricing", tt.model)
			}
			standard := float64(272000)/1_000_000*tt.inputRate + float64(outputTokens)/1_000_000*tt.outputRate
			if got := rate.Compute(272000, outputTokens); got != standard {
				t.Fatalf("exactly 272K input must retain standard pricing: got %.12f want %.12f", got, standard)
			}

			tiered := float64(272001)/1_000_000*(tt.inputRate*2.0) + float64(outputTokens)/1_000_000*(tt.outputRate*1.5)
			if got := rate.Compute(272001, outputTokens); got != tiered {
				t.Fatalf("input above 272K must price the full request at 2x input and 1.5x output: got %.12f want %.12f", got, tiered)
			}
		})
	}
}

func TestGPT5ProDoesNotUseGPT56LongContextMultiplier(t *testing.T) {
	rate, ok := DefaultPricing().Lookup("openai", "gpt-5-pro-2025-10-06")
	if !ok {
		t.Fatal("missing dated gpt-5-pro pricing")
	}

	const inputTokens = 300000
	const outputTokens = 1000
	want := float64(inputTokens)/1_000_000*15.0 + float64(outputTokens)/1_000_000*120.0
	if got := rate.Compute(inputTokens, outputTokens); got != want {
		t.Fatalf("gpt-5-pro must retain uniform pricing above 272K: got %.12f want %.12f", got, want)
	}
}
