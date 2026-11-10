package cost

import (
	"fmt"
	"math"
	"sort"
	"strings"
)

// Tier coverage of a table entry.
const (
	// ContextTiersAll means the entry encodes every rate tier the table knows for
	// the model, so its maximum rates are the highest the table can charge.
	ContextTiersAll = "all"
	// ContextTiersBaseOnly means the entry is knowingly simplified to the model's
	// base tier: a higher tier exists and is not in the table. Its rates are
	// fine for estimating cost and are not a ceiling.
	ContextTiersBaseOnly = "base_only"
)

// Entry is one row of the compiled price table, as exported.
type Entry struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
	// Input and Output are the base rates in USD per million tokens.
	Input  float64 `json:"input"`
	Output float64 `json:"output"`
	// MaxInput and MaxOutput are the highest rates this entry can charge,
	// including its long-context multipliers.
	MaxInput                   float64 `json:"max_input"`
	MaxOutput                  float64 `json:"max_output"`
	LongContextThresholdTokens int     `json:"long_context_threshold_tokens,omitempty"`
	ContextTiers               string  `json:"context_tiers"`
}

// baseTierOnly reports whether a table entry is knowingly simplified to the
// model's base tier. Google's models are priced here at the standard
// <=200k-token text tier only, on the native route and through OpenRouter.
func baseTierOnly(provider, model string) bool {
	return provider == "google" || (provider == "openrouter" && strings.HasPrefix(model, "google/"))
}

func maxRate(base, multiplier float64, threshold int) float64 {
	if threshold > 0 && multiplier > 1 {
		return base * multiplier
	}
	return base
}

// Entries returns every row of the table, sorted by provider then model. It
// fails if any rate is negative or not finite: a table that cannot be
// trusted as a bound must not be exported as one.
func (p *Pricing) Entries() ([]Entry, error) {
	var out []Entry
	for provider, models := range p.rates {
		for model, r := range models {
			e := Entry{
				Provider: provider, Model: model,
				Input: r.InputPerMTok, Output: r.OutputPerMTok,
				MaxInput:                   maxRate(r.InputPerMTok, r.LongContextInputMultiplier, r.LongContextThresholdTokens),
				MaxOutput:                  maxRate(r.OutputPerMTok, r.LongContextOutputMultiplier, r.LongContextThresholdTokens),
				LongContextThresholdTokens: r.LongContextThresholdTokens,
				ContextTiers:               ContextTiersAll,
			}
			if baseTierOnly(provider, model) {
				e.ContextTiers = ContextTiersBaseOnly
			}
			for _, v := range []float64{e.Input, e.Output, e.MaxInput, e.MaxOutput, r.LongContextInputMultiplier, r.LongContextOutputMultiplier} {
				if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
					return nil, fmt.Errorf("price table entry %s/%s has an invalid rate", provider, model)
				}
			}
			if r.LongContextThresholdTokens < 0 {
				return nil, fmt.Errorf("price table entry %s/%s has an invalid threshold", provider, model)
			}
			out = append(out, e)
		}
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Provider != out[j].Provider {
			return out[i].Provider < out[j].Provider
		}
		return out[i].Model < out[j].Model
	})
	return out, nil
}
