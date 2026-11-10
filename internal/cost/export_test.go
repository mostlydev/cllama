package cost

import (
	"math"
	"sort"
	"testing"
)

func TestEntriesCoverTheWholeTableInOrder(t *testing.T) {
	p := DefaultPricing()
	entries, err := p.Entries()
	if err != nil {
		t.Fatal(err)
	}
	want := 0
	for _, models := range p.rates {
		want += len(models)
	}
	if len(entries) != want || want == 0 {
		t.Fatalf("entries = %d, table rows = %d", len(entries), want)
	}
	if !sort.SliceIsSorted(entries, func(i, j int) bool {
		if entries[i].Provider != entries[j].Provider {
			return entries[i].Provider < entries[j].Provider
		}
		return entries[i].Model < entries[j].Model
	}) {
		t.Fatal("entries are not sorted by provider then model")
	}
	for _, e := range entries {
		// An exported row is exactly what the proxy charges for that id.
		r, ok := p.Lookup(e.Provider, e.Model)
		if !ok || r.InputPerMTok != e.Input || r.OutputPerMTok != e.Output {
			t.Fatalf("%s/%s: exported %v/%v, lookup %v", e.Provider, e.Model, e.Input, e.Output, r)
		}
		if e.MaxInput < e.Input || e.MaxOutput < e.Output {
			t.Fatalf("%s/%s: maximum below base: %+v", e.Provider, e.Model, e)
		}
		// The maximum is what Compute charges above the threshold.
		if e.LongContextThresholdTokens > 0 {
			in := e.LongContextThresholdTokens + 1
			got := r.Compute(in, 1_000_000)
			wantCost := float64(in)/1_000_000*e.MaxInput + e.MaxOutput
			if math.Abs(got-wantCost) > 1e-9 {
				t.Fatalf("%s/%s: long-context cost %v, maximum rates give %v", e.Provider, e.Model, got, wantCost)
			}
		} else if e.MaxInput != e.Input || e.MaxOutput != e.Output {
			t.Fatalf("%s/%s: flat entry has a different maximum: %+v", e.Provider, e.Model, e)
		}
	}
}

func TestEntriesMarkKnownIncompleteTiers(t *testing.T) {
	entries, err := DefaultPricing().Entries()
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]string{}
	for _, e := range entries {
		seen[e.Provider+"/"+e.Model] = e.Tiers
	}
	for id, want := range map[string]string{
		"openai/gpt-5.6-terra":               TiersAll,
		"google/gemini-2.5-pro":              TiersBaseOnly,
		"openrouter/google/gemini-2.5-pro":   TiersBaseOnly,
		"openrouter/anthropic/claude-opus-5": TiersAll,
	} {
		if seen[id] != want {
			t.Errorf("%s tiers = %q, want %q", id, seen[id], want)
		}
	}
}

func TestEntriesRefuseAnInvalidRate(t *testing.T) {
	for name, r := range map[string]Rate{
		"negative":       {InputPerMTok: -1, OutputPerMTok: 1},
		"not a number":   {InputPerMTok: math.NaN(), OutputPerMTok: 1},
		"infinite":       {InputPerMTok: 1, OutputPerMTok: math.Inf(1)},
		"bad multiplier": {InputPerMTok: 1, OutputPerMTok: 1, LongContextThresholdTokens: 10, LongContextInputMultiplier: math.Inf(1)},
		"bad threshold":  {InputPerMTok: 1, OutputPerMTok: 1, LongContextThresholdTokens: -5},
	} {
		p := &Pricing{rates: map[string]map[string]Rate{"x": {"m": r}}}
		if entries, err := p.Entries(); err == nil {
			t.Errorf("%s: exported %+v", name, entries)
		}
	}
}
