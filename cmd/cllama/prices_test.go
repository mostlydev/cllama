package main

import (
	"bytes"
	"encoding/json"
	"testing"
)

func TestPricesPrintsTheCompiledTable(t *testing.T) {
	// The command must not depend on the environment or any file.
	t.Setenv("CLAW_AUTH_DIR", "/nonexistent")
	t.Setenv("CLAW_CONTEXT_ROOT", "/nonexistent")
	var first, second, stderr bytes.Buffer
	if err := run([]string{"prices", "--json"}, &first, &stderr); err != nil {
		t.Fatal(err)
	}
	if err := run([]string{"prices", "--json"}, &second, &stderr); err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(first.Bytes(), second.Bytes()) {
		t.Fatal("output is not stable between runs")
	}
	var got priceExport
	dec := json.NewDecoder(bytes.NewReader(first.Bytes()))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != 1 || got.Currency != "USD" || got.Unit != "per_million_tokens" || got.Note == "" {
		t.Fatalf("header = %+v", got)
	}
	if got.Cllama.Version != version || got.Cllama.Revision != revision {
		t.Fatalf("build = %+v", got.Cllama)
	}
	var terra bool
	for _, e := range got.Entries {
		if e.Provider == "openai" && e.Model == "gpt-5.6-terra" {
			terra = true
			if e.Input != 2 || e.Output != 12 || e.MaxInput != 4 || e.MaxOutput != 18 || e.LongContextThresholdTokens != 272000 || e.Tiers != "all" {
				t.Fatalf("terra = %+v", e)
			}
		}
	}
	if !terra {
		t.Fatal("no entry for openai/gpt-5.6-terra")
	}
}

func TestPricesRequiresTheJSONFlag(t *testing.T) {
	for _, args := range [][]string{{"prices"}, {"prices", "--json", "extra"}} {
		var stdout, stderr bytes.Buffer
		if err := run(args, &stdout, &stderr); err == nil || stdout.Len() != 0 {
			t.Fatalf("%v: err=%v stdout=%q", args, err, stdout.String())
		}
	}
}
