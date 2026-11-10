package main

import (
	"encoding/json"
	"errors"
	"flag"
	"io"

	"github.com/mostlydev/cllama/internal/cost"
)

// priceExport is the output of "cllama prices --json": the price table this
// build costs requests with. It reads no configuration, environment or file.
type priceExport struct {
	Schema   int          `json:"schema"`
	Cllama   buildInfo    `json:"cllama"`
	Currency string       `json:"currency"`
	Unit     string       `json:"unit"`
	Note     string       `json:"note"`
	Scope    priceScope   `json:"scope"`
	Entries  []cost.Entry `json:"entries"`
}

type buildInfo struct {
	Version  string `json:"version"`
	Revision string `json:"revision"`
}

type priceScope struct {
	ServiceTier string   `json:"service_tier"`
	Tokens      string   `json:"tokens"`
	Includes    []string `json:"includes"`
	Excludes    []string `json:"excludes"`
}

func exportedPriceScope() priceScope {
	return priceScope{ServiceTier: "standard", Tokens: "uncached_text",
		Includes: []string{"long_context"},
		Excludes: []string{"regional_or_compliance_uplift", "priority_or_fast_service_tiers", "cache_write", "cached_input_discount", "batch", "non_text_modalities", "hosted_tool_fees"}}
}

const priceExportNote = "Rates are the vendor's standard service tier for uncached text tokens. " +
	"max_input and max_output add the long-context tier where the entry has one, and nothing else. " +
	"Regional or compliance uplifts, priority or fast service tiers, cache writes, batch and non-text pricing are outside this table. " +
	"An id equal to an entry's model is priced exactly; a longer id costed by its longest matching entry is an estimate. " +
	"context_tiers base_only means a higher context tier exists and is not encoded. These are listed prices, not a billing cap."

func runPrices(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("cllama prices", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "print the compiled price table as JSON")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if !*asJSON || fs.NArg() != 0 {
		return errors.New("usage: cllama prices --json")
	}
	entries, err := cost.DefaultPricing().Entries()
	if err != nil {
		return err
	}
	out, err := json.MarshalIndent(priceExport{
		Schema: 1, Cllama: buildInfo{Version: version, Revision: revision},
		Currency: "USD", Unit: "per_million_tokens", Note: priceExportNote, Scope: exportedPriceScope(), Entries: entries,
	}, "", "  ")
	if err != nil {
		return err
	}
	_, err = stdout.Write(append(out, '\n'))
	return err
}
