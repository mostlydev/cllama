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
	Entries  []cost.Entry `json:"entries"`
}

type buildInfo struct {
	Version  string `json:"version"`
	Revision string `json:"revision"`
}

const priceExportNote = "An id equal to an entry's model is priced exactly. " +
	"cllama also costs a longer id by its longest matching entry; that is an estimate, not an exact price. " +
	"max_input and max_output are the highest rates the entry encodes. " +
	"An entry whose tiers is base_only omits a higher tier and is not a ceiling. " +
	"These are listed prices, not a billing cap."

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
		Currency: "USD", Unit: "per_million_tokens", Note: priceExportNote, Entries: entries,
	}, "", "  ")
	if err != nil {
		return err
	}
	_, err = stdout.Write(append(out, '\n'))
	return err
}
