package sessionhistory

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"strings"
)

const entryIDPrefix = "hist1_"

type entryWithoutID struct {
	Version           int             `json:"version"`
	TS                string          `json:"ts"`
	ClawID            string          `json:"claw_id"`
	Path              string          `json:"path"`
	RequestedModel    string          `json:"requested_model"`
	EffectiveProvider string          `json:"effective_provider"`
	EffectiveModel    string          `json:"effective_model"`
	StatusCode        int             `json:"status_code"`
	Stream            bool            `json:"stream"`
	RequestOriginal   json.RawMessage `json:"request_original,omitempty"`
	RequestEffective  json.RawMessage `json:"request_effective,omitempty"`
	Response          Payload         `json:"response"`
	Usage             Usage           `json:"usage,omitempty"`
}

// EnsureID assigns a stable source-event ID when the entry does not already
// have one. New entries hash the canonical JSON that would have been written
// without the id field; older ledger lines can be hydrated from their raw JSON
// using IDFromJSON.
func (e *Entry) EnsureID() error {
	if e == nil || strings.TrimSpace(e.ID) != "" {
		return nil
	}
	raw, err := json.Marshal(entryWithoutID{
		Version:           e.Version,
		TS:                e.TS,
		ClawID:            e.ClawID,
		Path:              e.Path,
		RequestedModel:    e.RequestedModel,
		EffectiveProvider: e.EffectiveProvider,
		EffectiveModel:    e.EffectiveModel,
		StatusCode:        e.StatusCode,
		Stream:            e.Stream,
		RequestOriginal:   e.RequestOriginal,
		RequestEffective:  e.RequestEffective,
		Response:          e.Response,
		Usage:             e.Usage,
	})
	if err != nil {
		return err
	}
	e.ID = IDFromJSON(raw)
	return nil
}

// IDFromJSON derives the stable source-event ID for a JSONL entry body.
// Callers should pass the raw JSON line without a trailing newline.
func IDFromJSON(raw []byte) string {
	sum := sha256.Sum256(bytes.TrimSpace(raw))
	return entryIDPrefix + hex.EncodeToString(sum[:])
}
