// Package sessionhistory provides a durable, append-only recorder for LLM
// proxy turns. Each agent's history is written to
// <baseDir>/<agent-id>/history.jsonl. An empty baseDir makes the recorder a
// no-op so that callers need not special-case the unconfigured state.
package sessionhistory

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
)

// Payload holds either a structured JSON response or a raw SSE text stream.
type Payload struct {
	Format string          `json:"format"`         // "json" or "sse"
	JSON   json.RawMessage `json:"json,omitempty"` // set when Format == "json"
	Text   string          `json:"text,omitempty"` // set when Format == "sse"
}

// Usage captures token counts and optional cost for a single LLM turn.
type Usage struct {
	PromptTokens     int      `json:"prompt_tokens,omitempty"`
	CompletionTokens int      `json:"completion_tokens,omitempty"`
	ReportedCostUSD  *float64 `json:"reported_cost_usd,omitempty"`
}

// Entry is the normalized envelope written as one JSONL line per LLM turn.
type Entry struct {
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

// Recorder appends Entry values to per-agent JSONL files. It is safe for
// concurrent use. When baseDir is empty every call to Record is a no-op.
type Recorder struct {
	baseDir string

	mu    sync.Mutex
	files map[string]*openFile // keyed by agent ID
}

// openFile wraps an os.File together with a per-file mutex so that multiple
// goroutines writing for the same agent are serialised without blocking writes
// for other agents.
type openFile struct {
	mu sync.Mutex
	f  *os.File
}

// New creates a Recorder that writes under baseDir. Pass an empty string to
// obtain a no-op recorder.
func New(baseDir string) *Recorder {
	return &Recorder{
		baseDir: baseDir,
		files:   make(map[string]*openFile),
	}
}

// Record marshals e as a single JSON line and appends it to
// <baseDir>/<agentID>/history.jsonl. If the recorder was created with an
// empty baseDir the call succeeds immediately without any I/O.
func (r *Recorder) Record(agentID string, e Entry) error {
	if r.baseDir == "" {
		return nil
	}

	of, err := r.openFor(agentID)
	if err != nil {
		return err
	}

	line, err := json.Marshal(e)
	if err != nil {
		return err
	}
	line = append(line, '\n')

	of.mu.Lock()
	defer of.mu.Unlock()
	_, err = of.f.Write(line)
	return err
}

// openFor returns (creating if necessary) the openFile handle for agentID.
// The recorder-level mutex serialises the open/create path; once the handle
// exists subsequent writes lock only the per-file mutex.
func (r *Recorder) openFor(agentID string) (*openFile, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	if of, ok := r.files[agentID]; ok {
		return of, nil
	}

	agentDir := filepath.Join(r.baseDir, agentID)
	if err := os.MkdirAll(agentDir, 0o777); err != nil {
		return nil, err
	}

	histPath := filepath.Join(agentDir, "history.jsonl")
	f, err := os.OpenFile(histPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o666)
	if err != nil {
		return nil, err
	}

	of := &openFile{f: f}
	r.files[agentID] = of
	return of, nil
}
