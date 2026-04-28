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
	TotalRounds      int      `json:"total_rounds,omitempty"`
}

type ToolCallTrace struct {
	Name             string          `json:"name"`
	Arguments        json.RawMessage `json:"arguments,omitempty"`
	Result           json.RawMessage `json:"result,omitempty"`
	LatencyMS        int64           `json:"latency_ms,omitempty"`
	Service          string          `json:"service,omitempty"`
	StatusCode       int             `json:"status_code,omitempty"`
	Duplicate        bool            `json:"duplicate,omitempty"`
	DuplicateOfRound int             `json:"duplicate_of_round,omitempty"`
	DuplicateCount   int             `json:"duplicate_count,omitempty"`
}

type ToolRoundTrace struct {
	Round      int             `json:"round"`
	ToolCalls  []ToolCallTrace `json:"tool_calls,omitempty"`
	RoundUsage Usage           `json:"round_usage,omitempty"`
}

// Entry is the normalized envelope written as one JSONL line per LLM turn.
type Entry struct {
	Version           int              `json:"version"`
	ID                string           `json:"id,omitempty"`
	Status            string           `json:"status,omitempty"`
	TS                string           `json:"ts"`
	ClawID            string           `json:"claw_id"`
	Path              string           `json:"path"`
	RequestedModel    string           `json:"requested_model"`
	EffectiveProvider string           `json:"effective_provider"`
	EffectiveModel    string           `json:"effective_model"`
	StatusCode        int              `json:"status_code"`
	Stream            bool             `json:"stream"`
	RequestOriginal   json.RawMessage  `json:"request_original,omitempty"`
	RequestEffective  json.RawMessage  `json:"request_effective,omitempty"`
	Response          Payload          `json:"response"`
	Usage             Usage            `json:"usage,omitempty"`
	ToolTrace         []ToolRoundTrace `json:"tool_trace,omitempty"`
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

// BaseDir returns the recorder's configured history root.
func (r *Recorder) BaseDir() string {
	if r == nil {
		return ""
	}
	return r.baseDir
}

// Record marshals e as a single JSON line and appends it to
// <baseDir>/<agentID>/history.jsonl. If the recorder was created with an
// empty baseDir the call succeeds immediately without any I/O.
func (r *Recorder) Record(agentID string, e Entry) error {
	if r.baseDir == "" {
		return nil
	}
	if err := e.EnsureID(); err != nil {
		return err
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

// Close closes all open file handles held by the recorder. It collects the
// first error encountered but continues closing remaining files. Calling Close
// on an already-closed or no-op recorder is safe and returns nil.
func (r *Recorder) Close() error {
	if r.baseDir == "" {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	var firstErr error
	for _, of := range r.files {
		of.mu.Lock()
		if err := of.f.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
		of.mu.Unlock()
	}
	r.files = nil
	return firstErr
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
