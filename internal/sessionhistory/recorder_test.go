package sessionhistory

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"
)

func TestRecorder_WritesOneJSONEntry(t *testing.T) {
	dir := t.TempDir()
	r := New(dir)

	cost := 0.001
	entry := Entry{
		Version:           1,
		TS:                time.Now().UTC().Format(time.RFC3339),
		ClawID:            "agent-abc",
		Path:              "/v1/chat/completions",
		RequestedModel:    "claude-3-opus",
		EffectiveProvider: "anthropic",
		EffectiveModel:    "claude-3-opus-20240229",
		StatusCode:        200,
		Stream:            false,
		Response: Payload{
			Format: "json",
			JSON:   json.RawMessage(`{"id":"resp-1","object":"chat.completion"}`),
		},
		Usage: Usage{
			PromptTokens:     100,
			CompletionTokens: 50,
			ReportedCostUSD:  &cost,
		},
	}

	if err := r.Record("agent-abc", entry); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	histFile := filepath.Join(dir, "agent-abc", "history.jsonl")
	data, err := os.ReadFile(histFile)
	if err != nil {
		t.Fatalf("history file not created: %v", err)
	}

	lines := nonEmptyLines(string(data))
	if len(lines) != 1 {
		t.Fatalf("expected 1 line, got %d", len(lines))
	}

	var got Entry
	if err := json.Unmarshal([]byte(lines[0]), &got); err != nil {
		t.Fatalf("line is not valid JSON: %v", err)
	}

	if got.ClawID != "agent-abc" {
		t.Errorf("ClawID mismatch: got %q", got.ClawID)
	}
	if got.StatusCode != 200 {
		t.Errorf("StatusCode mismatch: got %d", got.StatusCode)
	}
	if got.EffectiveProvider != "anthropic" {
		t.Errorf("EffectiveProvider mismatch: got %q", got.EffectiveProvider)
	}
	if got.Response.Format != "json" {
		t.Errorf("Response.Format mismatch: got %q", got.Response.Format)
	}
	if got.Version != 1 {
		t.Errorf("Version mismatch: got %d", got.Version)
	}
}

func TestRecorder_AppendsMultipleTurns(t *testing.T) {
	dir := t.TempDir()
	r := New(dir)

	for i := 0; i < 3; i++ {
		entry := Entry{
			Version: 1,
			TS:      time.Now().UTC().Format(time.RFC3339),
			ClawID:  "agent-xyz",
			Response: Payload{
				Format: "json",
				JSON:   json.RawMessage(`{}`),
			},
		}
		if err := r.Record("agent-xyz", entry); err != nil {
			t.Fatalf("write %d failed: %v", i, err)
		}
	}

	histFile := filepath.Join(dir, "agent-xyz", "history.jsonl")
	data, err := os.ReadFile(histFile)
	if err != nil {
		t.Fatalf("history file not found: %v", err)
	}

	lines := nonEmptyLines(string(data))
	if len(lines) != 3 {
		t.Fatalf("expected 3 lines, got %d", len(lines))
	}

	for i, line := range lines {
		var e Entry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Errorf("line %d is not valid JSON: %v", i, err)
		}
	}
}

func TestRecorder_ConcurrentWritesSafe(t *testing.T) {
	dir := t.TempDir()
	r := New(dir)

	const goroutines = 10
	var wg sync.WaitGroup
	wg.Add(goroutines)

	for i := 0; i < goroutines; i++ {
		go func() {
			defer wg.Done()
			entry := Entry{
				Version: 1,
				TS:      time.Now().UTC().Format(time.RFC3339),
				ClawID:  "agent-concurrent",
				Response: Payload{
					Format: "json",
					JSON:   json.RawMessage(`{"concurrent":true}`),
				},
			}
			if err := r.Record("agent-concurrent", entry); err != nil {
				t.Errorf("concurrent write failed: %v", err)
			}
		}()
	}
	wg.Wait()

	histFile := filepath.Join(dir, "agent-concurrent", "history.jsonl")
	data, err := os.ReadFile(histFile)
	if err != nil {
		t.Fatalf("history file not found: %v", err)
	}

	lines := nonEmptyLines(string(data))
	if len(lines) != goroutines {
		t.Fatalf("expected %d lines, got %d", goroutines, len(lines))
	}

	for i, line := range lines {
		var e Entry
		if err := json.Unmarshal([]byte(line), &e); err != nil {
			t.Errorf("line %d is corrupted (not valid JSON): %v", i, err)
		}
	}
}

func TestRecorder_SSEPayloadNoMarshalFailure(t *testing.T) {
	dir := t.TempDir()
	r := New(dir)

	entry := Entry{
		Version: 1,
		TS:      time.Now().UTC().Format(time.RFC3339),
		ClawID:  "agent-sse",
		Stream:  true,
		Response: Payload{
			Format: "sse",
			Text:   "data: {\"id\":\"chunk-1\"}\n\ndata: [DONE]\n\n",
		},
	}

	if err := r.Record("agent-sse", entry); err != nil {
		t.Fatalf("unexpected error recording SSE entry: %v", err)
	}

	histFile := filepath.Join(dir, "agent-sse", "history.jsonl")
	data, err := os.ReadFile(histFile)
	if err != nil {
		t.Fatalf("history file not found: %v", err)
	}

	lines := nonEmptyLines(string(data))
	if len(lines) != 1 {
		t.Fatalf("expected 1 line, got %d", len(lines))
	}

	var got Entry
	if err := json.Unmarshal([]byte(lines[0]), &got); err != nil {
		t.Fatalf("line is not valid JSON: %v", err)
	}

	if got.Response.Format != "sse" {
		t.Errorf("expected format=sse, got %q", got.Response.Format)
	}
	if got.Response.Text == "" {
		t.Errorf("expected non-empty Text for SSE payload")
	}
	if got.Response.JSON != nil {
		t.Errorf("expected nil JSON for SSE payload, got %s", got.Response.JSON)
	}
}

func TestRecorder_NoOpWhenBaseDirEmpty(t *testing.T) {
	r := New("")

	entry := Entry{
		Version: 1,
		TS:      time.Now().UTC().Format(time.RFC3339),
		ClawID:  "agent-noop",
		Response: Payload{
			Format: "json",
			JSON:   json.RawMessage(`{}`),
		},
	}

	if err := r.Record("agent-noop", entry); err != nil {
		t.Fatalf("expected no error for no-op recorder, got: %v", err)
	}
}

// nonEmptyLines splits s by newline and returns only non-empty lines.
func nonEmptyLines(s string) []string {
	var lines []string
	sc := bufio.NewScanner(newStringReader(s))
	for sc.Scan() {
		line := sc.Text()
		if line != "" {
			lines = append(lines, line)
		}
	}
	return lines
}

type stringReader struct {
	s   string
	pos int
}

func newStringReader(s string) *stringReader { return &stringReader{s: s} }

func (r *stringReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.s) {
		return 0, os.ErrClosed
	}
	n := copy(p, r.s[r.pos:])
	r.pos += n
	return n, nil
}
