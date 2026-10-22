package proxy

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/provider"
	"github.com/mostlydev/cllama/internal/tracejournal"
)

func newTestJournal(t *testing.T, mutate func(*tracejournal.Config)) (*tracejournal.Journal, string) {
	t.Helper()
	cfg := tracejournal.Config{
		HistoryDir:      t.TempDir(),
		OwnerUID:        -1,
		OwnerGID:        -1,
		MaxFieldBytes:   tracejournal.DefaultMaxFieldBytes,
		MaxSegmentBytes: tracejournal.DefaultMaxSegmentBytes,
		Version:         "v-test",
		Revision:        "rev-test",
	}
	if mutate != nil {
		mutate(&cfg)
	}
	j, err := tracejournal.Open(cfg)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	t.Cleanup(func() { j.Close() })
	return j, cfg.HistoryDir
}

func journalSegments(t *testing.T, historyDir, agentID, suffix string) []string {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(historyDir, agentID, "journal", "*"+suffix))
	if err != nil {
		t.Fatal(err)
	}
	return matches
}

// verifiedSegment applies the collector's acceptance rule to the only closed
// segment for agentID and returns its records.
func verifiedSegment(t *testing.T, historyDir, agentID string) ([]map[string]any, []byte) {
	t.Helper()
	if open := journalSegments(t, historyDir, agentID, tracejournal.SuffixOpen); len(open) != 0 {
		t.Fatalf("open segments remain: %v", open)
	}
	closed := journalSegments(t, historyDir, agentID, tracejournal.SuffixClosed)
	if len(closed) != 1 {
		t.Fatalf("closed segments = %v", closed)
	}
	raw, err := os.ReadFile(closed[0])
	if err != nil {
		t.Fatal(err)
	}
	var records []map[string]any
	h := sha256.New()
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	var prior []byte
	for sc.Scan() {
		if prior != nil {
			h.Write(prior)
		}
		line := append(append([]byte(nil), sc.Bytes()...), '\n')
		prior = line
		var rec map[string]any
		if err := json.Unmarshal(line, &rec); err != nil {
			t.Fatalf("bad record: %v", err)
		}
		if int(rec["seq"].(float64)) != len(records)+1 {
			t.Fatalf("seq gap at %v", rec)
		}
		records = append(records, rec)
	}
	last := records[len(records)-1]
	if last["type"] != "segment_close" || int(last["record_count"].(float64)) != len(records)-1 || last["sha256"] != hex.EncodeToString(h.Sum(nil)) {
		t.Fatalf("segment does not verify: %v", last)
	}
	return records, raw
}

func recordTypes(records []map[string]any) string {
	types := make([]string, len(records))
	for i, rec := range records {
		types[i] = rec["type"].(string)
	}
	return strings.Join(types, ",")
}

func TestTraceJournalCapturesManagedToolLoop(t *testing.T) {
	// openai and anthropic pass through; responses and bridge exercise the
	// Responses and chat-to-Messages adapters, whose upstream bodies differ
	// from what the runner sent.
	for _, format := range []string{"openai", "anthropic", "responses", "bridge"} {
		for _, stream := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/stream=%t", format, stream), func(t *testing.T) {
				// The tool echoes its credential so redaction is exercised.
				toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprintf(w, `{"price":42,"echo":%q}`, r.Header.Get("Authorization"))
				}))
				defer toolSrv.Close()
				manifest := managedToolManifestForURL(toolSrv.URL, http.MethodPost, "/tool", "secret-tool-token")
				name := managedToolHashlessAliasForCanonical(manifest.Tools[0].Name)

				var calls atomic.Int32
				backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					_, _ = io.Copy(io.Discard, r.Body)
					w.Header().Set("Content-Type", "application/json")
					first := calls.Add(1) == 1
					switch {
					case format == "responses" && first:
						fmt.Fprintf(w, `{"id":"resp_1","object":"response","status":"completed","output":[{"type":"reasoning","id":"rs_1","encrypted_content":"PRIVATE-CHAIN","summary":[{"type":"summary_text","text":"PRIVATE-CHAIN"}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"checking"}]},{"type":"function_call","call_id":"call-1","name":%q,"arguments":"{}"}],"usage":{"input_tokens":1,"output_tokens":1}}`, name)
					case format == "responses":
						fmt.Fprint(w, `{"id":"resp_2","object":"response","status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"price is 42"}]}],"usage":{"input_tokens":1,"output_tokens":1}}`)
					case format == "openai" && first:
						fmt.Fprintf(w, `{"id":"r1","choices":[{"message":{"role":"assistant","content":"checking","tool_calls":[{"id":"call-1","type":"function","function":{"name":%q,"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`, name)
					case format == "openai":
						fmt.Fprint(w, `{"id":"r2","choices":[{"message":{"role":"assistant","content":"price is 42","reasoning_content":"PRIVATE-CHAIN"},"finish_reason":"stop"}]}`)
					case first: // anthropic and bridge
						fmt.Fprintf(w, `{"id":"m1","type":"message","role":"assistant","content":[{"type":"thinking","thinking":"PRIVATE-CHAIN","signature":"SIG"},{"type":"text","text":"checking"},{"type":"tool_use","id":"call-1","name":%q,"input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`, name)
					default:
						fmt.Fprint(w, `{"id":"m2","type":"message","role":"assistant","content":[{"type":"text","text":"price is 42"}],"stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}`)
					}
				}))
				defer backend.Close()

				providerName, model, path, upstreamPath := "openai", "openai/model", "/v1/chat/completions", "/v1/chat/completions"
				switch format {
				case "anthropic":
					providerName, model, path, upstreamPath = "anthropic", "anthropic/model", "/v1/messages", "/v1/messages"
				case "bridge":
					providerName, model, upstreamPath = "anthropic", "anthropic/model", "/v1/messages"
				case "responses":
					model, upstreamPath = "openai/gpt-5.6-terra", "/v1/responses"
				}
				reg := provider.NewRegistry("")
				prov := &provider.Provider{Name: providerName, BaseURL: backend.URL + "/v1", APIKey: "provider-secret-key", Auth: "bearer"}
				if providerName == "anthropic" {
					prov.APIFormat = "anthropic"
					prov.Auth = "x-api-key"
				}
				reg.Set(providerName, prov)
				journal, historyDir := newTestJournal(t, nil)
				h := NewHandler(reg, stubContextLoaderWithTools("worker-1", "worker-1:agent-secret-value", manifest), nil, WithTraceJournal(journal))

				body := fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"price? my token is agent-secret-value"}],"max_tokens":20,"stream":%t}`, model, stream)
				req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
				req.Header.Set("Authorization", "Bearer worker-1:agent-secret-value")
				w := httptest.NewRecorder()
				h.ServeHTTP(w, req)
				if w.Code != http.StatusOK {
					t.Fatalf("status %d: %s", w.Code, w.Body.String())
				}

				records, raw := verifiedSegment(t, historyDir, "worker-1")
				want := "segment_open,upstream_request,upstream_response,tool_call,tool_result,upstream_request,upstream_response,client_response,segment_close"
				if got := recordTypes(records); got != want {
					t.Fatalf("records = %s\nwant %s", got, want)
				}
				if records[3]["call_id"] != "call-1" || records[4]["call_id"] != "call-1" {
					t.Fatalf("tool call id not carried: %v / %v", records[3], records[4])
				}
				if records[1]["provider"] != providerName || records[1]["call"].(float64) != 1 || records[5]["call"].(float64) != 2 {
					t.Fatalf("upstream calls not numbered: %v %v", records[1], records[5])
				}
				if records[1]["path"] != upstreamPath || records[5]["path"] != upstreamPath {
					t.Fatalf("upstream path = %v / %v, want %s", records[1]["path"], records[5]["path"], upstreamPath)
				}
				// The recorded body is the translated upstream shape.
				shape := map[string]string{"openai": `messages`, "anthropic": `max_tokens`, "bridge": `max_tokens`, "responses": `input`}[format]
				if _, ok := records[1]["body"].(map[string]any)[shape]; !ok {
					t.Fatalf("upstream body is not the %s shape: %v", format, records[1]["body"])
				}
				// The second provider request carries the tool result the model saw.
				if !strings.Contains(fmt.Sprint(records[5]["body"]), "42") {
					t.Fatalf("round-two request lost the tool result: %v", records[5]["body"])
				}
				for _, secret := range []string{"provider-secret-key", "secret-tool-token", "agent-secret-value", "PRIVATE-CHAIN", "SIG"} {
					if bytes.Contains(raw, []byte(secret)) {
						t.Fatalf("%q recorded in journal", secret)
					}
				}
				if !bytes.Contains(raw, []byte(`"omitted":"hidden_reasoning"`)) {
					t.Fatal("hidden reasoning omission not marked")
				}
				if !bytes.Contains(raw, []byte("checking")) || !bytes.Contains(raw, []byte("price is 42")) {
					t.Fatal("model-visible text not recorded")
				}
				last := records[len(records)-1]
				if last["status"] != "ok" || last["complete"] != true {
					t.Fatalf("segment_close = %v", last)
				}
				if client := records[len(records)-2]; client["status_code"].(float64) != 200 || client["bytes_written"].(float64) <= 0 {
					t.Fatalf("client_response = %v", client)
				}
			})
		}
	}
}

func TestTraceJournalCapturesPassthroughStream(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"PRIVATE\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"hello stream\"}}]}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "provider-secret-key", Auth: "bearer"})
	journal, historyDir := newTestJournal(t, nil)
	h := NewHandler(reg, stubContextLoaderWithTools("worker-1", "worker-1:secret-value", nil), nil, WithTraceJournal(journal))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai/model","stream":true,"messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer worker-1:secret-value")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "hello stream") {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	records, raw := verifiedSegment(t, historyDir, "worker-1")
	if got := recordTypes(records); got != "segment_open,upstream_request,upstream_response,client_response,segment_close" {
		t.Fatalf("records = %s", got)
	}
	if records[2]["complete"] != true || !strings.Contains(records[2]["body"].(string), "hello stream") {
		t.Fatalf("upstream_response = %v", records[2])
	}
	if bytes.Contains(raw, []byte("PRIVATE")) {
		t.Fatal("streamed hidden reasoning recorded")
	}
}

func TestTraceJournalRecordsRoundZeroFailure(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, `{"error":{"message":"bad request for key provider-secret-key"}}`)
	}))
	defer backend.Close()
	manifest := managedToolManifestForURL("http://127.0.0.1:1", http.MethodPost, "/tool", "")
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "provider-secret-key", Auth: "bearer"})
	journal, historyDir := newTestJournal(t, nil)
	h := NewHandler(reg, stubContextLoaderWithTools("worker-1", "worker-1:secret-value", manifest), nil, WithTraceJournal(journal))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai/model","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer worker-1:secret-value")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code < 400 {
		t.Fatalf("status %d", w.Code)
	}
	records, raw := verifiedSegment(t, historyDir, "worker-1")
	if got := recordTypes(records); !strings.HasPrefix(got, "segment_open,upstream_request,upstream_response") || !strings.HasSuffix(got, "client_response,segment_close") {
		t.Fatalf("records = %s", got)
	}
	if records[2]["status_code"].(float64) != 400 {
		t.Fatalf("upstream_response = %v", records[2])
	}
	if records[len(records)-1]["status"] != "error" {
		t.Fatalf("segment_close = %v", records[len(records)-1])
	}
	if bytes.Contains(raw, []byte("provider-secret-key")) {
		t.Fatal("provider key echoed by upstream error was recorded")
	}
}

func TestTraceJournalFailsClosedBeforeDispatch(t *testing.T) {
	var hits atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "k", Auth: "bearer"})
	journal, historyDir := newTestJournal(t, nil)
	// A file where the journal directory belongs makes the segment unopenable.
	if err := os.MkdirAll(filepath.Join(historyDir, "worker-1"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(historyDir, "worker-1", "journal"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	h := NewHandler(reg, stubContextLoaderWithTools("worker-1", "worker-1:secret-value", nil), nil, WithTraceJournal(journal))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai/model","messages":[]}`))
	req.Header.Set("Authorization", "Bearer worker-1:secret-value")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	if hits.Load() != 0 {
		t.Fatal("provider was called without a journal segment")
	}
}

func TestTraceJournalWriteFailureStopsEffectsAndNeverCloses(t *testing.T) {
	var providerHits, toolHits atomic.Int32
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		toolHits.Add(1)
		fmt.Fprint(w, `{}`)
	}))
	defer toolSrv.Close()
	manifest := managedToolManifestForURL(toolSrv.URL, http.MethodPost, "/tool", "")
	name := managedToolHashlessAliasForCanonical(manifest.Tools[0].Name)
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		providerHits.Add(1)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"id":"r1","choices":[{"message":{"tool_calls":[{"id":"call-1","type":"function","function":{"name":%q,"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`, name)
	}))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "k", Auth: "bearer"})
	// Records 1-3 (open, upstream request, upstream response) succeed; the
	// tool_call record then fails to sync, so the tool must never run.
	var syncs atomic.Int32
	journal, historyDir := newTestJournal(t, func(cfg *tracejournal.Config) {
		cfg.SyncFile = func(f *os.File) error {
			if syncs.Add(1) > 3 {
				return errors.New("disk failed")
			}
			return f.Sync()
		}
	})
	h := NewHandler(reg, stubContextLoaderWithTools("worker-1", "worker-1:secret-value", manifest), nil, WithTraceJournal(journal))
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai/model","messages":[{"role":"user","content":"go"}]}`))
	req.Header.Set("Authorization", "Bearer worker-1:secret-value")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("status %d: %s", w.Code, w.Body.String())
	}
	if toolHits.Load() != 0 {
		t.Fatal("tool ran after its record failed")
	}
	if providerHits.Load() != 1 {
		t.Fatalf("provider hits = %d after journal failure", providerHits.Load())
	}
	if closed := journalSegments(t, historyDir, "worker-1", tracejournal.SuffixClosed); len(closed) != 0 {
		t.Fatalf("failed segment published as closed: %v", closed)
	}
	incomplete := journalSegments(t, historyDir, "worker-1", tracejournal.SuffixIncomplete)
	if len(incomplete) != 1 {
		t.Fatalf("incomplete segments = %v", incomplete)
	}
	if raw, _ := os.ReadFile(incomplete[0]); bytes.Contains(raw, []byte("segment_close")) {
		t.Fatal("failed segment carries a close record")
	}
}

func TestTraceJournalRevocationWaitsForRegisteredHandler(t *testing.T) {
	var hits atomic.Int32
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { hits.Add(1) }))
	defer backend.Close()
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: backend.URL + "/v1", APIKey: "k", Auth: "bearer"})
	journal, historyDir := newTestJournal(t, nil)
	h := NewHandler(reg, stubContextLoaderWithTools("worker-1", "worker-1:secret-value", nil), nil, WithRevocationDir(t.TempDir()), WithAdminToken("operator"), WithTraceJournal(journal))

	// The handler is registered but held before it can open a segment.
	body := &revocationHeldBody{Reader: strings.NewReader(`{"model":"openai/model","messages":[]}`), entered: make(chan struct{}), release: make(chan struct{})}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Body = io.NopCloser(body)
	req.Header.Set("Authorization", "Bearer worker-1:secret-value")
	done := make(chan struct{})
	go func() { h.ServeHTTP(httptest.NewRecorder(), req); close(done) }()
	<-body.entered

	acked := make(chan int, 1)
	go func() { acked <- revokeForTest(h, "worker-1", 1, "operator").Code }()
	select {
	case code := <-acked:
		t.Fatalf("revocation acknowledged (%d) while a registered handler could still open a segment", code)
	case <-time.After(200 * time.Millisecond):
	}
	close(body.release)
	select {
	case code := <-acked:
		if code != http.StatusOK {
			t.Fatalf("revoke: %d", code)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("revocation never acknowledged after the handler exited")
	}
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("handler still running after acknowledgement")
	}
	before := len(journalSegments(t, historyDir, "worker-1", "*"))
	if open := journalSegments(t, historyDir, "worker-1", tracejournal.SuffixOpen); len(open) != 0 {
		t.Fatalf("open segment after acknowledgement: %v", open)
	}
	fresh := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai/model","messages":[]}`))
	fresh.Header.Set("Authorization", "Bearer worker-1:secret-value")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, fresh)
	if w.Code != http.StatusForbidden {
		t.Fatalf("fresh request after revoke got %d", w.Code)
	}
	if after := len(journalSegments(t, historyDir, "worker-1", "*")); after != before {
		t.Fatalf("segments changed after acknowledgement: %d -> %d", before, after)
	}
	if hits.Load() != 0 {
		t.Fatal("provider called by a revoked identity")
	}
}
