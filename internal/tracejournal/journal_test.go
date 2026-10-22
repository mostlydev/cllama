package tracejournal

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"
)

func testConfig(t *testing.T) Config {
	t.Helper()
	return Config{
		HistoryDir:      t.TempDir(),
		OwnerUID:        -1,
		OwnerGID:        -1,
		MaxFieldBytes:   DefaultMaxFieldBytes,
		MaxSegmentBytes: DefaultMaxSegmentBytes,
		Version:         "v-test",
		Revision:        "abc123",
		Pod:             "pod-a",
	}
}

func openJournal(t *testing.T, cfg Config) *Journal {
	t.Helper()
	j, err := Open(cfg)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	t.Cleanup(func() { j.Close() })
	return j
}

func segmentFiles(t *testing.T, cfg Config, agentID, suffix string) []string {
	t.Helper()
	matches, err := filepath.Glob(filepath.Join(cfg.HistoryDir, agentID, journalDirName, "*"+suffix))
	if err != nil {
		t.Fatal(err)
	}
	return matches
}

func readRecords(t *testing.T, path string) ([]map[string]any, [][]byte) {
	t.Helper()
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var records []map[string]any
	var lines [][]byte
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 1<<20), 1<<26)
	for sc.Scan() {
		line := append([]byte(nil), sc.Bytes()...)
		lines = append(lines, append(line, '\n'))
		var rec map[string]any
		if err := json.Unmarshal(line, &rec); err != nil {
			t.Fatalf("invalid record %q: %v", line, err)
		}
		records = append(records, rec)
	}
	return records, lines
}

// verifySegment applies the collector's acceptance check to a closed segment.
func verifySegment(t *testing.T, path string) []map[string]any {
	t.Helper()
	records, lines := readRecords(t, path)
	if len(records) < 2 {
		t.Fatalf("closed segment has %d records", len(records))
	}
	for i, rec := range records {
		if int(rec["seq"].(float64)) != i+1 {
			t.Fatalf("record %d has seq %v", i, rec["seq"])
		}
	}
	closeRec := records[len(records)-1]
	if closeRec["type"] != "segment_close" {
		t.Fatalf("last record is %v", closeRec["type"])
	}
	if int(closeRec["record_count"].(float64)) != len(records)-1 {
		t.Fatalf("record_count %v, want %d", closeRec["record_count"], len(records)-1)
	}
	h := sha256.New()
	for _, line := range lines[:len(lines)-1] {
		h.Write(line)
	}
	if got := hex.EncodeToString(h.Sum(nil)); got != closeRec["sha256"] {
		t.Fatalf("digest %s, close says %v", got, closeRec["sha256"])
	}
	return records
}

func TestSegmentLifecycleProducesVerifiableClosedSegment(t *testing.T) {
	cfg := testConfig(t)
	j := openJournal(t, cfg)

	seg, err := j.Begin("worker-1", OpenInfo{
		Method:         "POST",
		Path:           "/v1/chat/completions",
		RequestedModel: "openai/gpt-x",
		Body:           []byte(`{"model":"openai/gpt-x","messages":[{"role":"user","content":"hi"}]}`),
	})
	if err != nil {
		t.Fatalf("Begin: %v", err)
	}
	if open := segmentFiles(t, cfg, "worker-1", SuffixOpen); len(open) != 1 {
		t.Fatalf("open segments = %v", open)
	}
	call, err := seg.BeginUpstream(UpstreamInfo{Provider: "openai", Model: "gpt-x", Path: "/v1/chat/completions", Body: []byte(`{"model":"gpt-x"}`)})
	if err != nil {
		t.Fatal(err)
	}
	resp := &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"choices":[{"message":{"content":"hello"}}]}`))}
	resp, _ = seg.FinishUpstream(call, time.Now(), resp, nil)
	if _, err := io.ReadAll(resp.Body); err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if err := seg.ToolCall("call_1", "lookup", "svc", map[string]any{"q": "x"}); err != nil {
		t.Fatal(err)
	}
	seg.ToolResult("call_1", 200, time.Millisecond, []byte(`{"ok":true}`), nil)
	seg.ClientResponse(200, 5, []byte(`{"a":1}`), false, nil)
	if err := seg.Close("ok"); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if open := segmentFiles(t, cfg, "worker-1", SuffixOpen); len(open) != 0 {
		t.Fatalf("open segments remain: %v", open)
	}
	closed := segmentFiles(t, cfg, "worker-1", SuffixClosed)
	if len(closed) != 1 {
		t.Fatalf("closed segments = %v", closed)
	}
	records := verifySegment(t, closed[0])
	var types []string
	for _, rec := range records {
		types = append(types, rec["type"].(string))
	}
	want := "segment_open,upstream_request,upstream_response,tool_call,tool_result,client_response,segment_close"
	if got := strings.Join(types, ","); got != want {
		t.Fatalf("types = %s, want %s", got, want)
	}
	open := records[0]
	if open["schema"] != Schema || open["cllama_version"] != "v-test" || open["cllama_revision"] != "abc123" || open["agent_id"] != "worker-1" || open["pod"] != "pod-a" {
		t.Fatalf("segment_open = %v", open)
	}
	if id, _ := open["request_id"].(string); len(id) != 36 || !strings.Contains(filepath.Base(closed[0]), id) {
		t.Fatalf("request_id %q not in %s", id, closed[0])
	}
	if records[2]["call"] != records[1]["call"] || records[2]["complete"] != true {
		t.Fatalf("upstream_response = %v", records[2])
	}
	last := records[len(records)-1]
	if last["status"] != "ok" || last["complete"] != true || last["truncated"] != false {
		t.Fatalf("segment_close = %v", last)
	}
	info, err := os.Stat(filepath.Dir(closed[0]))
	if err != nil || info.Mode().Perm() != 0o700 {
		t.Fatalf("journal dir mode = %v, %v", info.Mode(), err)
	}
	if fi, _ := os.Stat(closed[0]); fi.Mode().Perm() != 0o600 {
		t.Fatalf("segment mode = %v", fi.Mode())
	}
}

func TestOpenRefusesSecondWriter(t *testing.T) {
	cfg := testConfig(t)
	openJournal(t, cfg)
	if j, err := Open(cfg); err == nil {
		j.Close()
		t.Fatal("second writer was allowed")
	}
}

func TestOpenRecoversOpenSegmentsAsIncompleteWithoutRewriting(t *testing.T) {
	cfg := testConfig(t)
	dir := filepath.Join(cfg.HistoryDir, "worker-1", journalDirName)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	// A crash between writing segment_close and the rename leaves a close
	// record inside an open file; the filename state still wins.
	content := []byte("{\"seq\":1,\"type\":\"segment_open\"}\n{\"seq\":2,\"type\":\"segment_close\"}\n")
	openPath := filepath.Join(dir, "1-00000000-0000-4000-8000-000000000000"+SuffixOpen)
	if err := os.WriteFile(openPath, content, 0o600); err != nil {
		t.Fatal(err)
	}
	openJournal(t, cfg)
	if _, err := os.Stat(openPath); !os.IsNotExist(err) {
		t.Fatalf("open segment still present: %v", err)
	}
	got, err := os.ReadFile(strings.TrimSuffix(openPath, SuffixOpen) + SuffixIncomplete)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(got, content) {
		t.Fatalf("recovery rewrote evidence: %q", got)
	}
}

func TestPoisonedSegmentNeverPublishesDigest(t *testing.T) {
	cfg := testConfig(t)
	j := openJournal(t, cfg)
	seg, err := j.Begin("worker-1", OpenInfo{Method: "POST", Path: "/v1/messages"})
	if err != nil {
		t.Fatal(err)
	}
	// Simulate a failed write or sync by closing the file under the segment.
	seg.f.Close()
	if _, err := seg.BeginUpstream(UpstreamInfo{Provider: "anthropic"}); err == nil {
		t.Fatal("BeginUpstream succeeded on a failed journal")
	}
	if seg.Err() == nil {
		t.Fatal("segment not poisoned")
	}
	if err := seg.ToolCall("c", "t", "s", nil); err == nil {
		t.Fatal("ToolCall succeeded on a failed journal")
	}
	if err := seg.Close("ok"); err == nil {
		t.Fatal("Close reported success for a poisoned segment")
	}
	if closed := segmentFiles(t, cfg, "worker-1", SuffixClosed); len(closed) != 0 {
		t.Fatalf("poisoned segment was published closed: %v", closed)
	}
	incomplete := segmentFiles(t, cfg, "worker-1", SuffixIncomplete)
	if len(incomplete) != 1 {
		t.Fatalf("incomplete segments = %v", incomplete)
	}
	records, _ := readRecords(t, incomplete[0])
	for _, rec := range records {
		if rec["type"] == "segment_close" {
			t.Fatal("poisoned segment carries a close record")
		}
	}
}

func TestBodiesOverLimitsAreMarkedAndTruncateTheSegment(t *testing.T) {
	cfg := testConfig(t)
	cfg.MaxFieldBytes = 64
	j := openJournal(t, cfg)
	big := `{"text":"` + strings.Repeat("x", 200) + `"}`
	seg, err := j.Begin("worker-1", OpenInfo{Method: "POST", Path: "/v1/chat/completions", Body: []byte(big)})
	if err != nil {
		t.Fatal(err)
	}
	call, _ := seg.BeginUpstream(UpstreamInfo{Body: []byte(`{"small":true}`)})
	resp := &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(big))}
	resp, _ = seg.FinishUpstream(call, time.Now(), resp, nil)
	_, _ = io.ReadAll(resp.Body)
	resp.Body.Close()
	if err := seg.Close("ok"); err != nil {
		t.Fatal(err)
	}
	records := verifySegment(t, segmentFiles(t, cfg, "worker-1", SuffixClosed)[0])
	marker, _ := records[0]["client_request"].(map[string]any)
	if marker["omitted"] != "too_large" || int(marker["bytes"].(float64)) != len(big) {
		t.Fatalf("client_request = %v", records[0]["client_request"])
	}
	if _, hashed := marker["sha256"]; hashed {
		t.Fatal("omitted body was hashed")
	}
	if body, _ := records[1]["body"].(map[string]any); body["small"] != true {
		t.Fatalf("small body not kept: %v", records[1]["body"])
	}
	if body, _ := records[2]["body"].(map[string]any); body["omitted"] != "too_large" {
		t.Fatalf("large upstream body not omitted: %v", records[2]["body"])
	}
	last := records[len(records)-1]
	if last["truncated"] != true || last["complete"] != false {
		t.Fatalf("segment_close = %v", last)
	}
}

func TestKnownSecretsAreRedactedEverywhere(t *testing.T) {
	cfg := testConfig(t)
	j := openJournal(t, cfg)
	const providerKey = "sk-provider-secret-123456"
	const toolToken = "tool-token-abcdefgh"
	seg, err := j.Begin("worker-1", OpenInfo{
		Method:  "POST",
		Path:    "/v1/chat/completions",
		Body:    []byte(`{"messages":[{"role":"user","content":"my key is ` + providerKey + `"}]}`),
		Secrets: []string{providerKey, toolToken, "short"},
	})
	if err != nil {
		t.Fatal(err)
	}
	_ = seg.ToolCall("c1", "t", "svc", map[string]any{"note": toolToken})
	seg.ToolResult("c1", 500, 0, []byte(`{"error":"bad `+toolToken+`"}`), nil)
	call, _ := seg.BeginUpstream(UpstreamInfo{})
	_, _ = seg.FinishUpstream(call, time.Now(), nil, &testError{"dial https://x?key=" + providerKey})
	if err := seg.Close("error"); err != nil {
		t.Fatal(err)
	}
	path := segmentFiles(t, cfg, "worker-1", SuffixClosed)[0]
	raw, _ := os.ReadFile(path)
	for _, secret := range []string{providerKey, toolToken} {
		if bytes.Contains(raw, []byte(secret)) {
			t.Fatalf("secret %q recorded", secret)
		}
	}
	if !bytes.Contains(raw, []byte(redactedCredential)) {
		t.Fatal("no redaction marker recorded")
	}
	verifySegment(t, path)
}

type testError struct{ s string }

func (e *testError) Error() string { return e.s }

func TestAgentIDsThatEscapeTheJournalAreRejected(t *testing.T) {
	j := openJournal(t, testConfig(t))
	for _, id := range []string{"", ".", "..", "../x", "a/b", ".hidden"} {
		if _, err := j.Begin(id, OpenInfo{}); err == nil {
			t.Fatalf("agent id %q accepted", id)
		}
	}
}

func TestConfigFromEnv(t *testing.T) {
	t.Setenv(EnvEnabled, "")
	if _, on, err := ConfigFromEnv("/h", "v", "r", "p"); on || err != nil {
		t.Fatalf("default: on=%v err=%v", on, err)
	}
	t.Setenv(EnvEnabled, "on")
	if _, _, err := ConfigFromEnv("", "v", "r", "p"); err == nil {
		t.Fatal("enabled without history dir")
	}
	t.Setenv(EnvOwner, "1000:1000")
	cfg, on, err := ConfigFromEnv("/h", "v", "r", "p")
	if err != nil || !on || cfg.OwnerUID != 1000 || cfg.OwnerGID != 1000 {
		t.Fatalf("owner: %+v on=%v err=%v", cfg, on, err)
	}
	t.Setenv(EnvOwner, "nobody")
	if _, _, err := ConfigFromEnv("/h", "v", "r", "p"); err == nil {
		t.Fatal("malformed owner accepted")
	}
	t.Setenv(EnvOwner, "")
	t.Setenv(EnvMaxFieldBytes, "0")
	if _, _, err := ConfigFromEnv("/h", "v", "r", "p"); err == nil {
		t.Fatal("zero field limit accepted")
	}
	t.Setenv(EnvEnabled, "maybe")
	if _, _, err := ConfigFromEnv("/h", "v", "r", "p"); err == nil {
		t.Fatal("unknown enable value accepted")
	}
}

func TestOwnerMustBeAbleToTraverseAncestors(t *testing.T) {
	cfg := testConfig(t)
	// The test process can always chown to itself, so it stands in for the
	// collector user.
	cfg.OwnerUID, cfg.OwnerGID = os.Getuid(), os.Getgid()
	j := openJournal(t, cfg)
	agentDir := filepath.Join(cfg.HistoryDir, "worker-1")
	if err := os.Mkdir(agentDir, 0o600); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(agentDir, 0o755) })
	if _, err := j.Begin("worker-1", OpenInfo{}); err == nil || !strings.Contains(err.Error(), "not traversable") {
		t.Fatalf("Begin through an untraversable agent dir: %v", err)
	}
	if err := os.Chmod(agentDir, 0o755); err != nil {
		t.Fatal(err)
	}
	seg, err := j.Begin("worker-1", OpenInfo{})
	if err != nil {
		t.Fatal(err)
	}
	if err := seg.Close("ok"); err != nil {
		t.Fatal(err)
	}
}

func TestRecoveredSegmentsAreOwnedAndReadableByCollector(t *testing.T) {
	cfg := testConfig(t)
	cfg.OwnerUID, cfg.OwnerGID = os.Getuid(), os.Getgid()
	dir := filepath.Join(cfg.HistoryDir, "worker-1", journalDirName)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		t.Fatal(err)
	}
	openPath := filepath.Join(dir, "1-00000000-0000-4000-8000-000000000000"+SuffixOpen)
	if err := os.WriteFile(openPath, []byte("{}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	openJournal(t, cfg)
	info, err := os.Stat(strings.TrimSuffix(openPath, SuffixOpen) + SuffixIncomplete)
	if err != nil {
		t.Fatal(err)
	}
	st := info.Sys().(*syscall.Stat_t)
	if int(st.Uid) != cfg.OwnerUID || info.Mode().Perm()&0o400 == 0 {
		t.Fatalf("recovered segment owner %d mode %v", st.Uid, info.Mode())
	}
}
