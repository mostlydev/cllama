package tracejournal

import (
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestReviewInterruptedReasoningIsNotPersisted(t *testing.T) {
	for _, input := range []string{
		`{"choices":[{"delta":{"reasoning_content":"HIDDEN-INTERRUPTED`,
		"data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"HIDDEN-INTERRUPTED",
		"data: {\"choices\":\n" + "data: [{\"delta\":{\"reasoning_content\":\"HIDDEN-INTERRUPTED\"}}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\n" + "data: \"reasoning_content\":\"HIDDEN-INTERRUPTED\"}}]}\n\n",
		"data: {\"type\":\"response.reasoning_text.delta\",\"delta\":\"HIDDEN-INTERRUPTED\"}\n\n",
	} {
		out, _ := json.Marshal(ProjectBody([]byte(input)))
		if strings.Contains(string(out), "HIDDEN-INTERRUPTED") {
			t.Errorf("hidden reasoning survived invalid/partial/multiline framing: %s", out)
		}
	}
}

func TestReviewEscapedAndShortKnownSecretsAreNotPersisted(t *testing.T) {
	for _, secret := range []string{"short", "credential-<>&-123", "credential-\"-123"} {
		t.Run(secret, func(t *testing.T) {
			cfg := testConfig(t)
			j := openJournal(t, cfg)
			body, _ := json.Marshal(map[string]string{"content": secret})
			seg, err := j.Begin("worker-1", OpenInfo{Body: body, Secrets: []string{secret}})
			if err != nil {
				t.Fatal(err)
			}
			if err := seg.Close("ok"); err != nil {
				t.Fatal(err)
			}
			records := verifySegment(t, segmentFiles(t, cfg, "worker-1", SuffixClosed)[0])
			got := records[0]["client_request"].(map[string]any)["content"]
			if got == secret {
				t.Errorf("known secret retained after decoding JSON: %q", got)
			}
		})
	}
}

func TestReviewPendingUpstreamCannotPublishCompleteSegment(t *testing.T) {
	cfg := testConfig(t)
	j := openJournal(t, cfg)
	seg, err := j.Begin("worker-1", OpenInfo{})
	if err != nil {
		t.Fatal(err)
	}
	call, err := seg.BeginUpstream(UpstreamInfo{Provider: "openai"})
	if err != nil {
		t.Fatal(err)
	}
	if err := seg.Close("error"); err != nil {
		return
	}
	resp := &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(`{"choices":[]}`))}
	resp, _ = seg.FinishUpstream(call, time.Now(), resp, nil)
	_, _ = io.Copy(io.Discard, resp.Body)
	_ = resp.Body.Close()
	for _, path := range segmentFiles(t, cfg, "worker-1", SuffixClosed) {
		raw, _ := os.ReadFile(path)
		if strings.Contains(string(raw), `"complete":true`) {
			t.Fatalf("published complete segment with unmatched upstream request: %s", raw)
		}
	}
}

func TestReviewKnownSecretInsideEncodedToolArguments(t *testing.T) {
	cfg := testConfig(t)
	j := openJournal(t, cfg)
	secret := "credential-\"-123"
	args, _ := json.Marshal(map[string]string{"content": secret})
	body, _ := json.Marshal(map[string]any{"tool_calls": []any{map[string]any{"function": map[string]string{"name": "lookup", "arguments": string(args)}}}})
	seg, err := j.Begin("worker-1", OpenInfo{Body: body, Secrets: []string{secret}})
	if err != nil {
		t.Fatal(err)
	}
	if err := seg.Close("ok"); err != nil {
		t.Fatal(err)
	}
	records := verifySegment(t, segmentFiles(t, cfg, "worker-1", SuffixClosed)[0])
	request := records[0]["client_request"].(map[string]any)
	encodedArgs := request["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)["arguments"].(string)
	var recovered map[string]string
	if err := json.Unmarshal([]byte(encodedArgs), &recovered); err != nil {
		t.Fatal(err)
	}
	if recovered["content"] == secret {
		t.Fatalf("known secret survived JSON-encoded tool arguments: %q", recovered["content"])
	}
}

func TestReviewTrailingUnparseableDataIsNotLossless(t *testing.T) {
	_, lossless := projectBody([]byte(`{"visible":"first"} trailing-unparseable`), nil)
	if lossless {
		t.Fatal("unparseable trailing bytes were silently discarded while declaring lossless capture")
	}
}

type reviewReadDuringClose struct {
	entered, release chan struct{}
	once             sync.Once
}

func (b *reviewReadDuringClose) Read(p []byte) (int, error) {
	close(b.entered)
	<-b.release
	return copy(p, `{"content":"visible"}`), io.EOF
}
func (b *reviewReadDuringClose) Close() error { b.once.Do(func() { close(b.release) }); return nil }

func TestReviewCaptureReadAndCloseAreRaceSafe(t *testing.T) {
	cfg := testConfig(t)
	j := openJournal(t, cfg)
	seg, err := j.Begin("worker-1", OpenInfo{})
	if err != nil {
		t.Fatal(err)
	}
	defer seg.Close("error")
	for i := 0; i < 20; i++ {
		call, err := seg.BeginUpstream(UpstreamInfo{})
		if err != nil {
			t.Fatal(err)
		}
		inner := &reviewReadDuringClose{entered: make(chan struct{}), release: make(chan struct{})}
		resp, _ := seg.FinishUpstream(call, time.Now(), &http.Response{StatusCode: 200, Body: inner}, nil)
		done := make(chan struct{})
		go func() { _, _ = io.ReadAll(resp.Body); close(done) }()
		<-inner.entered
		_ = resp.Body.Close()
		<-done
	}
}
