package proxy

import (
	"bytes"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/provider"
	"github.com/mostlydev/cllama/internal/tracejournal"
)

type reviewDelayedTransport struct{ entered, release, done chan struct{} }

func (t *reviewDelayedTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	close(t.entered)
	<-t.release
	defer close(t.done)
	return nil, errors.New("delayed transport failure")
}

func TestReviewStreamTimeoutCannotSealBeforeAttemptSettles(t *testing.T) {
	t.Setenv(EnvStreamFirstByteTimeoutMS, "20")
	reg := provider.NewRegistry("")
	reg.Set("openai", &provider.Provider{Name: "openai", BaseURL: "http://unused.invalid/v1", APIKey: "provider-secret-key", Auth: "bearer"})
	j, root := newTestJournal(t, nil)
	h := NewHandler(reg, stubContextLoaderWithTools("worker-1", "worker-1:secret-value", nil), nil, WithTraceJournal(j), WithRevocationDir(t.TempDir()), WithAdminToken("operator"))
	transport := &reviewDelayedTransport{make(chan struct{}), make(chan struct{}), make(chan struct{})}
	h.client = &http.Client{Transport: transport}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"openai/model","stream":true,"messages":[]}`))
	req.Header.Set("Authorization", "Bearer worker-1:secret-value")
	done := make(chan struct{})
	go func() { h.ServeHTTP(httptest.NewRecorder(), req); close(done) }()
	<-transport.entered
	t.Cleanup(func() { close(transport.release); <-transport.done })
	select {
	case <-done:
	case <-time.After(200 * time.Millisecond):
		return // Waiting for the attempt is an acceptable fix.
	}
	for _, path := range journalSegments(t, root, "worker-1", tracejournal.SuffixClosed) {
		raw, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if bytes.Contains(raw, []byte(`"complete":true`)) && !bytes.Contains(raw, []byte(`"type":"upstream_response"`)) {
			t.Errorf("closed complete trace has no outcome for dispatched attempt: %s", raw)
		}
	}
}
