package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/feeds"
	"github.com/mostlydev/cllama/internal/tracejournal"
)

var errTraceJournalFailed = errors.New("trace journal unavailable")

// WithTraceJournal records every authenticated request into the trace
// journal. A nil journal leaves tracing off.
func WithTraceJournal(j *tracejournal.Journal) HandlerOption {
	return func(h *Handler) {
		h.journal = j
	}
}

// beginTrace opens the request's journal segment before any provider or tool
// effect. On failure the client has already been answered with 503 and the
// request must stop.
func (h *Handler) beginTrace(w http.ResponseWriter, r *http.Request, agentID, presentedSecret string, agentCtx *agentctx.AgentContext, start time.Time) (*tracejournal.Segment, *traceResponseWriter, *http.Request, bool) {
	if h.journal == nil {
		return nil, nil, r, true
	}
	// Reading the body can outlast a revocation. A cancelled request opens
	// no segment and causes no effect.
	body, err := io.ReadAll(r.Body)
	if err != nil {
		h.fail(w, http.StatusBadRequest, "failed to read request body", agentID, "", start, err)
		return nil, nil, r, false
	}
	_ = r.Body.Close()
	r.Body = io.NopCloser(bytes.NewReader(body))
	if err := r.Context().Err(); err != nil {
		h.fail(w, http.StatusServiceUnavailable, "request cancelled", agentID, "", start, err)
		return nil, nil, r, false
	}
	var payload map[string]any
	_ = json.Unmarshal(body, &payload)
	requestedModel, _ := payload["model"].(string)
	seg, err := h.journal.Begin(agentID, tracejournal.OpenInfo{
		Method:         r.Method,
		Path:           r.URL.Path,
		RequestedModel: strings.TrimSpace(requestedModel),
		Stream:         requestedStream(payload),
		Body:           body,
		Secrets:        h.traceSecrets(presentedSecret, agentCtx),
	})
	if err != nil {
		h.logger.LogError(agentID, "", http.StatusServiceUnavailable, time.Since(start).Milliseconds(), err)
		writeJSONError(w, http.StatusServiceUnavailable, "trace journal unavailable")
		return nil, nil, r, false
	}
	tw := &traceResponseWriter{ResponseWriter: w, seg: seg, limit: h.journalFieldLimit()}
	return seg, tw, r.WithContext(tracejournal.WithSegment(r.Context(), seg)), true
}

// finishTrace records what was written to the client and closes the segment.
func (h *Handler) finishTrace(seg *tracejournal.Segment, tw *traceResponseWriter, r *http.Request, agentID string) {
	if seg == nil {
		return
	}
	status := tw.status
	if status == 0 && tw.written > 0 {
		status = http.StatusOK
	}
	seg.ClientResponse(status, tw.written, tw.captured, tw.overflow, tw.writeErr)
	outcome := "ok"
	switch {
	case h.agentRevoked(agentID):
		outcome = "revoked"
	case tw.writeErr != nil:
		outcome = "client_write_failed"
	case r.Context().Err() != nil:
		outcome = "client_cancelled"
	case status == 0 || status >= 400:
		outcome = "error"
	}
	if err := seg.Close(outcome); err != nil {
		h.logger.LogError(agentID, "", 0, 0, err)
	}
}

func (h *Handler) journalFieldLimit() int {
	return int(h.journal.MaxFieldBytes())
}

func (h *Handler) agentRevoked(agentID string) bool {
	if h.revocations == nil {
		return false
	}
	h.revocations.mu.Lock()
	defer h.revocations.mu.Unlock()
	_, revoked := h.revocations.ledger.Agents[agentID]
	return revoked
}

// traceSecrets lists the credential values the proxy knows, so they are
// redacted wherever a body, tool result or error echoes them.
func (h *Handler) traceSecrets(presentedSecret string, agentCtx *agentctx.AgentContext) []string {
	secrets := []string{presentedSecret, h.adminToken, os.Getenv("CLLAMA_UI_TOKEN")}
	if h.registry != nil {
		for _, state := range h.registry.All() {
			for _, key := range state.Keys {
				secrets = append(secrets, key.Secret)
			}
		}
	}
	if agentCtx == nil {
		return secrets
	}
	token := strings.TrimSpace(agentCtx.MetadataToken())
	if strings.HasPrefix(strings.ToLower(token), "bearer ") {
		token = strings.TrimSpace(token[7:])
	}
	secrets = append(secrets, token)
	if _, secret, ok := strings.Cut(token, ":"); ok {
		secrets = append(secrets, secret)
	}
	for _, auth := range agentCtx.ServiceAuth {
		secrets = append(secrets, auth.Token)
	}
	if agentCtx.Tools != nil {
		for _, tool := range agentCtx.Tools.Tools {
			if tool.Execution.Auth != nil {
				secrets = append(secrets, tool.Execution.Auth.Token)
			}
		}
	}
	if agentCtx.Memory != nil && agentCtx.Memory.Auth != nil {
		secrets = append(secrets, agentCtx.Memory.Auth.Token)
	}
	if entries, err := feeds.LoadManifest(agentCtx.ContextDir); err == nil {
		for _, entry := range entries {
			secrets = append(secrets, entry.Auth)
		}
	}
	return secrets
}

// tracedDo sends one provider request, journaling it first. A journal
// failure stops the dispatch before anything reaches the provider.
func (h *Handler) tracedDo(req *http.Request, info tracejournal.UpstreamInfo) (*http.Response, error) {
	seg := tracejournal.FromContext(req.Context())
	call, err := seg.BeginUpstream(info)
	if err != nil {
		return nil, err
	}
	started := time.Now()
	resp, err := h.client.Do(req)
	return seg.FinishUpstream(call, started, resp, err)
}

func upstreamTraceInfo(candidate dispatchCandidate, upstream upstreamEncoding) tracejournal.UpstreamInfo {
	return tracejournal.UpstreamInfo{
		Provider: candidate.ProviderName,
		Model:    candidate.UpstreamModel,
		Path:     upstream.Path,
		Body:     upstream.Body,
	}
}

// traceResponseWriter captures what the proxy writes to the client, bounded
// while writing. Once the segment is poisoned it refuses further output: an
// unanswered request gets a 503, and a response already under way is cut.
type traceResponseWriter struct {
	http.ResponseWriter
	seg   *tracejournal.Segment
	limit int

	status      int
	written     int64
	captured    []byte
	overflow    bool
	writeErr    error
	headersSent bool
	refused     bool
}

func (t *traceResponseWriter) WriteHeader(status int) {
	if t.refuse() {
		return
	}
	if t.headersSent {
		return
	}
	t.headersSent = true
	t.status = status
	t.ResponseWriter.WriteHeader(status)
}

func (t *traceResponseWriter) Write(p []byte) (int, error) {
	if t.refuse() {
		return 0, errTraceJournalFailed
	}
	if !t.headersSent {
		t.headersSent = true
		t.status = http.StatusOK
	}
	n, err := t.ResponseWriter.Write(p)
	t.written += int64(n)
	if keep := t.limit - len(t.captured); keep > 0 {
		if keep > n {
			keep = n
		}
		t.captured = append(t.captured, p[:keep]...)
	}
	if int64(len(t.captured)) < t.written {
		t.overflow = true
	}
	if err != nil && t.writeErr == nil {
		t.writeErr = err
	}
	return n, err
}

func (t *traceResponseWriter) Flush() {
	if t.refused {
		return
	}
	if f, ok := t.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

func (t *traceResponseWriter) Unwrap() http.ResponseWriter {
	return t.ResponseWriter
}

// refuse reports whether output must stop because the journal failed. The
// first refusal answers an unanswered request with 503; a response whose
// headers are already out is aborted so the client sees a cut stream rather
// than an untraced complete one.
func (t *traceResponseWriter) refuse() bool {
	if t.refused {
		return true
	}
	if t.seg.Err() == nil {
		return false
	}
	t.refused = true
	if !t.headersSent {
		t.headersSent = true
		t.status = http.StatusServiceUnavailable
		writeJSONError(t.ResponseWriter, http.StatusServiceUnavailable, "trace journal unavailable")
		return true
	}
	panic(http.ErrAbortHandler)
}
