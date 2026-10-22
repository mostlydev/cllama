package tracejournal

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

var errSegmentClosed = errors.New("trace journal: segment already closed")

// Segment is the journal of one client request. All methods are safe on a
// nil *Segment, which records nothing, so callers need not special-case a
// disabled journal.
type Segment struct {
	j       *Journal
	dir     string
	name    string
	secrets []string

	mu        sync.Mutex
	f         *os.File
	seq       int
	digest    hash.Hash
	written   int64
	truncated bool
	calls     int
	pending   map[int]bool // provider attempts without a settled response
	err       error
	done      bool
}

func newSegment(j *Journal, f *os.File, dir, name string, secrets []string) *Segment {
	return &Segment{j: j, f: f, dir: dir, name: name, secrets: usableSecrets(secrets), digest: sha256.New(), pending: make(map[int]bool)}
}

// Err reports the write failure that poisoned this segment, if any. Once it
// is non-nil the request must not cause further effects.
func (s *Segment) Err() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.err
}

func (s *Segment) append(typ string, fields map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.appendLocked(typ, fields)
}

func (s *Segment) appendLocked(typ string, fields map[string]any) error {
	if s.err != nil {
		return s.err
	}
	if s.done {
		return errSegmentClosed
	}
	record := make(map[string]any, len(fields)+3)
	for k, v := range fields {
		record[k] = redactFields(v, s.secrets)
	}
	record["seq"] = s.seq + 1
	record["ts"] = time.Now().UTC().Format(time.RFC3339Nano)
	record["type"] = typ
	line, err := json.Marshal(record)
	if err != nil {
		s.err = fmt.Errorf("trace journal: encode %s: %w", typ, err)
		return s.err
	}
	line = append(redactSecrets(line, s.secrets), '\n')
	if _, err := s.f.Write(line); err != nil {
		s.err = fmt.Errorf("trace journal: write %s: %w", typ, err)
		return s.err
	}
	if err := s.j.syncFile(s.f); err != nil {
		s.err = fmt.Errorf("trace journal: sync %s: %w", typ, err)
		return s.err
	}
	s.seq++
	s.digest.Write(line)
	s.written += int64(len(line))
	return nil
}

// body projects a captured body for recording: hidden reasoning and
// credential-named fields are removed, and a body over the field limit, or
// one that would push the segment past its limit, is replaced by a size
// marker. The returned value is valid JSON.
func (s *Segment) body(raw []byte) any {
	if len(raw) == 0 {
		return nil
	}
	if int64(len(raw)) > s.j.cfg.MaxFieldBytes {
		return s.omitTooLarge(len(raw))
	}
	value, lossless := projectBody(raw, s.secrets)
	projected, err := json.Marshal(value)
	if err != nil {
		return s.omitTooLarge(len(raw))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		return nil
	}
	if s.written+int64(len(projected)) > s.j.cfg.MaxSegmentBytes {
		s.truncated = true
		return map[string]any{"omitted": "too_large", "bytes": len(raw)}
	}
	if !lossless {
		s.truncated = true
	}
	return json.RawMessage(projected)
}

func (s *Segment) value(v any) any {
	if v == nil {
		return nil
	}
	raw, err := json.Marshal(v)
	if err != nil {
		return map[string]any{"omitted": "unencodable"}
	}
	return s.body(raw)
}

func (s *Segment) omitTooLarge(n int) any {
	s.mu.Lock()
	if !s.done {
		s.truncated = true
	}
	s.mu.Unlock()
	return map[string]any{"omitted": "too_large", "bytes": n}
}

// UpstreamInfo describes one provider dispatch attempt.
type UpstreamInfo struct {
	Provider string
	Model    string
	Path     string
	Body     []byte
}

// BeginUpstream durably records a provider request before it is sent. The
// returned call number pairs it with its response record. An error means the
// request must not be sent.
func (s *Segment) BeginUpstream(info UpstreamInfo) (int, error) {
	if s == nil {
		return 0, nil
	}
	body := s.body(info.Body)
	s.mu.Lock()
	defer s.mu.Unlock()
	s.calls++
	call := s.calls
	s.pending[call] = true
	err := s.appendLocked("upstream_request", map[string]any{
		"call":     call,
		"provider": info.Provider,
		"model":    info.Model,
		"path":     info.Path,
		"body":     body,
	})
	return call, err
}

// FinishUpstream records a failed dispatch immediately, or wraps a response
// body so its upstream_response record is written when the body has been
// read to the end or closed. The capture is bounded while reading.
func (s *Segment) FinishUpstream(call int, started time.Time, resp *http.Response, err error) (*http.Response, error) {
	if s == nil {
		return resp, err
	}
	if err != nil || resp == nil || resp.Body == nil {
		msg := "no response"
		if err != nil {
			msg = err.Error()
		}
		s.settle(call, map[string]any{
			"call":       call,
			"latency_ms": time.Since(started).Milliseconds(),
			"error":      msg,
		})
		return resp, err
	}
	s.mu.Lock()
	live := s.pending[call] && !s.done
	s.mu.Unlock()
	if !live {
		// Already settled (for example by a first-byte timeout) or the
		// segment is closed: the late response is not recorded.
		return resp, err
	}
	resp.Body = &upstreamCapture{
		seg:        s,
		call:       call,
		started:    started,
		headerMS:   time.Since(started).Milliseconds(),
		statusCode: resp.StatusCode,
		inner:      resp.Body,
		limit:      int(s.j.cfg.MaxFieldBytes),
	}
	return resp, nil
}

type upstreamCapture struct {
	seg        *Segment
	call       int
	started    time.Time
	headerMS   int64
	statusCode int
	inner      io.ReadCloser
	limit      int

	// mu guards the capture state, never the blocking inner Read: a stream
	// reader may still be inside Read when another goroutine closes the
	// body. Once sealed, a late Read changes nothing.
	mu      sync.Mutex
	buf     []byte
	total   int
	readErr error
	sealed  bool
}

func (c *upstreamCapture) Read(p []byte) (int, error) {
	n, err := c.inner.Read(p)
	c.mu.Lock()
	if !c.sealed {
		if n > 0 {
			c.total += n
			if keep := c.limit + 1 - len(c.buf); keep > 0 {
				if keep > n {
					keep = n
				}
				c.buf = append(c.buf, p[:keep]...)
			}
		}
		if err != nil && !errors.Is(err, io.EOF) {
			c.readErr = err
		}
	}
	c.mu.Unlock()
	if err != nil {
		c.finish(errors.Is(err, io.EOF))
	}
	return n, err
}

func (c *upstreamCapture) Close() error {
	err := c.inner.Close()
	c.finish(false)
	return err
}

// finish seals the capture and settles its call exactly once, at the end of
// the body or at Close, whichever comes first.
func (c *upstreamCapture) finish(eof bool) {
	c.mu.Lock()
	if c.sealed {
		c.mu.Unlock()
		return
	}
	c.sealed = true
	buf, total, readErr := c.buf, c.total, c.readErr
	c.buf = nil
	c.mu.Unlock()

	var body any
	if total > len(buf) || total > c.limit {
		body = c.seg.omitTooLarge(total)
	} else {
		body = c.seg.body(buf)
	}
	fields := map[string]any{
		"call":        c.call,
		"status_code": c.statusCode,
		"latency_ms":  c.headerMS,
		"duration_ms": time.Since(c.started).Milliseconds(),
		"bytes":       total,
		"complete":    eof,
		"body":        body,
	}
	if readErr != nil {
		fields["error"] = readErr.Error()
	}
	c.seg.settle(c.call, fields)
}

// settle writes the one upstream_response for a pending call. It is a
// no-op once the call is settled or the segment is closed, so a late or
// duplicate outcome can neither append nor change the closed state.
func (s *Segment) settle(call int, fields map[string]any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if !s.pending[call] || s.done {
		return
	}
	delete(s.pending, call)
	_ = s.appendLocked("upstream_response", fields)
}

// ToolCall durably records a managed tool dispatch before it is executed. An
// error means the tool must not be executed.
func (s *Segment) ToolCall(callID, name, service string, args any) error {
	if s == nil {
		return nil
	}
	return s.append("tool_call", map[string]any{
		"call_id":   callID,
		"name":      name,
		"service":   service,
		"arguments": s.value(args),
	})
}

// ToolResult records what a managed tool dispatch returned.
func (s *Segment) ToolResult(callID string, statusCode int, latency time.Duration, raw []byte, err error) {
	if s == nil {
		return
	}
	fields := map[string]any{
		"call_id":     callID,
		"status_code": statusCode,
		"latency_ms":  latency.Milliseconds(),
		"result":      s.body(raw),
	}
	if err != nil {
		fields["error"] = err.Error()
	}
	_ = s.append("tool_result", fields)
}

// ClientResponse records what the proxy wrote to the client. It cannot
// certify what the client received.
func (s *Segment) ClientResponse(statusCode int, bytesWritten int64, captured []byte, overflow bool, writeErr error) {
	if s == nil {
		return
	}
	var body any
	if overflow {
		body = s.omitTooLarge(int(bytesWritten))
	} else {
		body = s.body(captured)
	}
	fields := map[string]any{
		"status_code":   statusCode,
		"bytes_written": bytesWritten,
		"body":          body,
	}
	if writeErr != nil {
		fields["write_error"] = writeErr.Error()
	}
	_ = s.append("client_response", fields)
}

// Close finishes the segment. A healthy segment gets its segment_close record
// and is renamed to its closed name; a poisoned one is left open so the next
// startup marks it incomplete, and no digest is ever published for it.
func (s *Segment) Close(status string) error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		return nil
	}
	if s.err != nil {
		// A poisoned segment never gets a close record or digest. Renaming
		// it incomplete now lets a collector finalize without a restart; if
		// the rename fails it stays open until the next startup recovers it.
		s.done = true
		_ = s.f.Close()
		dst := filepath.Join(s.dir, s.name+SuffixIncomplete)
		if os.Rename(filepath.Join(s.dir, s.name+SuffixOpen), dst) == nil {
			_ = s.j.chownPath(dst)
			_ = syncDir(s.dir)
		}
		return s.err
	}
	unsettled := make([]int, 0, len(s.pending))
	for call := range s.pending {
		unsettled = append(unsettled, call)
	}
	sort.Ints(unsettled)
	fields := map[string]any{
		"status":       status,
		"record_count": s.seq,
		"truncated":    s.truncated,
		"complete":     !s.truncated && len(unsettled) == 0,
		"sha256":       hex.EncodeToString(s.digest.Sum(nil)),
	}
	if len(unsettled) > 0 {
		fields["unsettled_calls"] = unsettled
	}
	err := s.appendLocked("segment_close", fields)
	s.done = true
	closeErr := s.f.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return fmt.Errorf("trace journal: close segment: %w", closeErr)
	}
	openPath := filepath.Join(s.dir, s.name+SuffixOpen)
	if err := os.Rename(openPath, filepath.Join(s.dir, s.name+SuffixClosed)); err != nil {
		return fmt.Errorf("trace journal: publish closed segment: %w", err)
	}
	if err := syncDir(s.dir); err != nil {
		return fmt.Errorf("trace journal: sync closed segment: %w", err)
	}
	return nil
}

type segmentKey struct{}
type toolCallKey struct{}

// WithSegment attaches a segment to a request context.
func WithSegment(ctx context.Context, s *Segment) context.Context {
	return context.WithValue(ctx, segmentKey{}, s)
}

// FromContext returns the request's segment, or nil when none is attached.
func FromContext(ctx context.Context) *Segment {
	if ctx == nil {
		return nil
	}
	s, _ := ctx.Value(segmentKey{}).(*Segment)
	return s
}

// WithToolCallID carries the model's tool-call id to the dispatch point.
func WithToolCallID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, toolCallKey{}, id)
}

// ToolCallID returns the model's tool-call id carried by ctx.
func ToolCallID(ctx context.Context) string {
	id, _ := ctx.Value(toolCallKey{}).(string)
	return id
}
