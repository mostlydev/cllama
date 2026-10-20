package proxy

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sync"
	"time"
)

var errAgentRevoked = errors.New("agent revoked")
var errStaleRevocation = errors.New("stale revocation generation")
var errRevocationCapacity = errors.New("revocation ledger capacity exceeded")

type revocationRecord struct {
	AgentID    string `json:"agent_id"`
	Generation uint64 `json:"generation"`
	State      string `json:"state"`
}
type revocationLedger struct {
	Version int                         `json:"version"`
	Agents  map[string]revocationRecord `json:"agents"`
}
type agentFence struct {
	requests map[uint64]context.CancelFunc
	active   int
	admitted uint64
	refused  uint64
	drained  chan struct{}
}

// revocationFence is owned by one proxy process. Its durable directory must be
// retained across restarts, and must never be shared by concurrent proxy writers.
// Deleting that trusted-host storage cannot preserve a historical fence.
type revocationFence struct {
	mu       sync.Mutex
	dir      string
	ledger   revocationLedger
	disk     os.FileInfo
	failed   error
	dirty    bool
	maxBytes int
	agents   map[string]*agentFence
	next     uint64
}

func newRevocationFence(dir string) *revocationFence {
	f := &revocationFence{dir: dir, maxBytes: 16 << 20, ledger: revocationLedger{Version: 1, Agents: map[string]revocationRecord{}}, agents: map[string]*agentFence{}}
	if dir == "" {
		return f
	}
	if err := os.MkdirAll(dir, 0700); err != nil {
		f.failed = err
		return f
	}
	raw, err := os.ReadFile(filepath.Join(dir, "ledger.json"))
	if os.IsNotExist(err) {
		if _, markerErr := os.Stat(filepath.Join(dir, "initialized")); !os.IsNotExist(markerErr) {
			f.failed = errors.New("initialized revocation ledger missing")
			return f
		}
		f.failed = f.persistLocked()
		if f.failed == nil {
			f.failed = f.markInitialized()
		}
		return f
	}
	if err != nil {
		f.failed = err
		return f
	}
	if len(raw) > 16<<20 {
		f.failed = errors.New("revocation ledger too large")
		return f
	}
	var ledger revocationLedger
	if err = json.Unmarshal(raw, &ledger); err != nil || ledger.Version != 1 || ledger.Agents == nil {
		f.failed = errors.New("invalid revocation ledger")
		return f
	}
	for id, r := range ledger.Agents {
		if !safeBudgetAgentID(id) || r.AgentID != id || r.Generation == 0 || r.State != "revoked" {
			f.failed = errors.New("invalid revocation record")
			return f
		}
	}
	f.ledger = ledger
	f.disk, f.failed = os.Stat(filepath.Join(dir, "ledger.json"))
	if f.failed != nil {
		return f
	}
	f.failed = f.markInitialized()
	return f
}

// The marker distinguishes a fresh mounted directory from a deleted ledger.
// Loss of the entire trusted storage directory remains an operator recovery event.
func (f *revocationFence) markInitialized() error {
	path := filepath.Join(f.dir, "initialized")
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY, 0600)
	if err != nil {
		return err
	}
	err = file.Sync()
	_ = file.Close()
	if err != nil {
		return err
	}
	dir, err := os.Open(f.dir)
	if err != nil {
		return err
	}
	defer dir.Close()
	return dir.Sync()
}

func (f *revocationFence) stateLocked(id string) *agentFence {
	s := f.agents[id]
	if s == nil {
		d := make(chan struct{})
		close(d)
		s = &agentFence{requests: map[uint64]context.CancelFunc{}, drained: d}
		f.agents[id] = s
	}
	return s
}
func (f *revocationFence) failLocked(err error) error {
	f.failed = err
	for _, s := range f.agents {
		for _, cancel := range s.requests {
			cancel()
		}
	}
	return err
}
func (f *revocationFence) checkLocked() error {
	if f.failed != nil {
		return f.failed
	}
	if f.dir == "" {
		return nil
	}
	info, err := os.Stat(filepath.Join(f.dir, "ledger.json"))
	if err != nil {
		return f.failLocked(err)
	}
	if f.disk == nil || !os.SameFile(info, f.disk) || info.Size() != f.disk.Size() || !info.ModTime().Equal(f.disk.ModTime()) {
		return f.failLocked(errors.New("revocation ledger changed outside proxy"))
	}
	return nil
}
func (f *revocationFence) register(parent context.Context, id string) (context.Context, func(), error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.checkLocked(); err != nil {
		return nil, nil, err
	}
	if _, ok := f.ledger.Agents[id]; ok {
		return nil, nil, errAgentRevoked
	}
	ctx, cancel := context.WithCancel(parent)
	f.next++
	n := f.next
	s := f.stateLocked(id)
	s.requests[n] = cancel
	return ctx, func() { cancel(); f.mu.Lock(); delete(s.requests, n); f.mu.Unlock() }, nil
}
func (f *revocationFence) admit(ctx context.Context, id string) (func(), error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.failed != nil {
		return nil, f.failed
	}
	if _, ok := f.ledger.Agents[id]; ok {
		f.stateLocked(id).refused++
		return nil, errAgentRevoked
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	s := f.stateLocked(id)
	if s.active == 0 {
		s.drained = make(chan struct{})
	}
	s.active++
	s.admitted++
	return func() {
		f.mu.Lock()
		s.active--
		if s.active == 0 {
			close(s.drained)
		}
		f.mu.Unlock()
	}, nil
}
func (f *revocationFence) revoke(ctx context.Context, id string, generation uint64) (revocationRecord, error) {
	f.mu.Lock()
	if f.dir == "" {
		f.mu.Unlock()
		return revocationRecord{}, errors.New("durable revocation storage not configured")
	}
	if err := f.checkLocked(); err != nil {
		f.mu.Unlock()
		return revocationRecord{}, err
	}
	old, ok := f.ledger.Agents[id]
	if ok && generation < old.Generation {
		f.mu.Unlock()
		return old, errStaleRevocation
	}
	record := revocationRecord{AgentID: id, Generation: generation, State: "revoked"}
	f.ledger.Agents[id] = record
	if !ok || generation != old.Generation {
		f.dirty = true
	}
	s := f.stateLocked(id)
	for _, cancel := range s.requests {
		cancel()
	}
	if f.dirty {
		if err := f.persistLocked(); err != nil {
			if !errors.Is(err, errRevocationCapacity) {
				f.failLocked(err)
			}
			f.mu.Unlock()
			return revocationRecord{}, err
		}
	}
	drained := s.drained
	f.mu.Unlock()
	select {
	case <-drained:
		return record, nil
	case <-ctx.Done():
		return revocationRecord{}, ctx.Err()
	}
}
func (f *revocationFence) persistLocked() error {
	raw, err := json.Marshal(f.ledger)
	if err != nil {
		return err
	}
	if len(raw) > f.maxBytes {
		return errRevocationCapacity
	}
	tmp, err := os.CreateTemp(f.dir, ".ledger-*")
	if err != nil {
		return err
	}
	name := tmp.Name()
	defer os.Remove(name)
	if _, err = tmp.Write(raw); err == nil {
		err = tmp.Sync()
	}
	closeErr := tmp.Close()
	if err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	if err = os.Rename(name, filepath.Join(f.dir, "ledger.json")); err != nil {
		return err
	}
	d, err := os.Open(f.dir)
	if err != nil {
		return err
	}
	err = d.Sync()
	_ = d.Close()
	if err != nil {
		return err
	}
	f.disk, err = os.Stat(filepath.Join(f.dir, "ledger.json"))
	if err == nil {
		f.dirty = false
	}
	return err
}

// WithRevocationDir enables durable, irreversible per-agent revocation. The
// directory is proxy-owned writable storage, distinct from governance overrides.
func WithRevocationDir(dir string) HandlerOption {
	return func(h *Handler) { h.revocations = newRevocationFence(dir) }
}

// HandleRevoke never authenticates worker tokens. A successful response proves
// durable closure and drainage of local dispatches, not rollback of remote effects.
func (h *Handler) HandleRevoke(w http.ResponseWriter, r *http.Request) {
	if h.adminToken == "" || !constantTimeEqual(extractPresentedToken(r), h.adminToken) {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}
	id := r.PathValue("agentID")
	if !safeBudgetAgentID(id) {
		http.Error(w, "invalid agent ID", http.StatusBadRequest)
		return
	}
	var body struct {
		Generation uint64 `json:"generation"`
	}
	dec := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4096))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&body); err != nil || body.Generation == 0 {
		http.Error(w, "invalid revocation request", http.StatusBadRequest)
		return
	}
	if err := dec.Decode(&struct{}{}); err != io.EOF {
		http.Error(w, "invalid revocation request", http.StatusBadRequest)
		return
	}
	revokeCtx, cancel := context.WithTimeout(r.Context(), 30*time.Second)
	defer cancel()
	record, err := h.revocations.revoke(revokeCtx, id, body.Generation)
	if err != nil {
		status := http.StatusServiceUnavailable
		if errors.Is(err, errStaleRevocation) {
			status = http.StatusConflict
		}
		reason := "revocation_failed"
		if errors.Is(err, errStaleRevocation) {
			reason = "revocation_stale"
		}
		h.logger.LogIntervention(id, "", reason)
		if !errors.Is(err, errStaleRevocation) {
			h.logger.LogError(id, "", status, 0, fmt.Errorf("revocation storage or drain unavailable at %s", h.revocations.dir))
		}
		http.Error(w, "revocation not acknowledged", status)
		return
	}
	h.revocations.mu.Lock()
	state := h.revocations.stateLocked(id)
	admitted, refused := state.admitted, state.refused
	h.revocations.mu.Unlock()
	h.logger.LogRevocation(id, record.Generation, admitted, refused)
	h.clearRevokedSnapshot(id)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(record)
}

// clearRevokedSnapshot runs both after the fence drains and after each handler
// exits. Waiting for the last registered handler prevents a late context capture
// from restoring a revoked ephemeral identity's full prompt in the snapshot store.
func (h *Handler) clearRevokedSnapshot(id string) {
	h.revocations.mu.Lock()
	defer h.revocations.mu.Unlock()
	if _, revoked := h.revocations.ledger.Agents[id]; !revoked {
		return
	}
	if state := h.revocations.agents[id]; state != nil && len(state.requests) > 0 {
		return
	}
	h.snapshots.Delete(id)
}
