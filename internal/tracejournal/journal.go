// Package tracejournal writes an opt-in, durable trace of every proxied
// request: one segment file per client HTTP request under
// <history-dir>/<agent-id>/journal/. Each record is fsynced before the step it
// describes proceeds, and a segment is published under its closed name only
// after a final record carrying its record count and SHA-256 digest. A
// segment cut off by a crash keeps its open name until the next startup
// renames it incomplete, so a partial trace can never pass as a complete one.
package tracejournal

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const (
	Schema = "cllama.trace/v1"

	EnvEnabled         = "CLLAMA_TRACE_JOURNAL"
	EnvOwner           = "CLLAMA_TRACE_OWNER"
	EnvMaxFieldBytes   = "CLLAMA_TRACE_MAX_FIELD_BYTES"
	EnvMaxSegmentBytes = "CLLAMA_TRACE_MAX_SEGMENT_BYTES"

	DefaultMaxFieldBytes   = int64(1 << 20)
	DefaultMaxSegmentBytes = int64(64 << 20)

	journalDirName = "journal"
	lockFileName   = ".trace-journal.lock"

	SuffixOpen       = ".open.jsonl"
	SuffixClosed     = ".closed.jsonl"
	SuffixIncomplete = ".incomplete.jsonl"
)

// Config controls a Journal. OwnerUID and OwnerGID are -1 when no owner is
// configured.
type Config struct {
	HistoryDir      string
	OwnerUID        int
	OwnerGID        int
	MaxFieldBytes   int64
	MaxSegmentBytes int64
	Version         string
	Revision        string
	Pod             string
	// SyncFile makes a record durable; nil uses (*os.File).Sync. Tests set
	// it to inject a failing disk.
	SyncFile func(*os.File) error
}

// ConfigFromEnv reads the journal settings. It reports enabled=false when
// CLLAMA_TRACE_JOURNAL is unset or off, and fails when the journal is enabled
// without a session history directory or with a malformed setting.
func ConfigFromEnv(historyDir, version, revision, pod string) (Config, bool, error) {
	cfg := Config{
		HistoryDir:      historyDir,
		OwnerUID:        -1,
		OwnerGID:        -1,
		MaxFieldBytes:   DefaultMaxFieldBytes,
		MaxSegmentBytes: DefaultMaxSegmentBytes,
		Version:         version,
		Revision:        revision,
		Pod:             pod,
	}
	switch strings.ToLower(strings.TrimSpace(os.Getenv(EnvEnabled))) {
	case "", "off", "false", "0":
		return cfg, false, nil
	case "on", "true", "1":
	default:
		return cfg, false, fmt.Errorf("%s must be on or off", EnvEnabled)
	}
	if strings.TrimSpace(historyDir) == "" {
		return cfg, false, fmt.Errorf("%s=on requires CLAW_SESSION_HISTORY_DIR", EnvEnabled)
	}
	if raw := strings.TrimSpace(os.Getenv(EnvOwner)); raw != "" {
		uidRaw, gidRaw, ok := strings.Cut(raw, ":")
		uid, uidErr := strconv.Atoi(uidRaw)
		gid, gidErr := strconv.Atoi(gidRaw)
		if !ok || uidErr != nil || gidErr != nil || uid < 0 || gid < 0 {
			return cfg, false, fmt.Errorf("%s must be <uid>:<gid>", EnvOwner)
		}
		cfg.OwnerUID, cfg.OwnerGID = uid, gid
	}
	var err error
	if cfg.MaxFieldBytes, err = positiveEnv(EnvMaxFieldBytes, DefaultMaxFieldBytes); err != nil {
		return cfg, false, err
	}
	if cfg.MaxSegmentBytes, err = positiveEnv(EnvMaxSegmentBytes, DefaultMaxSegmentBytes); err != nil {
		return cfg, false, err
	}
	return cfg, true, nil
}

func positiveEnv(name string, fallback int64) (int64, error) {
	raw := strings.TrimSpace(os.Getenv(name))
	if raw == "" {
		return fallback, nil
	}
	n, err := strconv.ParseInt(raw, 10, 64)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("%s must be a positive byte count", name)
	}
	return n, nil
}

// Journal is the single writer for one history directory.
type Journal struct {
	cfg  Config
	lock *os.File

	epoch string // capture_epoch: one per journal process lifetime

	mu         sync.Mutex
	prepared   map[string]bool // agent journal dirs already created and owned
	admissions map[string]*admissionState
}

// admissionState counts the requests the revocation fence admitted for one
// agent in this epoch, and those that never opened a segment.
type admissionState struct {
	requests int
	unopened []Unopened
}

// Unopened names an admitted request that never opened a segment. None of
// them caused a provider or tool effect.
type Unopened struct {
	AgentSeq int    `json:"agent_seq"`
	Reason   string `json:"reason"`
}

const (
	EpochFileName  = ".trace-journal-epoch.json"
	EpochSchema    = "cllama.trace.epoch/v1"
	SealSchema     = "cllama.trace.seal/v1"
	sealFilePrefix = "seal-"

	ReasonCancelledBeforeOpen = "cancelled_before_open"
	ReasonBodyReadFailed      = "body_read_failed"
	ReasonJournalOpenFailed   = "journal_open_failed"
)

// Open takes the exclusive writer lock, proves the configured owner can be
// applied, and renames every segment a previous process left open to its
// incomplete name. It fails rather than start a second writer.
func Open(cfg Config) (*Journal, error) {
	if err := os.MkdirAll(cfg.HistoryDir, 0o777); err != nil {
		return nil, fmt.Errorf("trace journal: create history dir: %w", err)
	}
	lock, err := os.OpenFile(filepath.Join(cfg.HistoryDir, lockFileName), os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, fmt.Errorf("trace journal: open lock: %w", err)
	}
	if err := syscall.Flock(int(lock.Fd()), syscall.LOCK_EX|syscall.LOCK_NB); err != nil {
		lock.Close()
		return nil, fmt.Errorf("trace journal: another writer holds %s: %w", lockFileName, err)
	}
	j := &Journal{cfg: cfg, lock: lock, prepared: make(map[string]bool), admissions: make(map[string]*admissionState)}
	if err := j.chown(lock); err != nil {
		j.Close()
		return nil, fmt.Errorf("trace journal: apply %s: %w", EnvOwner, err)
	}
	if err := j.checkTraversable(cfg.HistoryDir); err != nil {
		j.Close()
		return nil, err
	}
	if err := j.recover(); err != nil {
		j.Close()
		return nil, err
	}
	if j.epoch, err = newRequestID(); err != nil {
		j.Close()
		return nil, fmt.Errorf("trace journal: capture epoch: %w", err)
	}
	if err := j.writeDurableJSON(cfg.HistoryDir, EpochFileName, map[string]any{
		"schema":          EpochSchema,
		"capture_epoch":   j.epoch,
		"started_at":      time.Now().UTC().Format(time.RFC3339Nano),
		"cllama_version":  cfg.Version,
		"cllama_revision": cfg.Revision,
	}); err != nil {
		j.Close()
		return nil, fmt.Errorf("trace journal: write capture epoch: %w", err)
	}
	return j, nil
}

// Epoch is this journal process's capture_epoch.
func (j *Journal) Epoch() string {
	return j.epoch
}

// Admit assigns the next agent_seq to a request the revocation fence has
// just admitted, before its body is read. A request that then never opens a
// segment must be reported through Unopen.
func (j *Journal) Admit(agentID string) int {
	j.mu.Lock()
	defer j.mu.Unlock()
	st := j.admissions[agentID]
	if st == nil {
		st = &admissionState{}
		j.admissions[agentID] = st
	}
	st.requests++
	return st.requests
}

// Unopen records why an admitted request never opened a segment.
func (j *Journal) Unopen(agentID string, agentSeq int, reason string) {
	j.mu.Lock()
	defer j.mu.Unlock()
	if st := j.admissions[agentID]; st != nil {
		st.unopened = append(st.unopened, Unopened{AgentSeq: agentSeq, Reason: reason})
	}
}

// Seal durably writes seal-<generation>.json for an agent whose revocation
// has drained every handler, naming how many requests this epoch admitted
// and which of them never opened a segment.
func (j *Journal) Seal(agentID string, generation uint64) error {
	if !validAgentID(agentID) {
		return fmt.Errorf("trace journal: unusable agent id")
	}
	dir, err := j.prepareDir(agentID)
	if err != nil {
		return err
	}
	j.mu.Lock()
	requests := 0
	unopened := []Unopened{}
	if st := j.admissions[agentID]; st != nil {
		requests = st.requests
		unopened = append(unopened, st.unopened...)
	}
	j.mu.Unlock()
	return j.writeDurableJSON(dir, fmt.Sprintf("%s%d.json", sealFilePrefix, generation), map[string]any{
		"schema":        SealSchema,
		"agent_id":      agentID,
		"generation":    generation,
		"capture_epoch": j.epoch,
		"sealed_at":     time.Now().UTC().Format(time.RFC3339Nano),
		"requests":      requests,
		"unopened":      unopened,
	})
}

// writeDurableJSON publishes name in dir atomically: a temporary file is
// written, owned, fsynced and renamed, then the directory is fsynced.
func (j *Journal) writeDurableJSON(dir, name string, v any) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	tmp, err := os.CreateTemp(dir, ".tmp-"+name+"-*")
	if err != nil {
		return err
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if _, err := tmp.Write(append(raw, '\n')); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if err := j.chown(tmp); err != nil {
		tmp.Close()
		return err
	}
	if err := j.syncFile(tmp); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Rename(tmpPath, filepath.Join(dir, name)); err != nil {
		return err
	}
	return syncDir(dir)
}

// MaxFieldBytes is the largest body recorded verbatim.
func (j *Journal) MaxFieldBytes() int64 {
	return j.cfg.MaxFieldBytes
}

// Close releases the writer lock.
func (j *Journal) Close() error {
	if j == nil || j.lock == nil {
		return nil
	}
	err := j.lock.Close()
	j.lock = nil
	return err
}

func (j *Journal) recover() error {
	matches, err := filepath.Glob(filepath.Join(j.cfg.HistoryDir, "*", journalDirName, "*"+SuffixOpen))
	if err != nil {
		return fmt.Errorf("trace journal: scan open segments: %w", err)
	}
	dirs := make(map[string]bool)
	for _, path := range matches {
		info, err := os.Lstat(path)
		if err != nil || !info.Mode().IsRegular() {
			continue
		}
		target := strings.TrimSuffix(path, SuffixOpen) + SuffixIncomplete
		if err := os.Rename(path, target); err != nil {
			return fmt.Errorf("trace journal: recover %s: %w", filepath.Base(path), err)
		}
		if err := j.chownPath(target); err != nil {
			return fmt.Errorf("trace journal: own %s: %w", filepath.Base(target), err)
		}
		dirs[filepath.Dir(path)] = true
	}
	for dir := range dirs {
		// A journal left by an earlier owner must become reachable by the
		// configured collector too, not only its recovered files.
		if err := j.checkTraversable(filepath.Dir(dir)); err != nil {
			return err
		}
		if err := j.chownPath(dir); err != nil {
			return fmt.Errorf("trace journal: own %s: %w", dir, err)
		}
		if err := syncDir(dir); err != nil {
			return fmt.Errorf("trace journal: sync %s: %w", dir, err)
		}
	}
	return nil
}

// OpenInfo describes the client request that starts a segment.
type OpenInfo struct {
	Method         string
	Path           string
	RequestedModel string
	// AgentSeq is the admission sequence from Admit.
	AgentSeq int
	Stream   bool
	Body     []byte
	// Secrets are exact credential values redacted from every record.
	Secrets []string
}

// Begin creates and publishes a new open segment and durably writes its
// segment_open record. Any error means nothing may be dispatched.
func (j *Journal) Begin(agentID string, info OpenInfo) (*Segment, error) {
	if !validAgentID(agentID) {
		return nil, fmt.Errorf("trace journal: unusable agent id")
	}
	dir, err := j.prepareDir(agentID)
	if err != nil {
		return nil, err
	}
	requestID, err := newRequestID()
	if err != nil {
		return nil, fmt.Errorf("trace journal: request id: %w", err)
	}
	started := time.Now()
	name := fmt.Sprintf("%d-%s", started.UnixNano(), requestID)
	openPath := filepath.Join(dir, name+SuffixOpen)
	f, err := os.OpenFile(openPath, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if err != nil {
		return nil, fmt.Errorf("trace journal: create segment: %w", err)
	}
	if err := j.chown(f); err != nil {
		f.Close()
		return nil, fmt.Errorf("trace journal: own segment: %w", err)
	}
	if err := syncDir(dir); err != nil {
		f.Close()
		return nil, fmt.Errorf("trace journal: publish segment: %w", err)
	}
	seg := newSegment(j, f, dir, name, info.Secrets)
	body := seg.body(info.Body)
	fields := map[string]any{
		"schema":          Schema,
		"cllama_version":  j.cfg.Version,
		"cllama_revision": j.cfg.Revision,
		"pod":             j.cfg.Pod,
		"agent_id":        agentID,
		"request_id":      requestID,
		"capture_epoch":   j.epoch,
		"agent_seq":       info.AgentSeq,
		"method":          info.Method,
		"path":            info.Path,
		"requested_model": info.RequestedModel,
		"stream":          info.Stream,
		"client_request":  body,
	}
	if err := seg.append("segment_open", fields); err != nil {
		_ = seg.Close("")
		return nil, err
	}
	return seg, nil
}

func (j *Journal) prepareDir(agentID string) (string, error) {
	dir := filepath.Join(j.cfg.HistoryDir, agentID, journalDirName)
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.prepared[dir] {
		return dir, nil
	}
	agentDir := filepath.Dir(dir)
	if _, err := os.Stat(agentDir); errors.Is(err, os.ErrNotExist) {
		if err := os.Mkdir(agentDir, 0o755); err != nil && !errors.Is(err, os.ErrExist) {
			return "", fmt.Errorf("trace journal: create agent dir: %w", err)
		}
		if err := syncDir(j.cfg.HistoryDir); err != nil {
			return "", fmt.Errorf("trace journal: sync history dir: %w", err)
		}
	}
	if err := j.checkTraversable(agentDir); err != nil {
		return "", err
	}
	if err := os.Mkdir(dir, 0o700); err != nil && !errors.Is(err, os.ErrExist) {
		return "", fmt.Errorf("trace journal: create journal dir: %w", err)
	}
	info, err := os.Lstat(dir)
	if err != nil {
		return "", fmt.Errorf("trace journal: stat journal dir: %w", err)
	}
	if !info.IsDir() {
		return "", fmt.Errorf("trace journal: %s is not a directory", dir)
	}
	if err := j.chownPath(dir); err != nil {
		return "", fmt.Errorf("trace journal: own journal dir: %w", err)
	}
	if err := syncDir(agentDir); err != nil {
		return "", fmt.Errorf("trace journal: sync agent dir: %w", err)
	}
	j.prepared[dir] = true
	return dir, nil
}

// checkTraversable fails when the configured collector could not reach the
// journal through dir, so a misconfigured owner is caught before any request
// rather than discovered by a collector that silently sees nothing.
func (j *Journal) checkTraversable(dir string) error {
	if j.cfg.OwnerUID < 0 {
		return nil
	}
	info, err := os.Stat(dir)
	if err != nil {
		return fmt.Errorf("trace journal: stat %s: %w", dir, err)
	}
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return nil
	}
	perm := info.Mode().Perm()
	var bit os.FileMode
	switch {
	case int(st.Uid) == j.cfg.OwnerUID:
		bit = 0o100
	case int(st.Gid) == j.cfg.OwnerGID:
		bit = 0o010
	default:
		bit = 0o001
	}
	if perm&bit == 0 {
		return fmt.Errorf("trace journal: %s (mode %o) is not traversable by %s %d:%d", dir, perm, EnvOwner, j.cfg.OwnerUID, j.cfg.OwnerGID)
	}
	return nil
}

func (j *Journal) syncFile(f *os.File) error {
	if j.cfg.SyncFile != nil {
		return j.cfg.SyncFile(f)
	}
	return f.Sync()
}

func (j *Journal) chown(f *os.File) error {
	if j.cfg.OwnerUID < 0 {
		return nil
	}
	return f.Chown(j.cfg.OwnerUID, j.cfg.OwnerGID)
}

func (j *Journal) chownPath(path string) error {
	if j.cfg.OwnerUID < 0 {
		return nil
	}
	return os.Lchown(path, j.cfg.OwnerUID, j.cfg.OwnerGID)
}

func validAgentID(id string) bool {
	return id != "" && id != "." && id != ".." && !strings.ContainsAny(id, `/\`) && !strings.HasPrefix(id, ".")
}

func newRequestID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	h := hex.EncodeToString(b[:])
	return h[0:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:32], nil
}

func syncDir(dir string) error {
	d, err := os.Open(dir)
	if err != nil {
		return err
	}
	defer d.Close()
	return d.Sync()
}
