package tracejournal

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// Coverage is the verdict on one run identity's journal. RequestsVerified
// means every admitted request is accounted for exactly once, by a closed
// or readable incomplete segment or an unopened entry. ContentComplete
// additionally means every admitted request has a closed segment with
// complete:true. RequestsVerified alone never means full trace content.
type Coverage struct {
	RequestsVerified bool
	ContentComplete  bool
	Requests         int
	Problems         []string
}

var segmentName = regexp.MustCompile(`^[0-9]+-[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\.(open|closed|incomplete)\.jsonl$`)

var allowedUnopened = map[string]bool{
	ReasonCancelledBeforeOpen: true,
	ReasonBodyReadFailed:      true,
	ReasonJournalOpenFailed:   true,
}

// VerifyCoverage is the reference implementation of the operator rule. It
// applies the documented checks to the journal of a fresh, never reused
// agent identity, using the seal for generation and the capture_epoch the
// operator read before launching the run.
func VerifyCoverage(journalDir, agentID string, generation uint64, epoch string) (c Coverage) {
	c.ContentComplete = true
	fail := func(format string, args ...any) { c.Problems = append(c.Problems, fmt.Sprintf(format, args...)) }
	defer func() {
		sort.Strings(c.Problems)
		c.RequestsVerified = len(c.Problems) == 0
		if !c.RequestsVerified {
			c.ContentComplete = false
		}
	}()
	if epoch == "" {
		fail("no launch epoch")
		return c
	}

	var seal struct {
		Schema       string       `json:"schema"`
		AgentID      string       `json:"agent_id"`
		Generation   uint64       `json:"generation"`
		CaptureEpoch string       `json:"capture_epoch"`
		Requests     *json.Number `json:"requests"`
		Unopened     []struct {
			AgentSeq json.Number `json:"agent_seq"`
			Reason   string      `json:"reason"`
		} `json:"unopened"`
	}
	sealName := fmt.Sprintf("%s%d.json", sealFilePrefix, generation)
	raw, err := readRegular(filepath.Join(journalDir, sealName))
	if err != nil {
		fail("seal %s: %v", sealName, err)
		return c
	}
	if err := decodeStrict(raw, &seal); err != nil || seal.Requests == nil {
		fail("seal %s is malformed", sealName)
		return c
	}
	if seal.Schema != SealSchema || seal.AgentID != agentID || seal.Generation != generation {
		fail("seal %s does not name %s generation %d", sealName, agentID, generation)
		return c
	}
	if seal.CaptureEpoch != epoch {
		fail("epoch changed: seal %s, launch %s", seal.CaptureEpoch, epoch)
		return c
	}
	n, ok := exactInt(*seal.Requests)
	if !ok || n < 0 {
		fail("seal requests %q is not a non-negative integer", seal.Requests.String())
		return c
	}
	c.Requests = n

	// Sized by observed entries, never by the seal's count, so a corrupt
	// count cannot drive allocation.
	accounted := map[int]string{}
	account := func(k int, how string) {
		if k < 1 || k > n {
			fail("agent_seq %d (%s) is outside 1..%d", k, how, n)
			return
		}
		if prior, ok := accounted[k]; ok {
			fail("agent_seq %d accounted twice (%s, %s)", k, prior, how)
			return
		}
		accounted[k] = how
	}
	openFailed := map[int]bool{}
	openFailedFiled := map[int]bool{}
	unreadableIncomplete := 0
	for _, u := range seal.Unopened {
		k, ok := exactInt(u.AgentSeq)
		if !ok {
			fail("unopened agent_seq %q is not an integer", u.AgentSeq.String())
			continue
		}
		if !allowedUnopened[u.Reason] {
			fail("agent_seq %d has unknown unopened reason %q", k, u.Reason)
			continue
		}
		// An unopened request caused no effect, but its body and any error
		// delivered to the client were not captured.
		c.ContentComplete = false
		account(k, "unopened")
		if u.Reason == ReasonJournalOpenFailed {
			openFailed[k] = true
		}
	}

	entries, err := os.ReadDir(journalDir)
	if err != nil {
		fail("read journal: %v", err)
		return c
	}
	for _, e := range entries {
		name := e.Name()
		switch {
		case strings.HasPrefix(name, sealFilePrefix) && strings.HasSuffix(name, ".json"):
			continue
		case strings.HasPrefix(name, ".tmp-"):
			continue // an unpublished seal write; never evidence
		case !segmentName.MatchString(name):
			fail("unexpected file %s", name)
			continue
		}
		if e.Type()&os.ModeType != 0 {
			fail("%s is not a regular file", name)
			continue
		}
		path := filepath.Join(journalDir, name)
		switch {
		case strings.HasSuffix(name, SuffixOpen):
			fail("open segment %s after the seal", name)
		case strings.HasSuffix(name, SuffixClosed):
			k, complete, err := verifyClosed(path, agentID, epoch)
			if err != nil {
				fail("%s: %v", name, err)
				continue
			}
			account(k, "closed")
			if !complete {
				c.ContentComplete = false
			}
		case strings.HasSuffix(name, SuffixIncomplete):
			c.ContentComplete = false
			k, err := incompleteSeq(path, agentID, epoch)
			if err == errNoSegmentOpen {
				// A segment_open write that failed part way leaves no
				// readable seq; it can only be one of the requests the
				// seal names as journal_open_failed.
				unreadableIncomplete++
				continue
			}
			if err != nil {
				fail("%s: %v", name, err)
				continue
			}
			if openFailed[k] && !openFailedFiled[k] {
				openFailedFiled[k] = true
				continue // the partial file of an already named request
			}
			account(k, "incomplete")
		}
	}
	if unreadableIncomplete > len(openFailed)-len(openFailedFiled) {
		fail("%d incomplete segments without a readable segment_open are not explained by journal_open_failed entries", unreadableIncomplete)
	}
	// accounted holds only distinct seqs in 1..N, so all are present
	// exactly when its size is N. Search for the first gap only up to the
	// observed count.
	if len(accounted) != n {
		for k := 1; k <= len(accounted)+1; k++ {
			if _, ok := accounted[k]; !ok {
				fail("missing agent_seq %d (%d of %d accounted)", k, len(accounted), n)
				break
			}
		}
	}
	return c
}

// decodeStrict decodes JSON keeping numbers as json.Number.
func decodeStrict(raw []byte, v any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(v); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		return fmt.Errorf("expected exactly one JSON value")
	}
	return nil
}

// exactInt accepts only an integral JSON number written as an integer.
func exactInt(n json.Number) (int, bool) {
	v, err := strconv.ParseInt(n.String(), 10, 64)
	if err != nil || v != int64(int(v)) {
		return 0, false
	}
	return int(v), true
}

func numberField(rec map[string]any, key string) (int, bool) {
	n, ok := rec[key].(json.Number)
	if !ok {
		return 0, false
	}
	return exactInt(n)
}

var errNoSegmentOpen = fmt.Errorf("no readable segment_open")

// verifyClosed checks a closed segment's chain and returns its agent_seq
// and complete flag.
func verifyClosed(path, agentID, epoch string) (int, bool, error) {
	raw, err := readRegular(path)
	if err != nil {
		return 0, false, err
	}
	// Records are split on the exact LF bytes the journal wrote; a CR or
	// any other change to the persisted bytes breaks the digest.
	if len(raw) == 0 || raw[len(raw)-1] != '\n' {
		return 0, false, fmt.Errorf("unterminated segment")
	}
	lines := bytes.SplitAfter(raw, []byte("\n"))
	lines = lines[:len(lines)-1] // SplitAfter leaves an empty tail
	if len(lines) < 2 {
		return 0, false, fmt.Errorf("segment has no close record")
	}
	var open, closeRec map[string]any
	for i, line := range lines {
		var rec map[string]any
		if err := decodeStrict(line, &rec); err != nil {
			return 0, false, fmt.Errorf("line %d is not JSON", i+1)
		}
		if seq, ok := numberField(rec, "seq"); !ok || seq != i+1 {
			return 0, false, fmt.Errorf("seq gap at line %d", i+1)
		}
		if i == 0 {
			open = rec
		}
		closeRec = rec
	}
	closeLine := lines[len(lines)-1]
	if closeRec["type"] != "segment_close" {
		return 0, false, fmt.Errorf("no close record")
	}
	if count, ok := numberField(closeRec, "record_count"); !ok || count != len(lines)-1 {
		return 0, false, fmt.Errorf("record_count mismatch")
	}
	sum := sha256.Sum256(raw[:len(raw)-len(closeLine)])
	if closeRec["sha256"] != hex.EncodeToString(sum[:]) {
		return 0, false, fmt.Errorf("digest mismatch")
	}
	k, err := openIdentity(open, agentID, epoch)
	if err != nil {
		return 0, false, err
	}
	complete, _ := closeRec["complete"].(bool)
	return k, complete, nil
}

// incompleteSeq reads the agent_seq of an incomplete segment from its
// segment_open record. A file without a valid one cannot be explained.
func incompleteSeq(path, agentID, epoch string) (int, error) {
	raw, err := readRegular(path)
	if err != nil {
		return 0, err
	}
	first, _, _ := bytes.Cut(raw, []byte("\n"))
	var open map[string]any
	if err := decodeStrict(first, &open); err != nil || open["type"] != "segment_open" {
		return 0, errNoSegmentOpen
	}
	return openIdentity(open, agentID, epoch)
}

func openIdentity(open map[string]any, agentID, epoch string) (int, error) {
	if open["type"] != "segment_open" || open["schema"] != Schema {
		return 0, fmt.Errorf("first record is not a %s segment_open", Schema)
	}
	if open["agent_id"] != agentID {
		return 0, fmt.Errorf("segment belongs to %v", open["agent_id"])
	}
	if open["capture_epoch"] != epoch {
		return 0, fmt.Errorf("segment from epoch %v", open["capture_epoch"])
	}
	k, ok := numberField(open, "agent_seq")
	if !ok {
		return 0, fmt.Errorf("segment has no integer agent_seq")
	}
	return k, nil
}

func readRegular(path string) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() {
		return nil, fmt.Errorf("not a regular file")
	}
	return os.ReadFile(path)
}
