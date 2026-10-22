package tracejournal

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Coverage is the verdict on one run identity's journal. RequestsVerified
// means every admitted request is accounted for exactly once.
// ContentComplete additionally means every recorded request was captured
// in full. The two are reported separately: an unopened request or an
// incomplete segment is accounted for without being full capture.
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
		Schema       string     `json:"schema"`
		AgentID      string     `json:"agent_id"`
		Generation   uint64     `json:"generation"`
		CaptureEpoch string     `json:"capture_epoch"`
		Requests     *int       `json:"requests"`
		Unopened     []Unopened `json:"unopened"`
	}
	sealName := fmt.Sprintf("%s%d.json", sealFilePrefix, generation)
	raw, err := readRegular(filepath.Join(journalDir, sealName))
	if err != nil {
		fail("seal %s: %v", sealName, err)
		return c
	}
	if err := json.Unmarshal(raw, &seal); err != nil || seal.Requests == nil {
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
	n := *seal.Requests
	c.Requests = n
	if n < 0 {
		fail("seal requests %d is negative", n)
		return c
	}

	accounted := make(map[int]string, n)
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
		if !allowedUnopened[u.Reason] {
			fail("agent_seq %d has unknown unopened reason %q", u.AgentSeq, u.Reason)
			continue
		}
		account(u.AgentSeq, "unopened")
		if u.Reason == ReasonJournalOpenFailed {
			openFailed[u.AgentSeq] = true
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
	for k := 1; k <= n; k++ {
		if _, ok := accounted[k]; !ok {
			fail("missing agent_seq %d", k)
		}
	}
	return c
}

var errNoSegmentOpen = fmt.Errorf("no readable segment_open")

// verifyClosed checks a closed segment's chain and returns its agent_seq
// and complete flag.
func verifyClosed(path, agentID, epoch string) (int, bool, error) {
	raw, err := readRegular(path)
	if err != nil {
		return 0, false, err
	}
	var lines [][]byte
	sc := bufio.NewScanner(bytes.NewReader(raw))
	sc.Buffer(make([]byte, 1<<20), 1<<30)
	for sc.Scan() {
		lines = append(lines, append(append([]byte(nil), sc.Bytes()...), '\n'))
	}
	if err := sc.Err(); err != nil || len(lines) < 2 || !bytes.HasSuffix(raw, []byte("\n")) {
		return 0, false, fmt.Errorf("unreadable segment")
	}
	digest := sha256.New()
	var open, closeRec map[string]any
	for i, line := range lines {
		var rec map[string]any
		if err := json.Unmarshal(line, &rec); err != nil {
			return 0, false, fmt.Errorf("line %d is not JSON", i+1)
		}
		if seq, _ := rec["seq"].(float64); int(seq) != i+1 {
			return 0, false, fmt.Errorf("seq gap at line %d", i+1)
		}
		if i == 0 {
			open = rec
		}
		if i < len(lines)-1 {
			digest.Write(line)
		} else {
			closeRec = rec
		}
	}
	if closeRec["type"] != "segment_close" {
		return 0, false, fmt.Errorf("no close record")
	}
	if count, _ := closeRec["record_count"].(float64); int(count) != len(lines)-1 {
		return 0, false, fmt.Errorf("record_count mismatch")
	}
	if closeRec["sha256"] != hex.EncodeToString(digest.Sum(nil)) {
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
	if err := json.Unmarshal(first, &open); err != nil || open["type"] != "segment_open" {
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
	k, ok := open["agent_seq"].(float64)
	if !ok || k != float64(int(k)) {
		return 0, fmt.Errorf("segment has no agent_seq")
	}
	return int(k), nil
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
