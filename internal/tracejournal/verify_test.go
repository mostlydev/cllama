package tracejournal

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// runJournal writes n closed segments for worker-1 and seals generation 1.
func runJournal(t *testing.T, n int) (Config, *Journal, string) {
	t.Helper()
	cfg := testConfig(t)
	j := openJournal(t, cfg)
	for i := 0; i < n; i++ {
		seq := j.Admit("worker-1")
		seg, err := j.Begin("worker-1", OpenInfo{AgentSeq: seq})
		if err != nil {
			t.Fatal(err)
		}
		if err := seg.Close("ok"); err != nil {
			t.Fatal(err)
		}
	}
	if err := j.Seal("worker-1", 1); err != nil {
		t.Fatal(err)
	}
	return cfg, j, filepath.Join(cfg.HistoryDir, "worker-1", journalDirName)
}

func TestVerifyCoverageAcceptsAnAccountedRun(t *testing.T) {
	_, j, dir := runJournal(t, 3)
	c := VerifyCoverage(dir, "worker-1", 1, j.Epoch())
	if !c.RequestsVerified || !c.ContentComplete || c.Requests != 3 {
		t.Fatalf("coverage = %+v", c)
	}
	if c := VerifyCoverage(dir, "worker-1", 2, j.Epoch()); c.RequestsVerified {
		t.Fatal("a missing seal generation verified")
	}
	if c := VerifyCoverage(dir, "worker-2", 1, j.Epoch()); c.RequestsVerified {
		t.Fatal("another agent's journal verified")
	}
}

func TestVerifyCoverageRejectsTamperedOrStrayEvidence(t *testing.T) {
	cases := map[string]func(t *testing.T, dir string){
		"digest tampered": func(t *testing.T, dir string) {
			path, _ := filepath.Glob(filepath.Join(dir, "*"+SuffixClosed))
			raw, _ := os.ReadFile(path[0])
			os.WriteFile(path[0], []byte(strings.Replace(string(raw), `"stream":false`, `"stream":true`, 1)), 0o600)
		},
		"open segment left": func(t *testing.T, dir string) {
			os.WriteFile(filepath.Join(dir, "9-00000000-0000-4000-8000-000000000000"+SuffixOpen), []byte("{}\n"), 0o600)
		},
		"stray file": func(t *testing.T, dir string) {
			os.WriteFile(filepath.Join(dir, "notes.txt"), nil, 0o600)
		},
		"unexplained unreadable incomplete": func(t *testing.T, dir string) {
			os.WriteFile(filepath.Join(dir, "9-00000000-0000-4000-8000-000000000000"+SuffixIncomplete), []byte(`{"seq":1,"ty`), 0o600)
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			_, j, dir := runJournal(t, 2)
			mutate(t, dir)
			if c := VerifyCoverage(dir, "worker-1", 1, j.Epoch()); c.RequestsVerified {
				t.Fatalf("accepted: %+v", c)
			}
		})
	}
}

func TestVerifyCoverageExplainsAPartialFileOfAFailedOpen(t *testing.T) {
	cfg := testConfig(t)
	j := openJournal(t, cfg)
	seq := j.Admit("worker-1")
	j.Unopen("worker-1", seq, ReasonJournalOpenFailed)
	if err := j.Seal("worker-1", 1); err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(cfg.HistoryDir, "worker-1", journalDirName)
	// The failed segment_open left a torn line behind.
	os.WriteFile(filepath.Join(dir, "9-00000000-0000-4000-8000-000000000000"+SuffixIncomplete), []byte(`{"seq":1,"ty`), 0o600)
	c := VerifyCoverage(dir, "worker-1", 1, j.Epoch())
	if !c.RequestsVerified || c.ContentComplete {
		t.Fatalf("coverage = %+v", c)
	}
}
