package proxy

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/mostlydev/cllama/internal/tracejournal"
)

func TestReviewCoverageVerifierRejectsUnaccountedEvidence(t *testing.T) {
	for _, name := range []string{"invalid_unopened_reason", "duplicate_unopened", "stray_incomplete_empty_run"} {
		t.Run(name, func(t *testing.T) {
			journal, root := newTestJournal(t, nil)
			h := NewHandler(nil, stubContextLoaderWithTools("worker-1", "worker-1:secret-value", nil), nil, WithRevocationDir(t.TempDir()), WithAdminToken("operator"), WithTraceJournal(journal))
			if code := revokeForTest(h, "worker-1", 1, "operator").Code; code != 200 {
				t.Fatal(code)
			}
			sealPath := filepath.Join(root, "worker-1", "journal", "seal-1.json")
			seal := readJSONFile(t, sealPath)
			switch name {
			case "invalid_unopened_reason":
				seal["requests"] = 1
				seal["unopened"] = []any{map[string]any{"agent_seq": 1, "reason": "unknown"}}
			case "duplicate_unopened":
				seal["requests"] = 1
				entry := map[string]any{"agent_seq": 1, "reason": tracejournal.ReasonCancelledBeforeOpen}
				seal["unopened"] = []any{entry, entry}
			case "stray_incomplete_empty_run":
				if err := os.WriteFile(filepath.Join(root, "worker-1", "journal", "1-00000000-0000-4000-8000-000000000000.incomplete.jsonl"), []byte("{}\n"), 0600); err != nil {
					t.Fatal(err)
				}
			}
			raw, _ := json.Marshal(seal)
			if err := os.WriteFile(sealPath, raw, 0600); err != nil {
				t.Fatal(err)
			}
			if ok, why := coverageVerified(t, root, "worker-1", 1, journal.Epoch()); ok {
				t.Fatalf("inconsistent evidence was accepted as verified coverage (%s)", why)
			}
		})
	}
}
