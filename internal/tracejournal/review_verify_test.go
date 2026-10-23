package tracejournal

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestReviewVerifierUsesExactPersistedBytesAndIntegerCounts(t *testing.T) {
	for _, variant := range []string{"newline_changed", "fractional_count", "trailing_close_data", "trailing_seal_data"} {
		t.Run(variant, func(t *testing.T) {
			_, j, dir := runJournal(t, 1)
			paths, _ := filepath.Glob(filepath.Join(dir, "*"+SuffixClosed))
			raw, err := os.ReadFile(paths[0])
			if err != nil {
				t.Fatal(err)
			}
			switch variant {
			case "newline_changed":
				raw = bytes.ReplaceAll(raw, []byte("\n"), []byte("\r\n"))
			case "fractional_count":
				lines := bytes.Split(raw, []byte("\n"))
				var closeRec map[string]any
				if err := json.Unmarshal(lines[len(lines)-2], &closeRec); err != nil {
					t.Fatal(err)
				}
				closeRec["record_count"] = 1.5
				lines[len(lines)-2], _ = json.Marshal(closeRec)
				raw = bytes.Join(lines, []byte("\n"))
			case "trailing_close_data":
				raw = append(bytes.TrimSuffix(raw, []byte("\n")), []byte(" trailing-data\n")...)
			case "trailing_seal_data":
				paths[0] = filepath.Join(dir, "seal-1.json")
				raw, err = os.ReadFile(paths[0])
				if err != nil {
					t.Fatal(err)
				}
				raw = append(raw, []byte("trailing-data")...)
			}
			if err := os.WriteFile(paths[0], raw, 0600); err != nil {
				t.Fatal(err)
			}
			if c := VerifyCoverage(dir, "worker-1", 1, j.Epoch()); c.RequestsVerified {
				t.Fatalf("accepted altered or malformed segment: %+v", c)
			}
		})
	}
}
