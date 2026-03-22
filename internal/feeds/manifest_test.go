package feeds

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadManifest(t *testing.T) {
	dir := t.TempDir()
	data := `[
		{"name":"market-context","source":"trading-api","path":"/api/v1/market_context/weston","ttl":180,"url":"http://trading-api:4000/api/v1/market_context/weston"},
		{"name":"fleet-alerts","source":"claw-api","path":"/fleet/alerts","ttl":30,"url":"http://claw-api:8080/fleet/alerts"}
	]`
	if err := os.WriteFile(filepath.Join(dir, "feeds.json"), []byte(data), 0o644); err != nil {
		t.Fatal(err)
	}

	entries, err := LoadManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(entries))
	}
	if entries[0].Name != "market-context" {
		t.Errorf("expected name market-context, got %q", entries[0].Name)
	}
	if entries[0].URL != "http://trading-api:4000/api/v1/market_context/weston" {
		t.Errorf("expected resolved URL, got %q", entries[0].URL)
	}
	if entries[1].TTL != 30 {
		t.Errorf("expected TTL 30, got %d", entries[1].TTL)
	}
}

func TestLoadManifestMissingFile(t *testing.T) {
	entries, err := LoadManifest(t.TempDir())
	if err != nil {
		t.Fatalf("missing feeds.json should not error, got: %v", err)
	}
	if len(entries) != 0 {
		t.Fatalf("expected empty entries, got %d", len(entries))
	}
}

func TestLoadManifestInvalidJSON(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "feeds.json"), []byte("not json"), 0o644); err != nil {
		t.Fatal(err)
	}

	_, err := LoadManifest(dir)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}
