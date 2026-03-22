package agentctx

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadReadsAllFiles(t *testing.T) {
	dir := t.TempDir()
	agentDir := filepath.Join(dir, "tiverton")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "AGENTS.md"), []byte("# Contract"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "CLAWDAPUS.md"), []byte("# Infra"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "metadata.json"), []byte(`{"service":"tiverton","pod":"ops","token":"tiverton:secret"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx, err := Load(dir, "tiverton")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(ctx.AgentsMD) != "# Contract" {
		t.Errorf("wrong AGENTS.md content: %q", ctx.AgentsMD)
	}
	if string(ctx.ClawdapusMD) != "# Infra" {
		t.Errorf("wrong CLAWDAPUS.md content: %q", ctx.ClawdapusMD)
	}
	if ctx.Metadata["service"] != "tiverton" {
		t.Errorf("wrong metadata: %v", ctx.Metadata)
	}
	if ctx.MetadataToken() != "tiverton:secret" {
		t.Errorf("wrong token: %q", ctx.MetadataToken())
	}
}

func TestLoadMissingDirErrors(t *testing.T) {
	_, err := Load("/nonexistent", "ghost")
	if err == nil {
		t.Error("expected error for missing dir")
	}
}

func TestAgentContextFeedsPath(t *testing.T) {
	dir := t.TempDir()
	agentDir := filepath.Join(dir, "weston")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "AGENTS.md"), []byte("# C"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "CLAWDAPUS.md"), []byte("# I"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "metadata.json"), []byte(`{"token":"weston:x"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "feeds.json"), []byte(`[{"name":"test"}]`), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx, err := Load(dir, "weston")
	if err != nil {
		t.Fatal(err)
	}

	expected := filepath.Join(agentDir, "feeds.json")
	if ctx.FeedsPath() != expected {
		t.Errorf("expected %q, got %q", expected, ctx.FeedsPath())
	}
}
