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
	if ctx.Tools != nil {
		t.Fatalf("expected no tools manifest, got %+v", ctx.Tools)
	}
	if ctx.HasPolicy() {
		t.Fatal("expected no model policy")
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

func TestLoadParsesModelPolicyAccessors(t *testing.T) {
	dir := t.TempDir()
	agentDir := filepath.Join(dir, "logan")
	if err := os.MkdirAll(agentDir, 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "AGENTS.md"), []byte("# Contract"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "CLAWDAPUS.md"), []byte("# Infra"), 0o644); err != nil {
		t.Fatal(err)
	}
	meta := `{
		"token":"logan:secret",
		"model_policy":{
			"mode":"clamp",
			"allowed":[
				{"slot":"primary","ref":"xai/grok-4.1-fast"},
				{"slot":"fallback","ref":"anthropic/claude-haiku-4-5"},
				{"slot":"analysis","ref":"openai/gpt-4o-mini"}
			]
		}
	}`
	if err := os.WriteFile(filepath.Join(agentDir, "metadata.json"), []byte(meta), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx, err := Load(dir, "logan")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !ctx.HasPolicy() {
		t.Fatal("expected model policy to be present")
	}
	if ctx.DefaultModel() != "xai/grok-4.1-fast" {
		t.Fatalf("unexpected default model: %q", ctx.DefaultModel())
	}
	allowed := ctx.AllowedModelRefs()
	if len(allowed) != 3 {
		t.Fatalf("expected 3 allowed refs, got %d", len(allowed))
	}
	failover := ctx.FailoverRefs()
	if len(failover) != 2 {
		t.Fatalf("expected 2 failover refs, got %d", len(failover))
	}
	if failover[0] != "xai/grok-4.1-fast" || failover[1] != "anthropic/claude-haiku-4-5" {
		t.Fatalf("unexpected failover refs: %#v", failover)
	}
}

func TestLoadReadsServiceAuthEntries(t *testing.T) {
	dir := t.TempDir()
	agentDir := filepath.Join(dir, "tiverton")
	if err := os.MkdirAll(filepath.Join(agentDir, "service-auth"), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "AGENTS.md"), []byte("# Contract"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "CLAWDAPUS.md"), []byte("# Infra"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "metadata.json"), []byte(`{"token":"tiverton:secret"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "service-auth", "team-memory.json"), []byte(`{"service":"team-memory","auth_type":"bearer","token":"memory-token"}`), 0o600); err != nil {
		t.Fatal(err)
	}

	ctx, err := Load(dir, "tiverton")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ctx.ServiceAuth) != 1 {
		t.Fatalf("expected 1 service auth entry, got %+v", ctx.ServiceAuth)
	}
	if ctx.ServiceAuth[0].Service != "team-memory" || ctx.ServiceAuth[0].Token != "memory-token" {
		t.Fatalf("unexpected service auth entry: %+v", ctx.ServiceAuth[0])
	}
}

func TestLoadReadsMemoryManifest(t *testing.T) {
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
	if err := os.WriteFile(filepath.Join(agentDir, "metadata.json"), []byte(`{"token":"tiverton:secret"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(agentDir, "memory.json"), []byte(`{
		"version": 1,
		"service": "team-memory",
		"base_url": "http://team-memory:8080",
		"recall": {"path": "/recall", "timeout_ms": 300},
		"retain": {"path": "/retain"}
	}`), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx, err := Load(dir, "tiverton")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ctx.Memory == nil {
		t.Fatal("expected memory manifest")
	}
	if ctx.Memory.Service != "team-memory" || ctx.Memory.Recall == nil || ctx.Memory.Recall.TimeoutMS != 300 {
		t.Fatalf("unexpected memory manifest: %+v", ctx.Memory)
	}
}

func TestLoadReadsToolsManifest(t *testing.T) {
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
	if err := os.WriteFile(filepath.Join(agentDir, "metadata.json"), []byte(`{"token":"tiverton:secret"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	tools := `{
		"version": 1,
		"tools": [
			{
				"name": "trading-api.get_market_context",
				"description": "Retrieve market context",
				"inputSchema": {"type": "object"},
				"annotations": {"readOnly": true},
				"execution": {
					"transport": "http",
					"service": "trading-api",
					"base_url": "http://trading-api:4000",
					"method": "GET",
					"path": "/api/v1/market_context/{claw_id}",
					"auth": {"type": "bearer", "token": "tool-token"}
				}
			}
		],
		"policy": {
			"max_rounds": 8,
			"timeout_per_tool_ms": 30000,
			"total_timeout_ms": 120000
		}
	}`
	if err := os.WriteFile(filepath.Join(agentDir, "tools.json"), []byte(tools), 0o644); err != nil {
		t.Fatal(err)
	}

	ctx, err := Load(dir, "tiverton")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ctx.Tools == nil {
		t.Fatal("expected tools manifest")
	}
	if ctx.Tools.Version != 1 || len(ctx.Tools.Tools) != 1 {
		t.Fatalf("unexpected tools manifest header: %+v", ctx.Tools)
	}
	tool := ctx.Tools.Tools[0]
	if tool.Name != "trading-api.get_market_context" {
		t.Fatalf("unexpected tool name: %+v", tool)
	}
	if tool.Execution.Service != "trading-api" || tool.Execution.Auth == nil || tool.Execution.Auth.Token != "tool-token" {
		t.Fatalf("unexpected tool execution: %+v", tool.Execution)
	}
	if ctx.Tools.Policy.MaxRounds != 8 || ctx.Tools.Policy.TimeoutPerToolMS != 30000 || ctx.Tools.Policy.TotalTimeoutMS != 120000 {
		t.Fatalf("unexpected tool policy: %+v", ctx.Tools.Policy)
	}
}
