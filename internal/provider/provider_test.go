package provider

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"
)

// -- helpers ------------------------------------------------------------------

func seedV2(t *testing.T, dir string, keys ...map[string]any) {
	t.Helper()
	v2 := map[string]any{
		"version": 2,
		"providers": map[string]any{
			"openai": map[string]any{
				"base_url":      "https://api.openai.com/v1",
				"auth":          "bearer",
				"api_format":    "openai",
				"active_key_id": keys[0]["id"],
				"keys":          keys,
			},
		},
	}
	data, err := json.MarshalIndent(v2, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "providers.json"), data, 0o600); err != nil {
		t.Fatal(err)
	}
}

func readyKey(id, label, secret string) map[string]any {
	return map[string]any{
		"id":                id,
		"label":             label,
		"secret":            secret,
		"source":            "seed",
		"state":             "ready",
		"cooldown_until":    "",
		"last_error_code":   0,
		"last_error_reason": "",
		"last_error_at":     "",
		"added_at":          "2026-03-23T12:00:00Z",
	}
}

// -- basic env loading --------------------------------------------------------

func TestLoadFromEnvSetsDefaultFields(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "sk-test-openai")
	t.Setenv("XAI_API_KEY", "xai-test")
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-test")

	r := NewRegistry("")
	r.LoadFromEnv()

	p, err := r.Get("openai")
	if err != nil {
		t.Fatalf("openai: %v", err)
	}
	if p.APIKey != "sk-test-openai" {
		t.Errorf("expected openai key, got %q", p.APIKey)
	}
	if p.BaseURL != "https://api.openai.com/v1" {
		t.Errorf("unexpected openai base URL: %q", p.BaseURL)
	}
	if p.Auth != "bearer" {
		t.Errorf("expected openai auth=bearer, got %q", p.Auth)
	}
	if p.APIFormat != "openai" {
		t.Errorf("expected openai api_format=openai, got %q", p.APIFormat)
	}

	p2, err := r.Get("anthropic")
	if err != nil {
		t.Fatalf("anthropic: %v", err)
	}
	if p2.APIKey != "sk-ant-test" {
		t.Errorf("expected anthropic key, got %q", p2.APIKey)
	}
	if p2.Auth != "x-api-key" {
		t.Errorf("expected anthropic auth=x-api-key, got %q", p2.Auth)
	}
	if p2.APIFormat != "anthropic" {
		t.Errorf("expected anthropic api_format=anthropic, got %q", p2.APIFormat)
	}

	px, err := r.Get("xai")
	if err != nil {
		t.Fatalf("xai: %v", err)
	}
	if px.APIKey != "xai-test" {
		t.Errorf("expected xai key, got %q", px.APIKey)
	}
	if px.BaseURL != "https://api.x.ai/v1" {
		t.Errorf("unexpected xai base URL: %q", px.BaseURL)
	}
	if px.Auth != "bearer" {
		t.Errorf("expected xai auth=bearer, got %q", px.Auth)
	}
	if px.APIFormat != "openai" {
		t.Errorf("expected xai api_format=openai, got %q", px.APIFormat)
	}
}

func TestLoadFromEnvSeedsXAIProvider(t *testing.T) {
	t.Setenv("XAI_API_KEY", "sk-xai-test")

	r := NewRegistry("")
	r.LoadFromEnv()

	p, err := r.Get("xai")
	if err != nil {
		t.Fatalf("xai: %v", err)
	}
	if p.APIKey != "sk-xai-test" {
		t.Errorf("expected xai key, got %q", p.APIKey)
	}
	if p.BaseURL != "https://api.x.ai/v1" {
		t.Errorf("unexpected xai base URL: %q", p.BaseURL)
	}
	if p.Auth != "bearer" {
		t.Errorf("expected xai auth=bearer, got %q", p.Auth)
	}
	if p.APIFormat != "openai" {
		t.Errorf("expected xai api_format=openai, got %q", p.APIFormat)
	}
}

func TestLoadFromEnvAppliesXAIBaseURLOverride(t *testing.T) {
	t.Setenv("XAI_API_KEY", "sk-xai-test")
	t.Setenv("XAI_BASE_URL", "https://proxy.example.test/v1")

	r := NewRegistry("")
	r.LoadFromEnv()

	p, err := r.Get("xai")
	if err != nil {
		t.Fatalf("xai: %v", err)
	}
	if p.BaseURL != "https://proxy.example.test/v1" {
		t.Errorf("expected xai base URL override, got %q", p.BaseURL)
	}
}

func TestLoadFromEnvSeedsVercelProvider(t *testing.T) {
	t.Setenv("AI_GATEWAY_API_KEY", "vck_test")

	r := NewRegistry("")
	r.LoadFromEnv()

	p, err := r.Get("vercel")
	if err != nil {
		t.Fatalf("vercel: %v", err)
	}
	if p.APIKey != "vck_test" {
		t.Fatalf("expected vercel key, got %q", p.APIKey)
	}
	if p.BaseURL != "https://ai-gateway.vercel.sh/v1" {
		t.Fatalf("unexpected vercel base URL: %q", p.BaseURL)
	}
	if p.Auth != "bearer" {
		t.Fatalf("expected vercel auth=bearer, got %q", p.Auth)
	}
	if p.APIFormat != "openai" {
		t.Fatalf("expected vercel api_format=openai, got %q", p.APIFormat)
	}

	state := r.All()["vercel"]
	if state == nil {
		t.Fatal("expected vercel provider state")
	}
	if state.ActiveKeyID != "seed:AI_GATEWAY_API_KEY" {
		t.Fatalf("active_key_id = %q, want seed:AI_GATEWAY_API_KEY", state.ActiveKeyID)
	}
	if len(state.Keys) != 1 || state.Keys[0].ID != "seed:AI_GATEWAY_API_KEY" {
		t.Fatalf("expected single vercel seed key, got %+v", state.Keys)
	}
}

func TestLoadFromEnvVercelKeyPool(t *testing.T) {
	t.Setenv("AI_GATEWAY_API_KEY", "vck_primary")
	t.Setenv("AI_GATEWAY_API_KEY_1", "vck_backup")

	r := NewRegistry("")
	r.LoadFromEnv()

	state := r.All()["vercel"]
	if state == nil {
		t.Fatal("expected vercel provider state")
	}
	if state.ActiveKeyID != "seed:AI_GATEWAY_API_KEY" {
		t.Fatalf("active_key_id = %q, want seed:AI_GATEWAY_API_KEY", state.ActiveKeyID)
	}
	if len(state.Keys) != 2 {
		t.Fatalf("expected 2 vercel keys, got %+v", state.Keys)
	}
	if state.Keys[0].ID != "seed:AI_GATEWAY_API_KEY" || state.Keys[0].Label != "primary" {
		t.Fatalf("unexpected primary key entry: %+v", state.Keys[0])
	}
	if state.Keys[1].ID != "seed:AI_GATEWAY_API_KEY_1" || state.Keys[1].Label != "backup-1" {
		t.Fatalf("unexpected backup key entry: %+v", state.Keys[1])
	}
}

func TestLoadFromEnvAppliesAIGatewayBaseURLOverride(t *testing.T) {
	t.Setenv("AI_GATEWAY_API_KEY", "vck_test")
	t.Setenv("AI_GATEWAY_BASE_URL", "https://gateway.example.test/v1")

	r := NewRegistry("")
	r.LoadFromEnv()

	p, err := r.Get("vercel")
	if err != nil {
		t.Fatalf("vercel: %v", err)
	}
	if p.BaseURL != "https://gateway.example.test/v1" {
		t.Fatalf("expected vercel base URL override, got %q", p.BaseURL)
	}
}

func TestLoadFromEnvSeedsGoogleProviderFromGeminiKey(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "sk-gemini-primary")

	r := NewRegistry("")
	r.LoadFromEnv()

	p, err := r.Get("google")
	if err != nil {
		t.Fatalf("google: %v", err)
	}
	if p.APIKey != "sk-gemini-primary" {
		t.Fatalf("expected google key from GEMINI_API_KEY, got %q", p.APIKey)
	}
	if p.BaseURL != "https://generativelanguage.googleapis.com/v1beta/openai" {
		t.Fatalf("unexpected google base URL: %q", p.BaseURL)
	}
	if p.Auth != "bearer" {
		t.Fatalf("expected google auth=bearer, got %q", p.Auth)
	}
	if p.APIFormat != "openai" {
		t.Fatalf("expected google api_format=openai, got %q", p.APIFormat)
	}

	state := r.All()["google"]
	if state == nil {
		t.Fatal("expected google provider state")
	}
	if state.ActiveKeyID != "seed:GEMINI_API_KEY" {
		t.Fatalf("active_key_id = %q, want seed:GEMINI_API_KEY", state.ActiveKeyID)
	}
	if len(state.Keys) != 1 || state.Keys[0].ID != "seed:GEMINI_API_KEY" {
		t.Fatalf("expected single google seed key, got %+v", state.Keys)
	}
}

func TestLoadFromEnvSeedsGoogleProviderFromGoogleAlias(t *testing.T) {
	t.Setenv("GOOGLE_API_KEY", "sk-google-alias")

	r := NewRegistry("")
	r.LoadFromEnv()

	p, err := r.Get("google")
	if err != nil {
		t.Fatalf("google: %v", err)
	}
	if p.APIKey != "sk-google-alias" {
		t.Fatalf("expected google key from GOOGLE_API_KEY, got %q", p.APIKey)
	}

	state := r.All()["google"]
	if state == nil {
		t.Fatal("expected google provider state")
	}
	if state.ActiveKeyID != "seed:GOOGLE_API_KEY" {
		t.Fatalf("active_key_id = %q, want seed:GOOGLE_API_KEY", state.ActiveKeyID)
	}
	if len(state.Keys) != 1 || state.Keys[0].ID != "seed:GOOGLE_API_KEY" {
		t.Fatalf("expected GOOGLE_API_KEY alias to seed google provider, got %+v", state.Keys)
	}
}

func TestLoadFromEnvPrefersGeminiKeyOverGoogleAlias(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "sk-gemini-primary")
	t.Setenv("GOOGLE_API_KEY", "sk-google-alias")

	r := NewRegistry("")
	r.LoadFromEnv()

	p, err := r.Get("google")
	if err != nil {
		t.Fatalf("google: %v", err)
	}
	if p.APIKey != "sk-gemini-primary" {
		t.Fatalf("expected GEMINI_API_KEY to win, got %q", p.APIKey)
	}

	state := r.All()["google"]
	if state == nil {
		t.Fatal("expected google provider state")
	}
	if state.ActiveKeyID != "seed:GEMINI_API_KEY" {
		t.Fatalf("active_key_id = %q, want seed:GEMINI_API_KEY", state.ActiveKeyID)
	}
	if len(state.Keys) != 2 {
		t.Fatalf("expected 2 google keys, got %+v", state.Keys)
	}
	if state.Keys[0].ID != "seed:GEMINI_API_KEY" || state.Keys[1].ID != "seed:GOOGLE_API_KEY" {
		t.Fatalf("expected GEMINI primary then GOOGLE alias ordering, got %+v", state.Keys)
	}
}

func TestLoadFromEnvAppliesGoogleBaseURLOverride(t *testing.T) {
	t.Setenv("GEMINI_API_KEY", "sk-gemini-primary")
	t.Setenv("GOOGLE_BASE_URL", "https://proxy.example.test/google")

	r := NewRegistry("")
	r.LoadFromEnv()

	p, err := r.Get("google")
	if err != nil {
		t.Fatalf("google: %v", err)
	}
	if p.BaseURL != "https://proxy.example.test/google" {
		t.Fatalf("expected google base URL override, got %q", p.BaseURL)
	}
}

func TestGetUnknownProviderErrors(t *testing.T) {
	r := NewRegistry("")
	_, err := r.Get("nonexistent")
	if err == nil {
		t.Error("expected error for unknown provider")
	}
}

// -- v1 backward compat -------------------------------------------------------

func TestLoadV1FromFile(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "providers.json"), []byte(`{
		"providers": {
			"ollama":     {"base_url": "http://ollama:11434/v1", "auth": "none"},
			"openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key": "sk-or-test"}
		}
	}`), 0o644)

	r := NewRegistry(dir)
	if err := r.LoadFromFile(); err != nil {
		t.Fatalf("load from file: %v", err)
	}

	p, err := r.Get("ollama")
	if err != nil {
		t.Fatalf("get ollama: %v", err)
	}
	if p.BaseURL != "http://ollama:11434/v1" {
		t.Errorf("unexpected ollama URL: %q", p.BaseURL)
	}
	if p.Auth != "none" {
		t.Errorf("expected auth=none for ollama, got %q", p.Auth)
	}

	p2, err := r.Get("openrouter")
	if err != nil {
		t.Fatalf("get openrouter: %v", err)
	}
	if p2.APIKey != "sk-or-test" {
		t.Errorf("expected openrouter key sk-or-test, got %q", p2.APIKey)
	}
}

// -- LoadFromEnv is fallback-only for file-backed providers -------------------

func TestLoadFromEnvFallbackOnlyForFileBacked(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "providers.json"), []byte(`{
		"version": 2,
		"providers": {
			"openai": {
				"base_url":      "https://api.openai.com/v1",
				"auth":          "bearer",
				"api_format":    "openai",
				"active_key_id": "seed:OPENAI_API_KEY",
				"keys": [{
					"id": "seed:OPENAI_API_KEY", "label": "primary",
					"secret": "sk-from-file", "source": "seed", "state": "ready",
					"cooldown_until": "", "last_error_code": 0,
					"last_error_reason": "", "last_error_at": "",
					"added_at": "2026-03-23T12:00:00Z"
				}]
			}
		}
	}`), 0o644)

	t.Setenv("OPENAI_API_KEY", "sk-from-env")

	r := NewRegistry(dir)
	_ = r.LoadFromFile()
	r.LoadFromEnv() // should NOT override file-backed provider

	p, err := r.Get("openai")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if p.APIKey != "sk-from-file" {
		t.Errorf("env overrode file-backed provider: got %q, want sk-from-file", p.APIKey)
	}
}

func TestLoadFromEnvPopulatesProviderNotInFile(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("ANTHROPIC_API_KEY", "sk-ant-env")

	r := NewRegistry(dir) // empty file
	_ = r.LoadFromFile()
	r.LoadFromEnv()

	p, err := r.Get("anthropic")
	if err != nil {
		t.Fatalf("Get anthropic: %v", err)
	}
	if p.APIKey != "sk-ant-env" {
		t.Errorf("expected env key for missing provider, got %q", p.APIKey)
	}
}

// -- pool state machine -------------------------------------------------------

func TestSelectKeyUsesActiveKey(t *testing.T) {
	dir := t.TempDir()
	seedV2(t, dir,
		readyKey("seed:OPENAI_API_KEY", "primary", "sk-primary"),
		readyKey("seed:OPENAI_API_KEY_1", "backup-1", "sk-backup"),
	)

	r := NewRegistry(dir)
	_ = r.LoadFromFile()

	prov, lease, err := r.SelectKey("openai")
	if err != nil {
		t.Fatalf("SelectKey: %v", err)
	}
	if prov.APIKey != "sk-primary" {
		t.Errorf("expected primary key, got %q", prov.APIKey)
	}
	if lease.KeyID != "seed:OPENAI_API_KEY" {
		t.Errorf("expected primary lease, got %q", lease.KeyID)
	}
}

func TestSelectKeyFallsBackAfterMarkDead(t *testing.T) {
	dir := t.TempDir()
	seedV2(t, dir,
		readyKey("seed:OPENAI_API_KEY", "primary", "sk-primary"),
		readyKey("seed:OPENAI_API_KEY_1", "backup-1", "sk-backup"),
	)

	r := NewRegistry(dir)
	_ = r.LoadFromFile()

	_ = r.MarkDead("openai", "seed:OPENAI_API_KEY", "http_401", 401)

	prov, lease, err := r.SelectKey("openai")
	if err != nil {
		t.Fatalf("SelectKey after dead: %v", err)
	}
	if prov.APIKey != "sk-backup" {
		t.Errorf("expected backup key, got %q", prov.APIKey)
	}
	if lease.KeyID != "seed:OPENAI_API_KEY_1" {
		t.Errorf("expected backup lease, got %q", lease.KeyID)
	}
}

func TestSelectKeyReturnsCooldownErrorWhenAllCooling(t *testing.T) {
	dir := t.TempDir()
	seedV2(t, dir, readyKey("seed:OPENAI_API_KEY", "only", "sk-only"))

	r := NewRegistry(dir)
	_ = r.LoadFromFile()

	until := time.Now().Add(time.Hour)
	_ = r.MarkCooldown("openai", "seed:OPENAI_API_KEY", "rate_limit", until)

	_, _, err := r.SelectKey("openai")
	if err == nil {
		t.Fatal("expected error, got nil")
	}
	ce, ok := err.(*CooldownError)
	if !ok {
		t.Fatalf("expected CooldownError, got %T: %v", err, err)
	}
	if ce.Provider != "openai" {
		t.Errorf("CooldownError.Provider = %q, want openai", ce.Provider)
	}
}

func TestCooldownExpiry(t *testing.T) {
	dir := t.TempDir()
	seedV2(t, dir, readyKey("seed:OPENAI_API_KEY", "only", "sk-only"))

	r := NewRegistry(dir)
	_ = r.LoadFromFile()

	// Cooldown already expired.
	past := time.Now().Add(-time.Minute)
	_ = r.MarkCooldown("openai", "seed:OPENAI_API_KEY", "rate_limit", past)

	prov, _, err := r.SelectKey("openai")
	if err != nil {
		t.Fatalf("SelectKey after expired cooldown: %v", err)
	}
	if prov.APIKey != "sk-only" {
		t.Errorf("expected key after cooldown expiry, got %q", prov.APIKey)
	}
}

func TestActivateKey(t *testing.T) {
	dir := t.TempDir()
	seedV2(t, dir,
		readyKey("seed:OPENAI_API_KEY", "primary", "sk-primary"),
		readyKey("seed:OPENAI_API_KEY_1", "backup-1", "sk-backup"),
	)

	r := NewRegistry(dir)
	_ = r.LoadFromFile()

	// Mark primary dead.
	_ = r.MarkDead("openai", "seed:OPENAI_API_KEY", "http_401", 401)

	// Activate backup explicitly.
	if err := r.ActivateKey("openai", "seed:OPENAI_API_KEY_1"); err != nil {
		t.Fatalf("ActivateKey: %v", err)
	}

	prov, lease, err := r.SelectKey("openai")
	if err != nil {
		t.Fatalf("SelectKey after activate: %v", err)
	}
	if lease.KeyID != "seed:OPENAI_API_KEY_1" {
		t.Errorf("expected activated key in lease, got %q", lease.KeyID)
	}
	if prov.APIKey != "sk-backup" {
		t.Errorf("expected backup secret, got %q", prov.APIKey)
	}
}

func TestAddRuntimeKey(t *testing.T) {
	dir := t.TempDir()
	seedV2(t, dir, readyKey("seed:OPENAI_API_KEY", "primary", "sk-primary"))

	r := NewRegistry(dir)
	_ = r.LoadFromFile()

	keyID, err := r.AddRuntimeKey("openai", "extra", "sk-runtime-extra")
	if err != nil {
		t.Fatalf("AddRuntimeKey: %v", err)
	}
	if keyID == "" {
		t.Error("expected non-empty key ID")
	}

	// Kill the primary to force fallthrough to runtime key.
	_ = r.MarkDead("openai", "seed:OPENAI_API_KEY", "http_401", 401)
	_ = r.ActivateKey("openai", keyID)

	prov, lease, err := r.SelectKey("openai")
	if err != nil {
		t.Fatalf("SelectKey after runtime key: %v", err)
	}
	if lease.KeyID != keyID {
		t.Errorf("expected runtime key in lease, got %q", lease.KeyID)
	}
	if prov.APIKey != "sk-runtime-extra" {
		t.Errorf("expected runtime secret, got %q", prov.APIKey)
	}
}

func TestDeleteKey(t *testing.T) {
	dir := t.TempDir()
	seedV2(t, dir,
		readyKey("seed:OPENAI_API_KEY", "primary", "sk-primary"),
		readyKey("seed:OPENAI_API_KEY_1", "backup-1", "sk-backup"),
	)

	r := NewRegistry(dir)
	_ = r.LoadFromFile()

	if err := r.DeleteKey("openai", "seed:OPENAI_API_KEY_1"); err != nil {
		t.Fatalf("DeleteKey: %v", err)
	}

	all := r.All()
	for _, k := range all["openai"].Keys {
		if k.ID == "seed:OPENAI_API_KEY_1" {
			t.Error("deleted key still present")
		}
	}
}

// -- SaveToFile ---------------------------------------------------------------

func TestSaveToFileV2Format(t *testing.T) {
	dir := t.TempDir()
	r := NewRegistry(dir)

	t.Setenv("OPENAI_API_KEY", "sk-env-key")
	r.LoadFromEnv()

	if err := r.SaveToFile(); err != nil {
		t.Fatalf("SaveToFile: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "providers.json"))
	if err != nil {
		t.Fatalf("read saved file: %v", err)
	}

	var probe struct {
		Version   int `json:"version"`
		Providers map[string]struct {
			Keys []struct {
				ID     string `json:"id"`
				Source string `json:"source"`
				State  string `json:"state"`
			} `json:"keys"`
		} `json:"providers"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		t.Fatalf("parse saved file: %v", err)
	}
	if probe.Version != 2 {
		t.Errorf("saved version = %d, want 2", probe.Version)
	}
	op, ok := probe.Providers["openai"]
	if !ok {
		t.Fatal("openai provider not in saved file")
	}
	if len(op.Keys) == 0 {
		t.Fatal("no keys in saved openai provider")
	}
	if op.Keys[0].Source != "seed" {
		t.Errorf("key source = %q, want seed", op.Keys[0].Source)
	}
	if op.Keys[0].State != "ready" {
		t.Errorf("key state = %q, want ready", op.Keys[0].State)
	}
}

func TestSaveToFileRoundTrip(t *testing.T) {
	dir := t.TempDir()
	r := NewRegistry(dir)

	t.Setenv("OPENAI_API_KEY", "sk-round-trip")
	r.LoadFromEnv()
	if err := r.SaveToFile(); err != nil {
		t.Fatalf("save: %v", err)
	}

	r2 := NewRegistry(dir)
	if err := r2.LoadFromFile(); err != nil {
		t.Fatalf("load: %v", err)
	}
	p, err := r2.Get("openai")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if p.APIKey != "sk-round-trip" {
		t.Fatalf("unexpected key after round-trip: %q", p.APIKey)
	}
}

func TestSaveIsAtomic(t *testing.T) {
	dir := t.TempDir()
	r := NewRegistry(dir)
	t.Setenv("OPENAI_API_KEY", "sk-atomic")
	r.LoadFromEnv()

	if err := r.SaveToFile(); err != nil {
		t.Fatalf("SaveToFile: %v", err)
	}
	// Ensure no tmp file is left behind.
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".tmp" {
			t.Errorf("leftover tmp file: %s", e.Name())
		}
	}
}

// -- AddRuntimeProvider -------------------------------------------------------

func TestAddRuntimeProviderCreatesProvider(t *testing.T) {
	dir := t.TempDir()
	r := NewRegistry(dir)

	err := r.AddRuntimeProvider("mistral", "https://api.mistral.ai/v1", "bearer", "openai", "primary", "msk-secret")
	if err != nil {
		t.Fatalf("AddRuntimeProvider: %v", err)
	}

	p, err := r.Get("mistral")
	if err != nil {
		t.Fatalf("Get after AddRuntimeProvider: %v", err)
	}
	if p.BaseURL != "https://api.mistral.ai/v1" {
		t.Errorf("unexpected BaseURL: %q", p.BaseURL)
	}
	if p.APIKey != "msk-secret" {
		t.Errorf("unexpected APIKey: %q", p.APIKey)
	}
}

func TestAddRuntimeProviderRejectsExistingProvider(t *testing.T) {
	dir := t.TempDir()
	seedV2(t, dir, readyKey("seed:OPENAI_API_KEY", "primary", "sk-seed"))

	r := NewRegistry(dir)
	_ = r.LoadFromFile()

	err := r.AddRuntimeProvider("openai", "https://api.openai.com/v1", "bearer", "openai", "primary", "sk-new")
	if err == nil {
		t.Error("expected error when adding a provider that already exists as seed; got nil")
	}
}

func TestAddRuntimeProviderRejectsExistingRuntimeProvider(t *testing.T) {
	dir := t.TempDir()
	r := NewRegistry(dir)

	if err := r.AddRuntimeProvider("mistral", "https://api.mistral.ai/v1", "", "", "", "msk-first"); err != nil {
		t.Fatalf("first AddRuntimeProvider: %v", err)
	}

	err := r.AddRuntimeProvider("mistral", "https://api.mistral.ai/v1", "", "", "", "msk-second")
	if err == nil {
		t.Error("expected error on duplicate AddRuntimeProvider; got nil")
	}
}

func TestAddRuntimeProviderSavesToFile(t *testing.T) {
	dir := t.TempDir()
	r := NewRegistry(dir)

	if err := r.AddRuntimeProvider("mistral", "https://api.mistral.ai/v1", "bearer", "openai", "primary", "msk-save"); err != nil {
		t.Fatalf("AddRuntimeProvider: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(dir, "providers.json"))
	if err != nil {
		t.Fatalf("read providers.json: %v", err)
	}

	var probe struct {
		Version   int `json:"version"`
		Providers map[string]struct {
			Source string `json:"source"`
			Keys   []struct {
				Source string `json:"source"`
			} `json:"keys"`
		} `json:"providers"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		t.Fatalf("parse providers.json: %v", err)
	}

	prov, ok := probe.Providers["mistral"]
	if !ok {
		t.Fatal("mistral not found in saved providers.json")
	}
	if prov.Source != "runtime" {
		t.Errorf("provider source = %q, want runtime", prov.Source)
	}
	if len(prov.Keys) == 0 {
		t.Fatal("no keys saved for mistral")
	}
	if prov.Keys[0].Source != "runtime" {
		t.Errorf("key source = %q, want runtime", prov.Keys[0].Source)
	}
}
