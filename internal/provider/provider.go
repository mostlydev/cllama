package provider

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// KeyState represents the lifecycle state of a provider API key.
type KeyState string

const (
	KeyStateReady    KeyState = "ready"
	KeyStateCooldown KeyState = "cooldown"
	KeyStateDead     KeyState = "dead"
	KeyStateDisabled KeyState = "disabled"
)

// KeyEntry is one API key in a provider's pool.
type KeyEntry struct {
	ID              string   `json:"id"`
	Label           string   `json:"label,omitempty"`
	Secret          string   `json:"secret"`
	Source          string   `json:"source"` // "seed" or "runtime"
	State           KeyState `json:"state"`
	CooldownUntil   string   `json:"cooldown_until"`    // RFC3339 or ""
	LastErrorCode   int      `json:"last_error_code"`   // 0 = none
	LastErrorReason string   `json:"last_error_reason"` // "" = none
	LastErrorAt     string   `json:"last_error_at"`     // RFC3339 or ""
	AddedAt         string   `json:"added_at"`          // RFC3339
}

// ProviderState is the v2 per-provider config including key pool.
type ProviderState struct {
	BaseURL     string     `json:"base_url"`
	Auth        string     `json:"auth,omitempty"`
	APIFormat   string     `json:"api_format,omitempty"`
	ActiveKeyID string     `json:"active_key_id,omitempty"`
	Source      string     `json:"source,omitempty"` // "seed" | "runtime"
	Keys        []KeyEntry `json:"keys"`
}

// Provider holds the resolved config for one request, populated from the pool.
// The APIKey field carries the selected key's secret for backward compatibility
// with setProviderAuth in the proxy handler.
type Provider struct {
	Name      string `json:"name,omitempty"`
	BaseURL   string `json:"base_url"`
	APIKey    string `json:"api_key,omitempty"`
	Auth      string `json:"auth,omitempty"`
	APIFormat string `json:"api_format,omitempty"`
}

// KeyLease identifies the key selected for a single request, so the caller
// can report the outcome back to the pool.
type KeyLease struct {
	ProviderName string
	KeyID        string
}

// Registry manages known providers and their key pools; safe for concurrent use.
type Registry struct {
	mu        sync.RWMutex
	providers map[string]*ProviderState
	authDir   string
}

var knownProviders = map[string]string{
	"openai":     "https://api.openai.com/v1",
	"xai":        "https://api.x.ai/v1",
	"anthropic":  "https://api.anthropic.com/v1",
	"openrouter": "https://openrouter.ai/api/v1",
	"google":     "https://generativelanguage.googleapis.com/v1beta/openai",
	"ollama":     "http://ollama:11434/v1",
	"vercel":     "https://ai-gateway.vercel.sh/v1",
}

// envKeyMap maps env var → provider name → indexed backup suffix.
// PRIMARY maps to "" (no suffix), BACKUP_n maps to "_n".
var envKeyMap = map[string]string{
	"OPENAI_API_KEY":       "openai",
	"OPENAI_API_KEY_1":     "openai",
	"OPENAI_API_KEY_2":     "openai",
	"XAI_API_KEY":          "xai",
	"XAI_API_KEY_1":        "xai",
	"ANTHROPIC_API_KEY":    "anthropic",
	"ANTHROPIC_API_KEY_1":  "anthropic",
	"OPENROUTER_API_KEY":   "openrouter",
	"OPENROUTER_API_KEY_1": "openrouter",
	"GEMINI_API_KEY":       "google",
	"GEMINI_API_KEY_1":     "google",
	"GOOGLE_API_KEY":       "google",
	"AI_GATEWAY_API_KEY":   "vercel",
	"AI_GATEWAY_API_KEY_1": "vercel",
	"AI_GATEWAY_API_KEY_2": "vercel",
}

var envBaseURLMap = map[string]string{
	"OPENAI_BASE_URL":     "openai",
	"XAI_BASE_URL":        "xai",
	"ANTHROPIC_BASE_URL":  "anthropic",
	"OPENROUTER_BASE_URL": "openrouter",
	"GOOGLE_BASE_URL":     "google",
	"OLLAMA_BASE_URL":     "ollama",
	"AI_GATEWAY_BASE_URL": "vercel",
}

func NewRegistry(authDir string) *Registry {
	return &Registry{
		providers: make(map[string]*ProviderState),
		authDir:   authDir,
	}
}

// v1File is the legacy format (version absent or 1).
type v1File struct {
	Providers map[string]struct {
		BaseURL   string `json:"base_url"`
		APIKey    string `json:"api_key"`
		Auth      string `json:"auth"`
		APIFormat string `json:"api_format"`
	} `json:"providers"`
}

// v2File is the current format.
type v2File struct {
	Version   int                       `json:"version"`
	Providers map[string]*ProviderState `json:"providers"`
}

// LoadFromFile reads providers.json from the auth directory.
// Both v1 (legacy api_key) and v2 (keys[]) are supported; v1 is
// normalised to a single-entry pool in memory.
func (r *Registry) LoadFromFile() error {
	if r.authDir == "" {
		return nil
	}
	path := filepath.Join(r.authDir, "providers.json")
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("read providers.json: %w", err)
	}

	// Peek at the version field to choose the parse path.
	var versionProbe struct {
		Version int `json:"version"`
	}
	_ = json.Unmarshal(data, &versionProbe)

	r.mu.Lock()
	defer r.mu.Unlock()

	if versionProbe.Version < 2 {
		return r.loadV1Locked(data)
	}
	return r.loadV2Locked(data)
}

func (r *Registry) loadV1Locked(data []byte) error {
	var f v1File
	if err := json.Unmarshal(data, &f); err != nil {
		return fmt.Errorf("parse providers.json (v1): %w", err)
	}
	for name, p := range f.Providers {
		n := normalizeName(name)
		if n == "" {
			continue
		}
		state := &ProviderState{
			BaseURL:   p.BaseURL,
			Auth:      p.Auth,
			APIFormat: p.APIFormat,
			Source:    "seed",
		}
		if state.BaseURL == "" {
			state.BaseURL = knownProviders[n]
		}
		if state.Auth == "" {
			state.Auth = defaultAuth(n)
		}
		if state.APIFormat == "" {
			state.APIFormat = defaultAPIFormat(n)
		}
		if p.APIKey != "" {
			keyID := "seed:" + strings.ToUpper(n) + "_API_KEY"
			state.ActiveKeyID = keyID
			state.Keys = []KeyEntry{{
				ID:      keyID,
				Label:   "primary",
				Secret:  p.APIKey,
				Source:  "seed",
				State:   KeyStateReady,
				AddedAt: time.Now().UTC().Format(time.RFC3339),
			}}
		}
		r.providers[n] = state
	}
	return nil
}

func (r *Registry) loadV2Locked(data []byte) error {
	var f v2File
	if err := json.Unmarshal(data, &f); err != nil {
		return fmt.Errorf("parse providers.json (v2): %w", err)
	}
	for name, state := range f.Providers {
		n := normalizeName(name)
		if n == "" || state == nil {
			continue
		}
		if state.BaseURL == "" {
			state.BaseURL = knownProviders[n]
		}
		if state.Auth == "" {
			state.Auth = defaultAuth(n)
		}
		if state.APIFormat == "" {
			state.APIFormat = defaultAPIFormat(n)
		}
		r.providers[n] = state
	}
	return nil
}

// LoadFromEnv overlays known provider keys/base URLs from env vars.
// Only providers NOT already present in the file-loaded registry are populated;
// file-backed providers are authoritative.
func (r *Registry) LoadFromEnv() {
	r.mu.Lock()
	defer r.mu.Unlock()

	for envKey, provName := range envBaseURLMap {
		// Only set base URL if provider has no keys from file (pure-env provider).
		if _, exists := r.providers[provName]; exists {
			continue
		}
		v := strings.TrimSpace(os.Getenv(envKey))
		if v == "" {
			continue
		}
		p, ok := r.providers[provName]
		if !ok {
			p = &ProviderState{Auth: defaultAuth(provName), APIFormat: defaultAPIFormat(provName), Source: "seed"}
		}
		p.BaseURL = v
		r.providers[provName] = p
	}

	// Ordered list of env key vars per provider, in priority order.
	type envKeyDef struct {
		envVar string
		keyID  string // deterministic seed ID
		label  string
	}
	envKeysByProvider := map[string][]envKeyDef{
		"openai": {
			{"OPENAI_API_KEY", "seed:OPENAI_API_KEY", "primary"},
			{"OPENAI_API_KEY_1", "seed:OPENAI_API_KEY_1", "backup-1"},
			{"OPENAI_API_KEY_2", "seed:OPENAI_API_KEY_2", "backup-2"},
		},
		"xai": {
			{"XAI_API_KEY", "seed:XAI_API_KEY", "primary"},
			{"XAI_API_KEY_1", "seed:XAI_API_KEY_1", "backup-1"},
		},
		"anthropic": {
			{"ANTHROPIC_API_KEY", "seed:ANTHROPIC_API_KEY", "primary"},
			{"ANTHROPIC_API_KEY_1", "seed:ANTHROPIC_API_KEY_1", "backup-1"},
		},
		"openrouter": {
			{"OPENROUTER_API_KEY", "seed:OPENROUTER_API_KEY", "primary"},
			{"OPENROUTER_API_KEY_1", "seed:OPENROUTER_API_KEY_1", "backup-1"},
		},
		"google": {
			{"GEMINI_API_KEY", "seed:GEMINI_API_KEY", "primary"},
			{"GEMINI_API_KEY_1", "seed:GEMINI_API_KEY_1", "backup-1"},
			{"GOOGLE_API_KEY", "seed:GOOGLE_API_KEY", "backup-2"},
		},
		"vercel": {
			{"AI_GATEWAY_API_KEY", "seed:AI_GATEWAY_API_KEY", "primary"},
			{"AI_GATEWAY_API_KEY_1", "seed:AI_GATEWAY_API_KEY_1", "backup-1"},
			{"AI_GATEWAY_API_KEY_2", "seed:AI_GATEWAY_API_KEY_2", "backup-2"},
		},
	}

	for provName, defs := range envKeysByProvider {
		// Skip providers already loaded from file.
		if existing, exists := r.providers[provName]; exists && len(existing.Keys) > 0 {
			continue
		}

		var keys []KeyEntry
		var firstKeyID string
		for _, def := range defs {
			v := strings.TrimSpace(os.Getenv(def.envVar))
			if v == "" {
				continue
			}
			if firstKeyID == "" {
				firstKeyID = def.keyID
			}
			keys = append(keys, KeyEntry{
				ID:      def.keyID,
				Label:   def.label,
				Secret:  v,
				Source:  "seed",
				State:   KeyStateReady,
				AddedAt: time.Now().UTC().Format(time.RFC3339),
			})
		}
		if len(keys) == 0 {
			continue
		}

		p, ok := r.providers[provName]
		if !ok {
			p = &ProviderState{
				BaseURL:   knownProviders[provName],
				Auth:      defaultAuth(provName),
				APIFormat: defaultAPIFormat(provName),
				Source:    "seed",
			}
		}
		p.Keys = keys
		p.ActiveKeyID = firstKeyID
		r.providers[provName] = p
	}
}

// Get returns a Provider populated with the active key's secret.
// It is a lightweight read — it does not advance the pool or apply cooldown logic.
// Use SelectKey for actual request dispatch.
func (r *Registry) Get(name string) (*Provider, error) {
	n := normalizeName(name)
	r.mu.RLock()
	state, ok := r.providers[n]
	r.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("unknown provider: %q", name)
	}
	return r.providerFromState(n, state), nil
}

func (r *Registry) providerFromState(name string, state *ProviderState) *Provider {
	p := &Provider{
		Name:      name,
		BaseURL:   state.BaseURL,
		Auth:      state.Auth,
		APIFormat: state.APIFormat,
	}
	// Populate APIKey from the active key (or first ready key).
	key := r.activeKey(state)
	if key != nil {
		p.APIKey = key.Secret
	}
	return p
}

// activeKey returns the KeyEntry that should be used for requests, or nil.
// Must NOT be called with the lock held — callers should have already acquired it.
func (r *Registry) activeKey(state *ProviderState) *KeyEntry {
	now := time.Now().UTC()
	// Prefer active_key_id if it is ready (or cooldown has expired).
	if state.ActiveKeyID != "" {
		for i := range state.Keys {
			k := &state.Keys[i]
			if k.ID != state.ActiveKeyID {
				continue
			}
			if k.State == KeyStateReady {
				return k
			}
			if k.State == KeyStateCooldown && k.CooldownUntil != "" {
				until, err := time.Parse(time.RFC3339, k.CooldownUntil)
				if err == nil && now.After(until) {
					k.State = KeyStateReady
					k.CooldownUntil = ""
					return k
				}
			}
		}
	}
	// Fall through to first ready key.
	for i := range state.Keys {
		k := &state.Keys[i]
		if k.State == KeyStateReady {
			return k
		}
		if k.State == KeyStateCooldown && k.CooldownUntil != "" {
			until, err := time.Parse(time.RFC3339, k.CooldownUntil)
			if err == nil && now.After(until) {
				k.State = KeyStateReady
				k.CooldownUntil = ""
				return k
			}
		}
	}
	return nil
}

// SelectKey picks the best available key for the named provider.
// It returns a Provider ready for use plus a KeyLease the caller must use
// to report back the outcome (MarkCooldown / MarkDead).
func (r *Registry) SelectKey(name string) (*Provider, KeyLease, error) {
	n := normalizeName(name)
	r.mu.Lock()
	defer r.mu.Unlock()

	state, ok := r.providers[n]
	if !ok {
		return nil, KeyLease{}, fmt.Errorf("unknown provider: %q", name)
	}

	key := r.activeKey(state)
	if key == nil {
		// All keys are cooling down or dead. Find earliest cooldown expiry.
		earliest := r.earliestCooldown(state)
		if earliest.IsZero() {
			return nil, KeyLease{}, fmt.Errorf("provider %q: no usable keys (all dead or disabled)", n)
		}
		return nil, KeyLease{}, &CooldownError{Provider: n, RetryAt: earliest}
	}

	p := &Provider{
		Name:      n,
		BaseURL:   state.BaseURL,
		Auth:      state.Auth,
		APIFormat: state.APIFormat,
		APIKey:    key.Secret,
	}
	return p, KeyLease{ProviderName: n, KeyID: key.ID}, nil
}

// CooldownError is returned by SelectKey when all keys are temporarily cooling down.
type CooldownError struct {
	Provider string
	RetryAt  time.Time
}

func (e *CooldownError) Error() string {
	return fmt.Sprintf("provider %q: all keys in cooldown until %s", e.Provider, e.RetryAt.Format(time.RFC3339))
}

func (r *Registry) earliestCooldown(state *ProviderState) time.Time {
	var earliest time.Time
	for _, k := range state.Keys {
		if k.State != KeyStateCooldown || k.CooldownUntil == "" {
			continue
		}
		t, err := time.Parse(time.RFC3339, k.CooldownUntil)
		if err != nil {
			continue
		}
		if earliest.IsZero() || t.Before(earliest) {
			earliest = t
		}
	}
	return earliest
}

// MarkCooldown temporarily removes a key from rotation.
func (r *Registry) MarkCooldown(providerName, keyID, reason string, until time.Time) error {
	return r.mutateKey(providerName, keyID, func(k *KeyEntry) {
		k.State = KeyStateCooldown
		k.CooldownUntil = until.UTC().Format(time.RFC3339)
		k.LastErrorReason = reason
		k.LastErrorAt = time.Now().UTC().Format(time.RFC3339)
	})
}

// MarkDead permanently removes a key from rotation.
func (r *Registry) MarkDead(providerName, keyID, reason string, statusCode int) error {
	return r.mutateKey(providerName, keyID, func(k *KeyEntry) {
		k.State = KeyStateDead
		k.CooldownUntil = ""
		k.LastErrorCode = statusCode
		k.LastErrorReason = reason
		k.LastErrorAt = time.Now().UTC().Format(time.RFC3339)
	})
}

// ActivateKey sets a key as the active_key_id and marks it ready.
func (r *Registry) ActivateKey(providerName, keyID string) error {
	n := normalizeName(providerName)
	r.mu.Lock()
	defer r.mu.Unlock()
	state, ok := r.providers[n]
	if !ok {
		return fmt.Errorf("unknown provider: %q", providerName)
	}
	for i := range state.Keys {
		if state.Keys[i].ID == keyID {
			state.Keys[i].State = KeyStateReady
			state.Keys[i].CooldownUntil = ""
			state.ActiveKeyID = keyID
			return nil
		}
	}
	return fmt.Errorf("key %q not found in provider %q", keyID, providerName)
}

// AddRuntimeKey adds a new key to the pool at runtime.
func (r *Registry) AddRuntimeKey(providerName, label, secret string) (string, error) {
	n := normalizeName(providerName)
	r.mu.Lock()
	defer r.mu.Unlock()
	state, ok := r.providers[n]
	if !ok {
		return "", fmt.Errorf("unknown provider: %q", providerName)
	}
	id := "runtime:" + randomHex(4)
	state.Keys = append(state.Keys, KeyEntry{
		ID:      id,
		Label:   label,
		Secret:  secret,
		Source:  "runtime",
		State:   KeyStateReady,
		AddedAt: time.Now().UTC().Format(time.RFC3339),
	})
	return id, nil
}

// AddRuntimeProvider creates a new provider at runtime with a single ready key.
// Returns an error if a provider with that name already exists (seed or runtime).
// To add more keys to an existing provider, use AddRuntimeKey.
func (r *Registry) AddRuntimeProvider(name, baseURL, auth, apiFormat, label, secret string) error {
	n := normalizeName(name)
	if n == "" {
		return fmt.Errorf("provider name must not be empty")
	}

	r.mu.Lock()
	if _, exists := r.providers[n]; exists {
		r.mu.Unlock()
		return fmt.Errorf("provider %q already exists; use /keys/add to add keys to it", n)
	}

	if auth == "" {
		auth = "bearer"
	}
	if apiFormat == "" {
		apiFormat = "openai"
	}
	if label == "" {
		label = "primary"
	}

	keyID := "runtime:" + n + ":" + randomHex(4)
	keyEntry := KeyEntry{
		ID:      keyID,
		Label:   label,
		Secret:  secret,
		Source:  "runtime",
		State:   KeyStateReady,
		AddedAt: time.Now().UTC().Format(time.RFC3339),
	}

	state := &ProviderState{
		BaseURL:     baseURL,
		Auth:        auth,
		APIFormat:   apiFormat,
		Source:      "runtime",
		ActiveKeyID: keyID,
		Keys:        []KeyEntry{keyEntry},
	}
	r.providers[n] = state
	r.mu.Unlock()

	return r.SaveToFile()
}

// DisableKey marks a key as disabled (operator-disabled, not retried).
func (r *Registry) DisableKey(providerName, keyID string) error {
	return r.mutateKey(providerName, keyID, func(k *KeyEntry) {
		k.State = KeyStateDisabled
	})
}

// DeleteKey removes a key from the pool entirely.
func (r *Registry) DeleteKey(providerName, keyID string) error {
	n := normalizeName(providerName)
	r.mu.Lock()
	defer r.mu.Unlock()
	state, ok := r.providers[n]
	if !ok {
		return fmt.Errorf("unknown provider: %q", providerName)
	}
	newKeys := state.Keys[:0]
	found := false
	for _, k := range state.Keys {
		if k.ID == keyID {
			found = true
			continue
		}
		newKeys = append(newKeys, k)
	}
	if !found {
		return fmt.Errorf("key %q not found in provider %q", keyID, providerName)
	}
	state.Keys = newKeys
	if state.ActiveKeyID == keyID {
		state.ActiveKeyID = ""
	}
	return nil
}

func (r *Registry) mutateKey(providerName, keyID string, fn func(*KeyEntry)) error {
	n := normalizeName(providerName)
	r.mu.Lock()
	defer r.mu.Unlock()
	state, ok := r.providers[n]
	if !ok {
		return fmt.Errorf("unknown provider: %q", providerName)
	}
	for i := range state.Keys {
		if state.Keys[i].ID == keyID {
			fn(&state.Keys[i])
			return nil
		}
	}
	return fmt.Errorf("key %q not found in provider %q", keyID, providerName)
}

// Set registers a Provider directly, wrapping it as a single-key ProviderState.
// This is intended for testing and programmatic seeding where the caller already
// has a resolved Provider struct.
func (r *Registry) Set(name string, p *Provider) {
	n := normalizeName(name)
	if n == "" {
		return
	}
	state := &ProviderState{
		BaseURL:   p.BaseURL,
		Auth:      p.Auth,
		APIFormat: p.APIFormat,
	}
	if p.APIKey != "" {
		keyID := "direct:" + n
		state.ActiveKeyID = keyID
		state.Keys = []KeyEntry{{
			ID:      keyID,
			Label:   "primary",
			Secret:  p.APIKey,
			Source:  "seed",
			State:   KeyStateReady,
			AddedAt: time.Now().UTC().Format(time.RFC3339),
		}}
	}
	r.mu.Lock()
	r.providers[n] = state
	r.mu.Unlock()
}

// All returns a copy of all ProviderState entries, keyed by provider name.
func (r *Registry) All() map[string]*ProviderState {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make(map[string]*ProviderState, len(r.providers))
	for k, v := range r.providers {
		cp := *v
		keys := make([]KeyEntry, len(v.Keys))
		copy(keys, v.Keys)
		cp.Keys = keys
		out[k] = &cp
	}
	return out
}

func (r *Registry) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.providers))
	for k := range r.providers {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// SaveToFile atomically writes the current pool state to providers.json.
// The file is always written in v2 format.
func (r *Registry) SaveToFile() error {
	if r.authDir == "" {
		return fmt.Errorf("no auth directory configured")
	}
	if err := os.MkdirAll(r.authDir, 0o777); err != nil {
		return fmt.Errorf("create auth dir: %w", err)
	}

	r.mu.RLock()
	providers := make(map[string]*ProviderState, len(r.providers))
	for name, state := range r.providers {
		cp := *state
		keys := make([]KeyEntry, len(state.Keys))
		copy(keys, state.Keys)
		cp.Keys = keys
		providers[name] = &cp
	}
	r.mu.RUnlock()

	f := v2File{Version: 2, Providers: providers}
	data, err := json.MarshalIndent(f, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal providers.json: %w", err)
	}

	dest := filepath.Join(r.authDir, "providers.json")
	tmp := dest + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return fmt.Errorf("write providers.json tmp: %w", err)
	}
	if err := os.Rename(tmp, dest); err != nil {
		_ = os.Remove(tmp)
		return fmt.Errorf("rename providers.json: %w", err)
	}
	return nil
}

func normalizeName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}

func defaultAuth(provider string) string {
	switch normalizeName(provider) {
	case "ollama":
		return "none"
	case "anthropic":
		return "x-api-key"
	default:
		return "bearer"
	}
}

func defaultAPIFormat(provider string) string {
	if normalizeName(provider) == "anthropic" {
		return "anthropic"
	}
	return "openai"
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
