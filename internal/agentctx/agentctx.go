package agentctx

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
)

// AgentContext holds the per-agent mounted contract and metadata files.
type AgentContext struct {
	AgentID     string
	ContextDir  string
	AgentsMD    []byte
	ClawdapusMD []byte
	Metadata    map[string]any
	ServiceAuth []ServiceAuthEntry
	ModelPolicy *ModelPolicy
}

type ServiceAuthEntry struct {
	Service   string `json:"service"`
	AuthType  string `json:"auth_type"`
	Token     string `json:"token,omitempty"`
	Principal string `json:"principal,omitempty"`
}

// Load reads an agent's context files from contextRoot/<agentID>/.
func Load(contextRoot, agentID string) (*AgentContext, error) {
	dir := filepath.Join(contextRoot, agentID)

	agentsMD, err := os.ReadFile(filepath.Join(dir, "AGENTS.md"))
	if err != nil {
		return nil, fmt.Errorf("load agent context %q: AGENTS.md: %w", agentID, err)
	}

	clawdapusMD, err := os.ReadFile(filepath.Join(dir, "CLAWDAPUS.md"))
	if err != nil {
		return nil, fmt.Errorf("load agent context %q: CLAWDAPUS.md: %w", agentID, err)
	}

	metaRaw, err := os.ReadFile(filepath.Join(dir, "metadata.json"))
	if err != nil {
		return nil, fmt.Errorf("load agent context %q: metadata.json: %w", agentID, err)
	}

	var meta map[string]any
	if err := json.Unmarshal(metaRaw, &meta); err != nil {
		return nil, fmt.Errorf("load agent context %q: parse metadata: %w", agentID, err)
	}
	var typed struct {
		ModelPolicy *ModelPolicy `json:"model_policy"`
	}
	if err := json.Unmarshal(metaRaw, &typed); err != nil {
		return nil, fmt.Errorf("load agent context %q: parse typed metadata: %w", agentID, err)
	}
	serviceAuth, err := loadServiceAuth(dir)
	if err != nil {
		return nil, fmt.Errorf("load agent context %q: service-auth: %w", agentID, err)
	}

	return &AgentContext{
		AgentID:     agentID,
		ContextDir:  dir,
		AgentsMD:    agentsMD,
		ClawdapusMD: clawdapusMD,
		Metadata:    meta,
		ServiceAuth: serviceAuth,
		ModelPolicy: typed.ModelPolicy,
	}, nil
}

// MetadataToken returns metadata["token"] when present and a string.
func (a *AgentContext) MetadataToken() string {
	if a == nil {
		return ""
	}
	tok, _ := a.Metadata["token"].(string)
	return tok
}

// MetadataString returns a string metadata field, or empty string.
func (a *AgentContext) MetadataString(key string) string {
	if a == nil {
		return ""
	}
	v, _ := a.Metadata[key].(string)
	return v
}

func (a *AgentContext) HasPolicy() bool {
	return a != nil && a.ModelPolicy.HasPolicy()
}

func (a *AgentContext) DefaultModel() string {
	if a == nil || a.ModelPolicy == nil {
		return ""
	}
	return a.ModelPolicy.DefaultModel()
}

func (a *AgentContext) AllowedModelRefs() []string {
	if a == nil || a.ModelPolicy == nil {
		return nil
	}
	return a.ModelPolicy.AllowedModelRefs()
}

func (a *AgentContext) FailoverRefs() []string {
	if a == nil || a.ModelPolicy == nil {
		return nil
	}
	return a.ModelPolicy.FailoverRefs()
}

// FeedsPath returns the path to the agent's feeds.json manifest.
func (a *AgentContext) FeedsPath() string {
	if a == nil || a.ContextDir == "" {
		return ""
	}
	return filepath.Join(a.ContextDir, "feeds.json")
}

func loadServiceAuth(dir string) ([]ServiceAuthEntry, error) {
	authDir := filepath.Join(dir, "service-auth")
	entries, err := os.ReadDir(authDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	sort.Slice(entries, func(i, j int) bool {
		return entries[i].Name() < entries[j].Name()
	})

	auth := make([]ServiceAuthEntry, 0, len(entries))
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		raw, err := os.ReadFile(filepath.Join(authDir, entry.Name()))
		if err != nil {
			return nil, err
		}
		var parsed ServiceAuthEntry
		if err := json.Unmarshal(raw, &parsed); err != nil {
			return nil, err
		}
		auth = append(auth, parsed)
	}
	return auth, nil
}

// AgentSummary is a lightweight view of an agent for listing purposes.
type AgentSummary struct {
	AgentID string
	Pod     string
	Type    string
	Service string
}

// ListAgents scans the context root directory for agent subdirectories
// and returns a summary for each. Agents that fail to load are skipped.
func ListAgents(contextRoot string) ([]AgentSummary, error) {
	entries, err := os.ReadDir(contextRoot)
	if err != nil {
		return nil, fmt.Errorf("list agents in %q: %w", contextRoot, err)
	}

	var agents []AgentSummary
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		metaPath := filepath.Join(contextRoot, e.Name(), "metadata.json")
		raw, err := os.ReadFile(metaPath)
		if err != nil {
			continue // skip agents without metadata
		}
		var meta map[string]any
		if json.Unmarshal(raw, &meta) != nil {
			continue
		}
		s := AgentSummary{AgentID: e.Name()}
		if v, ok := meta["pod"].(string); ok {
			s.Pod = v
		}
		if v, ok := meta["type"].(string); ok {
			s.Type = v
		}
		if v, ok := meta["service"].(string); ok {
			s.Service = v
		}
		agents = append(agents, s)
	}
	return agents, nil
}
