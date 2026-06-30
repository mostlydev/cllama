package agentctx

import "strings"

type AllowedModel struct {
	Slot string `json:"slot"`
	Ref  string `json:"ref"`
}

type ModelPolicy struct {
	Mode    string         `json:"mode"`
	Allowed []AllowedModel `json:"allowed"`
}

func (p *ModelPolicy) HasPolicy() bool {
	return p != nil && len(p.Allowed) > 0
}

func (p *ModelPolicy) AllowedModelRefs() []string {
	if p == nil {
		return nil
	}
	out := make([]string, 0, len(p.Allowed))
	seen := make(map[string]struct{}, len(p.Allowed))
	for _, entry := range p.Allowed {
		ref := strings.TrimSpace(entry.Ref)
		if ref == "" {
			continue
		}
		if _, ok := seen[ref]; ok {
			continue
		}
		seen[ref] = struct{}{}
		out = append(out, ref)
	}
	return out
}

func (p *ModelPolicy) DefaultModel() string {
	if p == nil {
		return ""
	}
	for _, entry := range p.Allowed {
		if strings.EqualFold(strings.TrimSpace(entry.Slot), "primary") && strings.TrimSpace(entry.Ref) != "" {
			return strings.TrimSpace(entry.Ref)
		}
	}
	for _, entry := range p.Allowed {
		if strings.TrimSpace(entry.Ref) != "" {
			return strings.TrimSpace(entry.Ref)
		}
	}
	return ""
}

func (p *ModelPolicy) FailoverRefs() []string {
	if p == nil {
		return nil
	}
	out := make([]string, 0, 2)
	seen := make(map[string]struct{}, 2)
	appendSlot := func(slot string) {
		for _, entry := range p.Allowed {
			if !strings.EqualFold(strings.TrimSpace(entry.Slot), slot) {
				continue
			}
			ref := strings.TrimSpace(entry.Ref)
			if ref == "" {
				return
			}
			if _, ok := seen[ref]; ok {
				return
			}
			seen[ref] = struct{}{}
			out = append(out, ref)
			return
		}
	}
	appendSlot("primary")
	appendSlot("fallback")
	return out
}
