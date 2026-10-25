package agentctx

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/url"
	"regexp"
	"strings"
)

// DataPolicy is compiler-owned policy, never a model/caller preference.
type DataPolicy struct {
	Version        int               `json:"version"`
	Regions        []string          `json:"regions"`
	Training       string            `json:"training"`
	Retention      string            `json:"retention"`
	LocalRetention string            `json:"local_retention"`
	Attestations   []DataAttestation `json:"attestations,omitempty"`
}

type DataAttestation struct {
	Provider   string   `json:"provider"`
	Endpoint   string   `json:"endpoint"`
	Guarantees []string `json:"guarantees"`
}

func DefaultDataPolicy() DataPolicy {
	return DataPolicy{Version: 1, Regions: []string{"eu", "us"}, Training: "none", Retention: "none", LocalRetention: "none"}
}

var policyRegion = regexp.MustCompile(`^[a-z]{2,3}$`)
var policyProvider = regexp.MustCompile(`^[a-z0-9][a-z0-9-]*$`)

// EffectiveDataPolicy validates mounted metadata even for older compilers.
// Absence means conservative defaults; malformed policy never means allow.
func (a *AgentContext) EffectiveDataPolicy() (DataPolicy, error) {
	p := DefaultDataPolicy()
	if a == nil {
		return p, nil
	}
	if a.DataPolicy != nil {
		p = *a.DataPolicy
	} else if raw, ok := a.Metadata["data_policy"]; ok {
		if raw == nil {
			return p, fmt.Errorf("data_policy must be an object")
		}
		b, err := json.Marshal(raw)
		if err != nil {
			return p, fmt.Errorf("invalid data_policy")
		}
		dec := json.NewDecoder(bytes.NewReader(b))
		dec.DisallowUnknownFields()
		if err := dec.Decode(&p); err != nil {
			return p, fmt.Errorf("invalid data_policy")
		}
		if err := dec.Decode(new(any)); err != io.EOF {
			return p, fmt.Errorf("invalid data_policy")
		}
	}
	if p.Version != 1 {
		return p, fmt.Errorf("unsupported data_policy version")
	}
	if len(p.Regions) == 0 {
		return p, fmt.Errorf("data_policy regions must not be empty")
	}
	for _, r := range p.Regions {
		if r == "global" {
			if len(p.Regions) != 1 {
				return p, fmt.Errorf("global region must stand alone")
			}
		} else if !policyRegion.MatchString(r) {
			return p, fmt.Errorf("invalid data_policy region")
		}
	}
	if p.Training != "none" && p.Training != "allowed" {
		return p, fmt.Errorf("invalid data_policy training")
	}
	if p.Retention != "none" && p.Retention != "provider-default" {
		return p, fmt.Errorf("invalid data_policy retention")
	}
	if p.LocalRetention != "none" && p.LocalRetention != "allowed" {
		return p, fmt.Errorf("invalid data_policy local_retention")
	}
	for _, att := range p.Attestations {
		if !policyProvider.MatchString(att.Provider) {
			return p, fmt.Errorf("invalid data_policy provider")
		}
		if att.Endpoint != "" {
			if _, err := NormalizePolicyEndpoint(att.Endpoint); err != nil {
				return p, err
			}
		}
		if len(att.Guarantees) == 0 {
			return p, fmt.Errorf("empty data_policy guarantees")
		}
		for _, g := range att.Guarantees {
			if g == "no-training" || g == "zero-data-retention" {
				continue
			}
			kind, r, ok := strings.Cut(g, ":")
			if !ok || (kind != "processing" && kind != "storage") || !policyRegion.MatchString(r) {
				return p, fmt.Errorf("invalid data_policy guarantee")
			}
		}
	}
	return p, nil
}

// NormalizePolicyEndpoint preserves path identity; a prefix or lookalike is
// never an attestation match. HTTP is only accepted for local/in-pod hosts.
func NormalizePolicyEndpoint(raw string) (string, error) {
	u, err := url.Parse(raw)
	if err != nil || u.Hostname() == "" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.Opaque != "" {
		return "", fmt.Errorf("invalid data_policy endpoint")
	}
	host := strings.ToLower(u.Hostname())
	if u.Scheme != "https" {
		ip := net.ParseIP(host)
		local := host == "localhost" || strings.HasSuffix(host, ".localhost") || strings.HasSuffix(host, ".local") || (!strings.Contains(host, ".") && ip == nil) || (ip != nil && (ip.IsLoopback() || ip.IsPrivate()))
		if u.Scheme != "http" || !local {
			return "", fmt.Errorf("data_policy endpoint requires HTTPS")
		}
	}
	u.Host = strings.ToLower(u.Host)
	u.Path = strings.TrimRight(u.Path, "/")
	if u.RawPath != "" {
		return "", fmt.Errorf("data_policy endpoint must not contain an encoded path")
	}
	return u.String(), nil
}
