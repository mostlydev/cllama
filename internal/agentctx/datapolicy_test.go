package agentctx

import "testing"

func TestDataPolicyMetadataDefaultsAndValidation(t *testing.T) {
	p, err := (&AgentContext{}).EffectiveDataPolicy()
	if err != nil || p.Training != "none" || p.Retention != "none" || p.LocalRetention != "none" || len(p.Regions) != 2 {
		t.Fatalf("%+v %v", p, err)
	}
	for _, raw := range []any{nil, "none", map[string]any{"version": 2}, map[string]any{"training": "yes"}, map[string]any{"regions": []string{}}, map[string]any{"regions": []string{"global", "us"}}, map[string]any{"unknown": true}, map[string]any{"attestations": []any{map[string]any{"provider": "openai", "guarantees": []string{"magic"}}}}} {
		if _, err := (&AgentContext{Metadata: map[string]any{"data_policy": raw}}).EffectiveDataPolicy(); err == nil {
			t.Errorf("accepted %#v", raw)
		}
	}
}
func TestDataPolicyEndpointScope(t *testing.T) {
	for _, endpoint := range []string{"https://api.example/v1", "http://localhost:8000/v1", "http://127.0.0.1/v1", "http://10.1.2.3/v1", "http://gateway:3000/v1"} {
		if _, err := NormalizePolicyEndpoint(endpoint); err != nil {
			t.Errorf("%s: %v", endpoint, err)
		}
	}
	for _, endpoint := range []string{"http://api.example/v1", "https://user:pass@api.example/v1", "https://api.example/v1?secret=x", "https://api.example/v1#x", "https://api.example/%76%31"} {
		if _, err := NormalizePolicyEndpoint(endpoint); err == nil {
			t.Errorf("accepted %s", endpoint)
		}
	}
}
