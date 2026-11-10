package agentctx

import "testing"

func TestHostedCapabilitiesGrant(t *testing.T) {
	for name, tc := range map[string]struct {
		meta map[string]any
		want []string
		bad  bool
	}{
		"absent":       {map[string]any{}, nil, false},
		"empty":        {map[string]any{"hosted_capabilities": []any{}}, nil, false},
		"one":          {map[string]any{"hosted_capabilities": []any{"web_search"}}, []string{"web_search"}, false},
		"all":          {map[string]any{"hosted_capabilities": []any{"web_search", "code_interpreter", "file_search", "image_generation"}}, []string{"web_search", "code_interpreter", "file_search", "image_generation"}, false},
		"null":         {map[string]any{"hosted_capabilities": nil}, nil, true},
		"a string":     {map[string]any{"hosted_capabilities": "web_search"}, nil, true},
		"a map":        {map[string]any{"hosted_capabilities": map[string]any{"web_search": true}}, nil, true},
		"not a string": {map[string]any{"hosted_capabilities": []any{true}}, nil, true},
		"mcp":          {map[string]any{"hosted_capabilities": []any{"mcp"}}, nil, true},
		"unknown":      {map[string]any{"hosted_capabilities": []any{"web_search", "anything"}}, nil, true},
		"wildcard":     {map[string]any{"hosted_capabilities": []any{"*"}}, nil, true},
		"wrong case":   {map[string]any{"hosted_capabilities": []any{"WEB_SEARCH"}}, nil, true},
		"padded":       {map[string]any{"hosted_capabilities": []any{" web_search"}}, nil, true},
	} {
		got, err := (&AgentContext{Metadata: tc.meta}).HostedCapabilities()
		if (err != nil) != tc.bad {
			t.Errorf("%s: err = %v", name, err)
			continue
		}
		if tc.bad {
			if got != nil {
				t.Errorf("%s: an invalid grant returned %v", name, got)
			}
			continue
		}
		if len(got) != len(tc.want) {
			t.Errorf("%s: grants = %v", name, got)
		}
		for _, w := range tc.want {
			if !got[w] {
				t.Errorf("%s: missing %s", name, w)
			}
		}
	}
	if got, err := (*AgentContext)(nil).HostedCapabilities(); err != nil || len(got) != 0 {
		t.Fatalf("nil context: %v %v", got, err)
	}
}
