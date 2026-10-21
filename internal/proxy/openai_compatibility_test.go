package proxy

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCurrentOpenAIWorkerEncoding(t *testing.T) {
	for _, tc := range []struct {
		model, effort       string
		sampling, wantError bool
	}{
		{"gpt-6.1-sol", "", false, false},
		{"gpt-6.1-sol-2026-10-01", "high", false, false},
		{"gpt-6-astra", "low", false, false},
		{"gpt-6-sol", "none", true, false},
		{"gpt-6-luna", "none", true, false},
		{"gpt-6-sol", "high", false, false},
		{"gpt-6.1-sol", "none", false, true},
		{"gpt-6-astra", "minimal", false, true},
	} {
		t.Run(tc.model+"/"+tc.effort, func(t *testing.T) {
			payload := map[string]any{"model": tc.model, "messages": []any{map[string]any{"role": "user", "content": "synthetic"}}, "max_tokens": 32768, "temperature": 0.7, "top_p": 0.9}
			if tc.effort != "" {
				payload["reasoning_effort"] = tc.effort
			}
			body, _ := json.Marshal(payload)
			upstream, err := encodeUpstreamRequest(dispatchCandidate{ProviderName: "openai", UpstreamModel: tc.model}, "/v1/chat/completions", payload, body)
			if tc.wantError {
				if _, ok := responsesAdapterClientError(err); !ok {
					t.Fatalf("expected fixed client error, got %v", err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if !upstream.Adapter || upstream.Path != "/v1/responses" {
				t.Fatal("current model did not use Responses")
			}
			var out map[string]any
			if err := json.Unmarshal(upstream.Body, &out); err != nil {
				t.Fatal(err)
			}
			if out["max_output_tokens"] != float64(32768) {
				t.Fatal("legacy token budget not translated")
			}
			for _, key := range []string{"temperature", "top_p"} {
				_, got := out[key]
				if got != tc.sampling {
					t.Fatalf("%s presence=%v wanted %v", key, got, tc.sampling)
				}
			}
			if !tc.sampling && !strings.Contains(strings.Join(upstream.Mutations, ","), "responses_adapter_dropped_sampling") {
				t.Fatal("sampling change unaudited")
			}
			if payload["temperature"] != 0.7 || payload["max_tokens"] != 32768 {
				t.Fatal("agent-facing request mutated")
			}
		})
	}
}

func TestCurrentOpenAICapabilityScope(t *testing.T) {
	for _, model := range []string{"gpt-6.1-solitude", "gpt-6-unknown", "gpt-6.1-sol-pro", "gpt-6.1-sol-2026-99-99"} {
		if responsesAPIRequired("openai", model) {
			t.Errorf("unknown sibling matched: %s", model)
		}
	}
	t.Setenv(EnvResponsesAPIModels, "other/gpt-6.1-sol")
	payload := map[string]any{"model": "gpt-6.1-sol", "temperature": 0.7}
	body, _ := json.Marshal(payload)
	upstream, err := encodeUpstreamRequest(dispatchCandidate{ProviderName: "other", UpstreamModel: "gpt-6.1-sol"}, "/v1/chat/completions", payload, body)
	if err != nil {
		t.Fatal(err)
	}
	var out map[string]any
	json.Unmarshal(upstream.Body, &out)
	if out["temperature"] != 0.7 || len(upstream.Mutations) != 0 {
		t.Fatal("another provider's model capabilities changed")
	}
}

func TestCurrentOpenAICapabilityOnRetry(t *testing.T) {
	body := []byte(`{"model":"gpt-6.1-sol","max_tokens":32768,"temperature":0.7}`)
	upstream, err := adaptChatBodyToResponsesForCandidate(body, nil, dispatchCandidate{ProviderName: "openai", UpstreamModel: "gpt-6.1-sol"})
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(upstream.Body), "temperature") || !strings.Contains(string(upstream.Body), "max_output_tokens") {
		t.Fatal("retry did not apply capabilities")
	}
}
