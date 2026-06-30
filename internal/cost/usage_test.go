package cost

import "testing"

func TestExtractUsageFromJSON(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-1",
		"choices": [{"message": {"content": "hello"}}],
		"usage": {
			"prompt_tokens": 150,
			"completion_tokens": 42,
			"total_tokens": 192
		}
	}`)

	u, err := ExtractUsage(body)
	if err != nil {
		t.Fatal(err)
	}
	if u.PromptTokens != 150 {
		t.Errorf("expected 150 prompt tokens, got %d", u.PromptTokens)
	}
	if u.CompletionTokens != 42 {
		t.Errorf("expected 42 completion tokens, got %d", u.CompletionTokens)
	}
}

func TestExtractUsageFromAnthropicJSON(t *testing.T) {
	body := []byte(`{
		"id": "msg_01",
		"type": "message",
		"usage": {
			"input_tokens": 321,
			"output_tokens": 89
		}
	}`)

	u, err := ExtractUsage(body)
	if err != nil {
		t.Fatal(err)
	}
	if u.PromptTokens != 321 {
		t.Errorf("expected 321 prompt tokens, got %d", u.PromptTokens)
	}
	if u.CompletionTokens != 89 {
		t.Errorf("expected 89 completion tokens, got %d", u.CompletionTokens)
	}
	if u.TotalTokens != 410 {
		t.Errorf("expected total tokens derived from input/output, got %d", u.TotalTokens)
	}
}

func TestExtractUsageIncludesReportedCost(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-1",
		"usage": {
			"prompt_tokens": 120,
			"completion_tokens": 30,
			"cost": 0.0042
		}
	}`)

	u, err := ExtractUsage(body)
	if err != nil {
		t.Fatal(err)
	}
	if u.ReportedCostUSD == nil {
		t.Fatal("expected reported cost to be parsed")
	}
	if *u.ReportedCostUSD != 0.0042 {
		t.Fatalf("expected reported cost 0.0042, got %f", *u.ReportedCostUSD)
	}
}

func TestExtractUsageIncludesCacheTokenDetails(t *testing.T) {
	body := []byte(`{
		"id": "chatcmpl-1",
		"usage": {
			"prompt_tokens": 120,
			"completion_tokens": 30,
			"prompt_tokens_details": {
				"cached_tokens": 100,
				"cache_write_tokens": 20
			}
		}
	}`)

	u, err := ExtractUsage(body)
	if err != nil {
		t.Fatal(err)
	}
	if u.CachedTokens == nil || *u.CachedTokens != 100 {
		t.Fatalf("expected cached_tokens=100, got %+v", u.CachedTokens)
	}
	if u.CacheWriteTokens == nil || *u.CacheWriteTokens != 20 {
		t.Fatalf("expected cache_write_tokens=20, got %+v", u.CacheWriteTokens)
	}
}

func TestExtractUsageIncludesAnthropicCacheTokenDetails(t *testing.T) {
	body := []byte(`{
		"id": "msg_01",
		"type": "message",
		"usage": {
			"input_tokens": 321,
			"output_tokens": 89,
			"cache_read_input_tokens": 300,
			"cache_creation_input_tokens": 21
		}
	}`)

	u, err := ExtractUsage(body)
	if err != nil {
		t.Fatal(err)
	}
	if u.CachedTokens == nil || *u.CachedTokens != 300 {
		t.Fatalf("expected cached_tokens=300, got %+v", u.CachedTokens)
	}
	if u.CacheWriteTokens == nil || *u.CacheWriteTokens != 21 {
		t.Fatalf("expected cache_write_tokens=21, got %+v", u.CacheWriteTokens)
	}
}

func TestExtractUsageMissing(t *testing.T) {
	body := []byte(`{"id": "chatcmpl-1", "choices": []}`)
	u, err := ExtractUsage(body)
	if err != nil {
		t.Fatal(err)
	}
	if u.PromptTokens != 0 || u.CompletionTokens != 0 {
		t.Errorf("expected zero usage when missing, got %+v", u)
	}
}

func TestExtractUsageFromSSE(t *testing.T) {
	// SSE stream: final data chunk before [DONE] contains usage
	stream := []byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n" +
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":100,\"completion_tokens\":20,\"total_tokens\":120}}\n\n" +
		"data: [DONE]\n\n")
	u, err := ExtractUsageFromSSE(stream)
	if err != nil {
		t.Fatal(err)
	}
	if u.PromptTokens != 100 {
		t.Errorf("expected 100 prompt tokens, got %d", u.PromptTokens)
	}
	if u.CompletionTokens != 20 {
		t.Errorf("expected 20 completion tokens, got %d", u.CompletionTokens)
	}
}

func TestExtractUsageFromAnthropicSSE(t *testing.T) {
	stream := []byte("event: message_start\n" +
		"data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":150}}}\n\n" +
		"event: message_delta\n" +
		"data: {\"type\":\"message_delta\",\"usage\":{\"output_tokens\":37}}\n\n" +
		"event: message_stop\n" +
		"data: {\"type\":\"message_stop\"}\n\n")

	u, err := ExtractUsageFromSSE(stream)
	if err != nil {
		t.Fatal(err)
	}
	if u.PromptTokens != 150 {
		t.Errorf("expected 150 prompt tokens, got %d", u.PromptTokens)
	}
	if u.CompletionTokens != 37 {
		t.Errorf("expected 37 completion tokens, got %d", u.CompletionTokens)
	}
	if u.TotalTokens != 187 {
		t.Errorf("expected derived total tokens, got %d", u.TotalTokens)
	}
}

func TestExtractUsageFromSSENoUsage(t *testing.T) {
	stream := []byte("data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\ndata: [DONE]\n\n")
	u, err := ExtractUsageFromSSE(stream)
	if err != nil {
		t.Fatal(err)
	}
	if u.PromptTokens != 0 {
		t.Errorf("expected 0, got %d", u.PromptTokens)
	}
}
