package tracejournal

import (
	"encoding/json"
	"strings"
	"testing"
)

func projectString(t *testing.T, raw string) string {
	t.Helper()
	out, err := json.Marshal(ProjectBody([]byte(raw)))
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}

func TestProjectionRemovesHiddenReasoningAndKeepsVisibleContent(t *testing.T) {
	cases := map[string]struct {
		in      string
		gone    []string
		kept    []string
		markers int
	}{
		"anthropic thinking blocks": {
			in:      `{"content":[{"type":"thinking","thinking":"SECRET-PLAN","signature":"SIG"},{"type":"redacted_thinking","data":"ENC"},{"type":"text","text":"visible answer"},{"type":"tool_use","id":"toolu_1","name":"lookup","input":{"q":1}}]}`,
			gone:    []string{"SECRET-PLAN", "SIG", "ENC"},
			kept:    []string{"visible answer", "toolu_1", "lookup"},
			markers: 2,
		},
		"responses reasoning items": {
			in:      `{"input":[{"type":"reasoning","id":"rs_1","encrypted_content":"OPAQUE","summary":[{"type":"summary_text","text":"HIDDEN"}]},{"type":"function_call_output","call_id":"call_1","output":"tool says hi"}],"include":["reasoning.encrypted_content"],"reasoning":{"effort":"high"}}`,
			gone:    []string{"OPAQUE", "HIDDEN", "rs_1"},
			kept:    []string{"call_1", "tool says hi", `"effort":"high"`, "reasoning.encrypted_content"},
			markers: 1,
		},
		"chat reasoning fields": {
			in:      `{"choices":[{"message":{"role":"assistant","content":"done","reasoning":"PRIVATE","reasoning_content":"PRIVATE2","reasoning_details":[{"text":"PRIVATE3"}],"tool_calls":[{"id":"call_9","function":{"name":"f","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
			gone:    []string{"PRIVATE"},
			kept:    []string{"done", "call_9", "tool_calls"},
			markers: 3,
		},
		"credential-named fields": {
			in:   `{"auth":{"Authorization":"Bearer abc","api_key":"k1","token":"t1","password":"p1"},"max_tokens":12,"token_count":3}`,
			gone: []string{"Bearer abc", `"k1"`, `"t1"`, `"p1"`},
			kept: []string{`"max_tokens":12`, `"token_count":3`},
		},
		"large integers survive": {
			in:   `{"id":12345678901234567890,"n":1.5}`,
			kept: []string{"12345678901234567890", "1.5"},
		},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got := projectString(t, tc.in)
			for _, s := range tc.gone {
				if strings.Contains(got, s) {
					t.Errorf("%q survived: %s", s, got)
				}
			}
			for _, s := range tc.kept {
				if !strings.Contains(got, s) {
					t.Errorf("%q lost: %s", s, got)
				}
			}
			if n := strings.Count(got, `"omitted":"hidden_reasoning"`); n != tc.markers {
				t.Errorf("markers = %d, want %d: %s", n, tc.markers, got)
			}
		})
	}
}

func TestProjectionHandlesEventStreams(t *testing.T) {
	stream := strings.Join([]string{
		`event: content_block_start`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`,
		``,
		`event: content_block_delta`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"STREAMED-SECRET"}}`,
		``,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"SIGNATURE"}}`,
		``,
		`data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"visible words"}}`,
		``,
		`data: {"choices":[{"delta":{"reasoning_content":"CHAT-SECRET","content":"chat words"}}]}`,
		``,
		`data: [DONE]`,
		``,
	}, "\n")
	got, ok := ProjectBody([]byte(stream)).(string)
	if !ok {
		t.Fatal("event stream not projected to text")
	}
	for _, s := range []string{"STREAMED-SECRET", "SIGNATURE", "CHAT-SECRET"} {
		if strings.Contains(got, s) {
			t.Errorf("%q survived: %s", s, got)
		}
	}
	for _, s := range []string{"visible words", "chat words", "data: [DONE]", "event: content_block_delta"} {
		if !strings.Contains(got, s) {
			t.Errorf("%q lost: %s", s, got)
		}
	}
}
