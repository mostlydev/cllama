package tracejournal

import (
	"bytes"
	"encoding/json"
	"sort"
	"strings"
)

// minSecretLen keeps short values (which would match ordinary text) out of
// literal redaction. Real credentials are far longer.
const minSecretLen = 8

const redactedCredential = "[redacted:credential]"

var hiddenReasoningMarker = map[string]any{"omitted": "hidden_reasoning"}

// hiddenReasoningTypes are content blocks, stream deltas and Responses items
// whose whole value is hidden reasoning.
var hiddenReasoningTypes = map[string]bool{
	"thinking":          true,
	"redacted_thinking": true,
	"thinking_delta":    true,
	"signature_delta":   true,
	"reasoning":         true,
}

// hiddenReasoningKeys hold hidden reasoning whatever their value. A
// "reasoning" key holding an object is request configuration (effort,
// summary) and is kept; holding text or a list it is reasoning content.
var hiddenReasoningKeys = map[string]bool{
	"reasoning_content": true,
	"reasoning_details": true,
	"encrypted_content": true,
}

var credentialKeys = map[string]bool{
	"authorization": true,
	"api_key":       true,
	"apikey":        true,
	"x-api-key":     true,
	"x_api_key":     true,
	"access_token":  true,
	"refresh_token": true,
	"client_secret": true,
	"secret":        true,
	"password":      true,
	"token":         true,
	"bearer":        true,
}

// ProjectBody returns the recordable form of a captured body. JSON bodies
// come back as JSON values; server-sent event streams and other text come
// back as a string with each JSON data payload projected. Hidden reasoning is
// replaced by an explicit marker and never retained or hashed.
func ProjectBody(raw []byte) any {
	if v, ok := decodeJSON(raw); ok {
		return projectValue(v)
	}
	return projectEventStream(raw)
}

// decodeJSON decodes exactly one JSON value, keeping numbers verbatim so
// large integers survive re-encoding.
func decodeJSON(raw []byte) (any, bool) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, false
	}
	if _, err := dec.Token(); err == nil {
		return nil, false
	}
	return v, true
}

func projectEventStream(raw []byte) string {
	lines := bytes.Split(raw, []byte("\n"))
	for i, line := range lines {
		trimmed := bytes.TrimRight(line, "\r")
		payload, ok := bytes.CutPrefix(trimmed, []byte("data:"))
		if !ok {
			continue
		}
		payload = bytes.TrimSpace(payload)
		v, ok := decodeJSON(payload)
		if !ok {
			continue
		}
		projected, err := json.Marshal(projectValue(v))
		if err != nil {
			continue
		}
		lines[i] = append([]byte("data: "), projected...)
	}
	return string(bytes.Join(lines, []byte("\n")))
}

func projectValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		if typ, _ := t["type"].(string); hiddenReasoningTypes[typ] {
			return hiddenReasoningMarker
		}
		out := make(map[string]any, len(t))
		for k, val := range t {
			lower := strings.ToLower(k)
			switch {
			case hiddenReasoningKeys[lower]:
				out[k] = hiddenReasoningMarker
			case lower == "reasoning":
				if m, ok := val.(map[string]any); ok {
					out[k] = projectValue(m)
				} else {
					out[k] = hiddenReasoningMarker
				}
			case credentialKeys[lower]:
				if _, ok := val.(string); ok {
					out[k] = redactedCredential
				} else {
					out[k] = projectValue(val)
				}
			default:
				out[k] = projectValue(val)
			}
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = projectValue(val)
		}
		return out
	default:
		return v
	}
}

func usableSecrets(secrets []string) []string {
	seen := make(map[string]bool, len(secrets))
	out := make([]string, 0, len(secrets))
	for _, s := range secrets {
		s = strings.TrimSpace(s)
		if len(s) < minSecretLen || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	// Longest first, so a secret that contains another is replaced whole.
	sort.Slice(out, func(i, j int) bool { return len(out[i]) > len(out[j]) })
	return out
}

// redactSecrets replaces every literal occurrence of a known secret in an
// encoded record. It runs on the final line, so it covers every field.
func redactSecrets(line []byte, secrets []string) []byte {
	for _, s := range secrets {
		if bytes.Contains(line, []byte(s)) {
			line = bytes.ReplaceAll(line, []byte(s), []byte(redactedCredential))
		}
	}
	return line
}
