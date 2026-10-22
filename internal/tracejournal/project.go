package tracejournal

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"reflect"
	"sort"
	"strings"
)

const redactedCredential = "[redacted:credential]"

var hiddenReasoningMarker = map[string]any{"omitted": "hidden_reasoning"}

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

// sseFields are the event-stream framing lines kept besides data lines.
var sseFields = []string{"event:", "id:", "retry:"}

// projector turns a captured body into its recordable form. lossless stays
// true only when nothing but hidden reasoning was removed.
type projector struct {
	secrets  []string
	lossless bool
	depth    int // nesting of JSON text inside strings, such as tool arguments
}

// maxEmbeddedDepth bounds how deep JSON text nested inside strings is
// decoded and projected. JSON text nested deeper is omitted and the capture
// marked lossy.
const maxEmbeddedDepth = 4

// ProjectBody returns the recordable form of a captured body with no known
// secrets. See projectBody.
func ProjectBody(raw []byte) any {
	v, _ := projectBody(raw, nil)
	return v
}

// projectBody returns the recordable form of a captured body and whether it
// is lossless. A JSON body comes back as a JSON value. An event stream comes
// back as text with each event's data projected. Anything that cannot be
// parsed, including a partial or interrupted payload, is omitted rather than
// kept raw, because hidden reasoning in it could not be removed. Known
// secrets are redacted inside decoded strings.
func projectBody(raw []byte, secrets []string) (any, bool) {
	p := &projector{secrets: secrets, lossless: true}
	if v, ok := decodeJSON(raw); ok {
		return p.value(v), p.lossless
	}
	if text, ok := p.eventStream(raw); ok {
		return text, p.lossless
	}
	return map[string]any{"omitted": "unparseable", "bytes": len(raw)}, false
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
	// Only whitespace may follow; trailing data of any kind is not JSON.
	if _, err := dec.Token(); !errors.Is(err, io.EOF) {
		return nil, false
	}
	return v, true
}

// eventStream projects a server-sent event stream event by event. Data
// lines of one event are joined as the SSE format specifies. An event whose
// data is not JSON (other than [DONE]) is replaced by a marker, and any line
// that is neither data nor known framing is dropped.
func (p *projector) eventStream(raw []byte) (string, bool) {
	normalized := strings.ReplaceAll(string(raw), "\r\n", "\n")
	if !strings.HasPrefix(normalized, "data:") && !strings.Contains(normalized, "\ndata:") {
		return "", false
	}
	var out []string
	for _, event := range strings.Split(normalized, "\n\n") {
		if strings.TrimSpace(event) == "" {
			continue
		}
		var framing, data []string
		hasData := false
		for _, line := range strings.Split(event, "\n") {
			switch {
			case strings.HasPrefix(line, "data:"):
				hasData = true
				data = append(data, strings.TrimPrefix(strings.TrimPrefix(line, "data:"), " "))
			case isSSEFraming(line):
				framing = append(framing, redact(line, p.secrets))
			case line == "" || strings.HasPrefix(line, ":"):
				// Blank lines and comments carry nothing to keep.
			default:
				p.lossless = false
			}
		}
		lines := framing
		if hasData {
			payload := strings.Join(data, "\n")
			switch {
			case strings.TrimSpace(payload) == "[DONE]":
				lines = append(lines, "data: [DONE]")
			default:
				if v, ok := decodeJSON([]byte(payload)); ok {
					encoded, err := json.Marshal(p.value(v))
					if err == nil {
						lines = append(lines, "data: "+string(encoded))
						break
					}
				}
				p.lossless = false
				lines = append(lines, `data: {"omitted":"unparseable_event"}`)
			}
		}
		if len(lines) > 0 {
			out = append(out, strings.Join(lines, "\n"))
		}
	}
	return strings.Join(out, "\n\n") + "\n\n", true
}

func isSSEFraming(line string) bool {
	for _, field := range sseFields {
		if strings.HasPrefix(line, field) {
			return true
		}
	}
	return false
}

// isHiddenReasoningType matches thinking blocks and deltas, Responses
// reasoning items and every Responses reasoning stream event type.
func isHiddenReasoningType(typ string) bool {
	typ = strings.ToLower(typ)
	return strings.Contains(typ, "reasoning") || strings.Contains(typ, "thinking") || typ == "signature_delta"
}

func (p *projector) value(v any) any {
	switch t := v.(type) {
	case map[string]any:
		if typ, _ := t["type"].(string); isHiddenReasoningType(typ) {
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
					out[k] = p.value(m)
				} else {
					out[k] = hiddenReasoningMarker
				}
			case credentialKeys[lower]:
				if _, ok := val.(string); ok {
					out[k] = redactedCredential
				} else {
					out[k] = p.value(val)
				}
			default:
				out[k] = p.value(val)
			}
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = p.value(val)
		}
		return out
	case string:
		return p.embedded(redact(t, p.secrets))
	default:
		return v
	}
}

// embedded projects JSON carried as text, such as Chat function.arguments,
// where an escaped credential or reasoning field would survive literal
// redaction of the outer string. A string that changes is re-encoded as
// valid JSON; one that does not is kept byte for byte.
func (p *projector) embedded(s string) string {
	trimmed := strings.TrimSpace(s)
	if len(trimmed) < 2 || (trimmed[0] != '{' && trimmed[0] != '[') {
		return s
	}
	v, ok := decodeJSON([]byte(trimmed))
	if !ok {
		return s
	}
	if p.depth >= maxEmbeddedDepth {
		// Deeper JSON text is not inspected, so it cannot be kept: an
		// escaped credential or reasoning field could hide in it.
		p.lossless = false
		return `{"omitted":"embedded_json_too_deep"}`
	}
	p.depth++
	projected := p.value(v)
	p.depth--
	if reflect.DeepEqual(v, projected) {
		return s
	}
	encoded, err := json.Marshal(projected)
	if err != nil {
		p.lossless = false
		return redactedCredential
	}
	return string(encoded)
}

// redactFields redacts known secrets in the plain string values of a record
// before it is encoded. Bodies arrive already projected as json.RawMessage.
func redactFields(v any, secrets []string) any {
	switch t := v.(type) {
	case string:
		return redact(t, secrets)
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, val := range t {
			out[k] = redactFields(val, secrets)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, val := range t {
			out[i] = redactFields(val, secrets)
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
		if strings.TrimSpace(s) == "" || seen[s] {
			continue
		}
		seen[s] = true
		out = append(out, s)
	}
	// Longest first, so a secret that contains another is replaced whole.
	sort.Slice(out, func(i, j int) bool { return len(out[i]) > len(out[j]) })
	return out
}

func redact(s string, secrets []string) string {
	for _, secret := range secrets {
		if strings.Contains(s, secret) {
			s = strings.ReplaceAll(s, secret, redactedCredential)
		}
	}
	return s
}

// redactSecrets is a backstop over the final encoded record. It only
// touches secrets long enough and plain enough that a match cannot be JSON
// structure; every decoded string was already redacted before encoding.
func redactSecrets(line []byte, secrets []string) []byte {
	for _, s := range secrets {
		if len(s) < 8 || strings.Trim(s, "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._~+/=-:") != "" {
			continue
		}
		if bytes.Contains(line, []byte(s)) {
			line = bytes.ReplaceAll(line, []byte(s), []byte(redactedCredential))
		}
	}
	return line
}
