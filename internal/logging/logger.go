package logging

import (
	"encoding/json"
	"io"
	"sync"
	"time"
)

// Logger writes structured JSON logs suitable for claw audit ingestion.
type Logger struct {
	mu  sync.Mutex
	enc *json.Encoder
}

type entry struct {
	TS           string   `json:"ts"`
	ClawID       string   `json:"claw_id,omitempty"`
	Type         string   `json:"type"`
	Model        string   `json:"model,omitempty"`
	FeedName     string   `json:"feed_name,omitempty"`
	FeedURL      string   `json:"feed_url,omitempty"`
	LatencyMS    *int64   `json:"latency_ms,omitempty"`
	StatusCode   *int     `json:"status_code,omitempty"`
	TokensIn     *int     `json:"tokens_in,omitempty"`
	TokensOut    *int     `json:"tokens_out,omitempty"`
	CostUSD      *float64 `json:"cost_usd,omitempty"`
	Intervention *string  `json:"intervention"`
	Error        string   `json:"error,omitempty"`
	// provider_pool event fields
	Provider      string  `json:"provider,omitempty"`
	KeyID         string  `json:"key_id,omitempty"`
	Action        string  `json:"action,omitempty"`
	Reason        string  `json:"reason,omitempty"`
	CooldownUntil string  `json:"cooldown_until,omitempty"`
}

// CostInfo holds token counts and estimated cost for a single LLM request.
type CostInfo struct {
	InputTokens  int
	OutputTokens int
	CostUSD      float64
}

func New(w io.Writer) *Logger {
	if w == nil {
		w = io.Discard
	}
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	return &Logger{enc: enc}
}

func (l *Logger) LogRequest(clawID, model string) {
	l.log(entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "request",
		Model:        model,
		Intervention: nil,
	})
}

func (l *Logger) LogResponse(clawID, model string, statusCode int, latencyMS int64) {
	l.log(entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "response",
		Model:        model,
		LatencyMS:    ptrI64(latencyMS),
		StatusCode:   ptrInt(statusCode),
		Intervention: nil,
	})
}

func (l *Logger) LogError(clawID, model string, statusCode int, latencyMS int64, err error) {
	errText := ""
	if err != nil {
		errText = err.Error()
	}
	l.log(entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "error",
		Model:        model,
		LatencyMS:    ptrI64(latencyMS),
		StatusCode:   ptrInt(statusCode),
		Intervention: nil,
		Error:        errText,
	})
}

func (l *Logger) LogResponseWithCost(clawID, model string, statusCode int, latencyMS int64, ci *CostInfo) {
	e := entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "response",
		Model:        model,
		LatencyMS:    ptrI64(latencyMS),
		StatusCode:   ptrInt(statusCode),
		Intervention: nil,
	}
	if ci != nil {
		e.TokensIn = ptrInt(ci.InputTokens)
		e.TokensOut = ptrInt(ci.OutputTokens)
		e.CostUSD = ptrF64(ci.CostUSD)
	}
	l.log(e)
}

func (l *Logger) LogIntervention(clawID, model, reason string) {
	reasonCopy := reason
	l.log(entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "intervention",
		Model:        model,
		Intervention: &reasonCopy,
	})
}

func (l *Logger) LogFeedFetch(clawID, feedName, feedURL string, statusCode int, latencyMS int64, err error) {
	e := entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		ClawID:       clawID,
		Type:         "feed_fetch",
		FeedName:     feedName,
		FeedURL:      feedURL,
		LatencyMS:    ptrI64(latencyMS),
		Intervention: nil,
	}
	if statusCode > 0 {
		e.StatusCode = ptrInt(statusCode)
	}
	if err != nil {
		e.Error = err.Error()
	}
	l.log(e)
}

// LogProviderPool emits a structured provider_pool transition event.
// action is one of: "cooldown", "dead", "activated", "added", "deleted".
func (l *Logger) LogProviderPool(provider, keyID, action, reason, cooldownUntil string) {
	e := entry{
		TS:           time.Now().UTC().Format(time.RFC3339),
		Type:         "provider_pool",
		Provider:     provider,
		KeyID:        keyID,
		Action:       action,
		Reason:       reason,
		CooldownUntil: cooldownUntil,
		Intervention: nil,
	}
	l.log(e)
}

func (l *Logger) log(e entry) {
	if l == nil || l.enc == nil {
		return
	}
	l.mu.Lock()
	_ = l.enc.Encode(e)
	l.mu.Unlock()
}

func ptrInt(v int) *int {
	return &v
}

func ptrI64(v int64) *int64 {
	return &v
}

func ptrF64(v float64) *float64 {
	return &v
}
