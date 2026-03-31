package proxy

import (
	"crypto/subtle"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/sessionhistory"
)

const historyReplayAuthService = "cllama-history"

func (h *Handler) HandleHistory(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	if h.sessionRecorder == nil || h.sessionRecorder.BaseDir() == "" {
		http.Error(w, "history not configured", http.StatusNotFound)
		return
	}

	targetAgentID := strings.TrimSpace(r.PathValue("agentID"))
	if targetAgentID == "" {
		http.Error(w, "missing agent id", http.StatusBadRequest)
		return
	}

	targetCtx, err := h.loadContext(targetAgentID)
	if err != nil {
		http.Error(w, "agent context not found", http.StatusNotFound)
		return
	}
	token := extractPresentedToken(r)
	if token == "" {
		http.Error(w, "missing bearer token", http.StatusUnauthorized)
		return
	}
	if err := h.authorizeHistoryRequest(token, targetCtx); err != nil {
		http.Error(w, err.Error(), http.StatusForbidden)
		return
	}

	after, err := parseAfterParam(r.URL.Query().Get("after"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	limit, err := parseHistoryLimit(r.URL.Query().Get("limit"))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	entries, err := sessionhistory.ReadEntries(h.sessionRecorder.BaseDir(), targetAgentID, after, limit)
	if err != nil {
		http.Error(w, fmt.Sprintf("read history: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/x-ndjson")
	enc := json.NewEncoder(w)
	for _, entry := range entries {
		if err := enc.Encode(entry); err != nil {
			return
		}
	}
}

func (h *Handler) authorizeHistoryRequest(token string, targetCtx *agentctx.AgentContext) error {
	if h.adminToken != "" && secureEqual(token, h.adminToken) {
		return nil
	}
	if secureEqual(token, targetCtx.MetadataToken()) {
		return nil
	}
	for _, entry := range targetCtx.ServiceAuth {
		if entry.Service == historyReplayAuthService && entry.AuthType == "bearer" && entry.Token != "" && secureEqual(token, entry.Token) {
			return nil
		}
	}
	return fmt.Errorf("forbidden")
}

func extractPresentedToken(r *http.Request) string {
	auth := strings.TrimSpace(r.Header.Get("Authorization"))
	if auth == "" {
		return strings.TrimSpace(r.Header.Get("x-api-key"))
	}
	if !strings.HasPrefix(strings.ToLower(auth), "bearer ") {
		return ""
	}
	return strings.TrimSpace(auth[len("Bearer "):])
}

func parseAfterParam(raw string) (*time.Time, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, nil
	}
	ts, err := time.Parse(time.RFC3339, raw)
	if err != nil {
		return nil, fmt.Errorf("after must be RFC3339")
	}
	return &ts, nil
}

func parseHistoryLimit(raw string) (int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return sessionhistory.DefaultReadLimit, nil
	}
	var limit int
	if _, err := fmt.Sscanf(raw, "%d", &limit); err != nil {
		return 0, fmt.Errorf("limit must be an integer")
	}
	if limit <= 0 {
		return 0, fmt.Errorf("limit must be > 0")
	}
	if limit > sessionhistory.MaxReadLimit {
		limit = sessionhistory.MaxReadLimit
	}
	return limit, nil
}

func secureEqual(a, b string) bool {
	if a == "" || b == "" {
		return false
	}
	return subtle.ConstantTimeCompare([]byte(a), []byte(b)) == 1
}
