package ui

import (
	"embed"
	"encoding/json"
	"fmt"
	"html/template"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/cost"
	"github.com/mostlydev/cllama/internal/provider"
)

//go:embed templates/*.html
var templateFS embed.FS

// UIOption configures optional Handler dependencies.
type UIOption func(*Handler)

// WithAccumulator attaches a cost accumulator to the UI handler,
// enabling the /costs dashboard and /costs/api endpoint.
func WithAccumulator(acc *cost.Accumulator) UIOption {
	return func(h *Handler) {
		h.accumulator = acc
	}
}

// WithContextRoot sets the agent context directory for pod member listing.
func WithContextRoot(root string) UIOption {
	return func(h *Handler) {
		h.contextRoot = root
	}
}

// WithUIToken requires Bearer authentication on all UI routes.
// If the token is empty, authentication is disabled (dev mode).
func WithUIToken(token string) UIOption {
	return func(h *Handler) {
		h.uiToken = token
	}
}

type Handler struct {
	registry    *provider.Registry
	accumulator *cost.Accumulator
	contextRoot string
	uiToken     string
	tpl         *template.Template
}

type providerRow struct {
	Name         string    `json:"name"`
	BaseURL      string    `json:"baseURL"`
	Auth         string    `json:"auth"`
	MaskedKey    string    `json:"maskedKey"`
	ActiveKeyID  string    `json:"activeKeyID"`
	ReadyCount   int       `json:"readyCount"`
	CooldownCount int      `json:"cooldownCount"`
	DeadCount    int       `json:"deadCount"`
	Keys         []keyRow  `json:"keys"`
}

type keyRow struct {
	ID            string `json:"id"`
	Label         string `json:"label"`
	Source        string `json:"source"`
	State         string `json:"state"`
	CooldownUntil string `json:"cooldownUntil"`
	LastErrorCode int    `json:"lastErrorCode"`
	LastErrorAt   string `json:"lastErrorAt"`
	MaskedSecret  string `json:"maskedSecret"`
}

// -- costs API types --

type costsAPIResponse struct {
	TotalCostUSD float64                     `json:"total_cost_usd"`
	Agents       map[string]agentAPIResponse `json:"agents"`
}

type agentAPIResponse struct {
	TotalCostUSD  float64            `json:"total_cost_usd"`
	TotalRequests int                `json:"total_requests"`
	Models        []modelAPIResponse `json:"models"`
}

type modelAPIResponse struct {
	Provider    string  `json:"provider"`
	Model       string  `json:"model"`
	InputTokens int     `json:"input_tokens"`
	OutputTokens int    `json:"output_tokens"`
	CostUSD     float64 `json:"cost_usd"`
	Requests    int     `json:"requests"`
}

func NewHandler(reg *provider.Registry, opts ...UIOption) http.Handler {
	if reg == nil {
		reg = provider.NewRegistry("")
	}
	tpl := template.Must(template.ParseFS(templateFS, "templates/*.html"))
	h := &Handler{registry: reg, tpl: tpl}
	for _, o := range opts {
		o(h)
	}
	return h
}

func (h *Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !h.checkBearer(w, r) {
		return
	}
	switch {
	case r.Method == http.MethodGet && r.URL.Path == "/":
		h.renderDashboard(w)
	case r.Method == http.MethodGet && r.URL.Path == "/events":
		h.handleSSE(w, r)
	case r.Method == http.MethodGet && r.URL.Path == "/costs/api":
		h.handleCostsAPI(w)
	case r.Method == http.MethodPost && r.URL.Path == "/keys/add":
		h.handleKeyAdd(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/keys/activate":
		h.handleKeyActivate(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/keys/disable":
		h.handleKeyDisable(w, r)
	case r.Method == http.MethodPost && r.URL.Path == "/keys/delete":
		h.handleKeyDelete(w, r)
	default:
		http.NotFound(w, r)
	}
}

// checkBearer enforces the CLLAMA_UI_TOKEN when configured.
// It returns true if the request is authorized (or no token is configured).
func (h *Handler) checkBearer(w http.ResponseWriter, r *http.Request) bool {
	if h.uiToken == "" {
		return true
	}
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, "Bearer ") {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return false
	}
	presented := strings.TrimPrefix(auth, "Bearer ")
	if !constantTimeEqualStr(presented, h.uiToken) {
		http.Error(w, "unauthorized", http.StatusUnauthorized)
		return false
	}
	return true
}

func constantTimeEqualStr(a, b string) bool {
	if len(a) != len(b) {
		return false
	}
	var result byte
	for i := 0; i < len(a); i++ {
		result |= a[i] ^ b[i]
	}
	return result == 0
}

func (h *Handler) renderDashboard(w http.ResponseWriter) {
	state := h.buildDashboardState()
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = h.tpl.ExecuteTemplate(w, "dashboard.html", state)
}

func (h *Handler) handleCostsAPI(w http.ResponseWriter) {
	resp := h.buildCostsAPIResponse()
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	_ = enc.Encode(resp)
}

func (h *Handler) buildCostsAPIResponse() costsAPIResponse {
	resp := costsAPIResponse{
		Agents: make(map[string]agentAPIResponse),
	}
	if h.accumulator == nil {
		return resp
	}

	resp.TotalCostUSD = h.accumulator.TotalCost()
	grouped := h.accumulator.All()
	for id, entries := range grouped {
		agent := agentAPIResponse{}
		for _, e := range entries {
			agent.TotalRequests += e.RequestCount
			agent.TotalCostUSD += e.TotalCostUSD
			agent.Models = append(agent.Models, modelAPIResponse{
				Provider:     e.Provider,
				Model:        e.Model,
				InputTokens:  e.TotalInputTokens,
				OutputTokens: e.TotalOutputTokens,
				CostUSD:      e.TotalCostUSD,
				Requests:     e.RequestCount,
			})
		}
		resp.Agents[id] = agent
	}
	return resp
}

// -- SSE dashboard state types --

type dashboardState struct {
	PodName      string           `json:"podName,omitempty"`
	TotalCostUSD float64          `json:"totalCostUSD"`
	TotalReqs    int              `json:"totalRequests"`
	TotalTokens  int              `json:"totalTokens"`
	Providers    []providerRow    `json:"providers"`
	Agents       []dashboardAgent `json:"agents"`
}

type dashboardAgent struct {
	AgentID        string           `json:"agentId"`
	Service        string           `json:"service,omitempty"`
	Type           string           `json:"type,omitempty"`
	TotalRequests  int              `json:"totalRequests"`
	TotalCostUSD   float64          `json:"totalCostUSD"`
	TotalTokensIn  int              `json:"totalTokensIn"`
	TotalTokensOut int              `json:"totalTokensOut"`
	Models         []dashboardModel `json:"models"`
}

type dashboardModel struct {
	Provider  string  `json:"provider"`
	Model     string  `json:"model"`
	Requests  int     `json:"requests"`
	TokensIn  int     `json:"tokensIn"`
	TokensOut int     `json:"tokensOut"`
	CostUSD   float64 `json:"costUSD"`
}

// -- key management handlers --

func (h *Handler) handleKeyAdd(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}
	prov := strings.TrimSpace(r.FormValue("provider"))
	label := strings.TrimSpace(r.FormValue("label"))
	secret := strings.TrimSpace(r.FormValue("secret"))
	if prov == "" || secret == "" {
		http.Error(w, "provider and secret are required", http.StatusBadRequest)
		return
	}
	keyID, err := h.registry.AddRuntimeKey(prov, label, secret)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	_ = h.registry.SaveToFile()
	redirectWithMsg(w, r, fmt.Sprintf("Key %s added to %s", keyID, prov))
}

func (h *Handler) handleKeyActivate(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}
	prov := strings.TrimSpace(r.FormValue("provider"))
	keyID := strings.TrimSpace(r.FormValue("key_id"))
	if prov == "" || keyID == "" {
		http.Error(w, "provider and key_id are required", http.StatusBadRequest)
		return
	}
	if err := h.registry.ActivateKey(prov, keyID); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	_ = h.registry.SaveToFile()
	redirectWithMsg(w, r, fmt.Sprintf("Key %s activated for %s", keyID, prov))
}

func (h *Handler) handleKeyDisable(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}
	prov := strings.TrimSpace(r.FormValue("provider"))
	keyID := strings.TrimSpace(r.FormValue("key_id"))
	if prov == "" || keyID == "" {
		http.Error(w, "provider and key_id are required", http.StatusBadRequest)
		return
	}
	if err := h.registry.DisableKey(prov, keyID); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	_ = h.registry.SaveToFile()
	redirectWithMsg(w, r, fmt.Sprintf("Key %s disabled for %s", keyID, prov))
}

func (h *Handler) handleKeyDelete(w http.ResponseWriter, r *http.Request) {
	if err := r.ParseForm(); err != nil {
		http.Error(w, "bad form", http.StatusBadRequest)
		return
	}
	prov := strings.TrimSpace(r.FormValue("provider"))
	keyID := strings.TrimSpace(r.FormValue("key_id"))
	if prov == "" || keyID == "" {
		http.Error(w, "provider and key_id are required", http.StatusBadRequest)
		return
	}
	if err := h.registry.DeleteKey(prov, keyID); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	_ = h.registry.SaveToFile()
	redirectWithMsg(w, r, fmt.Sprintf("Key %s deleted from %s", keyID, prov))
}

func redirectWithMsg(w http.ResponseWriter, r *http.Request, _ string) {
	http.Redirect(w, r, "/", http.StatusSeeOther)
}

func (h *Handler) buildDashboardState() dashboardState {
	state := dashboardState{
		Providers: []providerRow{},
		Agents:    []dashboardAgent{},
	}

	// 1. Providers from registry (sorted by name, masked keys)
	all := h.registry.All()
	names := make([]string, 0, len(all))
	for name := range all {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		p := all[name]
		row := providerRow{
			Name:        name,
			BaseURL:     p.BaseURL,
			Auth:        p.Auth,
			MaskedKey:   maskActiveKey(p),
			ActiveKeyID: p.ActiveKeyID,
		}
		for _, k := range p.Keys {
			switch k.State {
			case "ready":
				row.ReadyCount++
			case "cooldown":
				row.CooldownCount++
			case "dead":
				row.DeadCount++
			}
			row.Keys = append(row.Keys, keyRow{
				ID:            k.ID,
				Label:         k.Label,
				Source:        k.Source,
				State:         string(k.State),
				CooldownUntil: k.CooldownUntil,
				LastErrorCode: k.LastErrorCode,
				LastErrorAt:   k.LastErrorAt,
				MaskedSecret:  maskKey(k.Secret),
			})
		}
		state.Providers = append(state.Providers, row)
	}

	// 2. Agents from context root (if available), merging cost data
	seenAgents := make(map[string]bool)
	if h.contextRoot != "" {
		agents, err := agentctx.ListAgents(h.contextRoot)
		if err == nil {
			for _, a := range agents {
				if state.PodName == "" && a.Pod != "" {
					state.PodName = a.Pod
				}
				da := dashboardAgent{
					AgentID: a.AgentID,
					Service: a.Service,
					Type:    a.Type,
					Models:  []dashboardModel{},
				}
				if h.accumulator != nil {
					for _, e := range h.accumulator.ByAgent(a.AgentID) {
						da.TotalRequests += e.RequestCount
						da.TotalCostUSD += e.TotalCostUSD
						da.TotalTokensIn += e.TotalInputTokens
						da.TotalTokensOut += e.TotalOutputTokens
						da.Models = append(da.Models, dashboardModel{
							Provider:  e.Provider,
							Model:     e.Model,
							Requests:  e.RequestCount,
							TokensIn:  e.TotalInputTokens,
							TokensOut: e.TotalOutputTokens,
							CostUSD:   e.TotalCostUSD,
						})
					}
				}
				state.Agents = append(state.Agents, da)
				seenAgents[a.AgentID] = true
			}
		}
	}

	// 3. Agents from cost data that aren't in context (standalone requests)
	if h.accumulator != nil {
		grouped := h.accumulator.All()
		agentIDs := make([]string, 0, len(grouped))
		for id := range grouped {
			agentIDs = append(agentIDs, id)
		}
		sort.Strings(agentIDs)
		for _, id := range agentIDs {
			if seenAgents[id] {
				continue
			}
			da := dashboardAgent{
				AgentID: id,
				Models:  []dashboardModel{},
			}
			for _, e := range grouped[id] {
				da.TotalRequests += e.RequestCount
				da.TotalCostUSD += e.TotalCostUSD
				da.TotalTokensIn += e.TotalInputTokens
				da.TotalTokensOut += e.TotalOutputTokens
				da.Models = append(da.Models, dashboardModel{
					Provider:  e.Provider,
					Model:     e.Model,
					Requests:  e.RequestCount,
					TokensIn:  e.TotalInputTokens,
					TokensOut: e.TotalOutputTokens,
					CostUSD:   e.TotalCostUSD,
				})
			}
			state.Agents = append(state.Agents, da)
		}
	}

	// 4. Compute totals
	if h.accumulator != nil {
		state.TotalCostUSD = h.accumulator.TotalCost()
	}
	for _, a := range state.Agents {
		state.TotalReqs += a.TotalRequests
		state.TotalTokens += a.TotalTokensIn + a.TotalTokensOut
	}

	// 5. Sort agents by ID
	sort.Slice(state.Agents, func(i, j int) bool {
		return state.Agents[i].AgentID < state.Agents[j].AgentID
	})

	return state
}

func (h *Handler) handleSSE(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")

	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()

	// Send initial state immediately
	if !h.writeSSEEvent(w, flusher) {
		return
	}

	for {
		select {
		case <-r.Context().Done():
			return
		case <-ticker.C:
			if !h.writeSSEEvent(w, flusher) {
				return
			}
		}
	}
}

func (h *Handler) writeSSEEvent(w http.ResponseWriter, flusher http.Flusher) bool {
	state := h.buildDashboardState()
	data, err := json.Marshal(state)
	if err != nil {
		return true // state is all primitives; marshal won't fail in practice
	}
	if _, err := fmt.Fprintf(w, "data:%s\n\n", data); err != nil {
		return false // client disconnected
	}
	flusher.Flush()
	return true
}

func maskActiveKey(state *provider.ProviderState) string {
	if state == nil {
		return ""
	}
	// Show the active key's secret, masked. Fall back to first ready key.
	for _, k := range state.Keys {
		if k.ID == state.ActiveKeyID {
			return maskKey(k.Secret)
		}
	}
	for _, k := range state.Keys {
		if k.State == "ready" {
			return maskKey(k.Secret)
		}
	}
	return ""
}

func maskKey(key string) string {
	key = strings.TrimSpace(key)
	if key == "" {
		return ""
	}
	if len(key) <= 8 {
		return "****"
	}
	return key[:4] + "..." + key[len(key)-4:]
}
