package subscription

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"
)

// Manager is the running proxy's view of the saved sign-ins. It reads the
// record for each request, so a sign-in made by the operator command takes
// effect without a restart.
type Manager struct {
	clients map[string]*Client

	mu sync.Mutex
	// secrets remembers token values seen by this process, including rotated
	// ones, so they can be redacted from records written later.
	secrets map[string]struct{}
	served  map[string]map[string]int
	// failed holds each agent's most recent plan-route failure reason.
	failed map[string]string
}

// NewManager returns a manager for every known plan under an auth directory.
func NewManager(authDir string) *Manager {
	store := NewStore(authDir)
	m := &Manager{clients: map[string]*Client{}, secrets: map[string]struct{}{}, served: map[string]map[string]int{}, failed: map[string]string{}}
	for _, name := range Providers() {
		plan, _ := Lookup(name)
		m.clients[name] = NewClient(plan, store)
	}
	return m
}

// NewManagerWithClients builds a manager over explicit clients. Tests use it
// to point a plan at a local server.
func NewManagerWithClients(clients ...*Client) *Manager {
	m := &Manager{clients: map[string]*Client{}, secrets: map[string]struct{}{}, served: map[string]map[string]int{}, failed: map[string]string{}}
	for _, c := range clients {
		m.clients[c.Plan.Provider] = c
	}
	return m
}

// Plan returns the plan routed under a provider name.
func (m *Manager) Plan(provider string) (Plan, bool) {
	if m == nil {
		return Plan{}, false
	}
	c, ok := m.clients[provider]
	if !ok {
		return Plan{}, false
	}
	return c.Plan, true
}

func (m *Manager) remember(rec *Record) {
	if rec == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, s := range []string{rec.AccessToken, rec.RefreshToken} {
		if s != "" {
			m.secrets[s] = struct{}{}
		}
	}
}

// Secrets lists every token value this process has seen, for redaction.
func (m *Manager) Secrets() []string {
	if m == nil {
		return nil
	}
	for name, c := range m.clients {
		if rec, err := c.Store.Load(name); err == nil {
			m.remember(rec)
		}
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]string, 0, len(m.secrets))
	for s := range m.secrets {
		out = append(out, s)
	}
	return out
}

func signinRequired(detail string) error {
	return &Error{Reason: ReasonSigninRequired, Terminal: true, Detail: detail}
}

// Token returns a current access token for a plan provider, refreshing it
// when it is near expiry.
func (m *Manager) Token(ctx context.Context, provider string) (string, error) {
	return m.token(ctx, provider, "")
}

// Reauthorize is called after the vendor rejected a token as unauthorized.
// If the saved record already holds a newer token it is returned; otherwise
// the refresh token is spent once. A terminal refusal marks the sign-in as
// needing the operator.
func (m *Manager) Reauthorize(ctx context.Context, provider, rejected string) (string, error) {
	if rejected == "" {
		return "", signinRequired("")
	}
	return m.token(ctx, provider, rejected)
}

func (m *Manager) token(ctx context.Context, provider, rejected string) (string, error) {
	if m == nil {
		return "", signinRequired("plan sign-in is not configured")
	}
	c, ok := m.clients[provider]
	if !ok {
		return "", signinRequired("unknown plan provider")
	}
	fresh := func(rec *Record) bool {
		return rec.AccessToken != "" && rec.AccessToken != rejected && c.now().Before(rec.ExpiresAt.Add(-refreshSkew))
	}
	rec, err := c.Store.Load(provider)
	if errors.Is(err, ErrNotSignedIn) {
		return "", signinRequired("not signed in")
	}
	if err != nil {
		return "", &Error{Reason: ReasonSigninRequired, Terminal: true, Detail: "sign-in record unreadable"}
	}
	m.remember(rec)
	if !rec.Usable() || !rec.HasScope(c.Plan.PlanScope) {
		return "", signinRequired(rec.Reason)
	}
	if fresh(rec) {
		return rec.AccessToken, nil
	}

	unlock, err := c.Store.Lock(ctx, provider)
	if err != nil {
		return "", &Error{Reason: ReasonUnavailable, Detail: "sign-in lock unavailable"}
	}
	defer unlock()
	// Another process may have refreshed while this one waited.
	rec, err = c.Store.Load(provider)
	if err != nil {
		return "", signinRequired("sign-in record unreadable")
	}
	m.remember(rec)
	if !rec.Usable() {
		return "", signinRequired(rec.Reason)
	}
	if fresh(rec) {
		return rec.AccessToken, nil
	}
	next, err := c.Refresh(ctx, rec)
	if err != nil {
		// A transient refresh failure does not cost a token that still works.
		if ReasonOf(err) == ReasonUnavailable && rec.AccessToken != "" && rec.AccessToken != rejected && c.now().Before(rec.ExpiresAt) {
			return rec.AccessToken, nil
		}
		return "", err
	}
	m.remember(next)
	return next.AccessToken, nil
}

// Status is the operator-visible state of one plan sign-in. It never carries
// a token.
type Status struct {
	Provider    string     `json:"provider"`
	State       string     `json:"state"`
	Account     string     `json:"account,omitempty"`
	Reason      string     `json:"reason,omitempty"`
	SignedInAt  *time.Time `json:"signed_in_at,omitempty"`
	RefreshedAt *time.Time `json:"refreshed_at,omitempty"`
	// LimitedUntil is set while the plan's usage limit is in cooldown.
	LimitedUntil string `json:"limited_until,omitempty"`
}

// Sign-in states reported by Status.
const (
	StatusSignedIn       = "signed_in"
	StatusNotSignedIn    = "not_signed_in"
	StatusSigninRequired = "signin_required"
)

// Status reports every plan's sign-in state.
func (m *Manager) Status() []Status {
	if m == nil {
		return nil
	}
	names := make([]string, 0, len(m.clients))
	for name := range m.clients {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make([]Status, 0, len(names))
	for _, name := range names {
		out = append(out, m.clients[name].Status())
	}
	return out
}

// Status reports this plan's sign-in state from the saved record.
func (c *Client) Status() Status {
	st := Status{Provider: c.Plan.Provider}
	rec, err := c.Store.Load(c.Plan.Provider)
	switch {
	case errors.Is(err, ErrNotSignedIn):
		st.State = StatusNotSignedIn
	case err != nil:
		st.State = StatusSigninRequired
		st.Reason = "sign-in record unreadable"
	case rec.State == StateActive && rec.Usable() && rec.HasScope(c.Plan.PlanScope):
		st.State = StatusSignedIn
	case rec.State == StateSignedOut:
		st.State = StatusNotSignedIn
		st.Reason = rec.Reason
	default:
		st.State = StatusSigninRequired
		st.Reason = rec.Reason
	}
	if rec != nil {
		st.Account = rec.Email
		if !rec.SignedInAt.IsZero() {
			t := rec.SignedInAt
			st.SignedInAt = &t
		}
		if !rec.RefreshedAt.IsZero() {
			t := rec.RefreshedAt
			st.RefreshedAt = &t
		}
	}
	return st
}

// CountServed records that a request for an agent was served by a
// credential class. Only a completed upstream reply counts.
func (m *Manager) CountServed(agentID, class string) {
	if m == nil || agentID == "" || class == "" {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.served[agentID] == nil {
		m.served[agentID] = map[string]int{}
	}
	m.served[agentID][class]++
}

// Served returns per-agent counts of served requests by credential class.
func (m *Manager) Served() map[string]map[string]int {
	out := map[string]map[string]int{}
	if m == nil {
		return out
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for agent, counts := range m.served {
		cp := make(map[string]int, len(counts))
		for class, n := range counts {
			cp[class] = n
		}
		out[agent] = cp
	}
	return out
}

// NoteFailure records an agent's most recent plan-route failure reason.
func (m *Manager) NoteFailure(agentID, reason string) {
	if m == nil || agentID == "" || reason == "" {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.failed[agentID] = reason
}

// Failures returns each agent's most recent plan-route failure reason.
func (m *Manager) Failures() map[string]string {
	out := map[string]string{}
	if m == nil {
		return out
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	for agent, reason := range m.failed {
		out[agent] = reason
	}
	return out
}
