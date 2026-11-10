package subscription

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	maxAuthResponseBytes = 1 << 20
	// refreshSkew is how long before expiry an access token is replaced.
	refreshSkew = 5 * time.Minute
)

// Error is a plan-route failure with a stable reason.
type Error struct {
	Reason string
	// Code is the vendor's machine-readable code, when one was given.
	Code string
	// Terminal means retrying cannot help until the operator signs in again.
	Terminal bool
	Detail   string
	// Stage is a fixed label for the step that failed. It is chosen here,
	// never taken from a vendor reply, so it is safe to log.
	Stage string
	// Status is the token endpoint's HTTP status, when it answered.
	Status int
}

// Stages of the credential step, for diagnostics.
const (
	StageSigninRequired    = "signin_required"
	StageLock              = "lock"
	StageRefreshTransport  = "refresh_transport"
	StageRefreshHTTP       = "refresh_http"
	StageRefreshMalformed  = "refresh_malformed"
	StageRefreshNoRotation = "refresh_no_rotation"
	StageSave              = "save"
)

var (
	errTokenUnreachable = errors.New("token endpoint unreachable")
	errTokenUnreadable  = errors.New("token endpoint reply unreadable")
	errTokenMalformed   = errors.New("token endpoint reply malformed")
)

func (e *Error) Error() string {
	msg := e.Reason
	if e.Code != "" {
		msg += " (" + e.Code + ")"
	}
	if e.Detail != "" {
		msg += ": " + e.Detail
	}
	return msg
}

// ReasonOf returns the plan-route reason carried by err, or "".
func ReasonOf(err error) string {
	var e *Error
	if errors.As(err, &e) {
		return e.Reason
	}
	return ""
}

// Client performs the sign-in, refresh and sign-out calls for one plan.
type Client struct {
	Plan  Plan
	Store *Store
	// HTTP is used for every call to the plan's auth origin. It must not
	// follow redirects: a token is never re-sent to a redirect target.
	HTTP *http.Client
	Now  func() time.Time
}

// NewClient returns a client for a plan with a redirect-refusing transport.
func NewClient(plan Plan, store *Store) *Client {
	return &Client{
		Plan:  plan,
		Store: store,
		HTTP: &http.Client{
			Timeout: 30 * time.Second,
			CheckRedirect: func(*http.Request, []*http.Request) error {
				return http.ErrUseLastResponse
			},
		},
		Now: time.Now,
	}
}

func (c *Client) now() time.Time {
	if c.Now != nil {
		return c.Now()
	}
	return time.Now()
}

type tokenResponse struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	IDToken      string `json:"id_token"`
	ExpiresIn    int64  `json:"expires_in"`
	Scope        string `json:"scope"`
}

// authErrorCode extracts a machine-readable code from an auth-origin error
// body, in either the OAuth form or the nested form.
func authErrorCode(body []byte) string {
	var shaped struct {
		Error any    `json:"error"`
		Code  string `json:"code"`
	}
	if json.Unmarshal(body, &shaped) != nil {
		return ""
	}
	switch v := shaped.Error.(type) {
	case string:
		return v
	case map[string]any:
		if code, ok := v["code"].(string); ok {
			return code
		}
	}
	return shaped.Code
}

func (c *Client) tokenRequest(ctx context.Context, form url.Values) (*tokenResponse, int, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.Plan.TokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, 0, "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	req.Header.Set("Accept", "application/json")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		// A transport error can echo the request; never the form body.
		return nil, 0, "", errTokenUnreachable
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxAuthResponseBytes))
	if err != nil {
		return nil, resp.StatusCode, "", errTokenUnreadable
	}
	if resp.StatusCode != http.StatusOK {
		return nil, resp.StatusCode, authErrorCode(body), nil
	}
	var out tokenResponse
	if err := json.Unmarshal(body, &out); err != nil || out.AccessToken == "" {
		return nil, resp.StatusCode, "", errTokenMalformed
	}
	return &out, resp.StatusCode, "", nil
}

var terminalRefreshCodes = map[string]bool{
	"invalid_grant":             true,
	"invalid_refresh_token":     true,
	"token_expired":             true,
	"refresh_token_expired":     true,
	"refresh_token_invalidated": true,
	"refresh_token_reused":      true,
}

// Refresh spends the record's refresh token and saves the replacement set.
// The caller holds the provider lock. On a terminal refusal the record is
// saved as needing sign-in; on a transient failure it is left untouched.
func (c *Client) Refresh(ctx context.Context, rec *Record) (*Record, error) {
	if !rec.Usable() {
		return nil, &Error{Reason: ReasonSigninRequired, Terminal: true, Stage: StageSigninRequired}
	}
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"client_id":     {rec.ClientID},
		"refresh_token": {rec.RefreshToken},
		"resource":      {c.Plan.Resource},
	}
	tok, status, code, err := c.tokenRequest(ctx, form)
	if err != nil {
		stage := StageRefreshMalformed
		if errors.Is(err, errTokenUnreachable) {
			stage = StageRefreshTransport
		}
		return nil, &Error{Reason: ReasonUnavailable, Detail: err.Error(), Stage: stage, Status: status}
	}
	if tok == nil {
		if terminalRefreshCodes[code] {
			next := *rec
			next.State = StateSigninRequired
			next.Reason = "refresh refused: " + code
			next.AccessToken = ""
			next.RefreshToken = ""
			if serr := c.Store.Save(&next); serr != nil {
				return nil, &Error{Reason: ReasonSigninRequired, Code: code, Terminal: true, Detail: "record not saved", Stage: StageRefreshHTTP, Status: status}
			}
			return nil, &Error{Reason: ReasonSigninRequired, Code: code, Terminal: true, Stage: StageRefreshHTTP, Status: status}
		}
		return nil, &Error{Reason: ReasonUnavailable, Code: code, Detail: "token endpoint returned HTTP " + strconv.Itoa(status), Stage: StageRefreshHTTP, Status: status}
	}
	if tok.RefreshToken == "" {
		return nil, &Error{Reason: ReasonUnavailable, Detail: "refresh reply carried no replacement refresh token", Stage: StageRefreshNoRotation, Status: status}
	}
	next := *rec
	next.AccessToken = tok.AccessToken
	next.RefreshToken = tok.RefreshToken
	if tok.Scope != "" {
		next.Scopes = strings.Fields(tok.Scope)
	}
	now := c.now().UTC()
	next.ExpiresAt = now.Add(time.Duration(tok.ExpiresIn) * time.Second)
	next.RefreshedAt = now
	if !next.HasScope(c.Plan.PlanScope) {
		next.State = StateSigninRequired
		next.Reason = "plan usage is no longer granted"
		next.AccessToken = ""
		_ = c.Store.Save(&next)
		return nil, &Error{Reason: ReasonSigninRequired, Terminal: true, Detail: next.Reason, Stage: StageSigninRequired}
	}
	// The old refresh token is spent. A record that cannot be saved must not
	// be used: the next process would replay the spent token.
	if err := c.Store.Save(&next); err != nil {
		return nil, &Error{Reason: ReasonUnavailable, Detail: "refreshed sign-in could not be saved", Stage: StageSave}
	}
	return &next, nil
}

// LoginOptions configures one interactive sign-in.
type LoginOptions struct {
	// Listen is the address the callback listener binds. It defaults to
	// 127.0.0.1 on the callback port. Inside a container whose callback port
	// is published on the host's loopback, bind all interfaces.
	Listen string
	// Port is the callback port named in the redirect URI.
	Port int
	// Out receives the authorization URL and progress lines.
	Out io.Writer
	// Timeout bounds the wait for the browser callback.
	Timeout time.Duration
}

func randomURLToken(n int) (string, error) {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

func newHostID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	return fmt.Sprintf("urn:uuid:%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]), nil
}

type callbackResult struct {
	rec *Record
	err error
}

// Login runs the documented loopback sign-in: it starts the listener, prints
// the authorization URL, and on the callback validates state, exchanges the
// code with PKCE, verifies the signed ID token and the granted plan scope,
// and only then replaces the saved record.
func (c *Client) Login(ctx context.Context, opts LoginOptions) (*Record, error) {
	port := opts.Port
	if port == 0 {
		port = c.Plan.DefaultCallbackPort
	}
	listen := opts.Listen
	if listen == "" {
		listen = net.JoinHostPort("127.0.0.1", strconv.Itoa(port))
	}
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	out := opts.Out
	if out == nil {
		out = io.Discard
	}

	// The lock is held only to read or create the registration here and to
	// commit the result later, never across the wait for the browser: model
	// requests refresh under the same lock.
	existing, err := c.registration(ctx)
	if err != nil {
		return nil, err
	}

	state, err := randomURLToken(32)
	if err != nil {
		return nil, err
	}
	nonce, err := randomURLToken(32)
	if err != nil {
		return nil, err
	}
	verifier, err := randomURLToken(48)
	if err != nil {
		return nil, err
	}
	challenge := sha256.Sum256([]byte(verifier))
	// The redirect URI is always the literal loopback address.
	redirectURI := "http://" + net.JoinHostPort("127.0.0.1", strconv.Itoa(port)) + c.Plan.CallbackPath

	q := url.Values{
		"response_type":         {"code"},
		"redirect_uri":          {redirectURI},
		"scope":                 {strings.Join(c.Plan.Scopes, " ")},
		"resource":              {c.Plan.Resource},
		"state":                 {state},
		"nonce":                 {nonce},
		"code_challenge_method": {"S256"},
		"code_challenge":        {base64.RawURLEncoding.EncodeToString(challenge[:])},
		"ext_agent_host_id":     {existing.HostID},
	}
	returning := existing.ClientID != ""
	if returning {
		q.Set("client_id", existing.ClientID)
		if existing.Email != "" {
			q.Set("login_hint", existing.Email)
		}
	} else {
		q.Set("client_id", c.Plan.RegistrationClientID)
		q.Set("agent_name_hint", c.Plan.AgentNameHint)
	}
	authURL := c.Plan.AuthorizeURL + "?" + q.Encode()

	ln, err := net.Listen("tcp", listen)
	if err != nil {
		return nil, fmt.Errorf("start callback listener on %s: %w", listen, err)
	}
	done := make(chan callbackResult, 1)
	var once sync.Once
	mux := http.NewServeMux()
	mux.HandleFunc(c.Plan.CallbackPath, func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		// A request without the right state is not the browser we sent out.
		// It is refused and does not end the wait.
		if subtle.ConstantTimeCompare([]byte(r.URL.Query().Get("state")), []byte(state)) != 1 {
			http.Error(w, "sign-in state does not match", http.StatusBadRequest)
			return
		}
		handled := false
		once.Do(func() {
			handled = true
			rec, err := c.completeLogin(r.Context(), r.URL.Query(), verifier, nonce, redirectURI)
			w.Header().Set("Content-Type", "text/plain; charset=utf-8")
			if err != nil {
				w.WriteHeader(http.StatusBadRequest)
				fmt.Fprintf(w, "Sign-in failed: %v\nReturn to the terminal.\n", err)
			} else {
				fmt.Fprintf(w, "Signed in. You can close this tab and return to the terminal.\n")
			}
			done <- callbackResult{rec: rec, err: err}
		})
		if !handled {
			// The state value is spent by its first use.
			http.Error(w, "sign-in already handled", http.StatusConflict)
		}
	})
	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 10 * time.Second, ReadTimeout: 15 * time.Second, IdleTimeout: 15 * time.Second}
	go func() { _ = srv.Serve(ln) }()
	defer srv.Close()

	fmt.Fprintf(out, "Open this URL in a browser on this computer and sign in:\n\n%s\n\nWaiting for the sign-in to finish (callback %s)...\n", authURL, redirectURI)

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case res := <-done:
		// The handler sent its result before net/http finished flushing the
		// page. Drain it before Close can cut the browser connection short.
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
		_ = srv.Shutdown(shutdownCtx)
		cancel()
		return res.rec, res.err
	case <-timer.C:
		return nil, errors.New("timed out waiting for the sign-in callback")
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// registration returns the saved registration, creating the host ID on a
// host's first sign-in.
func (c *Client) registration(ctx context.Context) (*Record, error) {
	unlock, err := c.Store.Lock(ctx, c.Plan.Provider)
	if err != nil {
		return nil, err
	}
	defer unlock()
	existing, err := c.Store.Load(c.Plan.Provider)
	if err == nil {
		return existing, nil
	}
	if !errors.Is(err, ErrNotSignedIn) {
		return nil, err
	}
	hostID, err := newHostID()
	if err != nil {
		return nil, err
	}
	// The host ID is chosen and saved before the host's first sign-in.
	existing = &Record{Provider: c.Plan.Provider, HostID: hostID, State: StateSignedOut}
	if err := c.Store.Save(existing); err != nil {
		return nil, err
	}
	return existing, nil
}

func (c *Client) completeLogin(ctx context.Context, q url.Values, verifier, nonce, redirectURI string) (*Record, error) {
	unlock, err := c.Store.Lock(ctx, c.Plan.Provider)
	if err != nil {
		return nil, err
	}
	defer unlock()
	// Read the registration again under the lock: it is what gets replaced.
	existing, err := c.Store.Load(c.Plan.Provider)
	if err != nil {
		return nil, err
	}
	if code := q.Get("error"); code != "" {
		return nil, fmt.Errorf("authorization was refused (%s)", sanitizeCode(code))
	}
	code := q.Get("code")
	if code == "" {
		return nil, errors.New("callback carried no authorization code")
	}
	clientID := q.Get("client_id")
	returning := existing.ClientID != ""
	switch {
	case returning && clientID != "" && clientID != existing.ClientID:
		return nil, errors.New("callback named a different client than the saved registration")
	case returning:
		clientID = existing.ClientID
	case clientID == "" || clientID == c.Plan.RegistrationClientID:
		return nil, errors.New("callback did not return an issued client ID")
	}
	if !returning {
		// Keep the issued client even if the one-time code then fails, so a
		// retry does not register a second app.
		registered := *existing
		registered.ClientID = clientID
		if err := c.Store.Save(&registered); err != nil {
			return nil, err
		}
		existing = &registered
	}

	tok, status, errCode, err := c.tokenRequest(ctx, url.Values{
		"grant_type":    {"authorization_code"},
		"client_id":     {clientID},
		"code":          {code},
		"code_verifier": {verifier},
		"redirect_uri":  {redirectURI},
		"resource":      {c.Plan.Resource},
	})
	if err != nil {
		return nil, err
	}
	if tok == nil {
		return nil, fmt.Errorf("code exchange refused (HTTP %d %s)", status, sanitizeCode(errCode))
	}
	if tok.IDToken == "" {
		return nil, errors.New("sign-in returned no verifiable identity")
	}
	claims, err := c.verifyIDToken(ctx, tok.IDToken, clientID, nonce)
	if err != nil {
		return nil, fmt.Errorf("identity token rejected: %w", err)
	}
	if existing.Subject != "" && existing.Subject != claims.Subject {
		return nil, errors.New("this registration belongs to a different account; run logout --forget to register another")
	}
	scopes := strings.Fields(tok.Scope)
	granted := false
	for _, s := range scopes {
		if s == c.Plan.PlanScope {
			granted = true
		}
	}
	if !granted {
		return nil, fmt.Errorf("plan usage was not granted (scope %s missing)", c.Plan.PlanScope)
	}
	if tok.RefreshToken == "" {
		return nil, errors.New("sign-in returned no refresh token")
	}
	now := c.now().UTC()
	rec := &Record{
		Provider:     c.Plan.Provider,
		ClientID:     clientID,
		HostID:       existing.HostID,
		Subject:      claims.Subject,
		Email:        claims.Email,
		AccessToken:  tok.AccessToken,
		RefreshToken: tok.RefreshToken,
		Scopes:       scopes,
		ExpiresAt:    now.Add(time.Duration(tok.ExpiresIn) * time.Second),
		SignedInAt:   now,
		RefreshedAt:  now,
		State:        StateActive,
	}
	if err := c.Store.Save(rec); err != nil {
		return nil, err
	}
	return rec, nil
}

// sanitizeCode keeps an untrusted short code printable.
func sanitizeCode(code string) string {
	if len(code) > 64 {
		code = code[:64]
	}
	return strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '_' || r == '-' || r == '.' {
			return r
		}
		return '_'
	}, code)
}

// Logout revokes the refresh token and clears the saved tokens. The client
// registration and host ID are kept unless forget is set. revoked reports
// whether the vendor confirmed the revocation.
func (c *Client) Logout(ctx context.Context, forget bool) (revoked bool, err error) {
	unlock, err := c.Store.Lock(ctx, c.Plan.Provider)
	if err != nil {
		return false, err
	}
	defer unlock()
	rec, err := c.Store.Load(c.Plan.Provider)
	if errors.Is(err, ErrNotSignedIn) {
		return true, nil
	}
	if err != nil {
		return false, err
	}
	if rec.RefreshToken != "" {
		revoked = c.revoke(ctx, rec)
	} else {
		revoked = true
	}
	next := Record{Provider: rec.Provider, HostID: rec.HostID, State: StateSignedOut, Reason: "signed out"}
	if !forget {
		next.ClientID = rec.ClientID
		next.Subject = rec.Subject
		next.Email = rec.Email
	}
	return revoked, c.Store.Save(&next)
}

func (c *Client) revoke(ctx context.Context, rec *Record) bool {
	disc, err := c.discovery(ctx)
	if err != nil || disc.RevocationEndpoint == "" || !sameOrigin(disc.RevocationEndpoint, c.Plan.Issuer) {
		return false
	}
	form := url.Values{
		"token":           {rec.RefreshToken},
		"token_type_hint": {"refresh_token"},
		"client_id":       {rec.ClientID},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, disc.RevocationEndpoint, strings.NewReader(form.Encode()))
	if err != nil {
		return false
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxAuthResponseBytes))
	return resp.StatusCode == http.StatusOK
}

func sameOrigin(raw, origin string) bool {
	a, err1 := url.Parse(raw)
	b, err2 := url.Parse(origin)
	return err1 == nil && err2 == nil && a.Scheme == b.Scheme && a.Host == b.Host && a.User == nil
}
