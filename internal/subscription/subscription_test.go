package subscription

import (
	"bytes"
	"context"
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/big"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// fakeIssuer is a local stand-in for a plan's auth origin.
type fakeIssuer struct {
	t      *testing.T
	srv    *httptest.Server
	key    *rsa.PrivateKey
	mu     sync.Mutex
	nonce  string
	verify string // expected code_challenge
	// mutate lets a test corrupt the ID token claims or signing.
	mutate       func(claims map[string]any)
	signWith     *rsa.PrivateKey
	scope        string
	refreshCalls int
	spent        map[string]bool
	refreshReply func(w http.ResponseWriter) bool
	serial       int
	tokenForms   []url.Values
}

func newFakeIssuer(t *testing.T) *fakeIssuer {
	t.Helper()
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		t.Fatal(err)
	}
	f := &fakeIssuer{t: t, key: key, spent: map[string]bool{}, scope: "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"}
	mux := http.NewServeMux()
	mux.HandleFunc("/.well-known/openid-configuration", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]string{
			"issuer": f.srv.URL, "jwks_uri": f.srv.URL + "/jwks", "revocation_endpoint": f.srv.URL + "/revoke",
		})
	})
	mux.HandleFunc("/jwks", func(w http.ResponseWriter, r *http.Request) {
		pub := f.key.PublicKey
		_ = json.NewEncoder(w).Encode(map[string]any{"keys": []map[string]string{{
			"kty": "RSA", "kid": "k1", "use": "sig",
			"n": base64.RawURLEncoding.EncodeToString(pub.N.Bytes()),
			"e": base64.RawURLEncoding.EncodeToString(big.NewInt(int64(pub.E)).Bytes()),
		}}})
	})
	mux.HandleFunc("/revoke", func(w http.ResponseWriter, r *http.Request) {})
	mux.HandleFunc("/token", func(w http.ResponseWriter, r *http.Request) {
		_ = r.ParseForm()
		f.mu.Lock()
		defer f.mu.Unlock()
		f.tokenForms = append(f.tokenForms, r.PostForm)
		switch r.PostForm.Get("grant_type") {
		case "authorization_code":
			sum := sha256.Sum256([]byte(r.PostForm.Get("code_verifier")))
			if base64.RawURLEncoding.EncodeToString(sum[:]) != f.verify || r.PostForm.Get("code") != "good-code" {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":"invalid_grant"}`))
				return
			}
			f.serial++
			_ = json.NewEncoder(w).Encode(map[string]any{
				"access_token": fmt.Sprintf("access-%d-canary", f.serial), "refresh_token": fmt.Sprintf("refresh-%d-canary", f.serial),
				"id_token": f.idToken(r.PostForm.Get("client_id")), "expires_in": 3600, "scope": f.scope,
			})
		case "refresh_token":
			f.refreshCalls++
			if f.refreshReply != nil && f.refreshReply(w) {
				return
			}
			tok := r.PostForm.Get("refresh_token")
			if f.spent[tok] {
				w.WriteHeader(http.StatusBadRequest)
				_, _ = w.Write([]byte(`{"error":{"code":"refresh_token_reused"}}`))
				return
			}
			f.spent[tok] = true
			f.serial++
			_ = json.NewEncoder(w).Encode(map[string]any{
				"access_token": fmt.Sprintf("access-%d-canary", f.serial), "refresh_token": fmt.Sprintf("refresh-%d-canary", f.serial),
				"expires_in": 3600, "scope": f.scope,
			})
		}
	})
	f.srv = httptest.NewServer(mux)
	t.Cleanup(f.srv.Close)
	return f
}

func (f *fakeIssuer) idToken(clientID string) string {
	claims := map[string]any{
		"iss": f.srv.URL, "sub": "user-1", "aud": clientID, "exp": time.Now().Add(time.Hour).Unix(),
		"nonce": f.nonce, "email": "operator@example.com",
	}
	if f.mutate != nil {
		f.mutate(claims)
	}
	header := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"RS256","kid":"k1"}`))
	payload, _ := json.Marshal(claims)
	signing := header + "." + base64.RawURLEncoding.EncodeToString(payload)
	digest := sha256.Sum256([]byte(signing))
	key := f.key
	if f.signWith != nil {
		key = f.signWith
	}
	sig, err := rsa.SignPKCS1v15(rand.Reader, key, crypto.SHA256, digest[:])
	if err != nil {
		f.t.Fatal(err)
	}
	return signing + "." + base64.RawURLEncoding.EncodeToString(sig)
}

func (f *fakeIssuer) plan() Plan {
	p := ChatGPT
	p.Issuer = f.srv.URL
	p.AuthorizeURL = f.srv.URL + "/authorize"
	p.TokenURL = f.srv.URL + "/token"
	p.DiscoveryURL = f.srv.URL + "/.well-known/openid-configuration"
	return p
}

func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
}

// syncBuffer is a writer the test can read while Login is still writing.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

type loginRun struct {
	authURL  *url.URL
	redirect string
	done     chan callbackResult
	cancel   context.CancelFunc
}

// startLogin runs Login and returns once the authorization URL is printed.
func startLogin(t *testing.T, c *Client, f *fakeIssuer) *loginRun {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	out := &syncBuffer{}
	run := &loginRun{done: make(chan callbackResult, 1), cancel: cancel}
	go func() {
		rec, err := c.Login(ctx, LoginOptions{Port: freePort(t), Out: out, Timeout: 20 * time.Second})
		run.done <- callbackResult{rec: rec, err: err}
	}()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		for _, field := range strings.Fields(out.String()) {
			if strings.HasPrefix(field, c.Plan.AuthorizeURL+"?") {
				u, err := url.Parse(field)
				if err != nil {
					t.Fatal(err)
				}
				run.authURL = u
				run.redirect = u.Query().Get("redirect_uri")
				f.mu.Lock()
				f.nonce = u.Query().Get("nonce")
				f.verify = u.Query().Get("code_challenge")
				f.mu.Unlock()
				return run
			}
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("no authorization URL printed: %q", out.String())
	return nil
}

func callback(t *testing.T, run *loginRun, q url.Values) (int, string) {
	t.Helper()
	resp, err := http.Get(run.redirect + "?" + q.Encode())
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(body)
}

func newTestClient(t *testing.T, f *fakeIssuer) *Client {
	t.Helper()
	return NewClient(f.plan(), NewStore(t.TempDir()))
}

func TestLoginSavesVerifiedSignIn(t *testing.T) {
	f := newFakeIssuer(t)
	c := newTestClient(t, f)
	run := startLogin(t, c, f)

	q := run.authURL.Query()
	if q.Get("client_id") != "dynamic_agent_client" || q.Get("agent_name_hint") != "cllama" {
		t.Fatalf("first sign-in must register: %v", q)
	}
	if !strings.HasPrefix(q.Get("ext_agent_host_id"), "urn:uuid:") || q.Get("code_challenge_method") != "S256" {
		t.Fatalf("authorize request incomplete: %v", q)
	}
	if !strings.HasPrefix(run.redirect, "http://127.0.0.1:") || q.Get("resource") != "https://api.openai.com/v1" {
		t.Fatalf("redirect or resource wrong: %v", q)
	}
	if q.Get("id_token_hint") != "" {
		t.Fatal("no token may appear in a printed URL")
	}

	status, body := callback(t, run, url.Values{"code": {"good-code"}, "state": {q.Get("state")}, "client_id": {"oaiapp_one"}})
	if status != http.StatusOK {
		t.Fatalf("callback: %d %s", status, body)
	}
	res := <-run.done
	if res.err != nil {
		t.Fatal(res.err)
	}
	rec, err := c.Store.Load("chatgpt")
	if err != nil {
		t.Fatal(err)
	}
	if rec.State != StateActive || rec.ClientID != "oaiapp_one" || rec.Subject != "user-1" || rec.HostID != q.Get("ext_agent_host_id") {
		t.Fatalf("record: %+v", rec)
	}
	info, _ := os.Stat(c.Store.Dir())
	if info.Mode().Perm() != 0o700 {
		t.Fatalf("store dir mode %v", info.Mode().Perm())
	}
	info, _ = os.Stat(filepath.Join(c.Store.Dir(), "chatgpt.json"))
	if info.Mode().Perm() != 0o600 {
		t.Fatalf("record mode %v", info.Mode().Perm())
	}
	raw, _ := os.ReadFile(filepath.Join(c.Store.Dir(), "chatgpt.json"))
	if bytes.Contains(raw, []byte("eyJ")) {
		t.Fatal("the ID token must not be retained")
	}

	// A returning sign-in reuses the issued client and the host ID.
	run2 := startLogin(t, c, f)
	q2 := run2.authURL.Query()
	if q2.Get("client_id") != "oaiapp_one" || q2.Get("agent_name_hint") != "" || q2.Get("ext_agent_host_id") != rec.HostID {
		t.Fatalf("returning sign-in: %v", q2)
	}
	if q2.Get("state") == q.Get("state") || q2.Get("nonce") == q.Get("nonce") || q2.Get("code_challenge") == q.Get("code_challenge") {
		t.Fatal("state, nonce and PKCE must be fresh per attempt")
	}
	run2.cancel()
}

func TestLoginRejectsBeforeSaving(t *testing.T) {
	other, _ := rsa.GenerateKey(rand.Reader, 2048)
	cases := map[string]struct {
		setup func(f *fakeIssuer)
		query func(q url.Values) url.Values
	}{
		"bad signature":  {setup: func(f *fakeIssuer) { f.signWith = other }},
		"wrong issuer":   {setup: func(f *fakeIssuer) { f.mutate = func(c map[string]any) { c["iss"] = "https://evil.example" } }},
		"wrong audience": {setup: func(f *fakeIssuer) { f.mutate = func(c map[string]any) { c["aud"] = "oaiapp_other" } }},
		"expired": {setup: func(f *fakeIssuer) {
			f.mutate = func(c map[string]any) { c["exp"] = time.Now().Add(-time.Minute).Unix() }
		}},
		"wrong nonce":   {setup: func(f *fakeIssuer) { f.mutate = func(c map[string]any) { c["nonce"] = "replayed" } }},
		"no plan scope": {setup: func(f *fakeIssuer) { f.scope = "openid profile email offline_access" }},
		"no client id":  {query: func(q url.Values) url.Values { q.Del("client_id"); return q }},
		"placeholder client id": {query: func(q url.Values) url.Values {
			q.Set("client_id", "dynamic_agent_client")
			return q
		}},
		"vendor error": {query: func(q url.Values) url.Values { q.Set("error", "access_denied"); return q }},
		"bad code":     {query: func(q url.Values) url.Values { q.Set("code", "stolen"); return q }},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			f := newFakeIssuer(t)
			if tc.setup != nil {
				tc.setup(f)
			}
			c := newTestClient(t, f)
			run := startLogin(t, c, f)
			q := url.Values{"code": {"good-code"}, "state": {run.authURL.Query().Get("state")}, "client_id": {"oaiapp_one"}}
			if tc.query != nil {
				q = tc.query(q)
			}
			status, _ := callback(t, run, q)
			if status == http.StatusOK {
				t.Fatal("callback must fail")
			}
			if res := <-run.done; res.err == nil {
				t.Fatal("login must fail")
			}
			rec, err := c.Store.Load("chatgpt")
			if err != nil {
				t.Fatal(err)
			}
			if rec.State == StateActive || rec.AccessToken != "" || rec.RefreshToken != "" {
				t.Fatalf("nothing usable may be saved: %+v", rec)
			}
		})
	}
}

func TestLoginRefusesForgedAndReplayedState(t *testing.T) {
	f := newFakeIssuer(t)
	c := newTestClient(t, f)
	run := startLogin(t, c, f)
	good := url.Values{"code": {"good-code"}, "state": {run.authURL.Query().Get("state")}, "client_id": {"oaiapp_one"}}

	// A forged state is refused and does not end the wait.
	if status, _ := callback(t, run, url.Values{"code": {"good-code"}, "state": {"forged"}, "client_id": {"oaiapp_evil"}}); status != http.StatusBadRequest {
		t.Fatalf("forged state: %d", status)
	}
	select {
	case res := <-run.done:
		t.Fatalf("forged state ended the sign-in: %+v", res)
	default:
	}
	if status, body := callback(t, run, good); status != http.StatusOK {
		t.Fatalf("good callback: %d %s", status, body)
	}
	if res := <-run.done; res.err != nil {
		t.Fatal(res.err)
	}
	// The state is spent: a replay cannot run a second exchange.
	before := len(f.tokenForms)
	resp, err := http.Get(run.redirect + "?" + good.Encode())
	if err == nil {
		resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			t.Fatal("replayed callback must not succeed")
		}
	}
	if len(f.tokenForms) != before {
		t.Fatal("replayed callback reached the token endpoint")
	}
}

func TestReturningSignInRejectsOtherClientAndAccount(t *testing.T) {
	for name, tc := range map[string]struct {
		clientID string
		subject  string
	}{
		"other client":  {clientID: "oaiapp_other", subject: "user-1"},
		"other account": {clientID: "oaiapp_one", subject: "user-2"},
	} {
		t.Run(name, func(t *testing.T) {
			f := newFakeIssuer(t)
			c := newTestClient(t, f)
			saved := &Record{Provider: "chatgpt", ClientID: "oaiapp_one", HostID: "urn:uuid:h", Subject: "user-1", AccessToken: "old-access", RefreshToken: "old-refresh", Scopes: strings.Fields(f.scope), ExpiresAt: time.Now().Add(time.Hour), State: StateActive}
			if err := c.Store.Save(saved); err != nil {
				t.Fatal(err)
			}
			f.mutate = func(c map[string]any) { c["sub"] = tc.subject }
			run := startLogin(t, c, f)
			callback(t, run, url.Values{"code": {"good-code"}, "state": {run.authURL.Query().Get("state")}, "client_id": {tc.clientID}})
			if res := <-run.done; res.err == nil {
				t.Fatal("login must fail")
			}
			rec, _ := c.Store.Load("chatgpt")
			if rec.AccessToken != "old-access" || rec.Subject != "user-1" || rec.ClientID != "oaiapp_one" {
				t.Fatalf("active record was replaced: %+v", rec)
			}
		})
	}
}

func expiredRecord(f *fakeIssuer) *Record {
	return &Record{Provider: "chatgpt", ClientID: "oaiapp_one", HostID: "urn:uuid:h", Subject: "user-1", AccessToken: "stale-access", RefreshToken: "refresh-0-canary", Scopes: strings.Fields(f.scope), ExpiresAt: time.Now().Add(-time.Minute), State: StateActive}
}

// Two proxies sharing one auth directory must spend a rotating refresh token
// once and both end up with the saved successor.
func TestConcurrentRefreshSpendsTheTokenOnce(t *testing.T) {
	f := newFakeIssuer(t)
	dir := t.TempDir()
	if err := NewStore(dir).Save(expiredRecord(f)); err != nil {
		t.Fatal(err)
	}
	const holders = 8
	tokens := make([]string, holders)
	errs := make([]error, holders)
	var wg sync.WaitGroup
	for i := 0; i < holders; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			m := NewManagerWithClients(NewClient(f.plan(), NewStore(dir)))
			tokens[i], errs[i] = m.Token(context.Background(), "chatgpt")
		}(i)
	}
	wg.Wait()
	for i := range tokens {
		if errs[i] != nil || tokens[i] != tokens[0] || tokens[i] == "stale-access" {
			t.Fatalf("holder %d: %q %v (first %q)", i, tokens[i], errs[i], tokens[0])
		}
	}
	if f.refreshCalls != 1 {
		t.Fatalf("refresh token spent %d times", f.refreshCalls)
	}
	rec, _ := NewStore(dir).Load("chatgpt")
	if rec.AccessToken != tokens[0] || rec.RefreshToken == "refresh-0-canary" || rec.State != StateActive {
		t.Fatalf("saved successor: %+v", rec)
	}
}

func TestTerminalRefreshRequiresSignIn(t *testing.T) {
	f := newFakeIssuer(t)
	f.refreshReply = func(w http.ResponseWriter) bool {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":"invalid_grant"}`))
		return true
	}
	c := newTestClient(t, f)
	if err := c.Store.Save(expiredRecord(f)); err != nil {
		t.Fatal(err)
	}
	m := NewManagerWithClients(c)
	_, err := m.Token(context.Background(), "chatgpt")
	if ReasonOf(err) != ReasonSigninRequired {
		t.Fatalf("want sign-in required, got %v", err)
	}
	rec, _ := c.Store.Load("chatgpt")
	if rec.State != StateSigninRequired || rec.RefreshToken != "" || rec.ClientID != "oaiapp_one" {
		t.Fatalf("record: %+v", rec)
	}
	if st := c.Status(); st.State != StatusSigninRequired {
		t.Fatalf("status: %+v", st)
	}
	// It stays required without another call to the vendor.
	calls := f.refreshCalls
	if _, err := m.Token(context.Background(), "chatgpt"); ReasonOf(err) != ReasonSigninRequired || f.refreshCalls != calls {
		t.Fatalf("second attempt: %v, calls %d", err, f.refreshCalls)
	}
}

func TestTransientRefreshKeepsTheSignIn(t *testing.T) {
	f := newFakeIssuer(t)
	f.refreshReply = func(w http.ResponseWriter) bool {
		w.WriteHeader(http.StatusServiceUnavailable)
		return true
	}
	c := newTestClient(t, f)
	if err := c.Store.Save(expiredRecord(f)); err != nil {
		t.Fatal(err)
	}
	_, err := NewManagerWithClients(c).Token(context.Background(), "chatgpt")
	if ReasonOf(err) != ReasonUnavailable {
		t.Fatalf("want unavailable, got %v", err)
	}
	rec, _ := c.Store.Load("chatgpt")
	if rec.State != StateActive || rec.RefreshToken != "refresh-0-canary" {
		t.Fatalf("a transient failure must not cost the sign-in: %+v", rec)
	}
}

// A redirect from the token endpoint must not carry the refresh token to
// another host.
func TestRefreshDoesNotFollowRedirects(t *testing.T) {
	var leaked bool
	thief := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { leaked = true }))
	defer thief.Close()
	f := newFakeIssuer(t)
	f.refreshReply = func(w http.ResponseWriter) bool {
		w.Header().Set("Location", thief.URL+"/token")
		w.WriteHeader(http.StatusTemporaryRedirect)
		return true
	}
	c := newTestClient(t, f)
	if err := c.Store.Save(expiredRecord(f)); err != nil {
		t.Fatal(err)
	}
	if _, err := NewManagerWithClients(c).Token(context.Background(), "chatgpt"); err == nil {
		t.Fatal("a redirected refresh must fail")
	}
	if leaked {
		t.Fatal("refresh token followed a redirect")
	}
}

func TestStatusNeverCarriesTokens(t *testing.T) {
	f := newFakeIssuer(t)
	c := newTestClient(t, f)
	rec := expiredRecord(f)
	rec.ExpiresAt = time.Now().Add(time.Hour)
	rec.Email = "operator@example.com"
	if err := c.Store.Save(rec); err != nil {
		t.Fatal(err)
	}
	m := NewManagerWithClients(c)
	out, _ := json.Marshal(m.Status())
	if bytes.Contains(out, []byte("canary")) || bytes.Contains(out, []byte("stale-access")) {
		t.Fatalf("status leaks a token: %s", out)
	}
	if m.Status()[0].State != StatusSignedIn {
		t.Fatalf("status: %s", out)
	}
	secrets := strings.Join(m.Secrets(), " ")
	if !strings.Contains(secrets, "stale-access") || !strings.Contains(secrets, "refresh-0-canary") {
		t.Fatalf("redaction set incomplete: %q", secrets)
	}
}

func TestLogoutClearsTokensAndKeepsRegistration(t *testing.T) {
	f := newFakeIssuer(t)
	c := newTestClient(t, f)
	if err := c.Store.Save(expiredRecord(f)); err != nil {
		t.Fatal(err)
	}
	revoked, err := c.Logout(context.Background(), false)
	if err != nil || !revoked {
		t.Fatalf("logout: %v %v", revoked, err)
	}
	rec, _ := c.Store.Load("chatgpt")
	if rec.AccessToken != "" || rec.RefreshToken != "" || rec.ClientID != "oaiapp_one" || rec.HostID != "urn:uuid:h" {
		t.Fatalf("record: %+v", rec)
	}
	if st := c.Status(); st.State != StatusNotSignedIn {
		t.Fatalf("status: %+v", st)
	}
	if _, err := c.Logout(context.Background(), true); err != nil {
		t.Fatal(err)
	}
	rec, _ = c.Store.Load("chatgpt")
	if rec.ClientID != "" || rec.Subject != "" || rec.HostID != "urn:uuid:h" {
		t.Fatalf("forget: %+v", rec)
	}
}

func TestStoreRefusesUntrustedPaths(t *testing.T) {
	rec := &Record{Provider: "chatgpt", HostID: "urn:uuid:h", State: StateSignedOut}

	t.Run("open directory mode", func(t *testing.T) {
		s := NewStore(t.TempDir())
		if err := os.Mkdir(s.Dir(), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := s.Save(rec); err == nil {
			t.Fatal("a directory others can read must be refused")
		}
	})
	t.Run("linked directory", func(t *testing.T) {
		s := NewStore(t.TempDir())
		target := t.TempDir()
		_ = os.Chmod(target, 0o700)
		if err := os.Symlink(target, s.Dir()); err != nil {
			t.Fatal(err)
		}
		if err := s.Save(rec); err == nil {
			t.Fatal("a linked directory must be refused")
		}
		if _, err := s.Load("chatgpt"); err == nil || errors.Is(err, ErrNotSignedIn) {
			t.Fatalf("load through a link: %v", err)
		}
	})
	t.Run("readable record", func(t *testing.T) {
		s := NewStore(t.TempDir())
		if err := s.Save(rec); err != nil {
			t.Fatal(err)
		}
		if err := os.Chmod(filepath.Join(s.Dir(), "chatgpt.json"), 0o644); err != nil {
			t.Fatal(err)
		}
		if _, err := s.Load("chatgpt"); err == nil {
			t.Fatal("a record others can read must be refused")
		}
	})
	t.Run("linked record", func(t *testing.T) {
		s := NewStore(t.TempDir())
		if err := s.Save(rec); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(s.Dir(), "chatgpt.json")
		elsewhere := filepath.Join(t.TempDir(), "x.json")
		data, _ := os.ReadFile(path)
		_ = os.WriteFile(elsewhere, data, 0o600)
		_ = os.Remove(path)
		if err := os.Symlink(elsewhere, path); err != nil {
			t.Fatal(err)
		}
		if _, err := s.Load("chatgpt"); err == nil {
			t.Fatal("a linked record must be refused")
		}
	})
	t.Run("fifo record does not hang", func(t *testing.T) {
		s := NewStore(t.TempDir())
		if err := s.Save(rec); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(s.Dir(), "chatgpt.json")
		_ = os.Remove(path)
		if err := syscall.Mkfifo(path, 0o600); err != nil {
			t.Skip("no fifo support")
		}
		done := make(chan error, 1)
		go func() { _, err := s.Load("chatgpt"); done <- err }()
		select {
		case err := <-done:
			if err == nil {
				t.Fatal("a fifo must be refused")
			}
		case <-time.After(3 * time.Second):
			t.Fatal("load hung on a fifo")
		}
	})
	t.Run("provider name", func(t *testing.T) {
		s := NewStore(t.TempDir())
		for _, name := range []string{"../x", "a/b", "", "A", ".hidden"} {
			if _, err := s.Load(name); err == nil || errors.Is(err, ErrNotSignedIn) {
				t.Fatalf("name %q accepted: %v", name, err)
			}
		}
	})
}

func TestLockWaitEndsWithContext(t *testing.T) {
	s := NewStore(t.TempDir())
	unlock, err := s.Lock(context.Background(), "chatgpt")
	if err != nil {
		t.Fatal(err)
	}
	defer unlock()
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()
	if _, err := s.Lock(ctx, "chatgpt"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("want deadline, got %v", err)
	}
}
