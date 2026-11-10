package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/cost"
	"github.com/mostlydev/cllama/internal/provider"
	"github.com/mostlydev/cllama/internal/proxy"
	"github.com/mostlydev/cllama/internal/subscription"
)

func TestAuthStatusReportsStateWithoutTokens(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("CLAW_AUTH_DIR", dir)

	var out bytes.Buffer
	if err := run([]string{"auth", "status"}, &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), "chatgpt: not_signed_in") {
		t.Fatalf("status: %q", out.String())
	}

	err := subscription.NewStore(dir).Save(&subscription.Record{
		Provider: "chatgpt", ClientID: "oaiapp_one", HostID: "urn:uuid:h", Email: "operator@example.com",
		AccessToken: "access-canary", RefreshToken: "refresh-canary", Scopes: []string{"chatgpt.tokens.use.direct"},
		ExpiresAt: time.Now().Add(time.Hour), State: subscription.StateActive,
	})
	if err != nil {
		t.Fatal(err)
	}
	out.Reset()
	if err := run([]string{"auth", "status", "--json"}, &out, io.Discard); err != nil {
		t.Fatal(err)
	}
	if strings.Contains(out.String(), "canary") || !strings.Contains(out.String(), `"state": "signed_in"`) {
		t.Fatalf("status: %s", out.String())
	}
}

func TestAuthRejectsUnknownPlanAndSubcommand(t *testing.T) {
	t.Setenv("CLAW_AUTH_DIR", t.TempDir())
	if err := run([]string{"auth", "login", "nonesuch"}, io.Discard, io.Discard); err == nil {
		t.Fatal("unknown plan accepted")
	}
	if err := run([]string{"auth", "import"}, io.Discard, io.Discard); err == nil {
		t.Fatal("unknown subcommand accepted")
	}
	if err := run([]string{"auth"}, io.Discard, io.Discard); err == nil {
		t.Fatal("missing subcommand accepted")
	}
}

func TestOperatorPortReportsSubscriptions(t *testing.T) {
	dir := t.TempDir()
	subs := subscription.NewManager(dir)
	reg := provider.NewRegistry("")
	reg.RegisterPlan("chatgpt", subscription.ChatGPT.InferenceBase)
	subs.CountServed("worker", subscription.CredentialAPIKey)
	subs.NoteFailure("worker", subscription.ReasonSigninRequired)
	ui := newUIHandler(reg, cost.NewAccumulator(), t.TempDir(), "ui-token", proxy.NewContextSnapshotStore(), subs)

	req := httptest.NewRequest(http.MethodGet, "/subscriptions", nil)
	w := httptest.NewRecorder()
	ui.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("the status needs the operator token, got %d", w.Code)
	}

	req = httptest.NewRequest(http.MethodGet, "/subscriptions", nil)
	req.Header.Set("Authorization", "Bearer ui-token")
	w = httptest.NewRecorder()
	ui.ServeHTTP(w, req)
	var body struct {
		Subscriptions []subscription.Status     `json:"subscriptions"`
		ServedBy      map[string]map[string]int `json:"served_by"`
		LastFailure   map[string]string         `json:"last_failure"`
	}
	if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
		t.Fatalf("%v: %s", err, w.Body.String())
	}
	if len(body.Subscriptions) != 1 || body.Subscriptions[0].State != subscription.StatusNotSignedIn {
		t.Fatalf("subscriptions: %+v", body.Subscriptions)
	}
	if body.ServedBy["worker"]["api_key"] != 1 || body.LastFailure["worker"] != subscription.ReasonSigninRequired {
		t.Fatalf("body: %s", w.Body.String())
	}
}
