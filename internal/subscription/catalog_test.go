package subscription

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// A plan's model list is read with its own sign-in and reported by name.
func TestPlanCatalog(t *testing.T) {
	f := newFakeIssuer(t)
	reply, status := `{"models":[
	  {"slug":"gpt-6.1-terra","display_name":"Terra","visibility":"list"},
	  {"slug":"gpt-5.6-terra","display_name":"Terra 5.6","visibility":"list"},
	  {"slug":"gpt-internal","visibility":"hide"},
	  {"slug":"gpt-no-visibility"}]}`, 200
	var auth, path string
	var elsewhere int
	other := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { elsewhere++ }))
	defer other.Close()
	api := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth, path = r.Header.Get("Authorization"), r.URL.Path
		if status == 302 {
			http.Redirect(w, r, other.URL, status)
			return
		}
		w.WriteHeader(status)
		_, _ = io.WriteString(w, reply)
	}))
	defer api.Close()

	plan := f.plan()
	plan.InferenceBase = api.URL + "/v1"
	c := NewClient(plan, NewStore(t.TempDir()))
	rec := expiredRecord(f)
	rec.ExpiresAt = time.Now().Add(time.Hour)
	rec.Email = "operator@example.com"
	if err := c.Store.Save(rec); err != nil {
		t.Fatal(err)
	}
	m := NewManagerWithClients(c)

	catalog, err := m.Catalog(context.Background(), plan.Provider)
	if err != nil {
		t.Fatal(err)
	}
	if !catalog.Complete || strings.Join(catalog.IDs, ",") != "gpt-5.6-terra,gpt-6.1-terra" || catalog.Provider != plan.Provider || catalog.ReadAt.IsZero() {
		t.Fatalf("catalog = %+v", catalog)
	}
	if auth != "Bearer "+rec.AccessToken || path != "/v1/models" {
		t.Fatalf("request: auth=%q path=%q", auth, path)
	}
	// The output names models and nothing about the account or the sign-in.
	out, _ := json.Marshal(catalog)
	for _, private := range []string{rec.AccessToken, rec.RefreshToken, "operator@example.com", rec.Subject, "display_name", "Terra"} {
		if private != "" && strings.Contains(string(out), private) {
			t.Fatalf("catalog output carries %q: %s", private, out)
		}
	}

	for name, tc := range map[string]struct {
		status int
		reply  string
	}{
		"redirect":        {302, ""},
		"refused":         {403, reply},
		"not a list":      {200, `{"data":[{"id":"gpt-6.1-terra"}]}`},
		"slug not string": {200, `{"models":[{"slug":7,"visibility":"list"}]}`},
		"empty slug":      {200, `{"models":[{"slug":"","visibility":"list"}]}`},
		"repeated":        {200, `{"models":[{"slug":"a","visibility":"list"},{"slug":"a","visibility":"list"}]}`},
		"not json":        {200, `<html>`},
		"too long a slug": {200, `{"models":[{"slug":"` + strings.Repeat("m", maxCatalogSlug+1) + `","visibility":"list"}]}`},
	} {
		status, reply = tc.status, tc.reply
		got, err := m.Catalog(context.Background(), plan.Provider)
		if err == nil || got != nil {
			t.Errorf("%s: %+v %v", name, got, err)
		}
		if err != nil && (strings.Contains(err.Error(), rec.AccessToken) || strings.Contains(err.Error(), "gpt-")) {
			t.Errorf("%s: error carries the token or reply text: %v", name, err)
		}
	}
	if elsewhere != 0 {
		t.Fatal("the sign-in token followed a redirect")
	}

	// With no sign-in there is no catalog, and no request.
	empty := NewManagerWithClients(NewClient(plan, NewStore(t.TempDir())))
	auth = ""
	if got, err := empty.Catalog(context.Background(), plan.Provider); err == nil || got != nil || ReasonOf(err) != ReasonSigninRequired || auth != "" {
		t.Fatalf("not signed in: %+v %v auth=%q", got, err, auth)
	}
	if _, err := m.Catalog(context.Background(), "nope"); ReasonOf(err) != ReasonSigninRequired {
		t.Fatalf("unknown plan: %v", err)
	}
}
