package ui

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/provider"
	"github.com/mostlydev/cllama/internal/subscription"
)

func TestSubscriptionStatusRequiresOperatorAndNeverReturnsTokens(t *testing.T) {
	dir := t.TempDir()
	if err := subscription.NewStore(dir).Save(&subscription.Record{
		Provider: "chatgpt", ClientID: "oaiapp_test", HostID: "host", State: subscription.StateActive,
		AccessToken: "access-secret-canary", RefreshToken: "refresh-secret-canary",
		Scopes: []string{"chatgpt.tokens.use.direct"}, ExpiresAt: time.Now().Add(time.Hour),
	}); err != nil {
		t.Fatal(err)
	}
	h := NewHandler(provider.NewRegistry(""), WithUIToken("operator-token"), WithSubscriptions(subscription.NewManager(dir)))
	for _, bearer := range []string{"", "worker:secret", "operator-token"} {
		req := httptest.NewRequest(http.MethodGet, "/subscriptions", nil)
		if bearer != "" {
			req.Header.Set("Authorization", "Bearer "+bearer)
		}
		w := httptest.NewRecorder()
		h.ServeHTTP(w, req)
		if strings.Contains(w.Body.String(), "canary") {
			t.Fatal("status endpoint returned a credential")
		}
		if bearer == "operator-token" {
			if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), `"state":"signed_in"`) {
				t.Fatalf("operator status %d %s", w.Code, w.Body.String())
			}
		} else if w.Code != http.StatusUnauthorized {
			t.Fatalf("nonoperator status %d", w.Code)
		}
	}
}
