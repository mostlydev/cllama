package feeds

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestFetcherReturnsFeedContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("Fleet nominal. 4 agents healthy."))
	}))
	defer srv.Close()

	f := NewFetcher("tiverton-house", nil, nil)
	entry := FeedEntry{Name: "fleet-alerts", URL: srv.URL, TTL: 60}
	result, err := f.Fetch(context.Background(), "weston", entry)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "Fleet nominal. 4 agents healthy." {
		t.Errorf("unexpected content: %q", result.Content)
	}
	if result.Stale {
		t.Error("fresh fetch should not be stale")
	}
}

func TestFetcherSendsIdentityHeaders(t *testing.T) {
	var gotClawID, gotClawPod string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotClawID = r.Header.Get("X-Claw-ID")
		gotClawPod = r.Header.Get("X-Claw-Pod")
		_, _ = w.Write([]byte("ok"))
	}))
	defer srv.Close()

	f := NewFetcher("trading-desk", nil, nil)
	if _, err := f.Fetch(context.Background(), "tiverton", FeedEntry{Name: "test", URL: srv.URL, TTL: 60}); err != nil {
		t.Fatal(err)
	}
	if gotClawID != "tiverton" {
		t.Errorf("expected X-Claw-ID=tiverton, got %q", gotClawID)
	}
	if gotClawPod != "trading-desk" {
		t.Errorf("expected X-Claw-Pod=trading-desk, got %q", gotClawPod)
	}
}

func TestFetcherReturnsCachedWithinTTL(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = w.Write([]byte("data"))
	}))
	defer srv.Close()

	f := NewFetcher("pod", nil, nil)
	entry := FeedEntry{Name: "cached", URL: srv.URL, TTL: 300}
	if _, err := f.Fetch(context.Background(), "agent", entry); err != nil {
		t.Fatal(err)
	}
	if _, err := f.Fetch(context.Background(), "agent", entry); err != nil {
		t.Fatal(err)
	}
	if calls != 1 {
		t.Errorf("expected 1 upstream call (cached), got %d", calls)
	}
}

func TestFetcherRefetchesAfterTTLExpiry(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = w.Write([]byte("data"))
	}))
	defer srv.Close()

	f := NewFetcher("pod", nil, nil)
	f.nowFunc = func() time.Time { return time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC) }
	entry := FeedEntry{Name: "ttl-test", URL: srv.URL, TTL: 10}
	if _, err := f.Fetch(context.Background(), "agent", entry); err != nil {
		t.Fatal(err)
	}

	f.nowFunc = func() time.Time { return time.Date(2026, 1, 1, 0, 0, 11, 0, time.UTC) }
	if _, err := f.Fetch(context.Background(), "agent", entry); err != nil {
		t.Fatal(err)
	}

	if calls != 2 {
		t.Errorf("expected 2 upstream calls after TTL expiry, got %d", calls)
	}
}

func TestFetcherReturnsStaleOnFailure(t *testing.T) {
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		if calls == 1 {
			_, _ = w.Write([]byte("good data"))
			return
		}
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	f := NewFetcher("pod", nil, nil)
	f.nowFunc = func() time.Time { return time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC) }
	entry := FeedEntry{Name: "flaky", URL: srv.URL, TTL: 1}
	if _, err := f.Fetch(context.Background(), "agent", entry); err != nil {
		t.Fatal(err)
	}

	f.nowFunc = func() time.Time { return time.Date(2026, 1, 1, 0, 0, 5, 0, time.UTC) }
	result, err := f.Fetch(context.Background(), "agent", entry)
	if err != nil {
		t.Fatal(err)
	}
	if result.Content != "good data" {
		t.Errorf("expected stale cached content, got %q", result.Content)
	}
	if !result.Stale {
		t.Error("expected stale=true after failed refresh")
	}
}

func TestFetcherReturnsUnavailableOnFirstFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	f := NewFetcher("pod", nil, nil)
	result, err := f.Fetch(context.Background(), "agent", FeedEntry{Name: "down", URL: srv.URL, TTL: 60})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Unavailable {
		t.Error("expected unavailable=true")
	}
}

func TestFetcherWrapsJSONInFencedBlock(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"wallet":{"cash":"25000.0"}}`))
	}))
	defer srv.Close()

	f := NewFetcher("pod", nil, nil)
	result, err := f.Fetch(context.Background(), "agent", FeedEntry{Name: "ctx", URL: srv.URL, TTL: 60})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(result.Content, "```json\n") {
		t.Fatalf("JSON feed should be wrapped in fenced block, got: %q", result.Content)
	}
	if !strings.HasSuffix(result.Content, "\n```") {
		t.Errorf("JSON feed should end with closing fence, got: %q", result.Content)
	}
	if !strings.Contains(result.Content, `"cash":"25000.0"`) {
		t.Error("JSON content should be preserved inside fence")
	}
}

func TestFetcherTruncatesOversizeResponse(t *testing.T) {
	big := strings.Repeat("x", MaxFeedResponseBytes+100)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(big))
	}))
	defer srv.Close()

	f := NewFetcher("pod", nil, nil)
	result, err := f.Fetch(context.Background(), "agent", FeedEntry{Name: "big", URL: srv.URL, TTL: 60})
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Content) > MaxFeedResponseBytes {
		t.Errorf("expected truncation to %d bytes, got %d", MaxFeedResponseBytes, len(result.Content))
	}
	if !result.Truncated {
		t.Error("expected truncated=true")
	}
}
