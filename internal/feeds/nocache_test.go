package feeds

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestNoCacheEvictsContentAndCannotUseStaleFallback(t *testing.T) {
	fail := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if fail {
			w.WriteHeader(503)
			return
		}
		w.Write([]byte("private-content"))
	}))
	defer srv.Close()
	f := NewFetcher("pod", nil, nil)
	e := FeedEntry{URL: srv.URL, Name: "fixture", TTL: 3600}
	if _, err := f.Fetch(context.Background(), "agent", e); err != nil {
		t.Fatal(err)
	}
	if len(f.cache) != 1 {
		t.Fatal("fixture did not cache")
	}
	e.NoCache = true
	fail = true
	result, err := f.Fetch(context.Background(), "agent", e)
	if err != nil || !result.Unavailable || result.Content != "" || len(f.cache) != 0 {
		t.Fatalf("retained/reused cache: %+v %v", result, err)
	}
}
