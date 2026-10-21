package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestRevocationDrainsAdmissionAndPersists(t *testing.T) {
	dir := t.TempDir()
	f := newRevocationFence(dir)
	ctx, finish, err := f.register(context.Background(), "worker-1")
	if err != nil {
		t.Fatal(err)
	}
	defer finish()
	release, err := f.admit(ctx, "worker-1")
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() { _, err := f.revoke(context.Background(), "worker-1", 7); done <- err }()
	select {
	case <-ctx.Done():
	case <-time.After(time.Second):
		t.Fatal("request was not cancelled")
	}
	select {
	case err := <-done:
		t.Fatalf("ack before drain: %v", err)
	default:
	}
	if _, err := f.admit(context.Background(), "worker-1"); !errors.Is(err, errAgentRevoked) {
		t.Fatalf("admission after closure: %v", err)
	}
	release()
	if err := <-done; err != nil {
		t.Fatal(err)
	}
	again := newRevocationFence(dir)
	if _, _, err := again.register(context.Background(), "worker-1"); !errors.Is(err, errAgentRevoked) {
		t.Fatalf("restart reopened: %v", err)
	}
	if _, err := again.revoke(context.Background(), "worker-1", 7); err != nil {
		t.Fatal(err)
	}
	if _, err := again.revoke(context.Background(), "worker-1", 6); !errors.Is(err, errStaleRevocation) {
		t.Fatalf("stale generation: %v", err)
	}
}

func TestRevocationPersistenceFailsClosed(t *testing.T) {
	for _, corrupt := range []bool{false, true} {
		t.Run(map[bool]string{false: "missing", true: "corrupt"}[corrupt], func(t *testing.T) {
			dir := filepath.Join(t.TempDir(), "fences")
			f := newRevocationFence(dir)
			_, finish, err := f.register(context.Background(), "worker-1")
			if err != nil {
				t.Fatal(err)
			}
			finish()
			if corrupt {
				err = os.WriteFile(filepath.Join(dir, "ledger.json"), []byte("broken"), 0600)
			} else {
				err = os.RemoveAll(dir)
			}
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := f.register(context.Background(), "worker-2"); err == nil {
				t.Fatal("damaged state admitted a request")
			}
		})
	}
}

func TestRevocationCancelsLiveLoops(t *testing.T) {
	// bridge is chat/completions served by an anthropic-format provider.
	for _, format := range []string{"openai", "anthropic", "bridge"} {
		for _, transport := range []string{"http", "mcp"} {
			for _, phase := range []string{"provider", "tool"} {
				for _, stream := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/%s/%s/stream=%t", format, transport, phase, stream), func(t *testing.T) {
						reached := make(chan struct{})
						cleanup := make(chan struct{})

						block := func(r *http.Request) {
							close(reached)
							select {
							case <-r.Context().Done():
							case <-cleanup:
							}
						}
						toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							var payload map[string]any
							_ = json.NewDecoder(r.Body).Decode(&payload)
							w.Header().Set("Content-Type", "application/json")
							if transport == "mcp" {
								switch payload["method"] {
								case "initialize":
									fmt.Fprint(w, `{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2025-11-25","capabilities":{},"serverInfo":{"name":"test","version":"1"}}}`)
									return
								case "notifications/initialized":
									w.WriteHeader(202)
									return
								}
							}
							block(r)
						}))

						manifest := managedToolManifestForURL(toolSrv.URL, http.MethodPost, "/tool", "secret-tool")
						manifest.Tools[0].Execution.Transport = transport
						manifest.Tools[0].Execution.ToolName = "get_market_context"
						name := managedToolHashlessAliasForCanonical(manifest.Tools[0].Name)
						backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							_, _ = io.Copy(io.Discard, r.Body)
							if phase == "provider" {
								block(r)
								return
							}
							w.Header().Set("Content-Type", "application/json")
							if format == "openai" {
								fmt.Fprintf(w, `{"id":"resp","choices":[{"message":{"tool_calls":[{"id":"call-1","type":"function","function":{"name":%q,"arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`, name)
							} else {
								fmt.Fprintf(w, `{"id":"resp","type":"message","content":[{"type":"tool_use","id":"call-1","name":%q,"input":{}}],"stop_reason":"tool_use","usage":{"input_tokens":1,"output_tokens":1}}`, name)
							}
						}))
						defer func() { close(cleanup); backend.Close(); toolSrv.Close() }()
						reg := provider.NewRegistry("")
						providerName := map[bool]string{true: "openai", false: "anthropic"}[format == "openai"]
						prov := &provider.Provider{Name: providerName, BaseURL: backend.URL + "/v1", APIKey: "provider-secret", Auth: "bearer"}
						if providerName == "anthropic" {
							prov.APIFormat = "anthropic"
							prov.Auth = "x-api-key"
						}
						reg.Set(providerName, prov)
						var logs bytes.Buffer
						h := NewHandler(reg, stubContextLoaderWithTools("worker-1", "worker-1:secret", manifest), logging.New(&logs), WithRevocationDir(t.TempDir()), WithAdminToken("admin-secret"))
						path := "/v1/chat/completions"
						if format == "anthropic" {
							path = "/v1/messages"
						}
						req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(fmt.Sprintf(`{"model":%q,"messages":[{"role":"user","content":"run"}],"max_tokens":20,"stream":%t}`, providerName+"/model", stream)))
						req.Header.Set("Authorization", "Bearer worker-1:secret")
						done := make(chan struct{})
						streamResponse := &revocationStreamRecorder{ResponseRecorder: httptest.NewRecorder(), started: make(chan struct{})}
						go func() { h.ServeHTTP(streamResponse, req); close(done) }()
						select {
						case <-reached:
						case <-time.After(3 * time.Second):
							t.Fatal("did not reach " + phase)
						}
						if stream {
							select {
							case <-streamResponse.started:
							case <-time.After(3 * time.Second):
								t.Fatal("stream never began")
							}
						}
						response := revokeForTest(h, "worker-1", 3, "admin-secret")
						if response.Code != 200 {
							t.Fatalf("revoke: %d %s", response.Code, response.Body.String())
						}
						select {
						case <-done:
						case <-time.After(3 * time.Second):
							t.Fatal("old loop survived revoke")
						}
						if stream && !strings.Contains(streamResponse.Header().Get("Content-Type"), "text/event-stream") {
							t.Fatalf("stream response was not SSE: %v", streamResponse.Header())
						}
						fresh := httptest.NewRecorder()
						h.ServeHTTP(fresh, req.Clone(context.Background()))
						if fresh.Code != 403 {
							t.Fatalf("fresh request got %d", fresh.Code)
						}
						if strings.Contains(logs.String(), "provider-secret") || strings.Contains(logs.String(), "secret-tool") {
							t.Fatal("audit leaked credentials")
						}
						if phase == "tool" && (!strings.Contains(logs.String(), `"dispatch_status":"admitted"`) || !strings.Contains(logs.String(), `"dispatch_status":"admitted_before_revoke"`) || !strings.Contains(logs.String(), `"dispatch_id":`)) {
							t.Fatal("missing admission audit")
						}
					})
				}
			}
		}
	}
}

func revokeForTest(h *Handler, id string, generation uint64, token string) *httptest.ResponseRecorder {
	r := httptest.NewRequest(http.MethodPost, "/internal/agents/"+id+"/revoke", strings.NewReader(fmt.Sprintf(`{"generation":%d}`, generation)))
	r.SetPathValue("agentID", id)
	r.Header.Set("Authorization", "Bearer "+token)
	w := httptest.NewRecorder()
	h.HandleRevoke(w, r)
	return w
}

func TestRevocationEndpointAuthorizationAndGenerations(t *testing.T) {
	h := NewHandler(nil, nil, nil, WithRevocationDir(t.TempDir()), WithAdminToken("operator"))
	if got := revokeForTest(h, "worker-1", 2, "worker-1:secret"); got.Code != 403 {
		t.Fatalf("worker could revoke: %d", got.Code)
	}
	for _, gen := range []uint64{2, 2, 3} {
		if got := revokeForTest(h, "worker-1", gen, "operator"); got.Code != 200 {
			t.Fatalf("generation %d: %d", gen, got.Code)
		}
	}
	if got := revokeForTest(h, "worker-1", 2, "operator"); got.Code != 409 {
		t.Fatalf("stale generation: %d", got.Code)
	}
	if got := revokeForTest(h, "worker-1", 0, "operator"); got.Code != 400 {
		t.Fatalf("zero generation: %d", got.Code)
	}
}

func TestRevocationMissingLedgerOnRestartFailsClosed(t *testing.T) {
	dir := t.TempDir()
	f := newRevocationFence(dir)
	if _, err := f.revoke(context.Background(), "worker-1", 1); err != nil {
		t.Fatal(err)
	}
	if err := os.Remove(filepath.Join(dir, "ledger.json")); err != nil {
		t.Fatal(err)
	}
	restarted := newRevocationFence(dir)
	if _, _, err := restarted.register(context.Background(), "worker-1"); err == nil {
		t.Fatal("missing ledger reopened after restart")
	}
}

func TestRevocationTimeoutNeverAcknowledgesUndrainedDispatch(t *testing.T) {
	f := newRevocationFence(t.TempDir())
	release, err := f.admit(context.Background(), "worker-1")
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := f.revoke(ctx, "worker-1", 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("unexpected acknowledgement: %v", err)
	}
	release()
	if _, err := f.revoke(context.Background(), "worker-1", 1); err != nil {
		t.Fatal(err)
	}
}

func TestRevocationConcurrentAdmissionBoundary(t *testing.T) {
	f := newRevocationFence(t.TempDir())
	start := make(chan struct{})
	var wg sync.WaitGroup
	for n := 0; n < 32; n++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			for i := 0; i < 50; i++ {
				release, err := f.admit(context.Background(), "worker-1")
				if err == nil {
					runtime.Gosched()
					release()
				} else if !errors.Is(err, errAgentRevoked) {
					t.Errorf("admit: %v", err)
				}
			}
		}()
	}
	close(start)
	if _, err := f.revoke(context.Background(), "worker-1", 1); err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 100; i++ {
		if release, err := f.admit(context.Background(), "worker-1"); err == nil {
			release()
			t.Fatal("admitted after acknowledgement")
		}
	}
	wg.Wait()
}

func TestRevocationCorruptLedgerOnRestartFailsClosed(t *testing.T) {
	dir := t.TempDir()
	f := newRevocationFence(dir)
	if _, err := f.revoke(context.Background(), "worker-1", 1); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "ledger.json"), []byte(`{"version":1,"agents":{"worker-1":{"agent_id":"worker-1","generation":0,"state":"revoked"}}}`), 0600); err != nil {
		t.Fatal(err)
	}
	restarted := newRevocationFence(dir)
	if _, _, err := restarted.register(context.Background(), "worker-2"); err == nil {
		t.Fatal("corrupt state admitted after restart")
	}
}

func TestRevocationCapacityNeverAcknowledgesOrClosesUnrelatedAgents(t *testing.T) {
	f := newRevocationFence(t.TempDir())
	f.maxBytes = 32
	for i := 0; i < 2; i++ {
		if _, err := f.revoke(context.Background(), "worker-1", 1); !errors.Is(err, errRevocationCapacity) {
			t.Fatalf("capacity retry acknowledged: %v", err)
		}
	}
	if _, _, err := f.register(context.Background(), "worker-1"); !errors.Is(err, errAgentRevoked) {
		t.Fatalf("affected agent reopened: %v", err)
	}
	_, finish, err := f.register(context.Background(), "worker-2")
	if err != nil {
		t.Fatal(err)
	}
	finish()
	f.maxBytes = 16 << 20
	if _, err := f.revoke(context.Background(), "worker-1", 1); err != nil {
		t.Fatal(err)
	}
	restarted := newRevocationFence(f.dir)
	if _, _, err := restarted.register(context.Background(), "worker-1"); !errors.Is(err, errAgentRevoked) {
		t.Fatalf("recovery did not persist: %v", err)
	}
}

// The signal ensures revocation happens after an SSE keepalive has actually
// committed headers, rather than before a stream has begun.
type revocationStreamRecorder struct {
	*httptest.ResponseRecorder
	started chan struct{}
	once    sync.Once
}

func (r *revocationStreamRecorder) Write(p []byte) (int, error) {
	n, err := r.ResponseRecorder.Write(p)
	if bytes.Contains(p, []byte(": managed")) {
		r.once.Do(func() { close(r.started) })
	}
	return n, err
}

func (r *revocationStreamRecorder) WriteString(s string) (int, error) { return r.Write([]byte(s)) }

func TestRevocationCleansSnapshotAfterLateHandlerExit(t *testing.T) {
	h := NewHandler(nil, stubContextLoaderWithTools("worker-1", "worker-1:secret", nil), nil, WithRevocationDir(t.TempDir()), WithAdminToken("operator"))
	h.registry.Set("openai", &provider.Provider{Name: "openai", BaseURL: "http://127.0.0.1:1", APIKey: "key", Auth: "bearer"})
	body := &revocationHeldBody{Reader: strings.NewReader(`{"model":"openai/model","messages":[]}`), entered: make(chan struct{}), release: make(chan struct{})}
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Body = io.NopCloser(body)
	req.Header.Set("Authorization", "Bearer worker-1:secret")
	done := make(chan struct{})
	go func() { h.ServeHTTP(httptest.NewRecorder(), req); close(done) }()
	<-body.entered
	h.snapshots.Put(ContextSnapshot{AgentID: "worker-1", System: "old"})
	if got := revokeForTest(h, "worker-1", 1, "operator"); got.Code != 200 {
		t.Fatalf("revoke: %d", got.Code)
	}
	// The authenticated handler is deliberately still inside Body.Read. It may
	// capture context after revocation, so cleanup must also run on handler exit.
	close(body.release)
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("handler did not finish")
	}
	if _, ok := h.snapshots.Get("worker-1"); ok {
		t.Fatal("revoked snapshot survived handler exit")
	}
	if len(h.snapshots.History("worker-1", 20)) != 0 {
		t.Fatal("revoked snapshot history survived")
	}
}

type revocationHeldBody struct {
	io.Reader
	entered, release chan struct{}
	once             sync.Once
}

func (b *revocationHeldBody) Read(p []byte) (int, error) {
	b.once.Do(func() { close(b.entered); <-b.release })
	return b.Reader.Read(p)
}
