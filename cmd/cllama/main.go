package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/cost"
	"github.com/mostlydev/cllama/internal/logging"
	"github.com/mostlydev/cllama/internal/provider"
	"github.com/mostlydev/cllama/internal/proxy"
	"github.com/mostlydev/cllama/internal/sessionhistory"
	"github.com/mostlydev/cllama/internal/ui"
)

type config struct {
	APIAddr           string
	UIAddr            string
	ContextRoot       string
	AuthDir           string
	PodName           string
	UIToken           string
	SessionHistoryDir string
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		log.Fatalf("cllama: %v", err)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("cllama", flag.ContinueOnError)
	fs.SetOutput(stderr)
	healthcheck := fs.Bool("healthcheck", false, "check API server health and exit")
	if err := fs.Parse(args); err != nil {
		return err
	}

	cfg := configFromEnv()
	if *healthcheck {
		return runHealthcheck(cfg.APIAddr)
	}

	reg := provider.NewRegistry(cfg.AuthDir)
	if err := reg.LoadFromFile(); err != nil {
		return fmt.Errorf("load providers from file: %w", err)
	}
	// LoadFromEnv is fallback-only: file-backed providers are authoritative
	// and will not be overridden by env vars.
	reg.LoadFromEnv()

	logger := logging.New(stdout)
	pricing := cost.DefaultPricing()
	acc := cost.NewAccumulator()

	var recorder *sessionhistory.Recorder
	if cfg.SessionHistoryDir != "" {
		recorder = sessionhistory.New(cfg.SessionHistoryDir)
	}

	apiServer := &http.Server{
		Addr:              cfg.APIAddr,
		Handler:           newAPIHandler(cfg.ContextRoot, reg, logger, acc, pricing, cfg.PodName, recorder, cfg.UIToken),
		ReadHeaderTimeout: 10 * time.Second,
	}
	uiServer := &http.Server{
		Addr:              cfg.UIAddr,
		Handler:           newUIHandler(reg, acc, cfg.ContextRoot, cfg.UIToken),
		ReadHeaderTimeout: 10 * time.Second,
	}

	errCh := make(chan error, 2)
	go serveServer("api", apiServer, stderr, errCh)
	go serveServer("ui", uiServer, stderr, errCh)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	select {
	case sig := <-sigCh:
		fmt.Fprintf(stderr, "received signal %s, shutting down\n", sig)
	case err := <-errCh:
		return err
	}

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := apiServer.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("shutdown api server: %w", err)
	}
	if err := uiServer.Shutdown(shutdownCtx); err != nil {
		return fmt.Errorf("shutdown ui server: %w", err)
	}

	if recorder != nil {
		if err := recorder.Close(); err != nil {
			fmt.Fprintf(stderr, "close session recorder: %v\n", err)
		}
	}

	return nil
}

func newAPIHandler(contextRoot string, reg *provider.Registry, logger *logging.Logger, acc *cost.Accumulator, pricing *cost.Pricing, podName string, recorder *sessionhistory.Recorder, adminToken string) http.Handler {
	mux := http.NewServeMux()
	opts := []proxy.HandlerOption{proxy.WithCostTracking(acc, pricing)}
	if podName != "" {
		opts = append(opts, proxy.WithFeeds(podName))
	}
	if recorder != nil {
		opts = append(opts, proxy.WithSessionRecorder(recorder))
	}
	if adminToken != "" {
		opts = append(opts, proxy.WithAdminToken(adminToken))
	}
	proxyHandler := proxy.NewHandler(reg, func(agentID string) (*agentctx.AgentContext, error) {
		return agentctx.Load(contextRoot, agentID)
	}, logger, opts...)
	mux.Handle("POST /v1/chat/completions", proxyHandler)
	mux.Handle("POST /v1/messages", proxyHandler)
	mux.HandleFunc("GET /history/{agentID}", proxyHandler.HandleHistory)
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]bool{"ok": true})
	})
	return mux
}

func newUIHandler(reg *provider.Registry, acc *cost.Accumulator, contextRoot, uiToken string) http.Handler {
	mux := http.NewServeMux()
	opts := []ui.UIOption{ui.WithAccumulator(acc), ui.WithContextRoot(contextRoot)}
	if uiToken != "" {
		opts = append(opts, ui.WithUIToken(uiToken))
	}
	mux.Handle("/", ui.NewHandler(reg, opts...))
	return mux
}

func serveServer(name string, server *http.Server, stderr io.Writer, errCh chan<- error) {
	fmt.Fprintf(stderr, "cllama %s listening on %s\n", name, server.Addr)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		errCh <- fmt.Errorf("%s server: %w", name, err)
	}
}

func runHealthcheck(apiAddr string) error {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(healthcheckURL(apiAddr))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("health endpoint returned %s", resp.Status)
	}
	return nil
}

func healthcheckURL(addr string) string {
	if addr == "" {
		addr = ":8080"
	}
	if addr[0] == ':' {
		return "http://127.0.0.1" + addr + "/health"
	}
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return "http://127.0.0.1:8080/health"
	}
	if host == "" || host == "0.0.0.0" || host == "::" {
		host = "127.0.0.1"
	}
	if ip := net.ParseIP(host); ip != nil && ip.To4() == nil {
		host = "[" + host + "]"
	}
	return "http://" + host + ":" + port + "/health"
}

func configFromEnv() config {
	return config{
		APIAddr:           envOr("LISTEN_ADDR", ":8080"),
		UIAddr:            envOr("UI_ADDR", ":8081"),
		ContextRoot:       envOr("CLAW_CONTEXT_ROOT", "/claw/context"),
		AuthDir:           envOr("CLAW_AUTH_DIR", "/claw/auth"),
		PodName:           os.Getenv("CLAW_POD"),
		UIToken:           os.Getenv("CLLAMA_UI_TOKEN"),
		SessionHistoryDir: os.Getenv("CLAW_SESSION_HISTORY_DIR"),
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
