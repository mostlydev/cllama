package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/mostlydev/cllama/internal/subscription"
)

const authUsage = `usage:
  cllama auth login <plan> [--listen addr] [--port n] [--timeout d]
  cllama auth status [--json]
  cllama auth logout <plan> [--forget]

Plans: %s

login starts a callback listener on this machine's loopback address, prints
the sign-in URL, and saves the sign-in for the proxy once the vendor has
confirmed it. The operator runs it; the proxy and its agents never sign in.
`

func planNames() string {
	return strings.Join(subscription.Providers(), ", ")
}

// runAuth implements the operator's sign-in commands. They act on the same
// auth directory the proxy reads, so a sign-in takes effect without a restart.
func runAuth(args []string, stdout, stderr io.Writer) error {
	if len(args) == 0 {
		fmt.Fprintf(stderr, authUsage, planNames())
		return errors.New("auth: missing subcommand")
	}
	cfg := configFromEnv()
	store := subscription.NewStore(cfg.AuthDir)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Flags may come before or after the plan name.
	var name string
	rest := args[1:]
	if len(rest) > 0 && !strings.HasPrefix(rest[0], "-") {
		name, rest = rest[0], rest[1:]
	}
	planFor := func() (subscription.Plan, error) {
		plan, ok := subscription.Lookup(name)
		if !ok {
			return subscription.Plan{}, fmt.Errorf("auth: unknown plan %q (known: %s)", name, planNames())
		}
		return plan, nil
	}

	switch args[0] {
	case "login":
		fs := flag.NewFlagSet("auth login", flag.ContinueOnError)
		fs.SetOutput(stderr)
		listen := fs.String("listen", "", "callback listener address (default 127.0.0.1:<port>)")
		port := fs.Int("port", 0, "callback port named in the redirect URI")
		timeout := fs.Duration("timeout", 10*time.Minute, "how long to wait for the browser")
		if err := fs.Parse(rest); err != nil {
			return err
		}
		plan, err := planFor()
		if err != nil {
			return err
		}
		rec, err := subscription.NewClient(plan, store).Login(ctx, subscription.LoginOptions{
			Listen: *listen, Port: *port, Out: stdout, Timeout: *timeout,
		})
		if err != nil {
			return fmt.Errorf("auth login: %w", err)
		}
		account := rec.Email
		if account == "" {
			account = "the selected account"
		}
		fmt.Fprintf(stdout, "\nSigned in to the %s as %s. The proxy will use it from the next request.\n", plan.Label, account)
		return nil

	case "status":
		fs := flag.NewFlagSet("auth status", flag.ContinueOnError)
		fs.SetOutput(stderr)
		asJSON := fs.Bool("json", false, "print JSON")
		if err := fs.Parse(args[1:]); err != nil {
			return err
		}
		statuses := subscription.NewManager(cfg.AuthDir).Status()
		if *asJSON {
			enc := json.NewEncoder(stdout)
			enc.SetIndent("", "  ")
			return enc.Encode(map[string]any{"subscriptions": statuses})
		}
		for _, st := range statuses {
			line := st.Provider + ": " + st.State
			if st.Account != "" {
				line += " (" + st.Account + ")"
			}
			if st.Reason != "" {
				line += " - " + st.Reason
			}
			fmt.Fprintln(stdout, line)
		}
		return nil

	case "logout":
		fs := flag.NewFlagSet("auth logout", flag.ContinueOnError)
		fs.SetOutput(stderr)
		forget := fs.Bool("forget", false, "also forget the client registration and account")
		if err := fs.Parse(rest); err != nil {
			return err
		}
		plan, err := planFor()
		if err != nil {
			return err
		}
		revoked, err := subscription.NewClient(plan, store).Logout(ctx, *forget)
		if err != nil {
			return fmt.Errorf("auth logout: %w", err)
		}
		fmt.Fprintf(stdout, "Signed out of the %s on this host.\n", plan.Label)
		if !revoked {
			fmt.Fprintln(stdout, "The vendor did not confirm the revocation. Disconnect the app in the plan's settings to be sure.")
		}
		return nil
	}
	fmt.Fprintf(stderr, authUsage, planNames())
	return fmt.Errorf("auth: unknown subcommand %q", args[0])
}
