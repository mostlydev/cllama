package feeds

import (
	"testing"
	"time"
)

func TestBudgetFromEnv(t *testing.T) {
	t.Setenv(EnvMaxFeedResponseBytes, "262144")
	t.Setenv(EnvMaxTotalFeedBytes, "393216")
	t.Setenv(EnvFeedFetchTimeoutMS, "10000")

	budget := BudgetFromEnv()
	if budget.MaxFeedResponseBytes != 262144 {
		t.Fatalf("MaxFeedResponseBytes = %d; want 262144", budget.MaxFeedResponseBytes)
	}
	if budget.MaxTotalFeedBytes != 393216 {
		t.Fatalf("MaxTotalFeedBytes = %d; want 393216", budget.MaxTotalFeedBytes)
	}
	if budget.FetchTimeout != 10*time.Second {
		t.Fatalf("FetchTimeout = %v; want 10s", budget.FetchTimeout)
	}
}

func TestBudgetFromEnvIgnoresInvalidValues(t *testing.T) {
	t.Setenv(EnvMaxFeedResponseBytes, "nope")
	t.Setenv(EnvMaxTotalFeedBytes, "-1")
	t.Setenv(EnvFeedFetchTimeoutMS, "not-a-number")

	budget := BudgetFromEnv()
	defaults := DefaultBudget()
	if budget != defaults {
		t.Fatalf("budget = %+v; want defaults %+v", budget, defaults)
	}
}

func TestBudgetFromEnvFetchTimeoutBounds(t *testing.T) {
	cases := []struct {
		name string
		raw  string
		want time.Duration
	}{
		{"unset", "", FetchTimeout},
		{"below floor", "50", FetchTimeout},
		{"at floor", "100", 100 * time.Millisecond},
		{"at ceiling", "120000", 120 * time.Second},
		{"above ceiling", "120001", FetchTimeout},
		{"zero", "0", FetchTimeout},
		{"negative", "-5000", FetchTimeout},
		{"duration overflow", "10000000000000", FetchTimeout},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Setenv(EnvFeedFetchTimeoutMS, tc.raw)
			budget := BudgetFromEnv()
			if budget.FetchTimeout != tc.want {
				t.Fatalf("FetchTimeout = %v; want %v", budget.FetchTimeout, tc.want)
			}
		})
	}
}

func TestBudgetNormalizeDefaultsFetchTimeout(t *testing.T) {
	budget := Budget{}.Normalize()
	if budget.FetchTimeout != FetchTimeout {
		t.Fatalf("FetchTimeout = %v; want default %v", budget.FetchTimeout, FetchTimeout)
	}
}

func TestNewFetcherUsesBudgetFetchTimeout(t *testing.T) {
	fetcher := NewFetcherWithBudget("pod", nil, nil, Budget{FetchTimeout: 7 * time.Second})
	if fetcher.client.Timeout != 7*time.Second {
		t.Fatalf("client.Timeout = %v; want 7s", fetcher.client.Timeout)
	}
}
