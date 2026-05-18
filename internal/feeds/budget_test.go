package feeds

import "testing"

func TestBudgetFromEnv(t *testing.T) {
	t.Setenv(EnvMaxFeedResponseBytes, "262144")
	t.Setenv(EnvMaxTotalFeedBytes, "393216")

	budget := BudgetFromEnv()
	if budget.MaxFeedResponseBytes != 262144 {
		t.Fatalf("MaxFeedResponseBytes = %d; want 262144", budget.MaxFeedResponseBytes)
	}
	if budget.MaxTotalFeedBytes != 393216 {
		t.Fatalf("MaxTotalFeedBytes = %d; want 393216", budget.MaxTotalFeedBytes)
	}
}

func TestBudgetFromEnvIgnoresInvalidValues(t *testing.T) {
	t.Setenv(EnvMaxFeedResponseBytes, "nope")
	t.Setenv(EnvMaxTotalFeedBytes, "-1")

	budget := BudgetFromEnv()
	defaults := DefaultBudget()
	if budget != defaults {
		t.Fatalf("budget = %+v; want defaults %+v", budget, defaults)
	}
}
