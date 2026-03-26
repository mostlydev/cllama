package proxy

import (
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
)

func currentTimeLine(agentCtx *agentctx.AgentContext, now time.Time) string {
	loc, tzName := resolveAgentLocation(agentCtx)
	current := now.In(loc)
	abbr, _ := current.Zone()

	line := fmt.Sprintf("Current time: %s", current.Format("2006-01-02 3:04 PM MST"))
	if tzName != "" && tzName != abbr {
		line += " (" + tzName + ")"
	}
	return line
}

func resolveAgentLocation(agentCtx *agentctx.AgentContext) (*time.Location, string) {
	candidates := []string{
		strings.TrimSpace(agentCtx.MetadataString("timezone")),
		strings.TrimSpace(os.Getenv("TZ")),
		"UTC",
	}

	for _, tz := range candidates {
		if tz == "" {
			continue
		}
		loc, err := time.LoadLocation(tz)
		if err == nil {
			return loc, tz
		}
	}

	return time.UTC, "UTC"
}
