package proxy

import (
	"strings"
)

func joinRuntimeContext(sections ...string) string {
	var out []string
	for _, section := range sections {
		trimmed := strings.TrimSpace(section)
		if trimmed == "" {
			continue
		}
		out = append(out, trimmed)
	}
	return strings.Join(out, "\n\n")
}
