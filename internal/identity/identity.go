package identity

import (
	"fmt"
	"strings"
)

// ParseBearer extracts agent ID and secret from "Bearer <agent-id>:<secret>".
// It splits on the first colon only, allowing colons inside the secret.
func ParseBearer(header string) (agentID, secret string, err error) {
	header = strings.TrimSpace(header)
	if header == "" {
		return "", "", fmt.Errorf("missing authorization header")
	}

	scheme, token, ok := strings.Cut(header, " ")
	if !ok || !strings.EqualFold(scheme, "Bearer") {
		return "", "", fmt.Errorf("invalid authorization: expected Bearer scheme")
	}
	token = strings.TrimSpace(token)
	if token == "" {
		return "", "", fmt.Errorf("invalid bearer token: empty token")
	}

	agentID, secret, ok = strings.Cut(token, ":")
	if !ok || agentID == "" || secret == "" {
		return "", "", fmt.Errorf("invalid bearer token: expected <agent-id>:<secret>")
	}
	return agentID, secret, nil
}
