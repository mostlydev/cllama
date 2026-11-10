package subscription

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"
)

const (
	maxCatalogBytes  = 4 << 20
	maxCatalogModels = 2000
	maxCatalogSlug   = 128
	catalogTimeout   = 20 * time.Second
)

// Catalog is the set of models a plan sign-in may use, as the vendor lists
// them for that account. It carries model names only.
type Catalog struct {
	Provider string    `json:"provider"`
	Complete bool      `json:"complete"`
	IDs      []string  `json:"ids"`
	ReadAt   time.Time `json:"read_at"`
}

// Catalog reads the account's model list with the plan's own sign-in, which
// is refreshed first if it is due. The token is sent to the plan's inference
// origin only, and redirects are not followed. Only models the vendor marks
// for listing are returned. Anything short of one whole, well-formed listing
// is an error whose text carries neither the token nor the reply.
func (m *Manager) Catalog(ctx context.Context, provider string) (*Catalog, error) {
	c, ok := m.clients[provider]
	if !ok {
		return nil, &Error{Reason: ReasonSigninRequired, Terminal: true, Stage: StageSigninRequired, Detail: "unknown plan provider"}
	}
	token, err := m.Token(ctx, provider)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(ctx, catalogTimeout)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(c.Plan.InferenceBase, "/")+"/models", nil)
	if err != nil {
		return nil, errors.New("plan catalog endpoint is not usable")
	}
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Accept", "application/json")
	client := *c.HTTP
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := client.Do(req)
	if err != nil {
		return nil, errors.New("plan catalog could not be read")
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, errors.New("plan catalog was refused")
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxCatalogBytes+1))
	if err != nil || len(body) > maxCatalogBytes {
		return nil, errors.New("plan catalog is too large or was cut short")
	}
	var listing struct {
		Models []struct {
			Slug       any `json:"slug"`
			Visibility any `json:"visibility"`
		} `json:"models"`
	}
	if json.Unmarshal(body, &listing) != nil || listing.Models == nil {
		return nil, errors.New("plan catalog is not a model list")
	}
	if len(listing.Models) > maxCatalogModels {
		return nil, errors.New("plan catalog has too many entries")
	}
	seen := map[string]bool{}
	ids := []string{}
	for _, entry := range listing.Models {
		slug, ok := entry.Slug.(string)
		if !ok || slug == "" || len(slug) > maxCatalogSlug {
			return nil, errors.New("plan catalog has a malformed entry")
		}
		if seen[slug] {
			return nil, errors.New("plan catalog repeats an entry")
		}
		seen[slug] = true
		if visibility, _ := entry.Visibility.(string); visibility == "list" {
			ids = append(ids, slug)
		}
	}
	sort.Strings(ids)
	return &Catalog{Provider: provider, Complete: true, IDs: ids, ReadAt: c.now().UTC().Truncate(time.Second)}, nil
}
