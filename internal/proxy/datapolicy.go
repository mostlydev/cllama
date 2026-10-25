package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/provider"
	"github.com/mostlydev/cllama/internal/sessionhistory"
)

type dataPolicyKey struct{}

func requestDataPolicy(ctx context.Context) agentctx.DataPolicy {
	if p, ok := ctx.Value(dataPolicyKey{}).(agentctx.DataPolicy); ok {
		return p
	}
	return agentctx.DefaultDataPolicy()
}

func localContentAllowed(a *agentctx.AgentContext) bool {
	p, err := a.EffectiveDataPolicy()
	return err == nil && p.LocalRetention == "allowed"
}

// metadataHistory deliberately excludes content, tool arguments/results and
// content hashes; accounting remains available for budgets and audit.
func metadataHistory(a *agentctx.AgentContext, e sessionhistory.Entry) sessionhistory.Entry {
	if localContentAllowed(a) {
		return e
	}
	e.RequestOriginal = nil
	e.RequestEffective = nil
	e.ToolTrace = nil
	e.Response = sessionhistory.Payload{Format: "metadata"}
	e.ID = ""
	if e.TS == "" {
		e.TS = time.Now().UTC().Format(time.RFC3339Nano)
	}
	return e
}

type dataPolicyError struct{ dimension string }

func (e *dataPolicyError) Error() string {
	return "data policy requirement not established: " + e.dimension
}
func denyData(dimension string) error { return &dataPolicyError{dimension: dimension} }

// Official controls are only attached to exact known origins and API roots.
// Arbitrary BASE_URL overrides never inherit an official provider's terms.
func officialDataEndpoint(name, raw string) (bool, string) {
	u, err := url.Parse(raw)
	if err != nil || u.Scheme != "https" || u.User != nil || u.RawQuery != "" || u.Fragment != "" || u.RawPath != "" || (u.Port() != "" && u.Port() != "443") {
		return false, ""
	}
	host := strings.ToLower(u.Hostname())
	path := strings.TrimRight(u.Path, "/")
	if path == "" {
		path = "/v1"
	}
	switch name {
	case "openai":
		if path != "/v1" {
			return false, ""
		}
		switch host {
		case "api.openai.com":
			return true, ""
		case "us.api.openai.com":
			return true, "us"
		case "eu.api.openai.com":
			return true, "eu"
		case "jp.api.openai.com", "ca.api.openai.com", "au.api.openai.com", "in.api.openai.com", "sg.api.openai.com", "kr.api.openai.com":
			return true, ""
		}
	case "anthropic":
		return host == "api.anthropic.com" && path == "/v1", ""
	case "google":
		return host == "generativelanguage.googleapis.com" && (path == "/v1beta/openai" || path == "/v1beta" || path == "/v1"), ""
	case "openrouter":
		if path != "/api/v1" {
			return false, ""
		}
		switch host {
		case "openrouter.ai":
			return true, ""
		case "us.openrouter.ai":
			return true, "us"
		case "eu.openrouter.ai":
			return true, "eu"
		}
	case "vercel":
		return host == "ai-gateway.vercel.sh" && path == "/v1", ""
	case "xai":
		if path != "/v1" {
			return false, ""
		}
		switch host {
		case "api.x.ai":
			return true, ""
		case "us.api.x.ai":
			return true, "us"
		}
	case "meta":
		return host == "api.meta.ai" && path == "/v1", ""
	}
	return false, ""
}

func policyAllowsRegion(p agentctx.DataPolicy, region string) bool {
	for _, r := range p.Regions {
		if r == "global" || r == region {
			return true
		}
	}
	return false
}

func attestedGuarantees(p agentctx.DataPolicy, prov *provider.Provider, official bool) map[string]bool {
	out := map[string]bool{}
	actual, err := agentctx.NormalizePolicyEndpoint(prov.BaseURL)
	if err != nil {
		return out
	}
	for _, a := range p.Attestations {
		if a.Provider != prov.Name {
			continue
		}
		if a.Endpoint == "" {
			if !official {
				continue
			}
		} else {
			endpoint, err := agentctx.NormalizePolicyEndpoint(a.Endpoint)
			if err != nil || endpoint != actual {
				continue
			}
		}
		for _, g := range a.Guarantees {
			out[g] = true
		}
	}
	return out
}

func anthropicGeoModel(model string) bool {
	model = strings.TrimPrefix(model, "anthropic/")
	for _, family := range []string{"claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-6", "claude-opus-4.6", "claude-sonnet-4.6", "claude-haiku-4.6", "claude-opus-4-7", "claude-opus-4.7", "claude-sonnet-4-7", "claude-sonnet-4.7"} {
		if model == family || strings.HasPrefix(model, family+"-") {
			return true
		}
	}
	for _, family := range []string{"claude-sonnet-5", "claude-opus-5", "claude-haiku-5", "claude-fable-5", "claude-mythos-5"} {
		if model == family || strings.HasPrefix(model, family+"-") || strings.HasPrefix(model, family+".") {
			return true
		}
	}
	return false
}

func retentionIneligibleModel(model string) bool {
	model = strings.ToLower(model)
	return strings.Contains(model, "claude-fable-5") || strings.Contains(model, "claude-mythos-5")
}

func hasRetentionIneligibleModel(body map[string]any) bool {
	if model, _ := body["model"].(string); retentionIneligibleModel(model) {
		return true
	}
	if models, ok := body["models"].([]any); ok {
		for _, m := range models {
			if model, ok := m.(string); !ok || retentionIneligibleModel(model) {
				return true
			}
		}
	}
	return false
}

// An untrusted caller cannot widen the processor chain through hosted tools.
func dataPolicyConstrained(p agentctx.DataPolicy) bool {
	return p.Training == "none" || p.Retention == "none" || !policyAllowsRegion(p, "global")
}
func onlineModelVariant(model string) bool {
	parts := strings.Split(model, ":")
	for _, suffix := range parts[1:] {
		if strings.EqualFold(strings.TrimSpace(suffix), "online") {
			return true
		}
	}
	return false
}

func hasUnscopedFeature(x map[string]any) bool {
	// OpenRouter enables its hosted web plugin for an "online" model variant;
	// variant suffixes combine in any order (model:online:nitro).
	if model, _ := x["model"].(string); onlineModelVariant(model) {
		return true
	}
	if models, ok := x["models"].([]any); ok {
		for _, m := range models {
			if name, _ := m.(string); onlineModelVariant(name) {
				return true
			}
		}
	}
	for _, key := range []string{"mcp_servers", "web_search_options", "plugins"} {
		if v, present := x[key]; present && v != nil {
			switch v := v.(type) {
			case []any:
				if len(v) > 0 {
					return true
				}
			case map[string]any:
				return true
			default:
				return true
			}
		}
	}
	if tools, ok := x["tools"].([]any); ok {
		for _, raw := range tools {
			if tool, ok := raw.(map[string]any); ok {
				typ, _ := tool["type"].(string)
				if typ != "" && typ != "function" && typ != "custom" {
					return true
				}
			}
		}
	}
	return false
}

// Short-lived Anthropic KV caches are documented as ZDR-eligible. Longer or
// unknown cache markers require an explicit retention relaxation.
func safeShortCache(v any, allow bool) bool {
	if v == nil {
		return true
	}
	if !allow {
		return false
	}
	m, ok := v.(map[string]any)
	if !ok || m["type"] != "ephemeral" {
		return false
	}
	for key := range m {
		if key != "type" && key != "ttl" {
			return false
		}
	}
	ttl, present := m["ttl"]
	return !present || ttl == "5m"
}

func hasPersistentFeature(x map[string]any, allowShortCache bool) bool {
	for k, val := range x {
		switch k {
		case "background", "store", "store_messages":
			if val != nil && val != false {
				return true
			}
		case "previous_response_id", "conversation", "container", "file_id", "cached_content", "cachedContent":
			if val != nil && val != "" {
				return true
			}
		case "cache_control":
			if !safeShortCache(val, allowShortCache) {
				return true
			}
		case "prompt_cache_retention":
			if val != nil && val != "in_memory" {
				return true
			}
		case "prompt_cache_options":
			if val != nil {
				return true
			}
		}
	}
	// Provider-specific options are controls, not business tool arguments.
	if opts, ok := x["providerOptions"].(map[string]any); ok {
		for _, raw := range opts {
			if m, ok := raw.(map[string]any); ok && hasPersistentFeature(m, allowShortCache) {
				return true
			}
		}
	}
	// Never descend tool schemas/arguments: a business property may be named store.
	for _, key := range []string{"messages", "system", "input", "tools"} {
		if hasStoredContent(x[key], allowShortCache) {
			return true
		}
	}
	return false
}
func hasStoredContent(v any, allowShortCache bool) bool {
	switch x := v.(type) {
	case []any:
		for _, v := range x {
			if hasStoredContent(v, allowShortCache) {
				return true
			}
		}
	case map[string]any:
		for _, k := range []string{"file_id", "cached_content", "cachedContent"} {
			if v, present := x[k]; present && v != nil && v != "" {
				return true
			}
		}
		if v, present := x["cache_control"]; present && !safeShortCache(v, allowShortCache) {
			return true
		}
		for _, k := range []string{"content", "source", "file"} {
			if hasStoredContent(x[k], allowShortCache) {
				return true
			}
		}
	}
	return false
}

// Caller account selectors, credentials and experimental controls must not
// invalidate assurances about the key's configured account.
func copyDataPolicyHeaders(dst, src http.Header, p agentctx.DataPolicy) {
	if !dataPolicyConstrained(p) {
		copyRequestHeaders(dst, src)
		return
	}
	for _, key := range []string{"Accept", "Content-Type", "Anthropic-Version"} {
		for _, v := range src.Values(key) {
			dst.Add(key, v)
		}
	}
}

// enforceProviderDataPolicy acts on the final wire body, after translation,
// and is reapplied at every selected-key and fallback dispatch.
func enforceProviderDataPolicy(p agentctx.DataPolicy, prov *provider.Provider, upstream upstreamEncoding) (upstreamEncoding, error) {
	var body map[string]any
	if err := json.Unmarshal(upstream.Body, &body); err != nil || body == nil {
		return upstream, denyData("request")
	}
	official, region := officialDataEndpoint(prov.Name, prov.BaseURL)
	regionMode := "enforced"
	guarantees := attestedGuarantees(p, prov, official)
	model, _ := body["model"].(string)
	if p.Retention == "none" && upstream.Path != "/v1/chat/completions" && upstream.Path != "/v1/responses" && upstream.Path != "/v1/messages" {
		return upstream, denyData("retention_endpoint")
	}
	if p.Retention == "none" && hasRetentionIneligibleModel(body) {
		return upstream, denyData("retention_model")
	}
	if prov.Name == "vercel" && dataPolicyConstrained(p) {
		if raw, present := body["providerOptions"]; present {
			opts, ok := raw.(map[string]any)
			if !ok {
				return upstream, denyData("provider_preferences")
			}
			for name, raw := range opts {
				gateway, ok := raw.(map[string]any)
				if name != "gateway" || !ok {
					return upstream, denyData("provider_preferences")
				}
				for key := range gateway {
					if key != "zeroDataRetention" {
						return upstream, denyData("provider_preferences")
					}
				}
			}
		}
	}
	if dataPolicyConstrained(p) && hasUnscopedFeature(body) {
		return upstream, denyData("processor_feature")
	}
	if p.Retention == "none" && hasPersistentFeature(body, prov.Name == "anthropic" && upstream.Path == "/v1/messages") {
		return upstream, denyData("retention_feature")
	}
	if prov.Name == "openrouter" && !policyAllowsRegion(p, "global") && region != "" {
		if !guarantees["processing:"+region] {
			return upstream, denyData("processing_region")
		}
		regionMode = "attested"
	}
	if !policyAllowsRegion(p, "global") {
		if region != "" {
			if !policyAllowsRegion(p, region) {
				return upstream, denyData("processing_region")
			}
		} else if official && prov.Name == "anthropic" && upstream.Path == "/v1/messages" && anthropicGeoModel(model) && policyAllowsRegion(p, "us") {
			if geo, present := body["inference_geo"]; present && geo != "us" {
				return upstream, denyData("processing_region")
			}
			body["inference_geo"] = "us"
			region = "us"
		} else {
			regionMode = "attested"
			if geo, present := body["inference_geo"]; present {
				value, ok := geo.(string)
				if !ok || !policyAllowsRegion(p, value) || !guarantees["processing:"+value] {
					return upstream, denyData("processing_region")
				}
				region = value
			} else {
				for _, r := range p.Regions {
					if guarantees["processing:"+r] {
						region = r
						break
					}
				}
			}
			if region == "" {
				return upstream, denyData("processing_region")
			}
			// A request must not override an account/workspace region pin.
			if geo, present := body["inference_geo"]; present && geo != region {
				return upstream, denyData("processing_region")
			}
		}
	}
	trainingOK := p.Training == "allowed" || guarantees["no-training"]
	retentionOK := p.Retention == "provider-default" || guarantees["zero-data-retention"]
	trainingMode, retentionMode := "attested", "attested"
	if p.Training == "allowed" {
		trainingMode = "allowed"
	}
	if p.Retention == "provider-default" {
		retentionMode = "allowed"
	}
	if prov.Name == "openrouter" && (p.Training == "none" || p.Retention == "none") {
		prefs := map[string]any{}
		if raw, present := body["provider"]; present {
			var ok bool
			prefs, ok = raw.(map[string]any)
			if !ok {
				return upstream, denyData("provider_preferences")
			}
		}
		if p.Training == "none" {
			if v, ok := prefs["data_collection"]; ok && v != "deny" {
				return upstream, denyData("training")
			}
			prefs["data_collection"] = "deny"
			if official {
				trainingOK = true
				trainingMode = "enforced"
			}
		}
		if p.Retention == "none" {
			if v, ok := prefs["zdr"]; ok && v != true {
				return upstream, denyData("retention")
			}
			prefs["zdr"] = true
			if official {
				retentionOK = guarantees["zero-data-retention"]
				retentionMode = "attested"
			}
		}
		body["provider"] = prefs
	}
	if prov.Name == "vercel" && p.Retention == "none" {
		opts := map[string]any{}
		if raw, exists := body["providerOptions"]; exists {
			var ok bool
			opts, ok = raw.(map[string]any)
			if !ok {
				return upstream, denyData("provider_preferences")
			}
		}
		gateway := map[string]any{}
		if raw, exists := opts["gateway"]; exists {
			var ok bool
			gateway, ok = raw.(map[string]any)
			if !ok {
				return upstream, denyData("provider_preferences")
			}
		}
		if v, exists := gateway["zeroDataRetention"]; exists && v != true {
			return upstream, denyData("retention")
		}
		gateway["zeroDataRetention"] = true
		opts["gateway"] = gateway
		body["providerOptions"] = opts
		if official {
			retentionOK = guarantees["zero-data-retention"]
			retentionMode = "attested"
		}
	}
	if !trainingOK {
		return upstream, denyData("training")
	}
	if !retentionOK {
		return upstream, denyData("retention")
	}
	if p.Retention == "none" && (upstream.Path == "/v1/chat/completions" || upstream.Path == "/v1/responses") && (prov.Name == "openai" || prov.Name == "xai" || prov.Name == "openrouter" || prov.Name == "vercel") {
		body["store"] = false
	}
	var err error
	upstream.Body, err = json.Marshal(body)
	if err != nil {
		return upstream, denyData("request")
	}
	// Rechecks never accumulate duplicate evidence in a retry's encoding.
	mutations := make([]string, 0, len(upstream.Mutations)+3)
	for _, m := range upstream.Mutations {
		if !strings.HasPrefix(m, "data_policy_") {
			mutations = append(mutations, m)
		}
	}
	if policyAllowsRegion(p, "global") {
		regionMode = "allowed"
		region = "global"
	}
	if official && prov.Name == "openrouter" {
		if p.Training == "none" {
			mutations = append(mutations, "data_policy_control:openrouter:data_collection_deny")
		}
		if p.Retention == "none" {
			mutations = append(mutations, "data_policy_control:openrouter:zdr")
		}
		if !policyAllowsRegion(p, "global") && region != "" {
			mutations = append(mutations, "data_policy_control:openrouter:regional_"+region)
		}
	}
	if official && prov.Name == "vercel" && p.Retention == "none" {
		mutations = append(mutations, "data_policy_control:vercel:zero_data_retention")
	}
	mutations = append(mutations, "data_policy_processing:"+prov.Name+":"+regionMode+":"+region, "data_policy_training:"+prov.Name+":"+trainingMode, "data_policy_retention:"+prov.Name+":"+retentionMode)
	upstream.Mutations = mutations
	return upstream, nil
}

func (h *Handler) policyUpstream(ctx context.Context, agentID, model string, prov *provider.Provider, u upstreamEncoding) (upstreamEncoding, error) {
	out, err := enforceProviderDataPolicy(requestDataPolicy(ctx), prov, u)
	if err != nil {
		h.logger.LogIntervention(agentID, model, "data_policy_refused:"+err.(*dataPolicyError).dimension)
	} else {
		for _, m := range out.Mutations {
			if strings.HasPrefix(m, "data_policy_") {
				h.logger.LogIntervention(agentID, model, m)
			}
		}
	}
	return out, err
}

// Redirects are not an authorized fallback route: the target has not passed
// provider policy or endpoint-scoped credential/attestation checks.
func (h *Handler) doProviderHTTP(req *http.Request) (*http.Response, error) {
	client := *h.client
	client.CheckRedirect = func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }
	resp, err := client.Do(req)
	if err == nil && resp.StatusCode >= 300 && resp.StatusCode < 400 {
		resp.Body.Close()
		return nil, fmt.Errorf("provider redirect refused by data policy")
	}
	return resp, err
}
