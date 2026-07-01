package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
)

const (
	EnvPolicyURL       = "CLLAMA_POLICY_URL"
	EnvPolicyToken     = "CLLAMA_POLICY_TOKEN"
	EnvPolicyTimeoutMS = "CLLAMA_POLICY_TIMEOUT_MS"
	EnvPolicyFailMode  = "CLLAMA_POLICY_FAIL_MODE"

	defaultPolicyTimeoutMS = 1000

	policyFailModeClosed = "closed"
	policyFailModeOpen   = "open"

	policyVerdictAllow = "allow"
	policyVerdictDeny  = "deny"
	policyVerdictAmend = "amend"

	policyToolFilterAllowList = "allow_list"
	policyToolFilterDenyList  = "deny_list"

	policyInterventionDenied      = "policy_denied"
	policyInterventionUnavailable = "policy_unavailable"
	policyInterventionDecorated   = "policy_decorated"
	policyInterventionAmended     = "policy_amended"
)

type PolicyEvaluator interface {
	GateRequest(context.Context, PolicyGateRequest) (*PolicyGateRequestResult, error)
	Decorate(context.Context, PolicyDecorateRequest) (*PolicyDecorateResult, error)
	GateResponse(context.Context, PolicyGateResponseRequest) (*PolicyGateResponseResult, error)
	Score(context.Context, PolicyScoreRequest) error
}

type PolicyRulesRef struct {
	Path   string `json:"path,omitempty"`
	Digest string `json:"digest,omitempty"`
}

type PolicyRequestMeta struct {
	AgentID string         `json:"agent_id"`
	Format  string         `json:"format"`
	Mode    string         `json:"mode"`
	Stream  bool           `json:"stream"`
	Model   string         `json:"model,omitempty"`
	Rules   PolicyRulesRef `json:"rules,omitempty"`
}

type PolicyGateRequest struct {
	PolicyRequestMeta
	Request map[string]any `json:"request,omitempty"`
}

type PolicyToolFilter struct {
	Mode  string   `json:"mode"`
	Tools []string `json:"tools"`
}

type PolicyGateRequestResult struct {
	Verdict      string            `json:"verdict,omitempty"`
	Reason       string            `json:"reason,omitempty"`
	Intervention string            `json:"intervention,omitempty"`
	ToolFilter   *PolicyToolFilter `json:"tool_filter,omitempty"`
}

type PolicyDecorateRequest struct {
	PolicyRequestMeta
	Request map[string]any `json:"request,omitempty"`
}

type PolicyDecorateResult struct {
	Intervention  string           `json:"intervention,omitempty"`
	MessagesPatch []map[string]any `json:"messages_patch,omitempty"`
	SystemPatch   string           `json:"system_patch,omitempty"`
}

type PolicyGateResponseRequest struct {
	PolicyRequestMeta
	RequestBody    string      `json:"request_body,omitempty"`
	ResponseStatus int         `json:"response_status"`
	ResponseHeader http.Header `json:"response_header,omitempty"`
	ResponseBody   string      `json:"response_body,omitempty"`
}

type PolicyGateResponseResult struct {
	Verdict      string          `json:"verdict,omitempty"`
	Reason       string          `json:"reason,omitempty"`
	Intervention string          `json:"intervention,omitempty"`
	AmendedBody  json.RawMessage `json:"amended_body,omitempty"`
}

type PolicyScoreRequest struct {
	PolicyRequestMeta
	RequestBody    string      `json:"request_body,omitempty"`
	ResponseStatus int         `json:"response_status"`
	ResponseHeader http.Header `json:"response_header,omitempty"`
	ResponseBody   string      `json:"response_body,omitempty"`
}

type httpPolicyEvaluator struct {
	baseURL string
	token   string
	timeout time.Duration
	client  *http.Client
}

func newHTTPPolicyEvaluatorFromEnv(client *http.Client) PolicyEvaluator {
	rawURL := strings.TrimSpace(os.Getenv(EnvPolicyURL))
	if rawURL == "" {
		return nil
	}
	parsed, err := url.Parse(rawURL)
	if err != nil || parsed.Scheme == "" || parsed.Host == "" {
		if err == nil {
			err = fmt.Errorf("policy URL must include scheme and host")
		}
		return policyErrorEvaluator{err: fmt.Errorf("%s: %w", EnvPolicyURL, err)}
	}
	if client == nil {
		client = &http.Client{}
	}
	return &httpPolicyEvaluator{
		baseURL: strings.TrimRight(rawURL, "/"),
		token:   strings.TrimSpace(os.Getenv(EnvPolicyToken)),
		timeout: policyTimeoutFromEnv(),
		client:  client,
	}
}

func policyTimeoutFromEnv() time.Duration {
	raw := strings.TrimSpace(os.Getenv(EnvPolicyTimeoutMS))
	if raw == "" {
		return time.Duration(defaultPolicyTimeoutMS) * time.Millisecond
	}
	ms, err := strconv.Atoi(raw)
	if err != nil || ms <= 0 {
		return time.Duration(defaultPolicyTimeoutMS) * time.Millisecond
	}
	return time.Duration(ms) * time.Millisecond
}

func policyFailModeFromEnv() string {
	switch strings.ToLower(strings.TrimSpace(os.Getenv(EnvPolicyFailMode))) {
	case policyFailModeOpen:
		return policyFailModeOpen
	default:
		return policyFailModeClosed
	}
}

func (e *httpPolicyEvaluator) GateRequest(ctx context.Context, req PolicyGateRequest) (*PolicyGateRequestResult, error) {
	var out PolicyGateRequestResult
	if err := e.post(ctx, "/policy/gate-request", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (e *httpPolicyEvaluator) Decorate(ctx context.Context, req PolicyDecorateRequest) (*PolicyDecorateResult, error) {
	var out PolicyDecorateResult
	if err := e.post(ctx, "/policy/decorate", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (e *httpPolicyEvaluator) GateResponse(ctx context.Context, req PolicyGateResponseRequest) (*PolicyGateResponseResult, error) {
	var out PolicyGateResponseResult
	if err := e.post(ctx, "/policy/gate-response", req, &out); err != nil {
		return nil, err
	}
	return &out, nil
}

func (e *httpPolicyEvaluator) Score(ctx context.Context, req PolicyScoreRequest) error {
	return e.post(ctx, "/policy/score", req, nil)
}

func (e *httpPolicyEvaluator) post(ctx context.Context, path string, in any, out any) error {
	if e == nil {
		return fmt.Errorf("policy evaluator not configured")
	}
	if e.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, e.timeout)
		defer cancel()
	}
	raw, err := json.Marshal(in)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+path, bytes.NewReader(raw))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	if e.token != "" {
		req.Header.Set("Authorization", "Bearer "+e.token)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("policy sidecar %s returned HTTP %d", path, resp.StatusCode)
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

type policyErrorEvaluator struct {
	err error
}

func (e policyErrorEvaluator) GateRequest(context.Context, PolicyGateRequest) (*PolicyGateRequestResult, error) {
	return nil, e.err
}

func (e policyErrorEvaluator) Decorate(context.Context, PolicyDecorateRequest) (*PolicyDecorateResult, error) {
	return nil, e.err
}

func (e policyErrorEvaluator) GateResponse(context.Context, PolicyGateResponseRequest) (*PolicyGateResponseResult, error) {
	return nil, e.err
}

func (e policyErrorEvaluator) Score(context.Context, PolicyScoreRequest) error {
	return e.err
}

func policyMode(agentCtx *agentctx.AgentContext) string {
	if hasManagedTools(agentCtx) {
		return "managed"
	}
	return "plain"
}

func policyRulesRef(agentCtx *agentctx.AgentContext) PolicyRulesRef {
	if agentCtx == nil {
		return PolicyRulesRef{}
	}
	return PolicyRulesRef{
		Path:   agentCtx.RulesPath(),
		Digest: agentCtx.RulesDigest(),
	}
}

func policyBodyFromRaw(raw json.RawMessage) []byte {
	if len(raw) == 0 {
		return nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []byte(s)
	}
	return append([]byte(nil), raw...)
}
