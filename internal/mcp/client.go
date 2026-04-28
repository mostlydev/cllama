package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
)

const DefaultProtocolVersion = "2025-11-25"

type Target struct {
	BaseURL string
	Path    string
	Auth    *Auth
}

type Auth struct {
	Type  string
	Token string
}

type Client struct {
	httpClient       *http.Client
	maxResponseBytes int

	mu       sync.Mutex
	sessions map[string]session
}

type session struct {
	ID              string
	ProtocolVersion string
}

type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

func (e *RPCError) Error() string {
	if e == nil {
		return ""
	}
	if strings.TrimSpace(e.Message) == "" {
		return fmt.Sprintf("mcp json-rpc error %d", e.Code)
	}
	return fmt.Sprintf("mcp json-rpc error %d: %s", e.Code, e.Message)
}

type HTTPStatusError struct {
	StatusCode int
	Body       []byte
}

func (e *HTTPStatusError) Error() string {
	if e == nil {
		return ""
	}
	return fmt.Sprintf("mcp endpoint returned HTTP %d", e.StatusCode)
}

type ResponseTooLargeError struct {
	Limit int
}

func (e *ResponseTooLargeError) Error() string {
	if e == nil {
		return ""
	}
	return fmt.Sprintf("mcp response exceeded %d bytes", e.Limit)
}

type rpcResponse struct {
	ID     json.RawMessage `json:"id,omitempty"`
	Result json.RawMessage `json:"result,omitempty"`
	Error  *RPCError       `json:"error,omitempty"`
}

func NewClient(httpClient *http.Client, maxResponseBytes int) *Client {
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	if maxResponseBytes <= 0 {
		maxResponseBytes = 16 * 1024
	}
	return &Client{
		httpClient:       httpClient,
		maxResponseBytes: maxResponseBytes,
		sessions:         make(map[string]session),
	}
}

func (c *Client) Call(ctx context.Context, target Target, toolName string, args map[string]any) (json.RawMessage, int, error) {
	toolName = strings.TrimSpace(toolName)
	if toolName == "" {
		return nil, 0, fmt.Errorf("mcp tool name is required")
	}
	if args == nil {
		args = map[string]any{}
	}
	key, err := targetKey(target)
	if err != nil {
		return nil, 0, err
	}
	sess, err := c.sessionFor(ctx, key, target)
	if err != nil {
		return nil, 0, err
	}

	result, status, err := c.callWithSession(ctx, target, sess, toolName, args)
	if shouldRetrySession(err, sess) {
		c.dropSession(key)
		sess, initErr := c.sessionFor(ctx, key, target)
		if initErr != nil {
			return nil, 0, initErr
		}
		result, status, err = c.callWithSession(ctx, target, sess, toolName, args)
	}
	return result, status, err
}

func (c *Client) callWithSession(ctx context.Context, target Target, sess session, toolName string, args map[string]any) (json.RawMessage, int, error) {
	params := map[string]any{
		"name":      toolName,
		"arguments": args,
	}
	resp, status, _, err := c.postRequest(ctx, target, sess, 2, "tools/call", params, toolName)
	if err != nil {
		return nil, status, err
	}
	if resp.Error != nil {
		return nil, status, resp.Error
	}
	if len(resp.Result) == 0 {
		return nil, status, fmt.Errorf("mcp tools/call response missing result")
	}
	return resp.Result, status, nil
}

func (c *Client) sessionFor(ctx context.Context, key string, target Target) (session, error) {
	c.mu.Lock()
	if sess, ok := c.sessions[key]; ok {
		c.mu.Unlock()
		return sess, nil
	}
	c.mu.Unlock()

	sess, err := c.initialize(ctx, target)
	if err != nil {
		return session{}, err
	}

	c.mu.Lock()
	c.sessions[key] = sess
	c.mu.Unlock()
	return sess, nil
}

func (c *Client) initialize(ctx context.Context, target Target) (session, error) {
	params := map[string]any{
		"protocolVersion": DefaultProtocolVersion,
		"capabilities":    map[string]any{},
		"clientInfo": map[string]any{
			"name":    "cllama",
			"version": "clawdapus",
		},
	}
	resp, _, sessionID, err := c.postRequest(ctx, target, session{}, 1, "initialize", params, "")
	if err != nil {
		return session{}, err
	}
	if resp.Error != nil {
		return session{}, resp.Error
	}

	protocolVersion := DefaultProtocolVersion
	if len(resp.Result) > 0 {
		var result struct {
			ProtocolVersion string `json:"protocolVersion"`
		}
		if err := json.Unmarshal(resp.Result, &result); err == nil && strings.TrimSpace(result.ProtocolVersion) != "" {
			protocolVersion = strings.TrimSpace(result.ProtocolVersion)
		}
	}
	sess := session{
		ID:              strings.TrimSpace(sessionID),
		ProtocolVersion: protocolVersion,
	}
	if err := c.postNotification(ctx, target, sess, "notifications/initialized", nil, ""); err != nil {
		return session{}, err
	}
	return sess, nil
}

func (c *Client) postRequest(ctx context.Context, target Target, sess session, id int, method string, params any, name string) (rpcResponse, int, string, error) {
	body := map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"method":  method,
	}
	if params != nil {
		body["params"] = params
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return rpcResponse{}, 0, "", err
	}
	resp, status, sessionID, err := c.post(ctx, target, sess, method, name, raw, id)
	if err != nil {
		return rpcResponse{}, status, sessionID, err
	}
	return resp, status, sessionID, nil
}

func (c *Client) postNotification(ctx context.Context, target Target, sess session, method string, params any, name string) error {
	body := map[string]any{
		"jsonrpc": "2.0",
		"method":  method,
	}
	if params != nil {
		body["params"] = params
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	_, _, _, err = c.post(ctx, target, sess, method, name, raw, 0)
	return err
}

func (c *Client) post(ctx context.Context, target Target, sess session, method string, name string, body []byte, expectID int) (rpcResponse, int, string, error) {
	endpoint, err := endpointURL(target)
	if err != nil {
		return rpcResponse{}, 0, "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return rpcResponse{}, 0, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json, text/event-stream")
	req.Header.Set("MCP-Method", method)
	if strings.TrimSpace(name) != "" {
		req.Header.Set("MCP-Name", name)
	}
	if sess.ID != "" {
		req.Header.Set("MCP-Session-Id", sess.ID)
	}
	if sess.ProtocolVersion != "" {
		req.Header.Set("MCP-Protocol-Version", sess.ProtocolVersion)
	}
	if err := applyAuth(req, target.Auth); err != nil {
		return rpcResponse{}, 0, "", err
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		if ctx.Err() != nil {
			return rpcResponse{}, 0, "", ctx.Err()
		}
		return rpcResponse{}, 0, "", err
	}
	defer resp.Body.Close()

	status := resp.StatusCode
	sessionID := resp.Header.Get("MCP-Session-Id")
	if expectID == 0 && status == http.StatusAccepted {
		return rpcResponse{}, status, sessionID, nil
	}

	limited, err := readLimited(resp.Body, c.maxResponseBytes)
	if err != nil {
		return rpcResponse{}, status, sessionID, err
	}
	if limited.truncated {
		return rpcResponse{}, status, sessionID, &ResponseTooLargeError{Limit: c.maxResponseBytes}
	}
	if status < 200 || status >= 300 {
		return rpcResponse{}, status, sessionID, &HTTPStatusError{StatusCode: status, Body: limited.body}
	}
	if expectID == 0 && len(bytes.TrimSpace(limited.body)) == 0 {
		return rpcResponse{}, status, sessionID, nil
	}

	var rpc rpcResponse
	contentType := strings.ToLower(resp.Header.Get("Content-Type"))
	if strings.Contains(contentType, "text/event-stream") {
		rpc, err = parseSSEResponse(limited.body, expectID)
	} else {
		rpc, err = parseJSONRPCResponse(limited.body, expectID)
	}
	if err != nil {
		return rpcResponse{}, status, sessionID, err
	}
	return rpc, status, sessionID, nil
}

func shouldRetrySession(err error, sess session) bool {
	if err == nil || sess.ID == "" {
		return false
	}
	if httpErr, ok := err.(*HTTPStatusError); ok {
		return httpErr.StatusCode == http.StatusBadRequest || httpErr.StatusCode == http.StatusNotFound
	}
	return false
}

func (c *Client) dropSession(key string) {
	c.mu.Lock()
	delete(c.sessions, key)
	c.mu.Unlock()
}

func endpointURL(target Target) (string, error) {
	base := strings.TrimSpace(target.BaseURL)
	if base == "" {
		return "", fmt.Errorf("mcp base_url is required")
	}
	u, err := url.Parse(base)
	if err != nil {
		return "", err
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("invalid mcp base_url %q", target.BaseURL)
	}
	path := strings.TrimSpace(target.Path)
	if path == "" {
		path = "/mcp"
	}
	if !strings.HasPrefix(path, "/") {
		return "", fmt.Errorf("mcp path %q must start with '/'", path)
	}
	u.Path = strings.TrimRight(u.Path, "/") + path
	return u.String(), nil
}

func targetKey(target Target) (string, error) {
	u, err := endpointURL(target)
	if err != nil {
		return "", err
	}
	return u, nil
}

func applyAuth(req *http.Request, auth *Auth) error {
	if auth == nil {
		return nil
	}
	switch strings.ToLower(strings.TrimSpace(auth.Type)) {
	case "", "none":
		return nil
	case "bearer":
		token := strings.TrimSpace(auth.Token)
		if token == "" {
			return fmt.Errorf("mcp bearer token is empty")
		}
		req.Header.Set("Authorization", "Bearer "+token)
		return nil
	default:
		return fmt.Errorf("unsupported mcp auth type %q", auth.Type)
	}
}

func parseJSONRPCResponse(body []byte, expectID int) (rpcResponse, error) {
	var resp rpcResponse
	if err := json.Unmarshal(bytes.TrimSpace(body), &resp); err != nil {
		return rpcResponse{}, err
	}
	if expectID > 0 && !jsonIDMatches(resp.ID, expectID) {
		return rpcResponse{}, fmt.Errorf("mcp response id does not match request id %d", expectID)
	}
	return resp, nil
}

func parseSSEResponse(body []byte, expectID int) (rpcResponse, error) {
	scanner := bufio.NewScanner(bytes.NewReader(body))
	scanner.Buffer(make([]byte, 0, 64*1024), len(body)+1024)
	var dataLines []string
	flush := func() (rpcResponse, bool, error) {
		if len(dataLines) == 0 {
			return rpcResponse{}, false, nil
		}
		payload := strings.TrimSpace(strings.Join(dataLines, "\n"))
		dataLines = nil
		if payload == "" || payload == "[DONE]" {
			return rpcResponse{}, false, nil
		}
		var resp rpcResponse
		if err := json.Unmarshal([]byte(payload), &resp); err != nil {
			return rpcResponse{}, false, nil
		}
		if expectID == 0 || jsonIDMatches(resp.ID, expectID) || resp.Error != nil {
			return resp, true, nil
		}
		return rpcResponse{}, false, nil
	}
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if resp, ok, err := flush(); ok || err != nil {
				return resp, err
			}
			continue
		}
		if strings.HasPrefix(line, "data:") {
			dataLines = append(dataLines, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		}
	}
	if err := scanner.Err(); err != nil {
		return rpcResponse{}, err
	}
	if resp, ok, err := flush(); ok || err != nil {
		return resp, err
	}
	return rpcResponse{}, fmt.Errorf("mcp SSE response did not include JSON-RPC response id %d", expectID)
}

func jsonIDMatches(raw json.RawMessage, expected int) bool {
	raw = bytes.TrimSpace(raw)
	if len(raw) == 0 {
		return expected == 0
	}
	var n int
	if err := json.Unmarshal(raw, &n); err == nil {
		return n == expected
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s == fmt.Sprint(expected)
	}
	return false
}

type limitedBody struct {
	body      []byte
	truncated bool
}

func readLimited(r io.Reader, limit int) (limitedBody, error) {
	data, err := io.ReadAll(io.LimitReader(r, int64(limit)+1))
	if err != nil {
		return limitedBody{}, err
	}
	if len(data) > limit {
		return limitedBody{body: data[:limit], truncated: true}, nil
	}
	return limitedBody{body: data}, nil
}
