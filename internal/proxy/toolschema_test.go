package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/mostlydev/cllama/internal/agentctx"
	"github.com/mostlydev/cllama/internal/logging"
)

func TestValidateManagedToolArgsNilSchemaPasses(t *testing.T) {
	if v := validateManagedToolArgs(nil, map[string]any{"anything": 1}); v != nil {
		t.Fatalf("nil schema should pass, got %+v", v)
	}
	if v := validateManagedToolArgs(map[string]any{}, map[string]any{"anything": 1}); v != nil {
		t.Fatalf("empty schema should pass, got %+v", v)
	}
}

func TestValidateManagedToolArgsValidPasses(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"symbol":       map[string]any{"type": "string"},
			"side":         map[string]any{"type": "string", "enum": []any{"buy", "sell"}},
			"quantity":     map[string]any{"type": "number"},
			"target_price": map[string]any{"type": "number"},
		},
		"required": []any{"symbol", "side", "quantity", "target_price"},
	}
	args := map[string]any{
		"symbol":       "CMCSA",
		"side":         "buy",
		"quantity":     float64(100),
		"target_price": 42.5,
	}
	if v := validateManagedToolArgs(schema, args); v != nil {
		t.Fatalf("valid args should pass, got %+v", v)
	}
}

func TestValidateManagedToolArgsMissingRequired(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"symbol":       map[string]any{"type": "string"},
			"target_price": map[string]any{"type": "number"},
		},
		"required": []any{"symbol", "target_price"},
	}
	args := map[string]any{"symbol": "CMCSA"}
	violations := validateManagedToolArgs(schema, args)
	if len(violations) != 1 {
		t.Fatalf("want 1 violation, got %+v", violations)
	}
	if violations[0].Code != "missing_required" || violations[0].Path != "target_price" {
		t.Fatalf("unexpected violation: %+v", violations[0])
	}
}

func TestValidateManagedToolArgsWrongLocationHint(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"target_price": map[string]any{"type": "number"},
		},
		"required": []any{"target_price"},
	}
	args := map[string]any{
		"proposal_context": map[string]any{"target_price": 42.5},
	}
	violations := validateManagedToolArgs(schema, args)
	if len(violations) != 1 || violations[0].Code != "missing_required" {
		t.Fatalf("want 1 missing_required violation, got %+v", violations)
	}
	if !strings.Contains(violations[0].Message, "proposal_context.target_price") {
		t.Fatalf("message should hint at wrong location, got %q", violations[0].Message)
	}
}

func TestValidateManagedToolArgsWrongType(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"quantity": map[string]any{"type": "number"},
		},
	}
	args := map[string]any{"quantity": "100"}
	violations := validateManagedToolArgs(schema, args)
	if len(violations) != 1 || violations[0].Code != "wrong_type" || violations[0].Path != "quantity" {
		t.Fatalf("want wrong_type at quantity, got %+v", violations)
	}
}

func TestValidateManagedToolArgsIntegerType(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"count": map[string]any{"type": "integer"},
		},
	}
	if v := validateManagedToolArgs(schema, map[string]any{"count": 1.5}); len(v) != 1 || v[0].Code != "wrong_type" {
		t.Fatalf("1.5 should fail integer, got %+v", v)
	}
	if v := validateManagedToolArgs(schema, map[string]any{"count": float64(3)}); v != nil {
		t.Fatalf("3.0 should pass integer, got %+v", v)
	}
}

func TestValidateManagedToolArgsEnum(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"side": map[string]any{"type": "string", "enum": []any{"buy", "sell"}},
		},
	}
	violations := validateManagedToolArgs(schema, map[string]any{"side": "hold"})
	if len(violations) != 1 || violations[0].Code != "invalid_enum" || violations[0].Path != "side" {
		t.Fatalf("want invalid_enum at side, got %+v", violations)
	}
	if !strings.Contains(violations[0].Message, "buy") || !strings.Contains(violations[0].Message, "sell") {
		t.Fatalf("enum message should list allowed values, got %q", violations[0].Message)
	}
}

func TestValidateManagedToolArgsNestedRequired(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"order": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"limit": map[string]any{"type": "number"},
				},
				"required": []any{"limit"},
			},
		},
		"required": []any{"order"},
	}
	args := map[string]any{"order": map[string]any{}}
	violations := validateManagedToolArgs(schema, args)
	if len(violations) != 1 || violations[0].Code != "missing_required" || violations[0].Path != "order.limit" {
		t.Fatalf("want missing_required at order.limit, got %+v", violations)
	}
}

func TestValidateManagedToolArgsArrayItems(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"tags": map[string]any{
				"type":  "array",
				"items": map[string]any{"type": "string"},
			},
		},
	}
	violations := validateManagedToolArgs(schema, map[string]any{"tags": []any{"ok", float64(7)}})
	if len(violations) != 1 || violations[0].Code != "wrong_type" || violations[0].Path != "tags[1]" {
		t.Fatalf("want wrong_type at tags[1], got %+v", violations)
	}
}

func TestValidateManagedToolArgsNullType(t *testing.T) {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"note": map[string]any{"type": "string"},
			"gap":  map[string]any{"type": "null"},
		},
	}
	if v := validateManagedToolArgs(schema, map[string]any{"note": nil}); len(v) != 1 || v[0].Code != "wrong_type" {
		t.Fatalf("null for string should fail, got %+v", v)
	}
	if v := validateManagedToolArgs(schema, map[string]any{"gap": nil}); v != nil {
		t.Fatalf("null for null type should pass, got %+v", v)
	}
}

func TestValidateManagedToolArgsFailOpenUnsupportedKeywords(t *testing.T) {
	// $ref / allOf / anyOf / oneOf / not at a level skip validation at that
	// level entirely: never block a call the provider might accept.
	schema := map[string]any{
		"$ref":     "#/definitions/trade",
		"required": []any{"target_price"},
	}
	if v := validateManagedToolArgs(schema, map[string]any{}); v != nil {
		t.Fatalf("$ref schema should fail open, got %+v", v)
	}
	nested := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"order": map[string]any{
				"allOf":    []any{map[string]any{"type": "object"}},
				"required": []any{"limit"},
			},
			"side": map[string]any{"type": "string", "enum": []any{"buy", "sell"}},
		},
	}
	violations := validateManagedToolArgs(nested, map[string]any{"order": map[string]any{}, "side": "hold"})
	if len(violations) != 1 || violations[0].Path != "side" {
		t.Fatalf("allOf subtree should fail open while siblings validate, got %+v", violations)
	}
}

func TestValidateManagedToolArgsUnsupportedTypeFormFailsOpen(t *testing.T) {
	// A type keyword we cannot interpret as a plain known string signals a
	// schema dialect we do not understand: skip the whole level, including
	// required, so validation never blocks a call the provider might accept.
	union := map[string]any{
		"type":     []any{"object", "null"},
		"required": []any{"symbol"},
		"properties": map[string]any{
			"symbol": map[string]any{"type": "string"},
		},
	}
	if v := validateManagedToolArgs(union, map[string]any{}); v != nil {
		t.Fatalf("union type form must fail open entirely, got %+v", v)
	}
	unknown := map[string]any{
		"type":     "file",
		"required": []any{"symbol"},
	}
	if v := validateManagedToolArgs(unknown, map[string]any{}); v != nil {
		t.Fatalf("unknown type string must fail open entirely, got %+v", v)
	}
	nestedUnknown := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"upload": map[string]any{"type": "file", "required": []any{"name"}},
			"side":   map[string]any{"type": "string", "enum": []any{"buy", "sell"}},
		},
	}
	violations := validateManagedToolArgs(nestedUnknown, map[string]any{"upload": map[string]any{}, "side": "hold"})
	if len(violations) != 1 || violations[0].Path != "side" {
		t.Fatalf("unknown nested type must fail open while siblings validate, got %+v", violations)
	}
}

func TestValidateManagedToolArgsExtraPropertiesIgnored(t *testing.T) {
	schema := map[string]any{
		"type":                 "object",
		"properties":           map[string]any{"symbol": map[string]any{"type": "string"}},
		"additionalProperties": false,
	}
	if v := validateManagedToolArgs(schema, map[string]any{"symbol": "X", "extra": 1}); v != nil {
		t.Fatalf("extra properties are not rejected in v1, got %+v", v)
	}
}

func schemaValidationAgentContext(serverURL string) *agentctx.AgentContext {
	return &agentctx.AgentContext{
		Tools: &agentctx.ToolManifest{
			Tools: []agentctx.ToolManifestEntry{{
				Name: "trading-api.propose_trade",
				InputSchema: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"symbol":       map[string]any{"type": "string"},
						"target_price": map[string]any{"type": "number"},
					},
					"required": []any{"symbol", "target_price"},
				},
				Execution: agentctx.ToolExecution{
					Transport: "http",
					Service:   "trading-api",
					BaseURL:   serverURL,
					Method:    http.MethodPost,
					Path:      "/propose_trade",
				},
			}},
		},
	}
}

func TestExecuteManagedOpenAIToolRejectsSchemaViolationsBeforeDispatch(t *testing.T) {
	var hit bool
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer toolSrv.Close()

	var logs bytes.Buffer
	h := NewHandler(nil, nil, logging.New(&logs))
	agentCtx := schemaValidationAgentContext(toolSrv.URL)
	args := map[string]any{"proposal_context": map[string]any{"target_price": 42.5}, "symbol": "CMCSA"}
	argsRaw, _ := json.Marshal(args)

	outcome, err := h.executeManagedOpenAITool(context.Background(), "agent-1", "test/model", agentCtx, openAIToolCall{
		Name:         "trading-api.propose_trade",
		Arguments:    args,
		ArgumentsRaw: argsRaw,
	}, managedToolPolicy{PerToolTimeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("executeManagedOpenAITool: %v", err)
	}
	if hit {
		t.Fatal("providing service must not be called for schema-invalid args")
	}
	raw := string(outcome.RawJSON)
	if !strings.Contains(raw, "schema_validation") {
		t.Fatalf("result should carry schema_validation code, got %s", raw)
	}
	if !strings.Contains(raw, "proposal_context.target_price") {
		t.Fatalf("result should carry the wrong-location hint, got %s", raw)
	}
	if !json.Valid(outcome.Trace.Result) {
		t.Fatalf("trace result must be valid JSON, got %s", outcome.Trace.Result)
	}
	entries := parseLogEntries(t, logs.Bytes())
	var sawIntervention bool
	for _, entry := range entries {
		if entry["type"] == "intervention" && entry["intervention"] == managedToolSchemaRejectedIntervention+":trading-api.propose_trade" {
			sawIntervention = true
		}
	}
	if !sawIntervention {
		t.Fatalf("expected %s intervention, got %+v", managedToolSchemaRejectedIntervention, entries)
	}
}

func TestExecuteManagedOpenAIToolSchemaValidationKillSwitch(t *testing.T) {
	t.Setenv(EnvToolSchemaValidation, "off")
	var hit bool
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer toolSrv.Close()

	h := NewHandler(nil, nil, logging.New(io.Discard))
	agentCtx := schemaValidationAgentContext(toolSrv.URL)
	args := map[string]any{"symbol": "CMCSA"}
	argsRaw, _ := json.Marshal(args)

	if _, err := h.executeManagedOpenAITool(context.Background(), "agent-1", "test/model", agentCtx, openAIToolCall{
		Name:         "trading-api.propose_trade",
		Arguments:    args,
		ArgumentsRaw: argsRaw,
	}, managedToolPolicy{PerToolTimeout: 5 * time.Second}); err != nil {
		t.Fatalf("executeManagedOpenAITool: %v", err)
	}
	if !hit {
		t.Fatal("kill switch off must dispatch schema-invalid args to the providing service")
	}
}

func TestExecuteManagedAnthropicToolRejectsSchemaViolations(t *testing.T) {
	var hit bool
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer toolSrv.Close()

	var logs bytes.Buffer
	h := NewHandler(nil, nil, logging.New(&logs))
	agentCtx := schemaValidationAgentContext(toolSrv.URL)
	args := map[string]any{"symbol": "CMCSA"}
	argsRaw, _ := json.Marshal(args)

	outcome, err := h.executeManagedAnthropicTool(context.Background(), "agent-1", "test/model", agentCtx, anthropicToolUse{
		Name:         "trading-api.propose_trade",
		Arguments:    args,
		ArgumentsRaw: argsRaw,
	}, managedToolPolicy{PerToolTimeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("executeManagedAnthropicTool: %v", err)
	}
	if hit {
		t.Fatal("providing service must not be called for schema-invalid args")
	}
	if !strings.Contains(string(outcome.RawJSON), "schema_validation") {
		t.Fatalf("result should carry schema_validation code, got %s", outcome.RawJSON)
	}
}

func TestExecuteManagedOpenAIToolSchemaValidCallDispatches(t *testing.T) {
	var hit bool
	toolSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hit = true
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"ok":true}`))
	}))
	defer toolSrv.Close()

	h := NewHandler(nil, nil, logging.New(io.Discard))
	agentCtx := schemaValidationAgentContext(toolSrv.URL)
	args := map[string]any{"symbol": "CMCSA", "target_price": 42.5}
	argsRaw, _ := json.Marshal(args)

	if _, err := h.executeManagedOpenAITool(context.Background(), "agent-1", "test/model", agentCtx, openAIToolCall{
		Name:         "trading-api.propose_trade",
		Arguments:    args,
		ArgumentsRaw: argsRaw,
	}, managedToolPolicy{PerToolTimeout: 5 * time.Second}); err != nil {
		t.Fatalf("executeManagedOpenAITool: %v", err)
	}
	if !hit {
		t.Fatal("schema-valid call must dispatch to the providing service")
	}
}
