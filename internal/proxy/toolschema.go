package proxy

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"sort"
	"strings"
)

// EnvToolSchemaValidation disables pre-dispatch managed tool argument
// validation when set to "off". Validation is on by default; the switch is
// the emergency rollback path that does not require an image rollback.
const EnvToolSchemaValidation = "CLLAMA_TOOL_SCHEMA_VALIDATION"

const maxSchemaViolations = 8

type schemaViolation struct {
	Path    string `json:"path"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

func toolSchemaValidationFromEnv() bool {
	return !strings.EqualFold(strings.TrimSpace(os.Getenv(EnvToolSchemaValidation)), "off")
}

// validateManagedToolArgs checks model-emitted tool arguments against the
// manifest entry's inputSchema before the providing-service dispatch. It
// implements a deliberate JSON Schema subset: required, properties
// (recursive), primitive type, enum, and items. Anything it does not
// understand fails open — validation must never block a call the provider
// would have accepted. The provider remains the authority; this exists to
// turn guess-and-retry loops into one precise in-round correction.
func validateManagedToolArgs(schema map[string]any, args map[string]any) []schemaViolation {
	if len(schema) == 0 {
		return nil
	}
	violations := validateSchemaValue(schema, args, "", args)
	if len(violations) == 0 {
		return nil
	}
	if len(violations) > maxSchemaViolations {
		violations = violations[:maxSchemaViolations]
	}
	return violations
}

// unsupported schema combinators: their presence at a level skips validation
// of that entire level (conservative fail-open).
var unsupportedSchemaKeywords = []string{"$ref", "allOf", "anyOf", "oneOf", "not"}

func validateSchemaValue(schema map[string]any, value any, path string, rootArgs map[string]any) []schemaViolation {
	if len(schema) == 0 {
		return nil
	}
	for _, kw := range unsupportedSchemaKeywords {
		if _, present := schema[kw]; present {
			return nil
		}
	}

	var violations []schemaViolation

	if rawType, present := schema["type"]; present {
		typeName, isString := rawType.(string)
		if !isString || !isKnownSchemaType(typeName) {
			// A type keyword we cannot interpret as a plain known string
			// signals a schema dialect we do not understand. Skip the whole
			// level, required included: fail open.
			return nil
		}
		if violation := checkSchemaType(typeName, value, path); violation != nil {
			// A type mismatch makes deeper keyword checks meaningless noise.
			return []schemaViolation{*violation}
		}
	}

	if enumValues, ok := schema["enum"].([]any); ok && len(enumValues) > 0 {
		if violation := checkSchemaEnum(enumValues, value, path); violation != nil {
			violations = append(violations, *violation)
		}
	}

	if obj, ok := value.(map[string]any); ok {
		if required, reqOK := schema["required"].([]any); reqOK {
			for _, entry := range required {
				name, nameOK := entry.(string)
				if !nameOK || name == "" {
					continue
				}
				if _, present := obj[name]; present {
					continue
				}
				violations = append(violations, missingRequiredViolation(name, path, rootArgs))
			}
		}
		if properties, propsOK := schema["properties"].(map[string]any); propsOK {
			for name, propSchema := range properties {
				child, present := obj[name]
				if !present {
					continue
				}
				childSchema, schemaOK := propSchema.(map[string]any)
				if !schemaOK {
					continue
				}
				violations = append(violations, validateSchemaValue(childSchema, child, joinSchemaPath(path, name), rootArgs)...)
			}
		}
	}

	if arr, ok := value.([]any); ok {
		if itemSchema, itemsOK := schema["items"].(map[string]any); itemsOK {
			for i, item := range arr {
				violations = append(violations, validateSchemaValue(itemSchema, item, fmt.Sprintf("%s[%d]", path, i), rootArgs)...)
			}
		}
	}

	return violations
}

func isKnownSchemaType(typeName string) bool {
	switch typeName {
	case "object", "array", "string", "boolean", "number", "integer", "null":
		return true
	}
	return false
}

func checkSchemaType(typeName string, value any, path string) *schemaViolation {
	matched := true
	switch typeName {
	case "object":
		_, matched = value.(map[string]any)
	case "array":
		_, matched = value.([]any)
	case "string":
		_, matched = value.(string)
	case "boolean":
		_, matched = value.(bool)
	case "number":
		matched = isJSONNumber(value)
	case "integer":
		matched = isJSONInteger(value)
	case "null":
		matched = value == nil
	}
	if matched {
		return nil
	}
	return &schemaViolation{
		Path:    path,
		Code:    "wrong_type",
		Message: fmt.Sprintf("property %q must be of type %s, got %s", displaySchemaPath(path), typeName, jsonTypeName(value)),
	}
}

func checkSchemaEnum(enumValues []any, value any, path string) *schemaViolation {
	for _, allowed := range enumValues {
		if reflect.DeepEqual(value, allowed) {
			return nil
		}
	}
	allowed := make([]string, 0, len(enumValues))
	for _, entry := range enumValues {
		allowed = append(allowed, fmt.Sprintf("%v", entry))
	}
	return &schemaViolation{
		Path:    path,
		Code:    "invalid_enum",
		Message: fmt.Sprintf("property %q must be one of [%s], got %v", displaySchemaPath(path), strings.Join(allowed, ", "), value),
	}
}

func missingRequiredViolation(name, path string, rootArgs map[string]any) schemaViolation {
	violationPath := joinSchemaPath(path, name)
	where := "at top level"
	if path != "" {
		where = fmt.Sprintf("at %q", path)
	}
	message := fmt.Sprintf("missing required property %q %s", name, where)
	if foundAt := findKeyPath(rootArgs, name, ""); foundAt != "" && foundAt != violationPath {
		message += fmt.Sprintf("; found at %q — move it to %q", foundAt, where)
		if path == "" {
			message = fmt.Sprintf("missing required property %q at top level; found at %q — move it to the top level", name, foundAt)
		}
	}
	return schemaViolation{Path: violationPath, Code: "missing_required", Message: message}
}

// findKeyPath locates the first occurrence of key anywhere in args other than
// the expected location, powering the wrong-nesting hint: models frequently
// place a required field one level too deep, and naming the actual location
// turns multi-round guessing into a single correction.
func findKeyPath(value any, key, path string) string {
	obj, ok := value.(map[string]any)
	if !ok {
		return ""
	}
	for name, child := range obj {
		childPath := joinSchemaPath(path, name)
		if name == key && path != "" {
			return childPath
		}
		if found := findKeyPath(child, key, childPath); found != "" {
			return found
		}
	}
	return ""
}

func joinSchemaPath(path, name string) string {
	if path == "" {
		return name
	}
	return path + "." + name
}

func displaySchemaPath(path string) string {
	if path == "" {
		return "(root)"
	}
	return path
}

func isJSONNumber(value any) bool {
	switch value.(type) {
	case float64, float32, int, int32, int64, uint, uint32, uint64:
		return true
	}
	return false
}

func isJSONInteger(value any) bool {
	switch typed := value.(type) {
	case int, int32, int64, uint, uint32, uint64:
		return true
	case float64:
		return typed == float64(int64(typed))
	case float32:
		return typed == float32(int32(typed))
	}
	return false
}

func jsonTypeName(value any) string {
	switch value.(type) {
	case nil:
		return "null"
	case map[string]any:
		return "object"
	case []any:
		return "array"
	case string:
		return "string"
	case bool:
		return "boolean"
	case float64, float32, int, int32, int64, uint, uint32, uint64:
		return "number"
	}
	return fmt.Sprintf("%T", value)
}

// EnvToolArgPruneSentinels selects which models get sentinel-argument pruning,
// as a comma-separated list of "<provider>/<model-prefix>" entries — the same
// shape as EnvResponsesAPIModels. Empty means the feature is off.
//
// It is a model selector rather than a global switch on purpose. Pruning
// discards data the model emitted, and whether that is correct depends
// entirely on which model produced it: fabricating minima is a trait of
// specific model families, and a pod routinely runs several models at once.
// A global flag would apply a judgement made about one model to every other
// model sharing the proxy.
const EnvToolArgPruneSentinels = "CLLAMA_TOOL_ARG_PRUNE_SENTINELS"

// toolArgPruneSentinelsFor reports whether pruning is selected for a declared
// model reference such as "vercel/openai/gpt-5.6-luna". The reference is split
// on its first separator into provider and upstream model, matching how
// EnvResponsesAPIModels entries are matched.
func toolArgPruneSentinelsFor(declaredModel string) bool {
	provider, model, ok := strings.Cut(strings.TrimSpace(declaredModel), "/")
	if !ok || provider == "" || model == "" {
		return false
	}
	provider = strings.ToLower(provider)
	for _, entry := range strings.Split(os.Getenv(EnvToolArgPruneSentinels), ",") {
		entryProvider, entryPrefix, ok := strings.Cut(strings.TrimSpace(entry), "/")
		if !ok || entryPrefix == "" {
			continue
		}
		if strings.EqualFold(strings.TrimSpace(entryProvider), provider) &&
			strings.HasPrefix(model, strings.TrimSpace(entryPrefix)) {
			return true
		}
	}
	return false
}

// pruneSentinelOptionalArgs removes top-level optional arguments whose value is
// exactly the `minimum` their own schema declares, returning the names dropped.
//
// Some model families populate every optional parameter rather than omitting
// the ones they were not asked for, and reach for the schema's lower bound as
// the filler. On a trading API that turns a valid order into a rejected one:
// an otherwise correct LIMIT proposal arrives carrying amount_requested 0.01
// and stop_price 0.0001, and the service refuses it.
//
// Scope is deliberately narrow. Only top-level properties, only ones absent
// from `required`, only numbers, and only an exact match against `minimum` —
// a value the caller had to go out of its way to choose, since it is the least
// useful legal value of a field it was not asked to set. Anything else is
// left alone and remains the service's business to accept or reject.
//
// This is lossy by construction and cannot distinguish fabrication from a
// caller that genuinely meant the minimum, which is why it is opt-in.
func pruneSentinelOptionalArgs(schema map[string]any, args map[string]any) []string {
	if len(schema) == 0 || len(args) == 0 {
		return nil
	}
	properties, ok := schema["properties"].(map[string]any)
	if !ok {
		return nil
	}
	required := map[string]bool{}
	if list, ok := schema["required"].([]any); ok {
		for _, name := range list {
			if s, ok := name.(string); ok {
				required[s] = true
			}
		}
	}

	var pruned []string
	for name, value := range args {
		if required[name] {
			continue
		}
		property, ok := properties[name].(map[string]any)
		if !ok {
			continue
		}
		minimum, ok := toFloat(property["minimum"])
		if !ok {
			continue
		}
		actual, ok := toFloat(value)
		if !ok || actual != minimum {
			continue
		}
		delete(args, name)
		pruned = append(pruned, name)
	}
	sort.Strings(pruned)
	return pruned
}

func toFloat(value any) (float64, bool) {
	switch v := value.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	case json.Number:
		f, err := v.Float64()
		return f, err == nil
	default:
		return 0, false
	}
}
