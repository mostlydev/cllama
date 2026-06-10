package proxy

import (
	"fmt"
	"os"
	"reflect"
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

	if typeName, ok := schema["type"].(string); ok {
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
	default:
		// Unknown type keyword: fail open.
		return nil
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
