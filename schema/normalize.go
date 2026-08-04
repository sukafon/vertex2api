package schema

import (
	"fmt"
	"strconv"
	"strings"
)

// Normalize returns a JSON-Schema-compatible copy suitable for Gemini's
// parametersJsonSchema/responseJsonSchema fields. It resolves local references,
// flattens allOf, preserves unions as anyOf, and translates nullable type arrays
// without mutating the caller's request.
func Normalize(value interface{}) interface{} {
	root, ok := value.(map[string]interface{})
	if !ok {
		return normalizeValue(value, nil, map[string]bool{})
	}
	definitions := collectDefinitions(root)
	return normalizeValue(root, definitions, map[string]bool{})
}

func collectDefinitions(root map[string]interface{}) map[string]interface{} {
	definitions := make(map[string]interface{})
	for _, key := range []string{"$defs", "definitions"} {
		if defs, ok := root[key].(map[string]interface{}); ok {
			for name, definition := range defs {
				definitions[name] = definition
			}
		}
	}
	return definitions
}

func normalizeValue(value interface{}, definitions map[string]interface{}, resolving map[string]bool) interface{} {
	switch typed := value.(type) {
	case map[string]interface{}:
		return normalizeMap(typed, definitions, resolving)
	case []interface{}:
		result := make([]interface{}, 0, len(typed))
		for _, item := range typed {
			result = append(result, normalizeValue(item, definitions, resolving))
		}
		return result
	case []map[string]interface{}:
		result := make([]interface{}, 0, len(typed))
		for _, item := range typed {
			result = append(result, normalizeMap(item, definitions, resolving))
		}
		return result
	default:
		return value
	}
}

func normalizeMap(input map[string]interface{}, definitions map[string]interface{}, resolving map[string]bool) map[string]interface{} {
	working := copyMap(input)
	if ref, _ := working["$ref"].(string); ref != "" {
		delete(working, "$ref")
		if resolved := resolveLocalReference(ref, definitions, resolving); resolved != nil {
			working = mergeSchemas(resolved, working)
		} else {
			appendDescription(working, fmt.Sprintf("Unresolved schema reference: %s", ref))
		}
	}

	if allOf, ok := toSlice(working["allOf"]); ok {
		delete(working, "allOf")
		merged := make(map[string]interface{})
		for _, branch := range allOf {
			if normalized, ok := normalizeValue(branch, definitions, resolving).(map[string]interface{}); ok {
				merged = mergeSchemas(merged, normalized)
			}
		}
		working = mergeSchemas(merged, working)
	}

	for _, key := range []string{"oneOf", "anyOf"} {
		branches, ok := toSlice(working[key])
		if !ok {
			continue
		}
		delete(working, key)
		normalized, nullable := normalizeUnion(branches, definitions, resolving)
		if len(normalized) == 1 {
			if branch, ok := normalized[0].(map[string]interface{}); ok {
				working = mergeSchemas(branch, working)
			}
		} else if len(normalized) > 1 {
			working["anyOf"] = normalized
		}
		if nullable {
			working["nullable"] = true
		}
	}

	if types, ok := toSlice(working["type"]); ok {
		delete(working, "type")
		branches := make([]interface{}, 0, len(types))
		nullable := false
		for _, item := range types {
			typeName, _ := item.(string)
			if strings.EqualFold(typeName, "null") {
				nullable = true
				continue
			}
			if typeName != "" {
				branches = append(branches, map[string]interface{}{"type": strings.ToLower(typeName)})
			}
		}
		if len(branches) == 1 {
			working["type"] = branches[0].(map[string]interface{})["type"]
		} else if len(branches) > 1 {
			working["anyOf"] = branches
		}
		if nullable {
			working["nullable"] = true
		}
	}

	if constant, ok := working["const"]; ok {
		delete(working, "const")
		if _, exists := working["enum"]; !exists {
			working["enum"] = []interface{}{constant}
		}
	}

	for _, key := range []string{"$schema", "$id", "$anchor", "$comment", "$defs", "definitions"} {
		delete(working, key)
	}

	result := make(map[string]interface{}, len(working))
	for key, item := range working {
		switch key {
		case "properties":
			if properties, ok := item.(map[string]interface{}); ok {
				normalized := make(map[string]interface{}, len(properties))
				for name, property := range properties {
					normalized[name] = normalizeValue(property, definitions, resolving)
				}
				result[key] = normalized
				continue
			}
		case "items", "additionalProperties", "not", "contains", "propertyNames":
			result[key] = normalizeValue(item, definitions, resolving)
			continue
		case "anyOf":
			if branches, ok := toSlice(item); ok {
				normalized, nullable := normalizeUnion(branches, definitions, resolving)
				if len(normalized) > 0 {
					result[key] = normalized
				}
				if nullable {
					result["nullable"] = true
				}
				continue
			}
		}
		result[key] = normalizeValue(item, definitions, resolving)
	}

	if properties, ok := result["properties"].(map[string]interface{}); ok {
		if _, exists := result["type"]; !exists {
			result["type"] = "object"
		}
		if required, ok := toSlice(result["required"]); ok {
			filtered := make([]interface{}, 0, len(required))
			seen := make(map[string]bool)
			for _, item := range required {
				name, _ := item.(string)
				if name != "" && !seen[name] {
					if _, exists := properties[name]; exists {
						filtered = append(filtered, name)
						seen[name] = true
					}
				}
			}
			if len(filtered) > 0 {
				result["required"] = filtered
			} else {
				delete(result, "required")
			}
		}
	}
	return result
}

func normalizeUnion(branches []interface{}, definitions map[string]interface{}, resolving map[string]bool) ([]interface{}, bool) {
	result := make([]interface{}, 0, len(branches))
	nullable := false
	for _, branch := range branches {
		if schema, ok := branch.(map[string]interface{}); ok {
			if typeName, _ := schema["type"].(string); strings.EqualFold(typeName, "null") {
				nullable = true
				continue
			}
		}
		result = append(result, normalizeValue(branch, definitions, resolving))
	}
	return result, nullable
}

func resolveLocalReference(ref string, definitions map[string]interface{}, resolving map[string]bool) map[string]interface{} {
	const defsPrefix = "#/$defs/"
	const definitionsPrefix = "#/definitions/"
	var name string
	switch {
	case strings.HasPrefix(ref, defsPrefix):
		name = strings.TrimPrefix(ref, defsPrefix)
	case strings.HasPrefix(ref, definitionsPrefix):
		name = strings.TrimPrefix(ref, definitionsPrefix)
	default:
		return nil
	}
	segments := strings.Split(name, "/")
	for i := range segments {
		segments[i] = strings.ReplaceAll(strings.ReplaceAll(segments[i], "~1", "/"), "~0", "~")
	}
	definition, ok := definitions[segments[0]]
	if !ok || resolving[ref] {
		return nil
	}
	var value interface{} = definition
	for _, segment := range segments[1:] {
		switch typed := value.(type) {
		case map[string]interface{}:
			value, ok = typed[segment]
		case []interface{}:
			index, err := strconv.Atoi(segment)
			if err != nil || index < 0 || index >= len(typed) {
				ok = false
			} else {
				value = typed[index]
			}
		default:
			ok = false
		}
		if !ok {
			return nil
		}
	}
	resolving[ref] = true
	defer delete(resolving, ref)
	resolved, _ := normalizeValue(value, definitions, resolving).(map[string]interface{})
	return resolved
}

func mergeSchemas(base, overlay map[string]interface{}) map[string]interface{} {
	result := copyMap(base)
	for key, value := range overlay {
		switch key {
		case "properties":
			baseProperties, _ := result[key].(map[string]interface{})
			overlayProperties, ok := value.(map[string]interface{})
			if ok {
				merged := copyMap(baseProperties)
				for name, property := range overlayProperties {
					merged[name] = property
				}
				result[key] = merged
				continue
			}
		case "required":
			result[key] = mergeRequired(result[key], value)
			continue
		case "description":
			baseDescription, _ := result[key].(string)
			overlayDescription, _ := value.(string)
			if baseDescription != "" && overlayDescription != "" && baseDescription != overlayDescription {
				result[key] = baseDescription + "\n" + overlayDescription
				continue
			}
		}
		result[key] = value
	}
	return result
}

func mergeRequired(left, right interface{}) []interface{} {
	result := make([]interface{}, 0)
	seen := make(map[string]bool)
	for _, value := range []interface{}{left, right} {
		items, _ := toSlice(value)
		for _, item := range items {
			name, _ := item.(string)
			if name != "" && !seen[name] {
				result = append(result, name)
				seen[name] = true
			}
		}
	}
	return result
}

func appendDescription(schema map[string]interface{}, note string) {
	if existing, _ := schema["description"].(string); existing != "" {
		schema["description"] = existing + "\n" + note
		return
	}
	schema["description"] = note
}

func toSlice(value interface{}) ([]interface{}, bool) {
	switch typed := value.(type) {
	case []interface{}:
		return typed, true
	case []string:
		result := make([]interface{}, len(typed))
		for i, item := range typed {
			result[i] = item
		}
		return result, true
	default:
		return nil, false
	}
}

func copyMap(input map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{}, len(input))
	for key, value := range input {
		result[key] = value
	}
	return result
}
