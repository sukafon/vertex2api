package schema

import "testing"

func TestNormalizeResolvesRefsAndAllOf(t *testing.T) {
	input := map[string]interface{}{
		"$defs": map[string]interface{}{
			"name": map[string]interface{}{"type": "string", "minLength": 1},
		},
		"allOf": []interface{}{
			map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]interface{}{"$ref": "#/$defs/name"},
				},
				"required": []interface{}{"name", "missing"},
			},
			map[string]interface{}{
				"properties": map[string]interface{}{
					"age": map[string]interface{}{"type": []interface{}{"integer", "null"}},
				},
			},
		},
	}

	got := Normalize(input).(map[string]interface{})
	if _, ok := got["$defs"]; ok {
		t.Fatal("$defs should be removed after reference resolution")
	}
	properties := got["properties"].(map[string]interface{})
	name := properties["name"].(map[string]interface{})
	if name["type"] != "string" || name["minLength"] != 1 {
		t.Fatalf("resolved name schema = %#v", name)
	}
	age := properties["age"].(map[string]interface{})
	if age["type"] != "integer" || age["nullable"] != true {
		t.Fatalf("nullable age schema = %#v", age)
	}
	required := got["required"].([]interface{})
	if len(required) != 1 || required[0] != "name" {
		t.Fatalf("required = %#v, want only name", required)
	}
}

func TestNormalizeConvertsConstAndOneOf(t *testing.T) {
	got := Normalize(map[string]interface{}{
		"oneOf": []interface{}{
			map[string]interface{}{"const": "a"},
			map[string]interface{}{"type": "null"},
		},
	}).(map[string]interface{})

	values := got["enum"].([]interface{})
	if len(values) != 1 || values[0] != "a" {
		t.Fatalf("enum = %#v, want [a]", values)
	}
	if got["nullable"] != true {
		t.Fatalf("nullable = %#v, want true", got["nullable"])
	}
}

func TestNormalizeResolvesNestedJSONPointer(t *testing.T) {
	got := Normalize(map[string]interface{}{
		"$defs": map[string]interface{}{
			"container": map[string]interface{}{
				"properties": map[string]interface{}{
					"value": map[string]interface{}{"type": "number"},
				},
			},
		},
		"$ref": "#/$defs/container/properties/value",
	}).(map[string]interface{})
	if got["type"] != "number" {
		t.Fatalf("nested reference = %#v, want number schema", got)
	}
}
