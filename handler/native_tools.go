package handler

import "strings"

func convertNativeGeminiTool(tool map[string]interface{}) (map[string]interface{}, bool) {
	if len(tool) == 0 {
		return nil, false
	}

	if field, value, ok := nativeGeminiToolFieldValue(tool); ok {
		return map[string]interface{}{field: value}, true
	}

	toolType, _ := tool["type"].(string)
	if field, ok := nativeGeminiToolFieldName(toolType); ok {
		if field != "functionDeclarations" {
			return map[string]interface{}{field: map[string]interface{}{}}, true
		}
	}

	return nil, false
}

func nativeGeminiToolFieldValue(tool map[string]interface{}) (string, interface{}, bool) {
	for key, value := range tool {
		field, ok := nativeGeminiToolFieldName(key)
		if !ok {
			continue
		}
		return field, value, true
	}
	return "", nil, false
}

func nativeGeminiToolFieldName(name string) (string, bool) {
	name = strings.TrimSpace(name)
	switch name {
	case "retrieval":
		return "retrieval", true
	case "googleSearch":
		return "googleSearch", true
	case "googleSearchRetrieval":
		return "googleSearchRetrieval", true
	case "enterpriseWebSearch":
		return "enterpriseWebSearch", true
	case "codeExecution", "code_execution", "code_interpreter":
		return "codeExecution", true
	case "urlContext", "url_context":
		return "urlContext", true
	default:
		if strings.HasPrefix(name, "web_search") {
			return "googleSearch", true
		}
		return "", false
	}
}
