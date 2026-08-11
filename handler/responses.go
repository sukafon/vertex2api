package handler

import (
	"context"
	"crypto/aes"
	"crypto/cipher"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	"vertex2api/model"
	"vertex2api/proxy"

	"github.com/bytedance/sonic"
	"github.com/rs/zerolog/log"
)

const (
	compactContentPrefix = "v2c1."
	compactAAD           = "vertex2api.responses.compact.v1"
)

// ResponsesAPI implements the stateless parts of the OpenAI Responses API.
// The compact codec is intentionally scoped to this service: OpenAI compaction
// items are opaque and cannot be recreated by a third-party backend.
type ResponsesAPI struct {
	vp                    *proxy.VertexProxy
	allowCustomModelNames bool
	compactCodec          *responseCompactCodec
}

func NewResponsesAPI(vp *proxy.VertexProxy, allowCustomModelNames bool, compactSecret string) *ResponsesAPI {
	return &ResponsesAPI{
		vp:                    vp,
		allowCustomModelNames: allowCustomModelNames,
		compactCodec:          newResponseCompactCodec(compactSecret),
	}
}

// Responses handles POST /v1/responses.
func (api *ResponsesAPI) Responses() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(proxy.WithCompatibilityLayer(r.Context(), proxy.CompatibilityLayerOpenAIResponses))
		var req model.ResponseRequest
		if _, ok := readJSONRequest(w, r, &req); !ok {
			return
		}
		if message := api.validateResponseRequest(req); message != "" {
			writeResponseRequestError(w, message)
			return
		}

		chatReq, toolKinds, err := api.responseRequestToChat(req)
		if err != nil {
			writeResponseRequestError(w, err.Error())
			return
		}
		chatReq, compactItem, err := api.maybeCompactRequest(r.Context(), req, chatReq)
		if err != nil {
			if requestContextCanceled(r.Context(), err) {
				return
			}
			if !proxy.IsUpstreamErrorLogged(err) {
				log.Error().Str("err", upstreamLogError(api.vp, err)).Str("model", req.Model).Msg("OpenAI Responses compaction failed")
			}
			writeUpstreamProtocolError(w, r, api.vp, err)
			return
		}

		contents, systemInstruction := convertMessages(chatReq.Model, chatReq.Messages)
		genConfig := buildResponsesGenerationConfig(req, chatReq)
		options := buildOpenAIRequestOptions(chatReq)

		if req.Stream {
			bodyJSON, tokenLease, err := api.vp.BuildBodyWithTokenWithOptionsContext(r.Context(), req.Model, contents, genConfig, nil, systemInstruction, options)
			if err != nil {
				if requestContextCanceled(r.Context(), err) {
					return
				}
				WriteJSON(w, http.StatusInternalServerError, publicServerErrorResponse(err))
				return
			}
			api.streamResponse(w, r, req, chatReq, compactItem, toolKinds, func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
				return api.vp.StreamWithTokenRequireAssistantOutputContext(ctx, bodyJSON, tokenLease, onChunk)
			})
			return
		}

		result, err := api.vp.CallWithTokenWithOptionsRequireAssistantOutputContext(r.Context(), req.Model, contents, genConfig, nil, systemInstruction, options)
		if err != nil {
			if requestContextCanceled(r.Context(), err) {
				return
			}
			if !proxy.IsUpstreamErrorLogged(err) {
				log.Error().Str("err", upstreamLogError(api.vp, err)).Str("model", req.Model).Msg("OpenAI Responses call failed")
			}
			writeUpstreamProtocolError(w, r, api.vp, err)
			return
		}

		usage, estimated := responseUsage(chatReq, result)
		if estimated {
			w.Header().Set("X-Usage-Estimated", "true")
		}
		output := buildResponseOutput(result, toolKinds)
		if len(output) == 0 {
			log.Error().
				Str("model", req.Model).
				Str("finish_reason", result.FinishReason).
				Msg("OpenAI Responses call completed without assistant output")
			WriteProtocolError(w, r, http.StatusBadGateway, proxy.ErrNoAssistantOutput.Error(), "server_error")
			return
		}
		if compactItem != nil {
			output = append([]map[string]interface{}{compactItem}, output...)
		}
		responseID := newResponseObjectID("resp")
		WriteJSON(w, http.StatusOK, buildResponseObject(req, responseID, "completed", output, &usage))
	})
}

// Compact handles POST /v1/responses/compact.
func (api *ResponsesAPI) Compact() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(proxy.WithCompatibilityLayer(r.Context(), proxy.CompatibilityLayerOpenAIResponses))
		var req model.ResponseCompactRequest
		if _, ok := readJSONRequest(w, r, &req); !ok {
			return
		}
		if message := api.validateCompactRequest(req); message != "" {
			writeResponseRequestError(w, message)
			return
		}

		messages, err := api.responseInputToMessages(req.Input)
		if err != nil {
			writeResponseRequestError(w, err.Error())
			return
		}
		if req.Instructions != "" {
			messages = append([]model.ChatMessage{{Role: "developer", Content: req.Instructions}}, messages...)
		}
		summary, summaryReq, result, err := api.compactMessages(r.Context(), req.Model, messages)
		if err != nil {
			if requestContextCanceled(r.Context(), err) {
				return
			}
			if !proxy.IsUpstreamErrorLogged(err) {
				log.Error().Str("err", upstreamLogError(api.vp, err)).Str("model", req.Model).Msg("OpenAI Responses compact call failed")
			}
			writeUpstreamProtocolError(w, r, api.vp, err)
			return
		}

		item, err := api.newCompactionItem(req.Model, summary)
		if err != nil {
			WriteJSON(w, http.StatusInternalServerError, publicServerErrorResponse(err))
			return
		}
		usage, estimated := responseUsage(summaryReq, result)
		if estimated {
			w.Header().Set("X-Usage-Estimated", "true")
		}
		WriteJSON(w, http.StatusOK, model.CompactedResponse{
			ID:        newResponseObjectID("resp"),
			CreatedAt: time.Now().Unix(),
			Object:    "response.compaction",
			Output:    []map[string]interface{}{item},
			Usage:     usage,
		})
	})
}

func (api *ResponsesAPI) validateResponseRequest(req model.ResponseRequest) string {
	if message := validateModelName(req.Model, api.allowCustomModelNames); message != "" {
		return message
	}
	if strings.TrimSpace(req.Model) == "" {
		return "model is required"
	}
	if req.Input == nil {
		return "input is required"
	}
	if req.PreviousResponseID != "" {
		return "previous_response_id is not supported; send the complete stateless input array"
	}
	if req.Background {
		return "background responses are not supported"
	}
	if req.Store != nil && *req.Store {
		return "store=true is not supported; send the complete stateless input array"
	}
	if req.MaxOutputTokens != nil && *req.MaxOutputTokens < 1 {
		return "max_output_tokens must be greater than zero"
	}
	if req.Temperature != nil && (*req.Temperature < 0 || *req.Temperature > 2) {
		return "temperature must be between 0 and 2"
	}
	if req.TopP != nil && (*req.TopP < 0 || *req.TopP > 1) {
		return "top_p must be between 0 and 1"
	}
	if rawEffort, ok := req.Reasoning["effort"]; ok {
		effort, ok := rawEffort.(string)
		if !ok {
			return "reasoning.effort must be a string"
		}
		if message := validateOpenAIReasoningEffort(req.Model, &effort); message != "" {
			return message
		}
	}
	for _, entry := range req.ContextManagement {
		if entry.Type != "compaction" {
			return "context_management.type must be compaction"
		}
		if entry.CompactThreshold < 1 {
			return "context_management.compact_threshold must be greater than zero"
		}
	}
	return ""
}

func (api *ResponsesAPI) validateCompactRequest(req model.ResponseCompactRequest) string {
	if message := validateModelName(req.Model, api.allowCustomModelNames); message != "" {
		return message
	}
	if strings.TrimSpace(req.Model) == "" {
		return "model is required"
	}
	if req.Input == nil {
		return "input is required"
	}
	if req.PreviousResponseID != "" {
		return "previous_response_id is not supported; send the complete stateless input array"
	}
	return ""
}

func writeResponseRequestError(w http.ResponseWriter, message string) {
	WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{Error: &model.APIError{
		Message: message,
		Type:    "invalid_request_error",
	}})
}

func (api *ResponsesAPI) responseRequestToChat(req model.ResponseRequest) (model.ChatCompletionRequest, responseToolBindings, error) {
	messages, err := api.responseInputToMessages(req.Input)
	if err != nil {
		return model.ChatCompletionRequest{}, nil, err
	}
	if req.Instructions != "" {
		messages = append([]model.ChatMessage{{Role: "developer", Content: req.Instructions}}, messages...)
	}
	tools, toolKinds, err := responseToolsToChatWithChoice(req.Tools, req.ToolChoice)
	if err != nil {
		return model.ChatCompletionRequest{}, nil, err
	}
	responseFormat, err := responsesTextFormat(req.Text)
	if err != nil {
		return model.ChatCompletionRequest{}, nil, err
	}
	chatReq := model.ChatCompletionRequest{
		Model:               req.Model,
		Messages:            messages,
		Stream:              req.Stream,
		Temperature:         req.Temperature,
		TopP:                req.TopP,
		MaxCompletionTokens: req.MaxOutputTokens,
		ResponseFormat:      responseFormat,
		Tools:               tools,
		ToolChoice:          responseToolChoiceToChat(req.ToolChoice),
	}
	if effort, ok := req.Reasoning["effort"].(string); ok {
		chatReq.ReasoningEffort = &effort
	}
	if message := validateOpenAIRequest(chatReq); message != "" {
		return model.ChatCompletionRequest{}, nil, errors.New(message)
	}
	return chatReq, toolKinds, nil
}

type responseToolBinding struct {
	Kind      string
	Namespace string
	Name      string
}

type responseToolBindings map[string]responseToolBinding

func responseToolsToChat(tools []map[string]interface{}) ([]map[string]interface{}, responseToolBindings, error) {
	return responseToolsToChatWithChoice(tools, nil)
}

func responseToolsToChatWithChoice(tools []map[string]interface{}, toolChoice interface{}) ([]map[string]interface{}, responseToolBindings, error) {
	if len(tools) == 0 {
		return nil, nil, nil
	}
	converted := make([]map[string]interface{}, 0, len(tools))
	bindings := make(responseToolBindings)
	for _, tool := range tools {
		toolType, _ := tool["type"].(string)
		switch toolType {
		case "function":
			name, _ := tool["name"].(string)
			if name == "" {
				return nil, nil, errors.New("tools[].name is required for function tools")
			}
			if err := registerResponseToolBinding(bindings, name, responseToolBinding{Kind: "function", Name: name}); err != nil {
				return nil, nil, err
			}
			function := copyInterfaceMap(tool)
			delete(function, "type")
			converted = append(converted, map[string]interface{}{"type": "function", "function": function})
		case "namespace":
			namespaceName, _ := tool["name"].(string)
			namespaceName = strings.TrimSpace(namespaceName)
			if namespaceName == "" {
				return nil, nil, errors.New("tools[].name is required for namespace tools")
			}
			namespaceTools, ok := tool["tools"].([]interface{})
			if !ok || len(namespaceTools) == 0 {
				return nil, nil, fmt.Errorf("namespace tool %q must contain at least one function in tools", namespaceName)
			}
			for nestedIndex, rawNestedTool := range namespaceTools {
				nestedTool, ok := rawNestedTool.(map[string]interface{})
				if !ok {
					return nil, nil, fmt.Errorf("namespace tool %q tools[%d] must be an object", namespaceName, nestedIndex)
				}
				nestedType, _ := nestedTool["type"].(string)
				if nestedType != "function" {
					return nil, nil, fmt.Errorf("namespace tool %q tools[%d] type %q is not supported", namespaceName, nestedIndex, nestedType)
				}
				nestedName, _ := nestedTool["name"].(string)
				nestedName = strings.TrimSpace(nestedName)
				if nestedName == "" {
					return nil, nil, fmt.Errorf("namespace tool %q tools[%d].name is required", namespaceName, nestedIndex)
				}

				// Vertex function declarations are flat. Codex addresses namespace
				// members using the same double-underscore form used by its tool
				// runtime, so the mapping remains reversible for tool calls.
				name := flattenResponseNamespaceToolName(namespaceName, nestedName)
				if err := registerResponseToolBinding(bindings, name, responseToolBinding{
					Kind: "function", Namespace: namespaceName, Name: nestedName,
				}); err != nil {
					return nil, nil, err
				}
				parameters := nestedTool["inputSchema"]
				if parameters == nil {
					parameters = nestedTool["input_schema"]
				}
				if parameters == nil {
					parameters = nestedTool["parameters"]
				}
				if parameters == nil {
					parameters = map[string]interface{}{"type": "object", "properties": map[string]interface{}{}}
				}
				function := map[string]interface{}{
					"name":       name,
					"parameters": parameters,
				}
				if description, _ := nestedTool["description"].(string); description != "" {
					function["description"] = description
				} else if description, _ := tool["description"].(string); description != "" {
					function["description"] = description
				}
				converted = append(converted, map[string]interface{}{"type": "function", "function": function})
			}
		case "custom":
			name, _ := tool["name"].(string)
			if name == "" {
				return nil, nil, errors.New("tools[].name is required for custom tools")
			}
			if err := registerResponseToolBinding(bindings, name, responseToolBinding{Kind: "custom", Name: name}); err != nil {
				return nil, nil, err
			}
			function := map[string]interface{}{
				"name": name,
				"parameters": map[string]interface{}{
					"type":       "object",
					"properties": map[string]interface{}{"input": map[string]interface{}{"type": "string"}},
					"required":   []string{"input"},
				},
			}
			if description, _ := tool["description"].(string); description != "" {
				function["description"] = description
			}
			converted = append(converted, map[string]interface{}{"type": "function", "function": function})
		case "local_shell":
			name := "local_shell"
			if err := registerResponseToolBinding(bindings, name, responseToolBinding{Kind: "local_shell", Name: name}); err != nil {
				return nil, nil, err
			}
			converted = append(converted, map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        name,
					"description": "Execute one or more commands in the caller's local shell.",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"command":           map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "string"}},
							"timeout_ms":        map[string]interface{}{"type": "integer"},
							"working_directory": map[string]interface{}{"type": "string"},
						},
						"required": []string{"command"},
					},
				},
			})
		case "shell":
			name := "shell"
			if err := registerResponseToolBinding(bindings, name, responseToolBinding{Kind: "shell", Name: name}); err != nil {
				return nil, nil, err
			}
			converted = append(converted, map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        name,
					"description": "Execute one or more shell commands in the caller's environment.",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"commands":          map[string]interface{}{"type": "array", "items": map[string]interface{}{"type": "string"}},
							"timeout_ms":        map[string]interface{}{"type": "integer"},
							"max_output_length": map[string]interface{}{"type": "integer"},
						},
						"required": []string{"commands"},
					},
				},
			})
		case "apply_patch":
			name := "apply_patch"
			if err := registerResponseToolBinding(bindings, name, responseToolBinding{Kind: "apply_patch", Name: name}); err != nil {
				return nil, nil, err
			}
			converted = append(converted, map[string]interface{}{
				"type": "function",
				"function": map[string]interface{}{
					"name":        name,
					"description": "Create, update, or delete a file using a unified diff.",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"type": map[string]interface{}{"type": "string", "enum": []string{"create_file", "update_file", "delete_file"}},
							"path": map[string]interface{}{"type": "string"},
							"diff": map[string]interface{}{"type": "string"},
						},
						"required": []string{"type", "path"},
					},
				},
			})
		case "web_search", "web_search_2025_08_26", "web_search_preview", "web_search_preview_2025_03_11":
			cacheOnly, err := validateResponsesWebSearchTool(toolType, tool)
			if err != nil {
				return nil, nil, err
			}
			if cacheOnly {
				if responseToolChoiceRequiresWebSearch(toolChoice) {
					return nil, nil, errors.New("当前后端不支持强制使用缓存索引进行网页搜索。请移除网页搜索工具，或前往“配置 → 网页搜索”，选择“已索引”、“实时”或“已禁用”。")
				}
				continue
			}
			allowedKeys := []string{"type", "external_web_access"}
			if !strings.HasPrefix(toolType, "web_search_preview") {
				allowedKeys = append(allowedKeys, "indexed_web_access")
			}
			if unsupported := responseToolUnsupportedKeys(tool, allowedKeys...); len(unsupported) > 0 {
				return nil, nil, fmt.Errorf("tool type %q fields %s have no Vertex Google Search equivalent", toolType, strings.Join(unsupported, ", "))
			}
			converted = append(converted, map[string]interface{}{"type": toolType})
		case "code_interpreter", "code_execution":
			if unsupported := responseToolUnsupportedKeys(tool, "type", "container"); len(unsupported) > 0 {
				return nil, nil, fmt.Errorf("tool type %q fields %s have no Vertex code execution equivalent", toolType, strings.Join(unsupported, ", "))
			}
			if container, ok := tool["container"].(map[string]interface{}); ok {
				containerType, _ := container["type"].(string)
				fileIDs, _ := container["file_ids"].([]interface{})
				if (containerType != "" && containerType != "auto") || len(fileIDs) > 0 || len(responseToolUnsupportedKeys(container, "type", "file_ids")) > 0 {
					return nil, nil, errors.New("code_interpreter.container is only supported as auto without file_ids")
				}
			}
			converted = append(converted, map[string]interface{}{"type": toolType})
		case "url_context":
			if unsupported := responseToolUnsupportedKeys(tool, "type"); len(unsupported) > 0 {
				return nil, nil, fmt.Errorf("url_context fields %s have no Vertex equivalent", strings.Join(unsupported, ", "))
			}
			converted = append(converted, map[string]interface{}{"type": toolType})
		default:
			return nil, nil, fmt.Errorf("tool type %q is not supported by this Responses adapter", toolType)
		}
	}
	return converted, bindings, nil
}

func validateResponsesWebSearchTool(toolType string, tool map[string]interface{}) (bool, error) {
	rawExternalAccess, exists := tool["external_web_access"]
	externalAccess := true
	if exists {
		var ok bool
		externalAccess, ok = rawExternalAccess.(bool)
		if !ok {
			return false, errors.New("web_search.external_web_access must be a boolean")
		}
	}
	if strings.HasPrefix(toolType, "web_search_preview") {
		// OpenAI preview search ignores this field and always uses live access,
		// which matches Vertex Google Search behavior.
		return false, nil
	}
	if rawIndexedAccess, indexedExists := tool["indexed_web_access"]; indexedExists {
		indexedAccess, ok := rawIndexedAccess.(bool)
		if !ok {
			return false, errors.New("web_search.indexed_web_access must be a boolean")
		}
		if indexedAccess && !externalAccess {
			return false, errors.New("web_search.indexed_web_access=true requires external_web_access=true")
		}
	}
	return !externalAccess, nil
}

func responseToolChoiceRequiresWebSearch(choice interface{}) bool {
	if value, ok := choice.(string); ok {
		return value == "required"
	}
	tool, ok := choice.(map[string]interface{})
	if !ok {
		return false
	}
	toolType, _ := tool["type"].(string)
	return strings.HasPrefix(toolType, "web_search")
}

func registerResponseToolBinding(bindings responseToolBindings, name string, binding responseToolBinding) error {
	if _, exists := bindings[name]; exists {
		return fmt.Errorf("duplicate Responses tool name %q after namespace expansion", name)
	}
	bindings[name] = binding
	return nil
}

func flattenResponseNamespaceToolName(namespace, name string) string {
	if strings.HasSuffix(namespace, "__") {
		return namespace + name
	}
	return namespace + "__" + name
}

func responseToolUnsupportedKeys(tool map[string]interface{}, allowed ...string) []string {
	allowedSet := make(map[string]struct{}, len(allowed))
	for _, key := range allowed {
		allowedSet[key] = struct{}{}
	}
	unsupported := make([]string, 0)
	for key := range tool {
		if _, ok := allowedSet[key]; !ok {
			unsupported = append(unsupported, key)
		}
	}
	sort.Strings(unsupported)
	return unsupported
}

func responseToolChoiceToChat(choice interface{}) interface{} {
	m, ok := choice.(map[string]interface{})
	if !ok {
		return choice
	}
	toolType, _ := m["type"].(string)
	name, _ := m["name"].(string)
	namespace, _ := m["namespace"].(string)
	if (toolType == "function" || toolType == "custom") && name != "" {
		if toolType == "function" && namespace != "" {
			name = flattenResponseNamespaceToolName(namespace, name)
		}
		return map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": name}}
	}
	if toolType == "local_shell" || toolType == "shell" || toolType == "apply_patch" {
		return map[string]interface{}{"type": "function", "function": map[string]interface{}{"name": toolType}}
	}
	return choice
}

func responsesTextFormat(textConfig map[string]interface{}) (map[string]interface{}, error) {
	if len(textConfig) == 0 {
		return nil, nil
	}
	format, _ := textConfig["format"].(map[string]interface{})
	if len(format) == 0 {
		return nil, nil
	}
	formatType, _ := format["type"].(string)
	switch formatType {
	case "", "text":
		return map[string]interface{}{"type": "text"}, nil
	case "json_object":
		return map[string]interface{}{"type": "json_object"}, nil
	case "json_schema":
		if format["schema"] == nil {
			return nil, errors.New("text.format.schema is required for json_schema")
		}
		return map[string]interface{}{"type": "json_schema", "json_schema": copyInterfaceMap(format)}, nil
	default:
		return nil, fmt.Errorf("text.format.type %q is not supported", formatType)
	}
}

func buildResponsesGenerationConfig(req model.ResponseRequest, chatReq model.ChatCompletionRequest) map[string]interface{} {
	config := map[string]interface{}{}
	if chatReq.Temperature != nil {
		config["temperature"] = *chatReq.Temperature
	}
	if chatReq.TopP != nil {
		config["topP"] = *chatReq.TopP
	}
	if chatReq.MaxCompletionTokens != nil {
		config["maxOutputTokens"] = *chatReq.MaxCompletionTokens
	}
	applyOpenAIResponseFormat(config, chatReq.ResponseFormat)
	if thinkingConfig := openAIReasoningConfig(chatReq.Model, chatReq.ReasoningEffort); thinkingConfig != nil {
		thinkingConfig["includeThoughts"] = true
		config["thinkingConfig"] = thinkingConfig
	}
	return config
}

func (api *ResponsesAPI) responseInputToMessages(input interface{}) ([]model.ChatMessage, error) {
	switch value := input.(type) {
	case string:
		if value == "" {
			return nil, errors.New("input must not be empty")
		}
		return []model.ChatMessage{{Role: "user", Content: value}}, nil
	case []interface{}:
		messages := make([]model.ChatMessage, 0, len(value))
		for index, raw := range value {
			item, ok := raw.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("input[%d] must be an object", index)
			}
			itemMessages, err := api.responseInputItemToMessages(item)
			if err != nil {
				return nil, fmt.Errorf("input[%d]: %w", index, err)
			}
			messages = append(messages, itemMessages...)
		}
		if len(messages) == 0 {
			return nil, errors.New("input must contain at least one supported item")
		}
		return messages, nil
	default:
		return nil, errors.New("input must be a string or an array of input items")
	}
}

func (api *ResponsesAPI) responseInputItemToMessages(item map[string]interface{}) ([]model.ChatMessage, error) {
	itemType, _ := item["type"].(string)
	role, _ := item["role"].(string)
	if itemType == "" && role != "" {
		itemType = "message"
	}
	switch itemType {
	case "message":
		if role == "" {
			return nil, errors.New("message.role is required")
		}
		switch role {
		case "user", "assistant", "system", "developer":
		default:
			return nil, fmt.Errorf("message role %q is not supported", role)
		}
		content, err := responseContentToChat(item["content"])
		if err != nil {
			return nil, err
		}
		phase, _ := item["phase"].(string)
		if role == "assistant" && phase == "commentary" {
			if reasoning := extractTextContent(content); reasoning != "" {
				return []model.ChatMessage{{Role: role, Content: nil, ReasoningContent: reasoning}}, nil
			}
		}
		return []model.ChatMessage{{Role: role, Content: content}}, nil
	case "function_call":
		name, _ := item["name"].(string)
		namespace, _ := item["namespace"].(string)
		callID, _ := item["call_id"].(string)
		if name == "" || callID == "" {
			return nil, errors.New("function_call.name and function_call.call_id are required")
		}
		if namespace != "" {
			name = flattenResponseNamespaceToolName(namespace, name)
		}
		arguments := responseStringOrJSON(item["arguments"])
		return []model.ChatMessage{{Role: "assistant", Content: nil, ToolCalls: []model.ChatToolCall{{
			ID: callID, Type: "function", Function: &model.ChatFunctionCall{Name: name, Arguments: arguments},
		}}}}, nil
	case "custom_tool_call":
		name, _ := item["name"].(string)
		callID, _ := item["call_id"].(string)
		input, _ := item["input"].(string)
		arguments, _ := sonic.Marshal(map[string]interface{}{"input": input})
		return []model.ChatMessage{{Role: "assistant", Content: nil, ToolCalls: []model.ChatToolCall{{
			ID: callID, Type: "function", Function: &model.ChatFunctionCall{Name: name, Arguments: string(arguments)},
		}}}}, nil
	case "local_shell_call":
		callID, _ := item["call_id"].(string)
		if callID == "" {
			callID, _ = item["id"].(string)
		}
		arguments := responseStringOrJSON(item["action"])
		return []model.ChatMessage{{Role: "assistant", Content: nil, ToolCalls: []model.ChatToolCall{{
			ID: callID, Type: "function", Function: &model.ChatFunctionCall{Name: "local_shell", Arguments: arguments},
		}}}}, nil
	case "shell_call":
		callID, _ := item["call_id"].(string)
		if callID == "" {
			callID, _ = item["id"].(string)
		}
		arguments := responseStringOrJSON(item["action"])
		return []model.ChatMessage{{Role: "assistant", Content: nil, ToolCalls: []model.ChatToolCall{{
			ID: callID, Type: "function", Function: &model.ChatFunctionCall{Name: "shell", Arguments: arguments},
		}}}}, nil
	case "apply_patch_call":
		callID, _ := item["call_id"].(string)
		if callID == "" {
			callID, _ = item["id"].(string)
		}
		arguments := responseStringOrJSON(item["operation"])
		return []model.ChatMessage{{Role: "assistant", Content: nil, ToolCalls: []model.ChatToolCall{{
			ID: callID, Type: "function", Function: &model.ChatFunctionCall{Name: "apply_patch", Arguments: arguments},
		}}}}, nil
	case "function_call_output", "custom_tool_call_output":
		callID, _ := item["call_id"].(string)
		if callID == "" {
			return nil, errors.New("tool output call_id is required")
		}
		return []model.ChatMessage{{Role: "tool", ToolCallID: callID, Content: responseOutputContent(item["output"])}}, nil
	case "local_shell_call_output":
		callID, _ := item["call_id"].(string)
		if callID == "" {
			callID, _ = item["id"].(string)
		}
		return []model.ChatMessage{{Role: "tool", ToolCallID: callID, Name: "local_shell", Content: responseOutputContent(item["output"])}}, nil
	case "shell_call_output", "apply_patch_call_output":
		callID, _ := item["call_id"].(string)
		if callID == "" {
			callID, _ = item["id"].(string)
		}
		name := "shell"
		if itemType == "apply_patch_call_output" {
			name = "apply_patch"
		}
		return []model.ChatMessage{{Role: "tool", ToolCallID: callID, Name: name, Content: responseOutputContent(item["output"])}}, nil
	case "reasoning":
		var reasoning strings.Builder
		for _, field := range []string{"summary", "content"} {
			items, _ := item[field].([]interface{})
			for _, raw := range items {
				part, _ := raw.(map[string]interface{})
				text, _ := part["text"].(string)
				if text != "" {
					reasoning.WriteString(text)
				}
			}
		}
		if reasoning.Len() == 0 {
			return nil, nil
		}
		return []model.ChatMessage{{Role: "assistant", Content: nil, ReasoningContent: reasoning.String()}}, nil
	case "web_search_call":
		// A Responses web_search_call is hosted-tool activity metadata. Vertex
		// represents Google Search grounding outside the conversational contents,
		// while the adjacent assistant message carries the conversational text.
		// Keep that message and omit this non-content item during a replay.
		return nil, nil
	case "compaction":
		encrypted, _ := item["encrypted_content"].(string)
		payload, err := api.compactCodec.open(encrypted)
		if err != nil {
			return nil, errors.New("compaction item is invalid or was not created with this vertex2api API-key configuration")
		}
		return []model.ChatMessage{{Role: "user", Content: "[Compacted historical conversation state]\n" + payload.Summary}}, nil
	default:
		return nil, fmt.Errorf("input item type %q is not supported by this Responses adapter", itemType)
	}
}

func responseContentToChat(content interface{}) (interface{}, error) {
	if text, ok := content.(string); ok {
		return text, nil
	}
	items, ok := content.([]interface{})
	if !ok {
		return nil, errors.New("message.content must be a string or an array")
	}
	parts := make([]interface{}, 0, len(items))
	for _, raw := range items {
		part, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		partType, _ := part["type"].(string)
		switch partType {
		case "input_text", "output_text", "text":
			if text, _ := part["text"].(string); text != "" {
				parts = append(parts, map[string]interface{}{"type": "text", "text": text})
			}
		case "input_image", "image_url":
			imageURL, _ := part["image_url"].(string)
			if imageURL == "" {
				if nested, ok := part["image_url"].(map[string]interface{}); ok {
					imageURL, _ = nested["url"].(string)
				}
			}
			if imageURL == "" {
				return nil, errors.New("input_image.image_url is required")
			}
			parts = append(parts, map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": imageURL}})
		case "input_file":
			file := map[string]interface{}{}
			for _, key := range []string{"file_data", "file_url", "file_id", "filename"} {
				if value, exists := part[key]; exists {
					file[key] = value
				}
			}
			if file["file_data"] == nil && file["file_url"] == nil {
				if file["file_id"] != nil {
					return nil, errors.New("input_file.file_id cannot be resolved without an OpenAI Files backend; use file_data or file_url")
				}
				return nil, errors.New("input_file.file_data or input_file.file_url is required")
			}
			parts = append(parts, map[string]interface{}{"type": "file", "file": file})
		case "input_audio":
			audio, _ := part["input_audio"].(map[string]interface{})
			if len(audio) == 0 {
				audio = copyInterfaceMap(part)
				delete(audio, "type")
			}
			if audio["data"] == nil {
				return nil, errors.New("input_audio.data is required")
			}
			parts = append(parts, map[string]interface{}{"type": "input_audio", "input_audio": audio})
		default:
			return nil, fmt.Errorf("content part type %q is not supported", partType)
		}
	}
	return parts, nil
}

func responseStringOrJSON(value interface{}) string {
	if text, ok := value.(string); ok {
		return text
	}
	data, err := sonic.Marshal(value)
	if err != nil {
		return "{}"
	}
	return string(data)
}

func responseOutputContent(value interface{}) interface{} {
	if text, ok := value.(string); ok {
		return text
	}
	return responseStringOrJSON(value)
}

func responseUsage(req model.ChatCompletionRequest, result *proxy.CallResult) (model.ResponseUsage, bool) {
	usage, estimated := openAIUsage(req, result)
	converted := model.ResponseUsage{}
	if usage == nil {
		return converted, estimated
	}
	converted.InputTokens = usage.PromptTokens
	converted.OutputTokens = usage.CompletionTokens
	converted.TotalTokens = usage.TotalTokens
	if usage.PromptTokensDetails != nil {
		converted.InputTokensDetails.CachedTokens = usage.PromptTokensDetails.CachedTokens
	}
	if usage.CompletionTokensDetails != nil {
		converted.OutputTokensDetails.ReasoningTokens = usage.CompletionTokensDetails.ReasoningTokens
	}
	return converted, estimated
}

func buildResponseOutput(result *proxy.CallResult, toolBindings responseToolBindings) []map[string]interface{} {
	if result == nil {
		return []map[string]interface{}{}
	}
	output := make([]map[string]interface{}, 0, 2+len(result.FunctionCalls))
	_, reasoning := buildResponseContent(result)
	if reasoning != "" {
		output = append(output, responseMessageItemWithPhase(reasoning, "completed", "commentary", nil))
	}
	output = append(output, responseWebSearchCallItems(result.GroundingMetadata)...)
	content, _ := buildResponseContent(result)
	if text, ok := content.(string); ok && text != "" {
		output = append(output, responseMessageItemWithAnnotations(text, "completed", responseAnnotationsFromGrounding(result.GroundingMetadata)))
	}
	for _, call := range dedupeResponseFunctionCalls(result.FunctionCalls) {
		output = append(output, responseToolCallItem(call, toolBindings[call.Name], "completed"))
	}
	return output
}

func responseWebSearchCallItems(gm *model.GroundingMetadata) []map[string]interface{} {
	if gm == nil || len(gm.WebSearchQueries) == 0 {
		return nil
	}
	items := make([]map[string]interface{}, 0, len(gm.WebSearchQueries))
	for _, query := range gm.WebSearchQueries {
		if strings.TrimSpace(query) == "" {
			continue
		}
		items = append(items, map[string]interface{}{
			"id": newResponseObjectID("ws"), "type": "web_search_call", "status": "completed",
			"action": map[string]interface{}{"type": "search", "query": query},
		})
	}
	return items
}

func responseMessageItemWithAnnotations(text, status string, annotations []map[string]interface{}) map[string]interface{} {
	return responseMessageItemWithPhase(text, status, "final_answer", annotations)
}

func responseMessageItemWithPhase(text, status, phase string, annotations []map[string]interface{}) map[string]interface{} {
	if annotations == nil {
		annotations = []map[string]interface{}{}
	}
	return map[string]interface{}{
		"id":     newResponseObjectID("msg"),
		"type":   "message",
		"status": status,
		"role":   "assistant",
		"phase":  phase,
		"content": []map[string]interface{}{{
			"type": "output_text", "text": text, "annotations": annotations, "logprobs": []interface{}{},
		}},
	}
}

func responseAnnotationsFromGrounding(gm *model.GroundingMetadata) []map[string]interface{} {
	chatAnnotations := openAIAnnotationsFromGrounding(gm)
	annotations := make([]map[string]interface{}, 0, len(chatAnnotations))
	for _, annotation := range chatAnnotations {
		if annotation.URLCitation == nil {
			continue
		}
		citation := annotation.URLCitation
		annotations = append(annotations, map[string]interface{}{
			"type": "url_citation", "start_index": citation.StartIndex, "end_index": citation.EndIndex,
			"url": citation.URL, "title": citation.Title,
		})
	}
	return annotations
}

func responseToolCallItem(call model.FunctionCall, binding responseToolBinding, status string) map[string]interface{} {
	arguments := responseStringOrJSON(call.Args)
	callID := openAIToolCallID(call, 0)
	name := call.Name
	if binding.Name != "" {
		name = binding.Name
	}
	switch binding.Kind {
	case "custom":
		input := arguments
		if value, ok := call.Args["input"].(string); ok {
			input = value
		}
		return map[string]interface{}{
			"id": newResponseObjectID("ctc"), "type": "custom_tool_call", "status": status,
			"call_id": callID, "name": name, "input": input,
		}
	case "local_shell":
		return map[string]interface{}{
			"id": newResponseObjectID("lsc"), "type": "local_shell_call", "status": status,
			"call_id": callID, "action": localShellAction(call.Args),
		}
	case "shell":
		return map[string]interface{}{
			"id": newResponseObjectID("sh"), "type": "shell_call", "status": status,
			"call_id": callID, "action": shellAction(call.Args), "environment": map[string]interface{}{"type": "local"},
		}
	case "apply_patch":
		return map[string]interface{}{
			"id": newResponseObjectID("ap"), "type": "apply_patch_call", "status": status,
			"call_id": callID, "operation": applyPatchOperation(call.Args),
		}
	default:
		item := map[string]interface{}{
			"id": newResponseObjectID("fc"), "type": "function_call", "status": status,
			"call_id": callID, "name": name, "arguments": arguments,
		}
		if binding.Namespace != "" {
			item["namespace"] = binding.Namespace
		}
		return item
	}
}

func localShellAction(args map[string]interface{}) map[string]interface{} {
	action := copyInterfaceMap(args)
	action["type"] = "exec"
	if _, ok := action["command"]; !ok {
		if command, ok := action["cmd"]; ok {
			action["command"] = command
		} else {
			action["command"] = []string{}
		}
	}
	if command, ok := action["command"].(string); ok {
		action["command"] = []string{command}
	}
	if _, ok := action["env"]; !ok {
		action["env"] = map[string]string{}
	}
	delete(action, "cmd")
	return action
}

func shellAction(args map[string]interface{}) map[string]interface{} {
	action := copyInterfaceMap(args)
	if _, ok := action["commands"]; !ok {
		if command, ok := action["command"]; ok {
			action["commands"] = command
		} else {
			action["commands"] = []string{}
		}
	}
	if commands, ok := action["commands"].(string); ok {
		action["commands"] = []string{commands}
	}
	delete(action, "command")
	return action
}

func applyPatchOperation(args map[string]interface{}) map[string]interface{} {
	operation := copyInterfaceMap(args)
	operationType, _ := operation["type"].(string)
	switch operationType {
	case "create_file", "update_file", "delete_file":
	default:
		operation["type"] = "update_file"
	}
	return operation
}

func dedupeResponseFunctionCalls(calls []model.FunctionCall) []model.FunctionCall {
	seen := make(map[string]bool)
	result := make([]model.FunctionCall, 0, len(calls))
	for _, call := range calls {
		key := call.ID + "\x00" + call.Name + "\x00" + responseStringOrJSON(call.Args)
		if seen[key] {
			continue
		}
		seen[key] = true
		result = append(result, call)
	}
	return result
}

func buildResponseObject(req model.ResponseRequest, id, status string, output []map[string]interface{}, usage *model.ResponseUsage) map[string]interface{} {
	now := time.Now().Unix()
	parallel := true
	if req.ParallelToolCalls != nil {
		parallel = *req.ParallelToolCalls
	}
	store := false
	if req.Store != nil {
		store = *req.Store
	}
	temperature := interface{}(1.0)
	if req.Temperature != nil {
		temperature = *req.Temperature
	}
	topP := interface{}(1.0)
	if req.TopP != nil {
		topP = *req.TopP
	}
	toolChoice := req.ToolChoice
	if toolChoice == nil {
		toolChoice = "auto"
	}
	textConfig := req.Text
	if len(textConfig) == 0 {
		textConfig = map[string]interface{}{"format": map[string]interface{}{"type": "text"}}
	}
	truncation := req.Truncation
	if truncation == "" {
		truncation = "disabled"
	}
	metadata := req.Metadata
	if metadata == nil {
		metadata = map[string]string{}
	}
	response := map[string]interface{}{
		"id": id, "object": "response", "created_at": now, "status": status,
		"background": false, "error": nil, "incomplete_details": nil,
		"instructions": nil, "max_output_tokens": req.MaxOutputTokens,
		"metadata": metadata, "model": req.Model, "output": output,
		"parallel_tool_calls": parallel, "previous_response_id": nil,
		"reasoning": req.Reasoning, "store": store, "temperature": temperature,
		"text": textConfig, "tool_choice": toolChoice, "tools": req.Tools,
		"top_p": topP, "truncation": truncation, "usage": usage,
	}
	if req.Instructions != "" {
		response["instructions"] = req.Instructions
	}
	if status == "completed" {
		response["completed_at"] = now
	} else {
		response["completed_at"] = nil
	}
	return response
}

func newResponseObjectID(prefix string) string {
	data := make([]byte, 12)
	if _, err := rand.Read(data); err != nil {
		return fmt.Sprintf("%s_%d", prefix, time.Now().UnixNano())
	}
	return prefix + "_" + base64.RawURLEncoding.EncodeToString(data)
}

func (api *ResponsesAPI) maybeCompactRequest(ctx context.Context, req model.ResponseRequest, chatReq model.ChatCompletionRequest) (model.ChatCompletionRequest, map[string]interface{}, error) {
	threshold := 0
	for _, entry := range req.ContextManagement {
		if entry.Type == "compaction" {
			threshold = entry.CompactThreshold
		}
	}
	if threshold == 0 || estimateOpenAIPromptTokens(chatReq) < threshold {
		return chatReq, nil, nil
	}
	summary, _, _, err := api.compactMessages(ctx, req.Model, chatReq.Messages)
	if err != nil {
		return model.ChatCompletionRequest{}, nil, err
	}
	item, err := api.newCompactionItem(req.Model, summary)
	if err != nil {
		return model.ChatCompletionRequest{}, nil, err
	}
	instructions := retainResponseInstructions(chatReq.Messages)
	retained := retainLatestResponseTurn(chatReq.Messages)
	chatReq.Messages = append(instructions, model.ChatMessage{Role: "user", Content: "[Compacted historical conversation state]\n" + summary})
	chatReq.Messages = append(chatReq.Messages, retained...)
	return chatReq, item, nil
}

func retainResponseInstructions(messages []model.ChatMessage) []model.ChatMessage {
	retained := make([]model.ChatMessage, 0)
	for _, message := range messages {
		role := strings.ToLower(strings.TrimSpace(message.Role))
		if role == "system" || role == "developer" {
			retained = append(retained, message)
		}
	}
	return retained
}

func retainLatestResponseTurn(messages []model.ChatMessage) []model.ChatMessage {
	lastUser := -1
	for index, message := range messages {
		if strings.EqualFold(message.Role, "user") {
			lastUser = index
		}
	}
	if lastUser < 0 {
		return nil
	}
	retained := make([]model.ChatMessage, len(messages)-lastUser)
	copy(retained, messages[lastUser:])
	return retained
}

func (api *ResponsesAPI) compactMessages(ctx context.Context, modelName string, messages []model.ChatMessage) (string, model.ChatCompletionRequest, *proxy.CallResult, error) {
	if api.vp == nil {
		return "", model.ChatCompletionRequest{}, nil, errors.New("vertex proxy is unavailable")
	}
	summaryMessages := append([]model.ChatMessage(nil), messages...)
	summaryMessages = append(summaryMessages, model.ChatMessage{Role: "user", Content: strings.TrimSpace(`Create a dense conversation-state checkpoint for another model that will continue this exact task.
Preserve the user's goals, constraints, decisions, files and code locations, tool-call results, errors, current implementation state, and unresolved next steps. Preserve the latest active request precisely. Do not answer the task and do not add commentary. Return only the checkpoint.`)})
	summaryReq := model.ChatCompletionRequest{Model: modelName, Messages: summaryMessages}
	contents, systemInstruction := convertMessages(modelName, summaryMessages)
	result, err := api.vp.CallWithTokenWithOptionsContext(ctx, modelName, contents, map[string]interface{}{
		"temperature": 0, "maxOutputTokens": 4096,
	}, nil, systemInstruction, nil)
	if err != nil {
		return "", summaryReq, nil, err
	}
	content, _ := buildResponseContent(result)
	summary, _ := content.(string)
	summary = strings.TrimSpace(summary)
	if summary == "" {
		return "", summaryReq, result, errors.New("compaction model returned an empty checkpoint")
	}
	return summary, summaryReq, result, nil
}

func (api *ResponsesAPI) newCompactionItem(modelName, summary string) (map[string]interface{}, error) {
	encrypted, err := api.compactCodec.seal(compactPayload{Version: 1, Model: modelName, Summary: summary})
	if err != nil {
		return nil, err
	}
	return map[string]interface{}{
		"id": newResponseObjectID("cmp"), "type": "compaction",
		"encrypted_content": encrypted, "created_by": "vertex2api",
	}, nil
}

type compactPayload struct {
	Version int    `json:"version"`
	Model   string `json:"model"`
	Summary string `json:"summary"`
}

type responseCompactCodec struct {
	aead cipher.AEAD
}

func newResponseCompactCodec(secret string) *responseCompactCodec {
	seed := []byte(secret)
	if len(seed) == 0 {
		seed = make([]byte, 32)
		_, _ = rand.Read(seed)
	}
	key := sha256.Sum256(append([]byte(compactAAD+"\x00"), seed...))
	block, err := aes.NewCipher(key[:])
	if err != nil {
		panic(err)
	}
	aead, err := cipher.NewGCM(block)
	if err != nil {
		panic(err)
	}
	return &responseCompactCodec{aead: aead}
}

func (codec *responseCompactCodec) seal(payload compactPayload) (string, error) {
	plain, err := sonic.Marshal(payload)
	if err != nil {
		return "", err
	}
	nonce := make([]byte, codec.aead.NonceSize())
	if _, err := rand.Read(nonce); err != nil {
		return "", err
	}
	sealed := codec.aead.Seal(nil, nonce, plain, []byte(compactAAD))
	encoded := append(nonce, sealed...)
	return compactContentPrefix + base64.RawURLEncoding.EncodeToString(encoded), nil
}

func (codec *responseCompactCodec) open(value string) (compactPayload, error) {
	if !strings.HasPrefix(value, compactContentPrefix) {
		return compactPayload{}, errors.New("unknown compaction format")
	}
	data, err := base64.RawURLEncoding.DecodeString(strings.TrimPrefix(value, compactContentPrefix))
	if err != nil || len(data) < codec.aead.NonceSize() {
		return compactPayload{}, errors.New("invalid compaction payload")
	}
	nonce, ciphertext := data[:codec.aead.NonceSize()], data[codec.aead.NonceSize():]
	plain, err := codec.aead.Open(nil, nonce, ciphertext, []byte(compactAAD))
	if err != nil {
		return compactPayload{}, err
	}
	var payload compactPayload
	if err := sonic.Unmarshal(plain, &payload); err != nil {
		return compactPayload{}, err
	}
	if payload.Version != 1 || strings.TrimSpace(payload.Summary) == "" {
		return compactPayload{}, errors.New("unsupported compaction payload")
	}
	return payload, nil
}

func (api *ResponsesAPI) streamResponse(
	w http.ResponseWriter,
	r *http.Request,
	req model.ResponseRequest,
	chatReq model.ChatCompletionRequest,
	compactItem map[string]interface{},
	toolBindings responseToolBindings,
	stream func(context.Context, func(*proxy.CallResult) error) error,
) {
	setSSEHeaders(w)
	responseID := newResponseObjectID("resp")
	sequence := int64(0)
	writeEvent := func(event map[string]interface{}) error {
		event["sequence_number"] = sequence
		sequence++
		return writeResponsesSSEEvent(w, event)
	}

	created := buildResponseObject(req, responseID, "in_progress", []map[string]interface{}{}, nil)
	if err := writeEvent(map[string]interface{}{"type": "response.created", "response": created}); err != nil {
		return
	}
	if err := writeEvent(map[string]interface{}{"type": "response.in_progress", "response": created}); err != nil {
		return
	}

	output := make([]map[string]interface{}, 0)
	if compactItem != nil {
		index := len(output)
		if err := writeEvent(map[string]interface{}{"type": "response.output_item.added", "output_index": index, "item": compactItem}); err != nil {
			return
		}
		if err := writeEvent(map[string]interface{}{"type": "response.output_item.done", "output_index": index, "item": compactItem}); err != nil {
			return
		}
		output = append(output, compactItem)
	}

	aggregate := &proxy.CallResult{}
	commentaryID := ""
	commentaryIndex := -1
	var commentaryBuilder strings.Builder
	messageID := newResponseObjectID("msg")
	messageIndex := -1
	var textBuilder strings.Builder
	finishCommentary := func() error {
		if commentaryIndex < 0 {
			return nil
		}
		text := commentaryBuilder.String()
		part := map[string]interface{}{
			"type": "output_text", "text": text, "annotations": []interface{}{}, "logprobs": []interface{}{},
		}
		item := map[string]interface{}{
			"id": commentaryID, "type": "message", "status": "completed", "role": "assistant", "phase": "commentary",
			"content": []map[string]interface{}{part},
		}
		if err := writeEvent(map[string]interface{}{
			"type": "response.output_text.done", "item_id": commentaryID,
			"output_index": commentaryIndex, "content_index": 0, "text": text,
		}); err != nil {
			return err
		}
		if err := writeEvent(map[string]interface{}{
			"type": "response.content_part.done", "item_id": commentaryID,
			"output_index": commentaryIndex, "content_index": 0, "part": part,
		}); err != nil {
			return err
		}
		if err := writeEvent(map[string]interface{}{
			"type": "response.output_item.done", "output_index": commentaryIndex, "item": item,
		}); err != nil {
			return err
		}
		output[commentaryIndex] = item
		commentaryID = ""
		commentaryIndex = -1
		commentaryBuilder.Reset()
		return nil
	}
	ctx := r.Context()
	err := stream(ctx, func(result *proxy.CallResult) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if result == nil || result.IsEmpty() {
			return nil
		}
		accumulateCallResult(aggregate, result)
		content, reasoning := buildStreamResponseContent(result)
		if reasoning != "" {
			if commentaryIndex < 0 {
				commentaryID = newResponseObjectID("msg")
				commentaryIndex = len(output)
				item := map[string]interface{}{
					"id": commentaryID, "type": "message", "status": "in_progress", "role": "assistant", "phase": "commentary",
					"content": []interface{}{},
				}
				if err := writeEvent(map[string]interface{}{
					"type": "response.output_item.added", "output_index": commentaryIndex, "item": item,
				}); err != nil {
					return err
				}
				part := map[string]interface{}{
					"type": "output_text", "text": "", "annotations": []interface{}{}, "logprobs": []interface{}{},
				}
				if err := writeEvent(map[string]interface{}{
					"type": "response.content_part.added", "item_id": commentaryID,
					"output_index": commentaryIndex, "content_index": 0, "part": part,
				}); err != nil {
					return err
				}
				output = append(output, item)
			}
			commentaryBuilder.WriteString(reasoning)
			if err := writeEvent(map[string]interface{}{
				"type": "response.output_text.delta", "item_id": commentaryID,
				"output_index": commentaryIndex, "content_index": 0, "delta": reasoning,
			}); err != nil {
				return err
			}
		}
		text, _ := content.(string)
		if (text != "" || len(result.FunctionCalls) > 0) && commentaryIndex >= 0 {
			if err := finishCommentary(); err != nil {
				return err
			}
		}
		if text == "" {
			return nil
		}
		if messageIndex < 0 {
			messageIndex = len(output)
			item := map[string]interface{}{
				"id": messageID, "type": "message", "status": "in_progress", "role": "assistant", "phase": "final_answer", "content": []interface{}{},
			}
			if err := writeEvent(map[string]interface{}{"type": "response.output_item.added", "output_index": messageIndex, "item": item}); err != nil {
				return err
			}
			part := map[string]interface{}{"type": "output_text", "text": "", "annotations": []interface{}{}, "logprobs": []interface{}{}}
			if err := writeEvent(map[string]interface{}{
				"type": "response.content_part.added", "item_id": messageID, "output_index": messageIndex, "content_index": 0, "part": part,
			}); err != nil {
				return err
			}
			output = append(output, item)
		}
		textBuilder.WriteString(text)
		return writeEvent(map[string]interface{}{
			"type": "response.output_text.delta", "item_id": messageID, "output_index": messageIndex, "content_index": 0, "delta": text,
		})
	})
	if err != nil {
		if requestContextCanceled(ctx, err) {
			return
		}
		if !proxy.IsUpstreamErrorLogged(err) {
			log.Error().Str("err", upstreamLogError(api.vp, err)).Str("model", req.Model).Msg("OpenAI Responses stream failed")
		}
		_ = writeEvent(map[string]interface{}{
			"type": "error", "code": openAIErrorType(proxy.HTTPStatusForError(err)), "message": publicUpstreamErrorMessage(api.vp, err), "param": nil,
		})
		return
	}
	if err := finishCommentary(); err != nil {
		return
	}

	if messageIndex >= 0 {
		text := textBuilder.String()
		annotations := responseAnnotationsFromGrounding(aggregate.GroundingMetadata)
		for annotationIndex, annotation := range annotations {
			if err := writeEvent(map[string]interface{}{
				"type": "response.output_text.annotation.added", "item_id": messageID,
				"output_index": messageIndex, "content_index": 0, "annotation_index": annotationIndex,
				"annotation": annotation,
			}); err != nil {
				return
			}
		}
		part := map[string]interface{}{"type": "output_text", "text": text, "annotations": annotations, "logprobs": []interface{}{}}
		item := map[string]interface{}{
			"id": messageID, "type": "message", "status": "completed", "role": "assistant", "phase": "final_answer", "content": []map[string]interface{}{part},
		}
		if err := writeEvent(map[string]interface{}{
			"type": "response.output_text.done", "item_id": messageID, "output_index": messageIndex, "content_index": 0, "text": text,
		}); err != nil {
			return
		}
		if err := writeEvent(map[string]interface{}{
			"type": "response.content_part.done", "item_id": messageID, "output_index": messageIndex, "content_index": 0, "part": part,
		}); err != nil {
			return
		}
		if err := writeEvent(map[string]interface{}{"type": "response.output_item.done", "output_index": messageIndex, "item": item}); err != nil {
			return
		}
		output[messageIndex] = item
	}

	for _, call := range dedupeResponseFunctionCalls(aggregate.FunctionCalls) {
		index := len(output)
		doneItem := responseToolCallItem(call, toolBindings[call.Name], "completed")
		addedItem := copyInterfaceMap(doneItem)
		addedItem["status"] = "in_progress"
		if err := writeEvent(map[string]interface{}{"type": "response.output_item.added", "output_index": index, "item": addedItem}); err != nil {
			return
		}
		switch doneItem["type"] {
		case "function_call":
			arguments, _ := doneItem["arguments"].(string)
			name, _ := doneItem["name"].(string)
			if err := writeEvent(map[string]interface{}{
				"type": "response.function_call_arguments.delta", "item_id": doneItem["id"], "output_index": index, "delta": arguments,
			}); err != nil {
				return
			}
			if err := writeEvent(map[string]interface{}{
				"type": "response.function_call_arguments.done", "item_id": doneItem["id"], "output_index": index, "name": name, "arguments": arguments,
			}); err != nil {
				return
			}
		case "custom_tool_call":
			input, _ := doneItem["input"].(string)
			if err := writeEvent(map[string]interface{}{
				"type": "response.custom_tool_call_input.delta", "item_id": doneItem["id"], "output_index": index, "delta": input,
			}); err != nil {
				return
			}
			if err := writeEvent(map[string]interface{}{
				"type": "response.custom_tool_call_input.done", "item_id": doneItem["id"], "output_index": index, "input": input,
			}); err != nil {
				return
			}
		}
		if err := writeEvent(map[string]interface{}{"type": "response.output_item.done", "output_index": index, "item": doneItem}); err != nil {
			return
		}
		output = append(output, doneItem)
	}

	for _, searchCall := range responseWebSearchCallItems(aggregate.GroundingMetadata) {
		index := len(output)
		added := copyInterfaceMap(searchCall)
		added["status"] = "in_progress"
		if err := writeEvent(map[string]interface{}{"type": "response.output_item.added", "output_index": index, "item": added}); err != nil {
			return
		}
		if err := writeEvent(map[string]interface{}{"type": "response.output_item.done", "output_index": index, "item": searchCall}); err != nil {
			return
		}
		output = append(output, searchCall)
	}

	usage, _ := responseUsage(chatReq, aggregate)
	if len(buildResponseOutput(aggregate, toolBindings)) == 0 {
		log.Error().
			Str("model", req.Model).
			Str("finish_reason", aggregate.FinishReason).
			Msg("OpenAI Responses stream completed without assistant output")
		failed := buildResponseObject(req, responseID, "failed", output, &usage)
		failed["error"] = map[string]interface{}{
			"code":    "server_error",
			"message": proxy.ErrNoAssistantOutput.Error(),
		}
		_ = writeEvent(map[string]interface{}{"type": "response.failed", "response": failed})
		return
	}
	completed := buildResponseObject(req, responseID, "completed", output, &usage)
	_ = writeEvent(map[string]interface{}{"type": "response.completed", "response": completed})
}

func writeResponsesSSEEvent(w http.ResponseWriter, event map[string]interface{}) error {
	eventType, _ := event["type"].(string)
	data, err := sonic.Marshal(event)
	if err != nil {
		return err
	}
	if _, err := io.WriteString(w, "event: "+eventType+"\n"); err != nil {
		return err
	}
	if _, err := io.WriteString(w, "data: "); err != nil {
		return err
	}
	if _, err := w.Write(data); err != nil {
		return err
	}
	if _, err := io.WriteString(w, "\n\n"); err != nil {
		return err
	}
	return flushResponse(w)
}
