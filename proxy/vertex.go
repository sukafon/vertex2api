package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
	"unicode"

	"vertex2api/client"
	"vertex2api/config"
	"vertex2api/model"
	"vertex2api/recaptcha"
	schemanorm "vertex2api/schema"
	"vertex2api/stats"

	"github.com/bytedance/sonic"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const (
	vertexAPIPath                       = "/v3/entityServices/AiplatformEntityService/schemas/AIPLATFORM_GRAPHQL:batchGraphql"
	streamGenerateContentQuerySignature = "2/l8eCsMMY49imcDQ/lwwXyL8cYtTjxZBF2dNqy69LodY="
	streamGenerateContentOperationName  = "StreamGenerateContentAnonymous"
	finishReasonUnspecified             = "FINISH_REASON_UNSPECIFIED"
	recaptchaVerifyActionMessage        = "Failed to verify action"
	recaptchaTokenInvalidMessage        = "Recaptcha token is invalid"
	thoughtSignatureInvalidMessage      = "Thought signature is not valid"
	skipThoughtSignatureValidator       = "skip_thought_signature_validator"
)

// VertexProxy 封装 Vertex AI 的调用逻辑
type VertexProxy struct {
	httpClient *client.HTTPClient
	tokenCache *recaptcha.TokenCache
	cfg        *config.Config
}

func NewVertexProxy(httpClient *client.HTTPClient, tokenCache *recaptcha.TokenCache, cfg *config.Config) *VertexProxy {
	return &VertexProxy{
		httpClient: httpClient,
		tokenCache: tokenCache,
		cfg:        cfg,
	}
}

func (vp *VertexProxy) selectedVertexBaseURL() string {
	prefixes := vp.cfg.PrefixVertexBaseURLs
	if len(prefixes) == 0 {
		return vp.cfg.VertexBaseURL
	}

	return selectVertexBaseURL(vp.cfg.VertexBaseURL, prefixes, rand.Intn(2) == 0, rand.Intn(len(prefixes)))
}

func selectVertexBaseURL(baseURL string, prefixes []string, direct bool, prefixIndex int) string {
	if direct || len(prefixes) == 0 {
		return baseURL
	}
	return prefixes[prefixIndex%len(prefixes)] + baseURL
}

func buildVertexAPIURL(baseURL, apiKey string) string {
	return fmt.Sprintf("%s%s?key=%s&prettyPrint=false", strings.TrimRight(baseURL, "/"), vertexAPIPath, url.QueryEscape(apiKey))
}

// CallResult 封装调用结果
type CallResult struct {
	TextParts         []model.TextPart
	ImageParts        []model.InlineData
	FunctionCalls     []model.FunctionCall
	FinishReason      string
	GroundingMetadata *model.GroundingMetadata
	Candidates        []CandidateResult
	UsageMetadata     map[string]interface{}
	ModelVersion      string
	ResponseID        string
	CreateTime        string
	PromptFeedback    map[string]interface{}
	ModelStatus       map[string]interface{}
}

// CandidateResult retains candidate boundaries for native Gemini responses.
// Top-level fields mirror the first candidate for single-candidate adapters;
// native Gemini and non-streaming OpenAI responses consume Candidates directly.
type CandidateResult struct {
	Index              int
	TextParts          []model.TextPart
	ImageParts         []model.InlineData
	FunctionCalls      []model.FunctionCall
	FinishReason       string
	FinishMessage      string
	GroundingMetadata  *model.GroundingMetadata
	SafetyRatings      []map[string]interface{}
	CitationMetadata   map[string]interface{}
	URLContextMetadata map[string]interface{}
	LogprobsResult     map[string]interface{}
	AvgLogprobs        *float64
}

func (r *CallResult) HasContent() bool {
	if r == nil {
		return false
	}
	return len(r.TextParts) > 0 || len(r.ImageParts) > 0 || len(r.FunctionCalls) > 0 || r.GroundingMetadata != nil
}

func (r *CallResult) HasMetadata() bool {
	if r == nil {
		return false
	}
	return len(r.UsageMetadata) > 0 || r.ModelVersion != "" || r.ResponseID != "" || r.CreateTime != "" ||
		len(r.PromptFeedback) > 0 || len(r.ModelStatus) > 0
}

func (r *CallResult) IsEmpty() bool {
	if r == nil {
		return true
	}
	return !r.HasContent() && !r.HasMetadata() && r.FinishReason == ""
}

func (r *CallResult) IsFinishOnly() bool {
	return r != nil && !r.HasContent() && !r.HasMetadata() && r.FinishReason != ""
}

// BuildVertexBody 构建 Vertex AI GraphQL 请求体
func BuildVertexBody(
	modelName string,
	contents []map[string]interface{},
	genConfig map[string]interface{},
	safetySettings []map[string]string,
	systemInstruction interface{},
	recaptchaToken string,
) ([]byte, error) {
	return BuildVertexBodyWithOptions(modelName, contents, genConfig, safetySettings, systemInstruction, recaptchaToken, nil)
}

type VertexRequestOptions struct {
	Tools      interface{}
	ToolConfig interface{}
}

func BuildVertexBodyWithOptions(
	modelName string,
	contents []map[string]interface{},
	genConfig map[string]interface{},
	safetySettings []map[string]string,
	systemInstruction interface{},
	recaptchaToken string,
	options *VertexRequestOptions,
) ([]byte, error) {
	req := model.AcquireVertexRequest()
	defer model.ReleaseVertexRequest(req)

	// Always use Vertex's streaming operation upstream. Handlers decide whether
	// downstream clients receive SSE chunks or an aggregated JSON response.
	req.QuerySignature = streamGenerateContentQuerySignature
	req.OperationName = streamGenerateContentOperationName
	normalizedContents, err := normalizeContents(modelName, contents)
	if err != nil {
		return nil, err
	}
	req.Variables["model"] = modelName
	req.Variables["contents"] = normalizedContents
	req.Variables["region"] = "global"
	req.Variables["recaptchaToken"] = recaptchaToken

	if genConfig == nil {
		genConfig = map[string]interface{}{}
	}

	sanitizeThinkingConfig(modelName, genConfig)
	model.SanitizeGenerationConfigResponseModalities(modelName, genConfig)

	if len(genConfig) > 0 {
		req.Variables["generationConfig"] = genConfig
	}

	// safetySettings
	if safetySettings != nil {
		req.Variables["safetySettings"] = safetySettings
	} else {
		req.Variables["safetySettings"] = defaultSafetySettings()
	}

	// systemInstruction
	if systemInstruction != nil {
		req.Variables["systemInstruction"] = systemInstruction
	}
	if options != nil {
		if options.Tools != nil {
			if tools := normalizeTools(options.Tools); tools != nil {
				req.Variables["tools"] = tools
			}
		}
		if options.ToolConfig != nil {
			req.Variables["toolConfig"] = options.ToolConfig
		}
	}

	return sonic.Marshal(req)
}

func sanitizeThinkingConfig(modelName string, genConfig map[string]interface{}) {
	thinkingConfig, ok := genConfig["thinkingConfig"].(map[string]interface{})
	if !ok {
		return
	}

	normalizeThinkingLevel(thinkingConfig)

	isGemini3 := strings.HasPrefix(modelName, "gemini-3")
	isGemini25 := strings.HasPrefix(modelName, "gemini-2.5")
	_, hasThinkingLevel := thinkingConfig["thinkingLevel"]
	_, hasThinkingBudget := thinkingConfig["thinkingBudget"]

	if !isGemini3 {
		delete(thinkingConfig, "thinkingLevel")
	}
	if (!isGemini25 && !isGemini3) || (isGemini3 && hasThinkingLevel && hasThinkingBudget) {
		delete(thinkingConfig, "thinkingBudget")
	}

	if len(thinkingConfig) == 0 {
		delete(genConfig, "thinkingConfig")
	}
}

func normalizeThinkingLevel(thinkingConfig map[string]interface{}) {
	value, ok := thinkingConfig["thinkingLevel"]
	if !ok {
		return
	}

	level, ok := value.(string)
	if !ok {
		return
	}

	thinkingConfig["thinkingLevel"] = strings.ToUpper(strings.TrimSpace(level))
}

func normalizeContents(modelName string, contents []map[string]interface{}) ([]map[string]interface{}, error) {
	normalized := make([]map[string]interface{}, 0, len(contents))
	for i, content := range contents {
		contentCopy := copyMap(content)
		contentCopy["role"] = normalizeRole(fmt.Sprintf("%v", contentCopy["role"]))
		partsValue, ok := contentCopy["parts"]
		if !ok {
			normalized = append(normalized, contentCopy)
			continue
		}

		parts, ok := toInterfaceSlice(partsValue)
		if !ok {
			return nil, fmt.Errorf("contents[%d].parts must be an array", i)
		}

		normalizedParts := make([]interface{}, 0, len(parts))
		for j, partValue := range parts {
			part, ok := partValue.(map[string]interface{})
			if !ok {
				return nil, fmt.Errorf("contents[%d].parts[%d] must be an object", i, j)
			}
			normalizedPart, err := normalizePart(part)
			if err != nil {
				return nil, fmt.Errorf("contents[%d].parts[%d]: %w", i, j, err)
			}
			normalizedParts = append(normalizedParts, normalizedPart)
		}
		contentCopy["parts"] = normalizedParts
		if requiresFunctionCallThoughtSignature(modelName) {
			ensureFunctionCallThoughtSignature(normalizedParts)
		}
		normalized = append(normalized, contentCopy)
	}
	normalized = trimTrailingEmptyModelTurns(normalized)
	if len(normalized) == 0 {
		return nil, errors.New("contents must contain at least one non-empty turn")
	}
	if requiresTrailingUserTurn(modelName) {
		normalized = trimTrailingEmptyTurns(normalized)
		normalized = dropTrailingModelPrefill(modelName, normalized)
		if len(normalized) == 0 {
			return nil, errors.New("gemini 3.6 requests must contain a user turn")
		}
	}
	return normalized, nil
}

func normalizeRole(role string) string {
	r := strings.ToLower(strings.TrimSpace(role))
	if r == "assistant" || r == "model" {
		return "model"
	}
	return "user"
}

func trimTrailingEmptyModelTurns(contents []map[string]interface{}) []map[string]interface{} {
	for len(contents) > 0 {
		lastIndex := len(contents) - 1
		lastRole := normalizeRole(fmt.Sprintf("%v", contents[lastIndex]["role"]))
		if lastRole != "model" {
			break
		}

		if !contentTurnHasContent(contents[lastIndex]) {
			contents = contents[:lastIndex]
			continue
		}
		break
	}
	return contents
}

func trimTrailingEmptyTurns(contents []map[string]interface{}) []map[string]interface{} {
	for len(contents) > 0 && !contentTurnHasContent(contents[len(contents)-1]) {
		contents = contents[:len(contents)-1]
	}
	return contents
}

func contentTurnHasContent(content map[string]interface{}) bool {
	parts, ok := toInterfaceSlice(content["parts"])
	if !ok || len(parts) == 0 {
		return false
	}

	for _, part := range parts {
		pMap, ok := part.(map[string]interface{})
		if !ok {
			continue
		}
		if text, ok := pMap["text"].(string); ok && strings.TrimSpace(text) != "" {
			return true
		}
		for key, value := range pMap {
			if key != "text" && value != nil {
				return true
			}
		}
	}
	return false
}

func requiresTrailingUserTurn(modelName string) bool {
	modelName = strings.ToLower(strings.TrimSpace(modelName))
	return strings.HasPrefix(modelName, "gemini-3.6")
}

// dropTrailingModelPrefill adapts Gemini 3.6's generateContent contract,
// which disallows prefilled model turns. Other models are unchanged.
func dropTrailingModelPrefill(modelName string, contents []map[string]interface{}) []map[string]interface{} {
	for len(contents) > 0 {
		last := contents[len(contents)-1]
		if normalizeRole(fmt.Sprintf("%v", last["role"])) != "model" {
			break
		}
		userText := ""
		for i := len(contents) - 2; i >= 0; i-- {
			if normalizeRole(fmt.Sprintf("%v", contents[i]["role"])) == "user" {
				userText = contentTurnText(contents[i])
				break
			}
		}
		log.Debug().
			Str("model", modelName).
			Str("model_text", config.UpstreamLogValue(contentTurnText(last), false, 1024)).
			Str("user_text", config.UpstreamLogValue(userText, false, 1024)).
			Msg("Removed trailing Gemini 3.6 model turn before upstream request")
		contents = contents[:len(contents)-1]
	}
	return contents
}

func contentTurnText(content map[string]interface{}) string {
	parts, ok := toInterfaceSlice(content["parts"])
	if !ok {
		return ""
	}

	var text strings.Builder
	for _, partValue := range parts {
		part, ok := partValue.(map[string]interface{})
		if !ok {
			continue
		}
		if value, ok := part["text"].(string); ok {
			text.WriteString(value)
		}
	}
	return text.String()
}

func requiresFunctionCallThoughtSignature(modelName string) bool {
	return strings.Contains(modelName, "gemini-3")
}
func normalizePart(part map[string]interface{}) (map[string]interface{}, error) {
	partCopy := copyMap(part)
	if err := normalizeThoughtSignature(partCopy); err != nil {
		return nil, err
	}

	inlineValue, hasInline := partCopy["inlineData"]
	if !hasInline {
		return partCopy, nil
	}

	inlineData, ok := inlineValue.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("inlineData must be an object")
	}

	normalizedInline, err := normalizeInlineData(inlineData)
	if err != nil {
		return nil, err
	}
	partCopy["inlineData"] = normalizedInline
	return partCopy, nil
}

func normalizeThoughtSignature(part map[string]interface{}) error {
	signature, ok := part["thoughtSignature"]
	if !ok {
		return nil
	}

	signatureString, ok := signature.(string)
	if !ok {
		return fmt.Errorf("thoughtSignature must be a base64 string")
	}
	signatureString = strings.TrimSpace(signatureString)
	if signatureString == "" {
		delete(part, "thoughtSignature")
		return nil
	}
	if signatureString == skipThoughtSignatureValidator {
		part["thoughtSignature"] = thoughtSignatureBypassValue()
		return nil
	}

	decoded, err := decodeBase64(removeWhitespace(signatureString))
	if err != nil {
		delete(part, "thoughtSignature")
		return nil
	}
	if string(decoded) == skipThoughtSignatureValidator {
		part["thoughtSignature"] = thoughtSignatureBypassValue()
		return nil
	}
	part["thoughtSignature"] = base64.StdEncoding.EncodeToString(decoded)
	return nil
}

func thoughtSignatureBypassValue() string {
	return base64.StdEncoding.EncodeToString([]byte(skipThoughtSignatureValidator))
}
func ensureFunctionCallThoughtSignature(parts []interface{}) {
	for _, partValue := range parts {
		part, ok := partValue.(map[string]interface{})
		if !ok {
			continue
		}
		if _, ok := part["functionCall"]; !ok {
			continue
		}
		// Preserve a real upstream signature when one is already present.
		if signature, ok := part["thoughtSignature"].(string); ok && strings.TrimSpace(signature) != "" {
			continue
		}
		part["thoughtSignature"] = thoughtSignatureBypassValue()
	}
}

func normalizeTools(tools interface{}) interface{} {
	switch v := tools.(type) {
	case []interface{}:
		return normalizeToolSlice(v)
	case []map[string]interface{}:
		items := make([]interface{}, 0, len(v))
		for _, item := range v {
			items = append(items, item)
		}
		return normalizeToolSlice(items)
	case map[string]interface{}:
		tool, ok := normalizeToolMap(v)
		if !ok {
			return nil
		}
		return []interface{}{tool}
	default:
		return normalizeToolValue(tools, false)
	}
}

func normalizeToolSlice(tools []interface{}) interface{} {
	normalized := make([]interface{}, 0, len(tools))
	for _, item := range tools {
		tool, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		normalizedTool, ok := normalizeToolMap(tool)
		if !ok {
			continue
		}
		normalized = append(normalized, normalizedTool)
	}
	if len(normalized) == 0 {
		return nil
	}
	return normalized
}

func normalizeToolMap(tool map[string]interface{}) (map[string]interface{}, bool) {
	normalized := make(map[string]interface{}, len(tool))
	hasToolType := false
	for key, item := range tool {
		if !isToolTypeField(key) {
			continue
		}
		if key == "functionDeclarations" {
			value := normalizeFunctionDeclarations(item)
			if !isEmptyToolTypeValue(key, value) {
				normalized[key] = value
				hasToolType = true
			}
			continue
		}
		value := normalizeToolValue(item, false)
		if isEmptyToolTypeValue(key, value) {
			continue
		}
		normalized[key] = value
		hasToolType = true
	}
	return normalized, hasToolType
}

func normalizeFunctionDeclarations(value interface{}) interface{} {
	items, ok := toInterfaceSlice(value)
	if !ok {
		return normalizeToolValue(value, false)
	}
	result := make([]interface{}, 0, len(items))
	for _, item := range items {
		declaration, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		normalized := copyMap(declaration)
		if parameters, ok := normalized["parameters"]; ok {
			normalized["parametersJsonSchema"] = schemanorm.Normalize(parameters)
			delete(normalized, "parameters")
		}
		if parameters, ok := normalized["parametersJsonSchema"]; ok {
			normalized["parametersJsonSchema"] = schemanorm.Normalize(parameters)
		}
		if response, ok := normalized["response"]; ok {
			normalized["responseJsonSchema"] = schemanorm.Normalize(response)
			delete(normalized, "response")
		}
		if response, ok := normalized["responseJsonSchema"]; ok {
			normalized["responseJsonSchema"] = schemanorm.Normalize(response)
		}
		result = append(result, normalizeToolValue(normalized, false))
	}
	return result
}

func isToolTypeField(key string) bool {
	switch key {
	case "functionDeclarations", "retrieval", "googleSearchRetrieval", "googleSearch", "enterpriseWebSearch", "codeExecution", "urlContext":
		return true
	default:
		return false
	}
}

func isEmptyToolTypeValue(key string, value interface{}) bool {
	if value == nil {
		return true
	}
	if key != "functionDeclarations" {
		return false
	}
	switch declarations := value.(type) {
	case []interface{}:
		return len(declarations) == 0
	case []map[string]interface{}:
		return len(declarations) == 0
	default:
		return false
	}
}

func normalizeToolValue(value interface{}, schema bool) interface{} {
	switch v := value.(type) {
	case map[string]interface{}:
		normalized := make(map[string]interface{}, len(v))
		for key, item := range v {
			switch {
			case isJSONSchemaField(key):
				normalized[key] = normalizeJSONSchemaValue(item)
			case schema && key == "type":
				normalized[key] = normalizeSchemaType(item)
			case schema && key == "properties":
				normalized[key] = normalizeSchemaProperties(item)
			case schema && key == "items":
				normalized[key] = normalizeSchemaItems(item)
			case key == "items":
				normalized[key] = normalizeJSONSchemaItems(item)
			case isSchemaField(key):
				normalized[key] = normalizeToolValue(item, true)
			default:
				normalized[key] = normalizeToolValue(item, schema)
			}
		}
		return normalized
	case []interface{}:
		normalized := make([]interface{}, 0, len(v))
		for _, item := range v {
			normalized = append(normalized, normalizeToolValue(item, schema))
		}
		return normalized
	case []map[string]interface{}:
		normalized := make([]interface{}, 0, len(v))
		for _, item := range v {
			normalized = append(normalized, normalizeToolValue(item, schema))
		}
		return normalized
	default:
		return value
	}
}

func normalizeSchemaType(value interface{}) interface{} {
	typeValue, ok := value.(string)
	if !ok {
		return value
	}
	switch strings.ToLower(strings.TrimSpace(typeValue)) {
	case "object", "array", "string", "number", "integer", "boolean", "null":
		return strings.ToUpper(strings.TrimSpace(typeValue))
	default:
		return value
	}
}

func normalizeSchemaProperties(value interface{}) interface{} {
	switch properties := value.(type) {
	case map[string]interface{}:
		keys := make([]string, 0, len(properties))
		for key := range properties {
			keys = append(keys, key)
		}
		sort.Strings(keys)

		entries := make([]map[string]interface{}, 0, len(keys))
		for _, key := range keys {
			entries = append(entries, map[string]interface{}{
				"key":   key,
				"value": normalizeToolValue(properties[key], true),
			})
		}
		return entries
	case []interface{}:
		entries := make([]interface{}, 0, len(properties))
		for _, item := range properties {
			entry, ok := item.(map[string]interface{})
			if !ok {
				entries = append(entries, normalizeToolValue(item, true))
				continue
			}
			entryCopy := copyMap(entry)
			if value, ok := entryCopy["value"]; ok {
				entryCopy["value"] = normalizeToolValue(value, true)
			}
			entries = append(entries, entryCopy)
		}
		return entries
	default:
		return value
	}
}

func normalizeJSONSchemaValue(value interface{}) interface{} {
	switch v := value.(type) {
	case map[string]interface{}:
		normalized := make(map[string]interface{}, len(v))
		for key, item := range v {
			switch key {
			case "items":
				normalized[key] = normalizeJSONSchemaItems(item)
			case "properties":
				normalized[key] = normalizeJSONSchemaProperties(item)
			default:
				normalized[key] = normalizeJSONSchemaValue(item)
			}
		}
		return normalized
	case []interface{}:
		normalized := make([]interface{}, 0, len(v))
		for _, item := range v {
			normalized = append(normalized, normalizeJSONSchemaValue(item))
		}
		return normalized
	case []map[string]interface{}:
		normalized := make([]interface{}, 0, len(v))
		for _, item := range v {
			normalized = append(normalized, normalizeJSONSchemaValue(item))
		}
		return normalized
	default:
		return value
	}
}

func normalizeJSONSchemaItems(value interface{}) interface{} {
	switch items := value.(type) {
	case []interface{}:
		for _, item := range items {
			if schema := normalizeJSONSchemaObject(item); len(schema) > 0 {
				return schema
			}
		}
		return map[string]interface{}{}
	case []map[string]interface{}:
		for _, item := range items {
			if schema := normalizeJSONSchemaObject(item); len(schema) > 0 {
				return schema
			}
		}
		return map[string]interface{}{}
	default:
		return normalizeJSONSchemaObject(value)
	}
}

func normalizeJSONSchemaProperties(value interface{}) interface{} {
	switch properties := value.(type) {
	case []interface{}:
		normalized := make(map[string]interface{}, len(properties))
		for _, item := range properties {
			entry, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			key, ok := entry["key"].(string)
			if !ok || key == "" {
				continue
			}
			normalized[key] = normalizeJSONSchemaValue(entry["value"])
		}
		return normalized
	case []map[string]interface{}:
		normalized := make(map[string]interface{}, len(properties))
		for _, entry := range properties {
			key, ok := entry["key"].(string)
			if !ok || key == "" {
				continue
			}
			normalized[key] = normalizeJSONSchemaValue(entry["value"])
		}
		return normalized
	case map[string]interface{}:
		normalized := make(map[string]interface{}, len(properties))
		for key, item := range properties {
			normalized[key] = normalizeJSONSchemaValue(item)
		}
		return normalized
	default:
		return value
	}
}

func normalizeJSONSchemaObject(value interface{}) map[string]interface{} {
	normalized, ok := normalizeJSONSchemaValue(value).(map[string]interface{})
	if !ok {
		return map[string]interface{}{}
	}
	return normalized
}

func normalizeSchemaItems(value interface{}) interface{} {
	switch items := value.(type) {
	case []interface{}:
		for _, item := range items {
			if schema := normalizeSchemaObject(item); len(schema) > 0 {
				return schema
			}
		}
		return map[string]interface{}{}
	case []map[string]interface{}:
		for _, item := range items {
			if schema := normalizeSchemaObject(item); len(schema) > 0 {
				return schema
			}
		}
		return map[string]interface{}{}
	default:
		return normalizeSchemaObject(value)
	}
}

func normalizeSchemaObject(value interface{}) map[string]interface{} {
	normalized, ok := normalizeToolValue(value, true).(map[string]interface{})
	if !ok {
		return map[string]interface{}{}
	}
	return normalized
}

func isJSONSchemaField(key string) bool {
	switch key {
	case "parametersJsonSchema", "responseJsonSchema":
		return true
	default:
		return false
	}
}

func isSchemaField(key string) bool {
	switch key {
	case "parameters", "response", "items", "additionalProperties":
		return true
	default:
		return false
	}
}

func normalizeInlineData(inlineData map[string]interface{}) (map[string]interface{}, error) {
	mimeType := stringValue(inlineData, "mimeType")
	data := stringValue(inlineData, "data")
	if data == "" {
		return nil, fmt.Errorf("inlineData.data is required")
	}

	normalizedMime, normalizedData, err := normalizeBytesData(mimeType, data)
	if err != nil {
		return nil, err
	}
	if normalizedMime == "" {
		normalizedMime = "image/png"
	}

	return map[string]interface{}{
		"mimeType": normalizedMime,
		"data":     normalizedData,
	}, nil
}

func normalizeBytesData(mimeType, data string) (string, string, error) {
	data = strings.TrimSpace(data)
	if data == "" {
		return mimeType, "", fmt.Errorf("inlineData.data is empty")
	}

	lowerData := strings.ToLower(data)
	if strings.HasPrefix(lowerData, "http://") || strings.HasPrefix(lowerData, "https://") {
		return mimeType, "", fmt.Errorf("inlineData.data must be base64 bytes, got remote URL")
	}

	if strings.HasPrefix(lowerData, "data:") {
		parsedMime, payload, err := splitDataURL(data)
		if err != nil {
			return mimeType, "", err
		}
		if parsedMime != "" {
			mimeType = parsedMime
		}
		data = payload
	}

	data = removeWhitespace(data)
	decoded, err := decodeBase64(data)
	if err != nil {
		return mimeType, "", fmt.Errorf("inlineData.data must be valid base64 bytes: %w", err)
	}
	return mimeType, base64.StdEncoding.EncodeToString(decoded), nil
}

func splitDataURL(dataURL string) (string, string, error) {
	comma := strings.IndexByte(dataURL, ',')
	if comma < 0 {
		return "", "", fmt.Errorf("inlineData.data has invalid data URL")
	}

	meta := dataURL[len("data:"):comma]
	payload := dataURL[comma+1:]
	if payload == "" {
		return "", "", fmt.Errorf("inlineData.data has empty data URL payload")
	}

	var mimeType string
	isBase64 := false
	for idx, item := range strings.Split(meta, ";") {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if idx == 0 && strings.Contains(item, "/") {
			mimeType = item
			continue
		}
		if strings.EqualFold(item, "base64") {
			isBase64 = true
		}
	}
	if !isBase64 {
		return mimeType, "", fmt.Errorf("inlineData.data data URL must be base64 encoded")
	}
	return mimeType, payload, nil
}

func decodeBase64(data string) ([]byte, error) {
	encodings := []*base64.Encoding{
		base64.StdEncoding,
		base64.RawStdEncoding,
		base64.URLEncoding,
		base64.RawURLEncoding,
	}
	var lastErr error
	for _, encoding := range encodings {
		decoded, err := encoding.DecodeString(data)
		if err == nil {
			return decoded, nil
		}
		lastErr = err
	}
	return nil, lastErr
}

func removeWhitespace(s string) string {
	if strings.IndexFunc(s, unicode.IsSpace) < 0 {
		return s
	}
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if !unicode.IsSpace(r) {
			b.WriteRune(r)
		}
	}
	return b.String()
}

func toInterfaceSlice(v interface{}) ([]interface{}, bool) {
	switch items := v.(type) {
	case []interface{}:
		return items, true
	case []map[string]interface{}:
		result := make([]interface{}, 0, len(items))
		for _, item := range items {
			result = append(result, item)
		}
		return result, true
	default:
		return nil, false
	}
}

func copyMap(src map[string]interface{}) map[string]interface{} {
	dst := make(map[string]interface{}, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func stringValue(m map[string]interface{}, key string) string {
	if value, ok := m[key].(string); ok {
		return value
	}
	return ""
}

func sleepContext(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return ctx.Err()
	}

	timer := time.NewTimer(d)
	defer timer.Stop()

	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}

func (vp *VertexProxy) replaceRecaptchaTokenForRetry(ctx context.Context, bodyJSON []byte, oldLease *recaptcha.TokenLease) ([]byte, *recaptcha.TokenLease, error) {
	if vp == nil || vp.tokenCache == nil {
		return bodyJSON, oldLease, errors.New("recaptcha token cache is unavailable")
	}
	newLease, err := vp.tokenCache.GetTokenContext(ctx)
	if err != nil {
		return bodyJSON, oldLease, fmt.Errorf("get recaptcha token for retry: %w", err)
	}

	newBodyJSON, err := replaceRecaptchaToken(bodyJSON, newLease.Token())
	if err != nil {
		newLease.Release()
		return bodyJSON, oldLease, fmt.Errorf("replace recaptcha token for retry: %w", err)
	}

	if oldLease != nil {
		oldLease.Retire()
	}
	return newBodyJSON, newLease, nil
}

type recaptchaRetryReason string

const (
	recaptchaRetryNone    recaptchaRetryReason = ""
	recaptchaVerifyFailed recaptchaRetryReason = "token_verify_failed"
	recaptchaTokenInvalid recaptchaRetryReason = "token_invalid"
)

// classifyRecaptchaRetryError deliberately recognizes only the three code-3
// responses that identify a reCAPTCHA token problem. Other code-3 responses,
// such as invalid message turns or model arguments, must be returned without
// retrying or refreshing a token.
func classifyRecaptchaRetryError(err error) recaptchaRetryReason {
	if err == nil {
		return recaptchaRetryNone
	}

	message := err.Error()
	var vertexErr *vertexAPIError
	if errors.As(err, &vertexErr) && vertexErr.Message != "" {
		message = vertexErr.Message
	}
	lower := strings.ToLower(strings.TrimSpace(message))

	// Observed response: "Failed to verify action".
	if strings.Contains(lower, strings.ToLower(recaptchaVerifyActionMessage)) {
		return recaptchaVerifyFailed
	}

	// Observed response for both expired and invalid tokens:
	// "Recaptcha token is invalid, please refresh ...".
	if strings.Contains(lower, strings.ToLower(recaptchaTokenInvalidMessage)) {
		return recaptchaTokenInvalid
	}

	return recaptchaRetryNone
}

func isRetryableVertexError(status int, err error) bool {
	if status == 8 {
		return true
	}
	// Code 3 is retryable on the current token only for verification failures.
	// Expired and invalid tokens are handled by the callers as immediate refreshes.
	return status == 3 && classifyRecaptchaRetryError(err) == recaptchaVerifyFailed
}

func isThoughtSignatureInvalidError(err error) bool {
	if err == nil {
		return false
	}
	message := strings.ToLower(err.Error())
	return strings.Contains(message, strings.ToLower(thoughtSignatureInvalidMessage)) ||
		strings.Contains(message, "thought_signature") ||
		strings.Contains(message, "thought signature") ||
		strings.Contains(message, "thoughtsignature")
}

type vertexAPIError struct {
	Code    int
	Message string
}

func (e *vertexAPIError) Error() string {
	return fmt.Sprintf("API Error (Code %d): %s", e.Code, e.Message)
}

type upstreamHTTPError struct {
	Status int
	Body   string
}

func (e *upstreamHTTPError) Error() string {
	return fmt.Sprintf("HTTP %d: %s", e.Status, e.Body)
}

// HTTPStatusForError maps the private GraphQL/gRPC status space and transport
// status codes to their public HTTP equivalents. Protocol adapters use this
// only before streaming response headers have been committed.
func HTTPStatusForError(err error) int {
	var httpErr *upstreamHTTPError
	if errors.As(err, &httpErr) && httpErr.Status >= 400 && httpErr.Status <= 599 {
		return httpErr.Status
	}

	var vertexErr *vertexAPIError
	if errors.As(err, &vertexErr) {
		switch vertexErr.Code {
		case 3:
			return http.StatusBadRequest
		case 5:
			return http.StatusNotFound
		case 7:
			return http.StatusForbidden
		case 8:
			return http.StatusTooManyRequests
		case 16:
			return http.StatusUnauthorized
		case 14:
			return http.StatusServiceUnavailable
		default:
			return http.StatusInternalServerError
		}
	}
	return http.StatusInternalServerError
}

// CallContext invokes Vertex AI with retries. The first "Failed to verify
// action" response for a token is treated as a warm-up error.
func (vp *VertexProxy) CallContext(ctx context.Context, bodyJSON []byte) (*CallResult, error) {
	return vp.call(ctx, bodyJSON, nil)
}

func (vp *VertexProxy) maxRetry() int {
	if vp.cfg == nil || vp.cfg.MaxRetry <= 0 {
		return 3
	}
	return vp.cfg.MaxRetry
}

func (vp *VertexProxy) maxRefresh() int {
	if vp.cfg != nil && vp.cfg.MaxRefresh > 0 {
		return vp.cfg.MaxRefresh
	}
	return vp.maxRetry()
}

func (vp *VertexProxy) retryDelay() time.Duration {
	if vp == nil || vp.cfg == nil || vp.cfg.RetryDelayMs <= 0 {
		return 0
	}
	return time.Duration(vp.cfg.RetryDelayMs) * time.Millisecond
}

func (vp *VertexProxy) upstreamLogError(err error) string {
	redact := vp != nil && vp.cfg != nil && vp.cfg.RedactUpstreamLogs
	return config.UpstreamLogError(err, redact, 120)
}

// withUpstreamError keeps the structured error field while applying the same
// redaction policy used by the existing upstream log messages. Passing a
// sanitized error to zerolog avoids leaking the raw upstream response when
// REDACT_UPSTREAM_LOGS is enabled.
func (vp *VertexProxy) withUpstreamError(event *zerolog.Event, err error) *zerolog.Event {
	if err == nil {
		return event
	}
	return event.Err(errors.New(vp.upstreamLogError(err)))
}

// UpstreamLogError applies the configured upstream error logging policy to
// errors surfaced to protocol handlers.
func (vp *VertexProxy) UpstreamLogError(err error) string {
	return vp.upstreamLogError(err)
}

func extractModelName(bodyJSON []byte) string {
	if len(bodyJSON) == 0 {
		return ""
	}
	node, err := sonic.Get(bodyJSON, "variables", "model")
	if err != nil {
		return ""
	}
	modelName, _ := node.String()
	return modelName
}

func (vp *VertexProxy) call(ctx context.Context, bodyJSON []byte, tokenLease *recaptcha.TokenLease) (*CallResult, error) {
	defer func() {
		if tokenLease != nil {
			tokenLease.Release()
		}
	}()

	modelName := extractModelName(bodyJSON)
	var lastErr error
	thoughtSignatureBypassApplied := false
	maxRetry := vp.maxRetry()
	maxRefresh := vp.maxRefresh()

	refreshCount := 0

	for {
		verifyExemptUsed := false
		tokenRetry := 0
		for tokenRetry <= maxRetry {
			if err := ctx.Err(); err != nil {
				return nil, err
			}
			result, status, err := vp.doCall(ctx, bodyJSON, tokenLease)
			if err == nil && result != nil {
				return result, nil
			}

			if err != nil {
				lastErr = err
			} else {
				lastErr = fmt.Errorf("vertex API returned status %d", status)
			}
			if err := ctx.Err(); err != nil {
				return nil, err
			}

			// Transport/HTTP errors do not identify a reCAPTCHA token failure.
			if status == 0 {
				return nil, lastErr
			}

			// Thought-signature errors use the dedicated one-time bypass retry;
			// they must be handled before the code-3 reCAPTCHA classifier.
			if status == 3 && isThoughtSignatureInvalidError(lastErr) && !thoughtSignatureBypassApplied {
				bypassedBodyJSON, bypassed, bypassErr := applyThoughtSignatureBypassToBody(bodyJSON)
				if bypassErr != nil {
					return nil, fmt.Errorf("apply thought signature bypass: %w", bypassErr)
				}
				if bypassed {
					bodyJSON = bypassedBodyJSON
					thoughtSignatureBypassApplied = true
					log.Info().Str("err", vp.upstreamLogError(lastErr)).Str("model", modelName).Msg("Thought signature rejected by Vertex, retrying with thought signature bypass")
					continue
				}
			}

			recaptchaReason := classifyRecaptchaRetryError(lastErr)
			if status == 3 && recaptchaReason == recaptchaTokenInvalid {
				log.Warn().Str("err", vp.upstreamLogError(lastErr)).Str("model", modelName).Str("recaptcha_reason", string(recaptchaReason)).Msg("Recaptcha token rejected, refreshing token immediately")
				break
			}
			if !isRetryableVertexError(status, lastErr) {
				return nil, lastErr
			}

			// 每个 recaptchaToken 第一次返回 "Failed to verify action" 不消耗重试次数
			if status == 3 && recaptchaReason == recaptchaVerifyFailed && !verifyExemptUsed {
				verifyExemptUsed = true
				log.Debug().Str("err", vp.upstreamLogError(lastErr)).Str("model", modelName).Msg("First failed to verify action for this token, exempt from retry count, retrying with same token...")
				if err := sleepContext(ctx, vp.retryDelay()); err != nil {
					return nil, err
				}
				continue
			}

			// Verification failures and code 8 consume the current token retry budget.
			tokenRetry++
			retryLog := log.Warn().Str("err", vp.upstreamLogError(lastErr)).Str("model", modelName).Int("token_retry", tokenRetry).Int("refresh", refreshCount)
			if status == 3 {
				retryLog = retryLog.Str("recaptcha_reason", string(recaptchaReason))
			}
			retryLog.Msg("Vertex API call failed, retrying on current token...")
			if tokenRetry <= maxRetry {
				if err := sleepContext(ctx, vp.retryDelay()); err != nil {
					return nil, err
				}
				continue
			}
			break
		}

		if refreshCount < maxRefresh {
			refreshCount++
			var replaceErr error
			bodyJSON, tokenLease, replaceErr = vp.replaceRecaptchaTokenForRetry(ctx, bodyJSON, tokenLease)
			if replaceErr != nil {
				log.Error().Str("err", vp.upstreamLogError(replaceErr)).Str("model", modelName).Int("refresh", refreshCount).Int("max_refresh", maxRefresh).Msg("Failed to refresh recaptcha token")
				return nil, replaceErr
			}
			log.Info().Str("model", modelName).Int("refresh", refreshCount).Int("max_refresh", maxRefresh).Msg("Refreshed recaptcha token, starting retries with new token")
			if err := sleepContext(ctx, vp.retryDelay()); err != nil {
				return nil, err
			}
			continue
		}

		vp.withUpstreamError(log.Error(), lastErr).Str("model", modelName).Int("refresh", refreshCount).Int("max_refresh", maxRefresh).Msg("Recaptcha token refresh limit reached, retries exhausted")
		break
	}

	vp.withUpstreamError(log.Error(), lastErr).Str("model", modelName).Int("refresh", refreshCount).Int("max_refresh", maxRefresh).Msg("Vertex API call failed after retries")
	return nil, lastErr
}

func (vp *VertexProxy) BuildBodyWithTokenWithOptionsContext(
	ctx context.Context,
	modelName string,
	contents []map[string]interface{},
	genConfig map[string]interface{},
	safetySettings []map[string]string,
	systemInstruction interface{},
	options *VertexRequestOptions,
) ([]byte, *recaptcha.TokenLease, error) {
	lease, err := vp.tokenCache.GetTokenContext(ctx)
	if err != nil {
		return nil, nil, fmt.Errorf("get recaptcha token: %w", err)
	}

	bodyJSON, err := BuildVertexBodyWithOptions(modelName, contents, genConfig, safetySettings, systemInstruction, lease.Token(), options)
	if err != nil {
		lease.Release()
		return nil, nil, fmt.Errorf("build request body: %w", err)
	}

	return bodyJSON, lease, nil
}

func (vp *VertexProxy) CallWithTokenWithOptionsContext(
	ctx context.Context,
	modelName string,
	contents []map[string]interface{},
	genConfig map[string]interface{},
	safetySettings []map[string]string,
	systemInstruction interface{},
	options *VertexRequestOptions,
) (*CallResult, error) {
	bodyJSON, tokenLease, err := vp.BuildBodyWithTokenWithOptionsContext(ctx, modelName, contents, genConfig, safetySettings, systemInstruction, options)
	if err != nil {
		return nil, err
	}
	return vp.call(ctx, bodyJSON, tokenLease)
}

// StreamContext calls Vertex AI and emits each decoded upstream chunk immediately.
func (vp *VertexProxy) StreamContext(ctx context.Context, bodyJSON []byte, onChunk func(*CallResult) error) error {
	return vp.stream(ctx, bodyJSON, nil, onChunk)
}

func (vp *VertexProxy) StreamWithTokenContext(ctx context.Context, bodyJSON []byte, tokenLease *recaptcha.TokenLease, onChunk func(*CallResult) error) error {
	return vp.stream(ctx, bodyJSON, tokenLease, onChunk)
}

func (vp *VertexProxy) stream(ctx context.Context, bodyJSON []byte, tokenLease *recaptcha.TokenLease, onChunk func(*CallResult) error) error {
	defer func() {
		if tokenLease != nil {
			tokenLease.Release()
		}
	}()

	modelName := extractModelName(bodyJSON)
	var lastErr error
	thoughtSignatureBypassApplied := false
	maxRetry := vp.maxRetry()
	maxRefresh := vp.maxRefresh()

	refreshCount := 0

	for {
		verifyExemptUsed := false
		tokenRetry := 0
		for tokenRetry <= maxRetry {
			if err := ctx.Err(); err != nil {
				return err
			}
			emitted := false
			status, err := vp.doStream(ctx, bodyJSON, tokenLease, func(result *CallResult) error {
				emitted = true
				return onChunk(result)
			})
			if err == nil {
				return nil
			}

			lastErr = err
			if err := ctx.Err(); err != nil {
				return err
			}
			if emitted {
				return lastErr
			}

			if status == 0 {
				return lastErr
			}

			// Thought-signature errors use the dedicated one-time bypass retry;
			// they must be handled before the code-3 reCAPTCHA classifier.
			if status == 3 && isThoughtSignatureInvalidError(lastErr) && !thoughtSignatureBypassApplied {
				bypassedBodyJSON, bypassed, bypassErr := applyThoughtSignatureBypassToBody(bodyJSON)
				if bypassErr != nil {
					return fmt.Errorf("apply thought signature bypass: %w", bypassErr)
				}
				if bypassed {
					bodyJSON = bypassedBodyJSON
					thoughtSignatureBypassApplied = true
					log.Info().Str("err", vp.upstreamLogError(lastErr)).Str("model", modelName).Msg("Thought signature rejected by Vertex, retrying stream with thought signature bypass")
					continue
				}
			}

			recaptchaReason := classifyRecaptchaRetryError(lastErr)
			if status == 3 && recaptchaReason == recaptchaTokenInvalid {
				log.Warn().Str("err", vp.upstreamLogError(lastErr)).Str("model", modelName).Str("recaptcha_reason", string(recaptchaReason)).Msg("Recaptcha token rejected, refreshing token immediately for stream")
				break
			}
			if !isRetryableVertexError(status, lastErr) {
				return lastErr
			}

			// 每个 recaptchaToken 第一次返回 "Failed to verify action" 不消耗重试次数
			if status == 3 && recaptchaReason == recaptchaVerifyFailed && !verifyExemptUsed {
				verifyExemptUsed = true
				log.Debug().Str("err", vp.upstreamLogError(lastErr)).Str("model", modelName).Msg("First failed to verify action for this token, exempt from retry count, retrying stream with same token...")
				if err := sleepContext(ctx, vp.retryDelay()); err != nil {
					return err
				}
				continue
			}

			// Verification failures and code 8 consume the current token retry budget.
			tokenRetry++
			retryLog := log.Warn().Str("err", vp.upstreamLogError(lastErr)).Str("model", modelName).Int("token_retry", tokenRetry).Int("refresh", refreshCount)
			if status == 3 {
				retryLog = retryLog.Str("recaptcha_reason", string(recaptchaReason))
			}
			retryLog.Msg("Vertex API stream failed, retrying on current token...")
			if tokenRetry <= maxRetry {
				if err := sleepContext(ctx, vp.retryDelay()); err != nil {
					return err
				}
				continue
			}
			break
		}

		if refreshCount < maxRefresh {
			refreshCount++
			var replaceErr error
			bodyJSON, tokenLease, replaceErr = vp.replaceRecaptchaTokenForRetry(ctx, bodyJSON, tokenLease)
			if replaceErr != nil {
				log.Error().Str("err", vp.upstreamLogError(replaceErr)).Str("model", modelName).Int("refresh", refreshCount).Int("max_refresh", maxRefresh).Msg("Failed to refresh recaptcha token")
				return replaceErr
			}
			log.Info().Str("model", modelName).Int("refresh", refreshCount).Int("max_refresh", maxRefresh).Msg("Refreshed recaptcha token, starting stream retries with new token")
			if err := sleepContext(ctx, vp.retryDelay()); err != nil {
				return err
			}
			continue
		}

		vp.withUpstreamError(log.Error(), lastErr).Str("model", modelName).Int("refresh", refreshCount).Int("max_refresh", maxRefresh).Msg("Recaptcha token refresh limit reached, stream retries exhausted")
		break
	}

	vp.withUpstreamError(log.Error(), lastErr).Str("model", modelName).Int("refresh", refreshCount).Int("max_refresh", maxRefresh).Msg("Vertex API stream failed after retries")
	return lastErr
}

func (vp *VertexProxy) doCall(ctx context.Context, bodyJSON []byte, tokenLease *recaptcha.TokenLease) (*CallResult, int, error) {
	result := &CallResult{}
	status, err := vp.doStream(ctx, bodyJSON, tokenLease, func(chunk *CallResult) error {
		mergeCallResult(result, chunk)
		return nil
	})
	if err != nil {
		return nil, status, err
	}

	result.TextParts = coalesceTextParts(result.TextParts)
	for i := range result.Candidates {
		result.Candidates[i].TextParts = coalesceTextParts(result.Candidates[i].TextParts)
	}
	if result.IsEmpty() {
		return nil, 999, fmt.Errorf("response contains no data")
	}

	return result, 0, nil
}

func (vp *VertexProxy) doStream(ctx context.Context, bodyJSON []byte, tokenLease *recaptcha.TokenLease, onChunk func(*CallResult) error) (int, error) {
	vertexBase := vp.selectedVertexBaseURL()
	recaptchaBase := ""
	if tokenLease != nil {
		recaptchaBase = tokenLease.BaseURL()
	}

	stats.RecordRequest(vertexBase)
	stats.RecordRequest(recaptchaBase)

	apiKey := ""
	if vp.cfg != nil {
		apiKey = vp.cfg.GraphQLAPIKey
	}
	req, err := http.NewRequestWithContext(ctx, "POST", buildVertexAPIURL(vertexBase, apiKey), bytes.NewReader(bodyJSON))
	if err != nil {
		return 0, fmt.Errorf("build request: %w", err)
	}
	if vp.cfg != nil && vp.cfg.RandomFingerprint {
		// Select a fresh browser profile for every upstream attempt. The retry
		// loops call doStream again, so a retry does not reuse the prior profile.
		fingerprint := client.NewRandomFingerprint()
		fingerprint.ApplyXHRHeaders(
			req,
			"application/json",
			"*/*",
			"https://console.cloud.google.com",
			"https://console.cloud.google.com/",
			"cross-site",
		)
		req.Header.Set("X-Goog-Authuser", "0")
		req.Header.Set("X-Browser-Channel", "stable")
		req.Header.Set("X-Browser-Copyright", "Copyright 2026 Google LLC. All Rights Reserved.")
		req.Header.Set("X-Browser-Year", "2026")
		req.Header.Set("X-Goog-Ext-353267353-Jspb", "[null,null,null,194274]")
	} else {
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Referer", "https://console.cloud.google.com/")
	}

	resp, err := vp.httpClient.DoRaw(req)
	if err != nil {
		return 0, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return 0, &upstreamHTTPError{Status: resp.StatusCode, Body: string(body)}
	}

	status, streamErr := streamVertexResponse(resp.Body, onChunk)
	if streamErr == nil {
		stats.RecordSuccess(vertexBase)
		stats.RecordSuccess(recaptchaBase)
	}
	return status, streamErr
}

// parseVertexResponse 解析 Vertex AI 的响应
func parseVertexResponse(body []byte) (*CallResult, int, error) {
	return parseVertexResponseWithLogPolicy(body, false)
}

func parseVertexResponseWithLogPolicy(body []byte, redactUpstreamLogs bool) (*CallResult, int, error) {
	result, status, err := aggregateVertexResponse(bytes.NewReader(body))
	if err != nil {
		if status == 999 {
			log.Info().Str("response", config.UpstreamLogValue(string(body), redactUpstreamLogs, 1024)).Msg("Vertex upstream response could not be decoded")
		}
		return nil, status, err
	}

	if result.IsEmpty() {
		return nil, 999, fmt.Errorf("response contains no data")
	}

	return result, 0, nil
}

func aggregateVertexResponse(r io.Reader) (*CallResult, int, error) {
	result := &CallResult{}
	status, err := streamVertexResponse(r, func(chunk *CallResult) error {
		mergeCallResult(result, chunk)
		return nil
	})
	if err != nil {
		return nil, status, err
	}
	result.TextParts = coalesceTextParts(result.TextParts)
	for i := range result.Candidates {
		result.Candidates[i].TextParts = coalesceTextParts(result.Candidates[i].TextParts)
	}
	return result, 0, nil
}

func mergeCallResult(dst, src *CallResult) {
	if dst == nil || src == nil {
		return
	}
	dst.TextParts = append(dst.TextParts, src.TextParts...)
	dst.ImageParts = append(dst.ImageParts, src.ImageParts...)
	dst.FunctionCalls = append(dst.FunctionCalls, src.FunctionCalls...)
	if src.FinishReason != "" {
		dst.FinishReason = src.FinishReason
	}
	if src.GroundingMetadata != nil {
		dst.GroundingMetadata = src.GroundingMetadata
	}
	for _, candidate := range src.Candidates {
		mergeCandidateResult(dst, candidate)
	}
	if len(src.UsageMetadata) > 0 {
		dst.UsageMetadata = src.UsageMetadata
	}
	if src.ModelVersion != "" {
		dst.ModelVersion = src.ModelVersion
	}
	if src.ResponseID != "" {
		dst.ResponseID = src.ResponseID
	}
	if src.CreateTime != "" {
		dst.CreateTime = src.CreateTime
	}
	if len(src.PromptFeedback) > 0 {
		dst.PromptFeedback = src.PromptFeedback
	}
	if len(src.ModelStatus) > 0 {
		dst.ModelStatus = src.ModelStatus
	}
}

func mergeCandidateResult(result *CallResult, incoming CandidateResult) {
	for i := range result.Candidates {
		candidate := &result.Candidates[i]
		if candidate.Index != incoming.Index {
			continue
		}
		candidate.TextParts = append(candidate.TextParts, incoming.TextParts...)
		candidate.ImageParts = append(candidate.ImageParts, incoming.ImageParts...)
		candidate.FunctionCalls = append(candidate.FunctionCalls, incoming.FunctionCalls...)
		if incoming.FinishReason != "" {
			candidate.FinishReason = incoming.FinishReason
		}
		if incoming.FinishMessage != "" {
			candidate.FinishMessage = incoming.FinishMessage
		}
		if incoming.GroundingMetadata != nil {
			candidate.GroundingMetadata = incoming.GroundingMetadata
		}
		if len(incoming.SafetyRatings) > 0 {
			candidate.SafetyRatings = incoming.SafetyRatings
		}
		if len(incoming.CitationMetadata) > 0 {
			candidate.CitationMetadata = incoming.CitationMetadata
		}
		if len(incoming.URLContextMetadata) > 0 {
			candidate.URLContextMetadata = incoming.URLContextMetadata
		}
		if len(incoming.LogprobsResult) > 0 {
			candidate.LogprobsResult = incoming.LogprobsResult
		}
		if incoming.AvgLogprobs != nil {
			candidate.AvgLogprobs = incoming.AvgLogprobs
		}
		return
	}
	result.Candidates = append(result.Candidates, incoming)
}

func coalesceTextParts(parts []model.TextPart) []model.TextPart {
	if len(parts) <= 1 {
		return parts
	}

	merged := make([]model.TextPart, 0, len(parts))
	for _, part := range parts {
		if part.Text == "" && part.ThoughtSignature == "" {
			continue
		}
		lastIndex := len(merged) - 1
		if lastIndex >= 0 &&
			part.ThoughtSignature == "" &&
			merged[lastIndex].ThoughtSignature == "" &&
			merged[lastIndex].Thought == part.Thought {
			merged[lastIndex].Text += part.Text
			continue
		}
		merged = append(merged, part)
	}
	return merged
}

func collectVertexStreamResult(elements []model.VertexResponseElement) (*CallResult, int, error) {
	result := &CallResult{}

	for _, elem := range elements {
		for _, item := range elem.Results {
			if len(item.Errors) > 0 {
				vertexErr := item.Errors[0]
				status := 0
				if vertexErr.Extensions != nil && vertexErr.Extensions.Status != nil {
					status = vertexErr.Extensions.Status.Code
				}
				return nil, status, &vertexAPIError{Code: status, Message: vertexErr.Message}
			}

			if item.Data == nil {
				continue
			}
			dataItems, err := expandVertexData(item.Data)
			if err != nil {
				return nil, 0, fmt.Errorf("decode streamGenerateContentAnonymous: %w", err)
			}
			for _, data := range dataItems {
				collectVertexData(result, &data)
			}
		}
	}

	return result, 0, nil
}

func expandVertexData(data *model.VertexData) ([]model.VertexData, error) {
	if data == nil {
		return nil, nil
	}
	items := []model.VertexData{*data}
	if data.UI == nil || len(bytes.TrimSpace(data.UI.StreamGenerateContentAnonymous)) == 0 {
		return items, nil
	}

	raw := bytes.TrimSpace(data.UI.StreamGenerateContentAnonymous)
	var nested []model.VertexData
	if raw[0] == '[' {
		if err := sonic.Unmarshal(raw, &nested); err != nil {
			return nil, err
		}
	} else {
		var single model.VertexData
		if err := sonic.Unmarshal(raw, &single); err != nil {
			return nil, err
		}
		nested = []model.VertexData{single}
	}
	// Keep the wrapper first so metadata remains associated with its response
	// order if the upstream adds data alongside the nested response.
	return append(items, nested...), nil
}

func collectVertexData(result *CallResult, data *model.VertexData) {
	if result == nil || data == nil {
		return
	}
	if len(data.UsageMetadata) > 0 {
		result.UsageMetadata = data.UsageMetadata
	}
	if data.ModelVersion != "" {
		result.ModelVersion = data.ModelVersion
	}
	if data.ResponseID != "" {
		result.ResponseID = data.ResponseID
	}
	if data.CreateTime != "" {
		result.CreateTime = data.CreateTime
	}
	if len(data.PromptFeedback) > 0 {
		result.PromptFeedback = data.PromptFeedback
	}
	if len(data.ModelStatus) > 0 {
		result.ModelStatus = data.ModelStatus
	}

	var primary *CandidateResult
	for position, candidate := range data.Candidates {
		index := candidate.Index
		if index == 0 && position > 0 {
			index = position
		}
		candidateResult := CandidateResult{
			Index:              index,
			FinishReason:       normalizeFinishReason(candidate.FinishReason),
			FinishMessage:      candidate.FinishMessage,
			GroundingMetadata:  candidate.GroundingMetadata,
			SafetyRatings:      candidate.SafetyRatings,
			CitationMetadata:   candidate.CitationMetadata,
			URLContextMetadata: candidate.URLContextMetadata,
			LogprobsResult:     candidate.LogprobsResult,
			AvgLogprobs:        candidate.AvgLogprobs,
		}
		if candidate.Content != nil {
			for _, part := range candidate.Content.Parts {
				hasFunctionCall := part.FunctionCall != nil && part.FunctionCall.Name != ""
				hasInlineData := part.InlineData != nil && part.InlineData.Data != ""
				if part.Text != "" || (part.ThoughtSignature != "" && !hasFunctionCall && !hasInlineData) {
					candidateResult.TextParts = append(candidateResult.TextParts, model.TextPart{
						Text:             part.Text,
						Thought:          part.Thought,
						ThoughtSignature: part.ThoughtSignature,
					})
				}
				if hasInlineData {
					candidateResult.ImageParts = append(candidateResult.ImageParts, *part.InlineData)
				}
				if hasFunctionCall {
					functionCall := *part.FunctionCall
					functionCall.ThoughtSignature = part.ThoughtSignature
					candidateResult.FunctionCalls = append(candidateResult.FunctionCalls, functionCall)
				}
			}
		}
		mergeCandidateResult(result, candidateResult)
		if primary == nil {
			candidateCopy := candidateResult
			primary = &candidateCopy
		}
	}

	if primary == nil {
		return
	}
	result.TextParts = append(result.TextParts, primary.TextParts...)
	result.ImageParts = append(result.ImageParts, primary.ImageParts...)
	result.FunctionCalls = append(result.FunctionCalls, primary.FunctionCalls...)
	if primary.FinishReason != "" {
		result.FinishReason = primary.FinishReason
	}
	if primary.GroundingMetadata != nil {
		result.GroundingMetadata = primary.GroundingMetadata
	}
}

func normalizeFinishReason(reason string) string {
	if reason == finishReasonUnspecified {
		return ""
	}
	return reason
}

func streamVertexResponse(r io.Reader, onChunk func(*CallResult) error) (int, error) {
	br := bufio.NewReader(r)
	first, err := peekFirstNonSpace(br)
	if err != nil {
		return 0, fmt.Errorf("read stream: %w", err)
	}

	emitter := &vertexStreamEmitter{onChunk: onChunk}
	emit := emitter.emit
	var status int
	switch first {
	case '[':
		status, err = streamVertexArray(br, emit)
	case '{':
		status, err = streamVertexObjects(br, emit)
	default:
		status, err = streamVertexSSE(br, emit)
	}
	if err != nil {
		return status, err
	}
	if err := emitter.flush(); err != nil {
		return 0, err
	}
	return 0, nil
}

type vertexStreamEmitter struct {
	onChunk       func(*CallResult) error
	pendingFinish *CallResult
}

func (e *vertexStreamEmitter) emit(result *CallResult) error {
	if result == nil || result.IsEmpty() {
		return nil
	}
	if result.IsFinishOnly() {
		pending := *result
		e.pendingFinish = &pending
		return nil
	}
	if result.FinishReason != "" {
		e.pendingFinish = nil
	}
	return e.onChunk(result)
}

func (e *vertexStreamEmitter) flush() error {
	if e.pendingFinish == nil {
		return nil
	}
	pending := e.pendingFinish
	e.pendingFinish = nil
	return e.onChunk(pending)
}

func streamVertexArray(r io.Reader, onChunk func(*CallResult) error) (int, error) {
	decoder := json.NewDecoder(r)
	token, err := decoder.Token()
	if err != nil {
		return 0, fmt.Errorf("decode stream start: %w", err)
	}
	if delim, ok := token.(json.Delim); !ok || delim != '[' {
		return 0, fmt.Errorf("decode stream start: expected array")
	}

	for decoder.More() {
		if status, err := streamVertexElementObject(decoder, onChunk); err != nil {
			return status, err
		}
	}

	if _, err := decoder.Token(); err != nil {
		return 0, fmt.Errorf("decode stream end: %w", err)
	}
	return 0, nil
}

func streamVertexObjects(r io.Reader, onChunk func(*CallResult) error) (int, error) {
	decoder := json.NewDecoder(r)
	for {
		status, err := streamVertexElementObject(decoder, onChunk)
		if err != nil {
			if err == io.EOF {
				return 0, nil
			}
			return status, err
		}
	}
}

func streamVertexElementObject(decoder *json.Decoder, onChunk func(*CallResult) error) (int, error) {
	token, err := decoder.Token()
	if err != nil {
		return 0, err
	}
	if delim, ok := token.(json.Delim); !ok || delim != '{' {
		return 0, fmt.Errorf("decode stream element: expected object")
	}

	for decoder.More() {
		keyToken, err := decoder.Token()
		if err != nil {
			return 0, fmt.Errorf("decode stream element key: %w", err)
		}
		key, ok := keyToken.(string)
		if !ok {
			return 0, fmt.Errorf("decode stream element key: expected string")
		}

		if key == "results" {
			if status, err := streamVertexResultsArray(decoder, onChunk); err != nil {
				return status, err
			}
			continue
		}

		var ignored interface{}
		if err := decoder.Decode(&ignored); err != nil {
			return 0, fmt.Errorf("decode stream element field %q: %w", key, err)
		}
	}

	if _, err := decoder.Token(); err != nil {
		return 0, fmt.Errorf("decode stream element end: %w", err)
	}
	return 0, nil
}

func streamVertexResultsArray(decoder *json.Decoder, onChunk func(*CallResult) error) (int, error) {
	token, err := decoder.Token()
	if err != nil {
		return 0, fmt.Errorf("decode results start: %w", err)
	}
	if delim, ok := token.(json.Delim); !ok || delim != '[' {
		return 0, fmt.Errorf("decode results start: expected array")
	}

	for decoder.More() {
		var result model.VertexResult
		if err := decoder.Decode(&result); err != nil {
			return 0, fmt.Errorf("decode result item: %w", err)
		}
		elem := model.VertexResponseElement{Results: []model.VertexResult{result}}
		if status, err := emitVertexStreamElements([]model.VertexResponseElement{elem}, onChunk); err != nil {
			return status, err
		}
	}

	if _, err := decoder.Token(); err != nil {
		return 0, fmt.Errorf("decode results end: %w", err)
	}
	return 0, nil
}

func streamVertexSSE(r io.Reader, onChunk func(*CallResult) error) (int, error) {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 10*1024*1024)

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}

		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "" || payload == "[DONE]" {
			continue
		}

		status, err := emitVertexStreamPayload([]byte(payload), onChunk)
		if err != nil {
			return status, err
		}
	}
	if err := scanner.Err(); err != nil {
		return 0, fmt.Errorf("read stream event: %w", err)
	}
	return 0, nil
}

func emitVertexStreamPayload(payload []byte, onChunk func(*CallResult) error) (int, error) {
	payload = bytes.TrimSpace(payload)
	if len(payload) == 0 {
		return 0, nil
	}

	if payload[0] == '[' {
		var elements []model.VertexResponseElement
		if err := sonic.Unmarshal(payload, &elements); err != nil {
			return 0, fmt.Errorf("unmarshal stream payload: %w", err)
		}
		return emitVertexStreamElements(elements, onChunk)
	}

	var elem model.VertexResponseElement
	if err := sonic.Unmarshal(payload, &elem); err != nil {
		return 0, fmt.Errorf("unmarshal stream payload: %w", err)
	}
	return emitVertexStreamElements([]model.VertexResponseElement{elem}, onChunk)
}

func emitVertexStreamElements(elements []model.VertexResponseElement, onChunk func(*CallResult) error) (int, error) {
	result, status, err := collectVertexStreamResult(elements)
	if err != nil {
		return status, err
	}
	if result.IsEmpty() {
		return 0, nil
	}
	if err := onChunk(result); err != nil {
		return 0, err
	}
	return 0, nil
}

func peekFirstNonSpace(r *bufio.Reader) (byte, error) {
	for {
		b, err := r.ReadByte()
		if err != nil {
			return 0, err
		}
		if isJSONWhitespace(b) {
			continue
		}
		if err := r.UnreadByte(); err != nil {
			return 0, err
		}
		return b, nil
	}
}

func isJSONWhitespace(b byte) bool {
	return b == ' ' || b == '\n' || b == '\r' || b == '\t'
}

// replaceRecaptchaToken 替换 JSON body 中的 recaptchaToken
func replaceRecaptchaToken(bodyJSON []byte, newToken string) ([]byte, error) {
	var raw map[string]interface{}
	if err := sonic.Unmarshal(bodyJSON, &raw); err != nil {
		return bodyJSON, err
	}
	if vars, ok := raw["variables"].(map[string]interface{}); ok {
		vars["recaptchaToken"] = newToken
	}
	return sonic.Marshal(raw)
}

func applyThoughtSignatureBypassToBody(bodyJSON []byte) ([]byte, bool, error) {
	var raw map[string]interface{}
	if err := sonic.Unmarshal(bodyJSON, &raw); err != nil {
		return bodyJSON, false, err
	}

	vars, ok := raw["variables"].(map[string]interface{})
	if !ok {
		return bodyJSON, false, nil
	}

	changed := applyThoughtSignatureBypassToContentValue(vars["contents"])
	if applyThoughtSignatureBypassToContentValue(vars["systemInstruction"]) {
		changed = true
	}
	if !changed {
		return bodyJSON, false, nil
	}

	bypassedBodyJSON, err := sonic.Marshal(raw)
	if err != nil {
		return bodyJSON, false, err
	}
	return bypassedBodyJSON, true, nil
}

func applyThoughtSignatureBypassToContentValue(value interface{}) bool {
	switch v := value.(type) {
	case []interface{}:
		changed := false
		for _, item := range v {
			if applyThoughtSignatureBypassToContentValue(item) {
				changed = true
			}
		}
		return changed
	case []map[string]interface{}:
		changed := false
		for _, item := range v {
			if applyThoughtSignatureBypassToContentValue(item) {
				changed = true
			}
		}
		return changed
	case map[string]interface{}:
		return applyThoughtSignatureBypassToParts(v["parts"])
	default:
		return false
	}
}

func applyThoughtSignatureBypassToParts(value interface{}) bool {
	switch parts := value.(type) {
	case []interface{}:
		return applyThoughtSignatureBypassToPartSlice(parts)
	case []map[string]interface{}:
		items := make([]interface{}, 0, len(parts))
		for _, part := range parts {
			items = append(items, part)
		}
		return applyThoughtSignatureBypassToPartSlice(items)
	default:
		return false
	}
}

func applyThoughtSignatureBypassToPartSlice(parts []interface{}) bool {
	changed := false
	firstFunctionCall := true
	for _, value := range parts {
		part, ok := value.(map[string]interface{})
		if !ok {
			continue
		}
		if _, ok := part["functionCall"]; ok && firstFunctionCall {
			firstFunctionCall = false
			if part["thoughtSignature"] != thoughtSignatureBypassValue() {
				part["thoughtSignature"] = thoughtSignatureBypassValue()
				changed = true
			}
			continue
		}
		if removeThoughtSignature(part) {
			changed = true
		}
	}
	return changed
}

func removeThoughtSignature(part map[string]interface{}) bool {
	removed := false
	if _, ok := part["thoughtSignature"]; ok {
		delete(part, "thoughtSignature")
		removed = true
	}
	return removed
}

func defaultSafetySettings() []map[string]string {
	return []map[string]string{
		{"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
		{"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
		{"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
		{"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
	}
}
