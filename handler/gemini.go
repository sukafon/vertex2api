package handler

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"vertex2api/model"
	"vertex2api/proxy"

	"github.com/bytedance/sonic"
	"github.com/rs/zerolog/log"
)

// GeminiGenerate 处理 Gemini 原生端点
// POST /v1beta1/models/{model}:generateContent
// POST /v1beta1/models/{model}:streamGenerateContent
func GeminiGenerate(vp *proxy.VertexProxy, allowCustomModelNames, strictAltSSE bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(proxy.WithCompatibilityLayer(r.Context(), proxy.CompatibilityLayerGeminiNative))
		// 解析路径: modelAction = "gemini-2.0-flash:generateContent"
		modelAction := r.PathValue("modelAction")
		if modelAction == "" {
			WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error": map[string]interface{}{"code": 400, "message": "model and action required in path", "status": "INVALID_ARGUMENT"},
			})
			return
		}

		// 分割 model 和 action
		parts := strings.SplitN(modelAction, ":", 2)
		if len(parts) != 2 {
			WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error": map[string]interface{}{
					"code":    400,
					"message": fmt.Sprintf("invalid path format, expected {model}:{action}, got: %s", modelAction),
					"status":  "INVALID_ARGUMENT",
				},
			})
			return
		}
		modelName := parts[0]
		action := parts[1]
		if message := validateModelName(modelName, allowCustomModelNames); message != "" {
			WriteProtocolError(w, r, http.StatusBadRequest, message, "invalid_request_error")
			return
		}

		isStream := action == "streamGenerateContent"
		if action != "generateContent" && action != "streamGenerateContent" && action != "countTokens" {
			WriteJSON(w, http.StatusBadRequest, map[string]interface{}{
				"error": map[string]interface{}{
					"code":    400,
					"message": fmt.Sprintf("unsupported action: %s", action),
					"status":  "INVALID_ARGUMENT",
				},
			})
			return
		}

		// 解析 Gemini 请求体
		var geminiReq map[string]interface{}
		_, ok := readJSONRequest(w, r, &geminiReq)
		if !ok {
			return
		}
		if action == "countTokens" {
			input := interface{}(geminiReq)
			if generateRequest, ok := geminiReq["generateContentRequest"].(map[string]interface{}); ok {
				input = generateRequest
			}
			w.Header().Set("X-Usage-Estimated", "true")
			WriteJSON(w, http.StatusOK, map[string]interface{}{"totalTokens": estimateJSONTokens(input)})
			return
		}
		contentsValue, exists := geminiReq["contents"]
		contents, contentsOK := toSliceMap(contentsValue)
		if !exists || !contentsOK || len(contents) == 0 {
			WriteProtocolError(w, r, http.StatusBadRequest, "contents is required", "invalid_request_error")
			return
		}

		// 提取各字段，构建 Vertex 调用参数
		genConfig, _ := geminiReq["generationConfig"].(map[string]interface{})
		if geminiReq["modelArmorConfig"] != nil && geminiReq["safetySettings"] != nil {
			WriteProtocolError(w, r, http.StatusBadRequest, "modelArmorConfig and safetySettings cannot be used together", "invalid_request_error")
			return
		}
		safetySettings := toSafetySettings(geminiReq["safetySettings"])
		safetySettings = model.SanitizeSafetySettings(modelName, safetySettings)
		systemInstruction := geminiReq["systemInstruction"]
		options := buildGeminiRequestOptions(geminiReq)

		// 确保 responseModalities 默认值
		if genConfig == nil {
			genConfig = map[string]interface{}{}
		}
		includeThoughts := geminiIncludeThoughts(genConfig)

		if isStream && geminiRequestUsesSSE(r, strictAltSSE) {
			bodyJSON, tokenLease, err := vp.BuildBodyWithTokenWithOptionsContext(r.Context(), modelName, contents, genConfig, safetySettings, systemInstruction, options)
			if err != nil {
				if requestContextCanceled(r.Context(), err) {
					log.Debug().Err(err).Str("model", modelName).Msg("Gemini stream request canceled before upstream call")
					return
				}
				log.Error().Err(err).Str("model", modelName).Msg("Gemini Native request conversion failed")
				WriteJSON(w, http.StatusInternalServerError, geminiErrorResponse(err))
				return
			}
			stream := func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
				return vp.StreamWithTokenContext(ctx, bodyJSON, tokenLease, onChunk)
			}
			streamGeminiResponseWithProxyOptions(w, r, modelName, vp, includeThoughts, geminiReq, stream)
			return
		}

		result, err := vp.CallWithTokenWithOptionsContext(r.Context(), modelName, contents, genConfig, safetySettings, systemInstruction, options)
		if err != nil {
			if requestContextCanceled(r.Context(), err) {
				log.Debug().Err(err).Str("model", modelName).Msg("Gemini request canceled")
				return
			}
			writeUpstreamProtocolError(w, r, vp, err)
			return
		}

		writeGeminiCompletedResponse(w, geminiReq, result, includeThoughts)
	})
}

func writeGeminiCompletedResponse(
	w http.ResponseWriter,
	requestInput interface{},
	result *proxy.CallResult,
	includeThoughts bool,
) {
	response := buildGeminiResponseForRequest(result, includeThoughts)
	usage, estimated := geminiUsageMetadata(requestInput, result)
	response["usageMetadata"] = usage
	if estimated {
		w.Header().Set("X-Usage-Estimated", "true")
	}
	WriteJSON(w, http.StatusOK, response)
}

func buildGeminiResponse(result *proxy.CallResult) map[string]interface{} {
	return buildGeminiResponseWithFinish(result, true)
}

func buildGeminiStreamResponse(result *proxy.CallResult) map[string]interface{} {
	return buildGeminiResponseWithFinish(result, false)
}

func buildGeminiResponseForRequest(result *proxy.CallResult, includeThoughts bool) map[string]interface{} {
	return buildGeminiResponseWithOptions(result, true, includeThoughts)
}

func buildGeminiStreamResponseForRequest(result *proxy.CallResult, includeThoughts bool) map[string]interface{} {
	return buildGeminiResponseWithOptions(result, false, includeThoughts)
}

func geminiIncludeThoughts(genConfig map[string]interface{}) bool {
	thinkingConfig, ok := genConfig["thinkingConfig"].(map[string]interface{})
	if !ok {
		return false
	}
	includeThoughts, _ := thinkingConfig["includeThoughts"].(bool)
	return includeThoughts
}

func buildGeminiRequestOptions(req map[string]interface{}) *proxy.VertexRequestOptions {
	options := &proxy.VertexRequestOptions{}
	if tools, ok := req["tools"]; ok {
		options.Tools = tools
	}
	if toolConfig, ok := req["toolConfig"]; ok {
		options.ToolConfig = toolConfig
	}
	for _, key := range []string{"cachedContent", "labels", "modelArmorConfig", "serviceTier", "store"} {
		if value, ok := req[key]; ok {
			if options.AdditionalVariables == nil {
				options.AdditionalVariables = make(map[string]interface{})
			}
			options.AdditionalVariables[key] = value
		}
	}
	if options.Tools == nil && options.ToolConfig == nil && len(options.AdditionalVariables) == 0 {
		return nil
	}
	return options
}

func buildGeminiResponseWithFinish(result *proxy.CallResult, defaultStop bool) map[string]interface{} {
	return buildGeminiResponseWithOptions(result, defaultStop, true)
}

func buildGeminiResponseWithOptions(result *proxy.CallResult, defaultStop, includeThoughts bool) map[string]interface{} {
	response := make(map[string]interface{})
	if result == nil {
		return response
	}
	promptFeedback := normalizeGeminiPromptFeedback(result.PromptFeedback)
	promptWasBlocked := geminiPromptWasBlocked(promptFeedback)
	candidates := make([]map[string]interface{}, 0, maxInt(1, len(result.Candidates)))
	if !promptWasBlocked && len(result.Candidates) > 0 {
		for _, candidate := range result.Candidates {
			candidateResult := callResultFromCandidate(candidate)
			candidateMap := buildGeminiCandidateWithThoughts(candidateResult, defaultStop, includeThoughts)
			if candidate.FinishMessage != "" {
				candidateMap["finishMessage"] = candidate.FinishMessage
			}
			if safetyRatings := normalizeGeminiSafetyRatings(candidate.SafetyRatings); len(safetyRatings) > 0 {
				candidateMap["safetyRatings"] = safetyRatings
			}
			if citationMetadata := model.NormalizeCitationMetadata(candidate.CitationMetadata); len(citationMetadata) > 0 {
				candidateMap["citationMetadata"] = citationMetadata
			}
			if len(candidate.URLContextMetadata) > 0 {
				candidateMap["urlContextMetadata"] = candidate.URLContextMetadata
			}
			if len(candidate.LogprobsResult) > 0 {
				candidateMap["logprobsResult"] = candidate.LogprobsResult
			}
			if candidate.AvgLogprobs != nil {
				candidateMap["avgLogprobs"] = *candidate.AvgLogprobs
			}
			if len(candidateMap) == 0 {
				// Keep the v1.0.3 index behavior at the candidate envelope level:
				// index accompanies meaningful candidate data, but is never emitted
				// as a candidate by itself.
				continue
			}
			candidateMap["index"] = candidate.Index
			candidates = append(candidates, candidateMap)
		}
	} else if !promptWasBlocked && (result.HasContent() || result.FinishReason != "") {
		candidateMap := buildGeminiCandidateWithThoughts(result, defaultStop, includeThoughts)
		if len(candidateMap) > 0 {
			candidates = append(candidates, candidateMap)
		}
	}
	if len(candidates) > 0 {
		response["candidates"] = candidates
	}
	if len(result.UsageMetadata) > 0 {
		response["usageMetadata"] = result.UsageMetadata
	}
	if result.ModelVersion != "" {
		response["modelVersion"] = result.ModelVersion
	}
	if result.ResponseID != "" {
		response["responseId"] = result.ResponseID
	}
	if len(promptFeedback) > 0 {
		response["promptFeedback"] = promptFeedback
	}
	if len(result.ModelStatus) > 0 {
		response["modelStatus"] = result.ModelStatus
	}
	return response
}

func geminiPromptWasBlocked(promptFeedback map[string]interface{}) bool {
	blockReason, _ := promptFeedback["blockReason"].(string)
	return blockReason != ""
}

// normalizeGeminiPromptFeedback translates Vertex PromptFeedback into the
// Gemini Developer API schema instead of exposing the upstream object as-is.
func normalizeGeminiPromptFeedback(source map[string]interface{}) map[string]interface{} {
	if len(source) == 0 {
		return nil
	}

	result := make(map[string]interface{}, 2)
	if blockReason := normalizeGeminiBlockReason(source["blockReason"]); blockReason != "" {
		result["blockReason"] = blockReason
	}
	if safetyRatings := normalizeGeminiSafetyRatings(source["safetyRatings"]); len(safetyRatings) > 0 {
		result["safetyRatings"] = safetyRatings
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

func normalizeGeminiBlockReason(value interface{}) string {
	blockReason, ok := value.(string)
	if !ok {
		return ""
	}
	switch strings.ToUpper(strings.TrimSpace(blockReason)) {
	case "", "BLOCK_REASON_UNSPECIFIED", "BLOCKED_REASON_UNSPECIFIED":
		return ""
	case "SAFETY", "OTHER", "BLOCKLIST", "PROHIBITED_CONTENT", "IMAGE_SAFETY":
		return strings.ToUpper(strings.TrimSpace(blockReason))
	case "MODEL_ARMOR", "JAILBREAK":
		// These reasons exist only in Vertex AI. OTHER is the closest valid
		// Gemini Developer API value and still indicates that the prompt blocked.
		return "OTHER"
	default:
		// Do not leak future Vertex-only enum names into the Gemini API surface.
		return "OTHER"
	}
}

func normalizeGeminiSafetyRatings(value interface{}) []map[string]interface{} {
	var source []map[string]interface{}
	switch ratings := value.(type) {
	case []map[string]interface{}:
		source = ratings
	case []interface{}:
		source = make([]map[string]interface{}, 0, len(ratings))
		for _, value := range ratings {
			if rating, ok := value.(map[string]interface{}); ok {
				source = append(source, rating)
			}
		}
	default:
		return nil
	}

	result := make([]map[string]interface{}, 0, len(source))
	for _, sourceRating := range source {
		category := normalizeGeminiHarmCategory(sourceRating["category"])
		probability := normalizeGeminiHarmProbability(sourceRating["probability"])
		if category == "" || probability == "" {
			continue
		}
		rating := map[string]interface{}{"category": category, "probability": probability}
		if blocked, ok := sourceRating["blocked"].(bool); ok && blocked {
			rating["blocked"] = true
		}
		result = append(result, rating)
	}
	return result
}

func normalizeGeminiHarmCategory(value interface{}) string {
	category, ok := value.(string)
	if !ok {
		return ""
	}
	category = strings.ToUpper(strings.TrimSpace(category))
	if category == "" || category == "HARM_CATEGORY_UNSPECIFIED" {
		return ""
	}
	if !isKnownGeminiHarmCategory(category) {
		// Keep unknown non-default values so a newly added Gemini enum is not
		// silently lost before this compatibility layer is updated.
		log.Debug().Str("category", category).Msg("forwarding unknown Gemini harm category")
	}
	return category
}

func isKnownGeminiHarmCategory(category string) bool {
	switch category {
	case "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_DANGEROUS_CONTENT", "HARM_CATEGORY_HARASSMENT",
		"HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_CIVIC_INTEGRITY", "HARM_CATEGORY_IMAGE_HATE",
		"HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "HARM_CATEGORY_IMAGE_HARASSMENT",
		"HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT", "HARM_CATEGORY_JAILBREAK":
		return true
	default:
		return false
	}
}

func normalizeGeminiHarmProbability(value interface{}) string {
	probability, ok := value.(string)
	if !ok {
		return ""
	}
	probability = strings.ToUpper(strings.TrimSpace(probability))
	if probability == "" || probability == "HARM_PROBABILITY_UNSPECIFIED" {
		return ""
	}
	if !isKnownGeminiHarmProbability(probability) {
		log.Debug().Str("probability", probability).Msg("forwarding unknown Gemini harm probability")
	}
	return probability
}

func isKnownGeminiHarmProbability(probability string) bool {
	switch probability {
	case "NEGLIGIBLE", "LOW", "MEDIUM", "HIGH":
		return true
	default:
		return false
	}
}

func buildGeminiCandidate(result *proxy.CallResult, defaultStop bool) map[string]interface{} {
	return buildGeminiCandidateWithThoughts(result, defaultStop, true)
}

func buildGeminiCandidateWithThoughts(result *proxy.CallResult, defaultStop, includeThoughts bool) map[string]interface{} {
	parts := buildCanonicalGeminiPartsWithThoughts(result, includeThoughts)
	if defaultStop {
		parts = coalesceCanonicalGeminiTextParts(parts)
	}

	candidate := make(map[string]interface{}, 3)
	if len(parts) > 0 {
		candidate["content"] = map[string]interface{}{
			"parts": parts,
			"role":  "model",
		}
	}
	finishReason := result.FinishReason
	if defaultStop && len(parts) > 0 && (finishReason == "" || finishReason == "FINISH_REASON_UNSPECIFIED") {
		finishReason = "STOP"
	} else if finishReason == "FINISH_REASON_UNSPECIFIED" {
		finishReason = ""
	}
	if finishReason != "" {
		candidate["finishReason"] = finishReason
	}
	if groundingMetadata := model.NormalizeGroundingMetadata(result.GroundingMetadata); groundingMetadata != nil {
		candidate["groundingMetadata"] = groundingMetadata
	}

	return candidate
}

func buildCanonicalGeminiParts(result *proxy.CallResult) []map[string]interface{} {
	return buildCanonicalGeminiPartsWithThoughts(result, true)
}

func buildCanonicalGeminiPartsWithThoughts(result *proxy.CallResult, includeThoughts bool) []map[string]interface{} {
	if result == nil {
		return nil
	}
	parts := make([]map[string]interface{}, 0, maxInt(len(result.Parts), len(result.TextParts)+len(result.ImageParts)+len(result.FunctionCalls)))
	if result.Parts != nil {
		for _, sourcePart := range result.Parts {
			if part, ok := canonicalGeminiPart(sourcePart); ok {
				if part, ok = filterUnrequestedGeminiThought(part, includeThoughts); ok {
					parts = append(parts, part)
				}
			}
		}
		return parts
	}

	// Compatibility fallback for callers that still construct only the legacy
	// semantic slices. Parsed Vertex responses always use the ordered Parts path.
	for _, textPart := range result.TextParts {
		if part, ok := canonicalGeminiPart(model.VertexPart{
			Text:             textPart.Text,
			Thought:          textPart.Thought,
			ThoughtSignature: textPart.ThoughtSignature,
		}); ok {
			if part, ok = filterUnrequestedGeminiThought(part, includeThoughts); ok {
				parts = append(parts, part)
			}
		}
	}
	for i := range result.ImageParts {
		if part, ok := canonicalGeminiPart(model.VertexPart{InlineData: &result.ImageParts[i]}); ok {
			parts = append(parts, part)
		}
	}
	for i := range result.FunctionCalls {
		functionCall := result.FunctionCalls[i]
		if part, ok := canonicalGeminiPart(model.VertexPart{
			FunctionCall:     &functionCall,
			ThoughtSignature: functionCall.ThoughtSignature,
		}); ok {
			parts = append(parts, part)
		}
	}
	return parts
}

func filterUnrequestedGeminiThought(part map[string]interface{}, includeThoughts bool) (map[string]interface{}, bool) {
	thought, _ := part["thought"].(bool)
	if includeThoughts || !thought {
		return part, true
	}

	// Thought summaries are opt-in in the Gemini API. The anonymous Vertex
	// backend can still return them when includeThoughts was not requested, so
	// remove only the summary text at the compatibility boundary. Preserve an
	// opaque thoughtSignature as an empty text part because Gemini 3 clients
	// must be able to round-trip that signature in later requests.
	filtered := make(map[string]interface{}, len(part))
	for key, value := range part {
		if key == "text" || key == "thought" {
			continue
		}
		filtered[key] = value
	}
	if signature, _ := filtered["thoughtSignature"].(string); strings.TrimSpace(signature) != "" {
		filtered["text"] = ""
		return filtered, true
	}
	if canonicalGeminiPartHasData(filtered) {
		return filtered, true
	}
	return nil, false
}

func canonicalGeminiPartHasData(part map[string]interface{}) bool {
	for _, key := range []string{
		"text", "inlineData", "fileData", "functionCall", "functionResponse",
		"executableCode", "codeExecutionResult",
	} {
		if _, ok := part[key]; ok {
			return true
		}
	}
	return false
}

func canonicalGeminiPart(source model.VertexPart) (map[string]interface{}, bool) {
	type dataArm struct {
		name  string
		value interface{}
	}
	arms := make([]dataArm, 0, 2)
	if source.Text != "" {
		arms = append(arms, dataArm{name: "text", value: source.Text})
	}
	if inlineData, ok := canonicalGeminiInlineData(source.InlineData); ok {
		arms = append(arms, dataArm{name: "inlineData", value: inlineData})
	}
	if fileData, ok := canonicalGeminiFileData(source.FileData); ok {
		arms = append(arms, dataArm{name: "fileData", value: fileData})
	}
	if functionCall, ok := canonicalGeminiFunctionCall(source.FunctionCall); ok {
		arms = append(arms, dataArm{name: "functionCall", value: functionCall})
	}
	if functionResponse, ok := canonicalGeminiFunctionResponse(source.FunctionResponse); ok {
		arms = append(arms, dataArm{name: "functionResponse", value: functionResponse})
	}
	if executableCode, ok := canonicalGeminiExecutableCode(source.ExecutableCode); ok {
		arms = append(arms, dataArm{name: "executableCode", value: executableCode})
	}
	if codeExecutionResult, ok := canonicalGeminiCodeExecutionResult(source.CodeExecutionResult); ok {
		arms = append(arms, dataArm{name: "codeExecutionResult", value: codeExecutionResult})
	}
	if len(arms) == 0 && strings.TrimSpace(source.ThoughtSignature) != "" {
		arms = append(arms, dataArm{name: "text", value: ""})
	}
	if len(arms) != 1 {
		// The Gemini Part data field is a oneof. Drop malformed upstream Parts
		// rather than serializing a Vertex object containing multiple data arms.
		return nil, false
	}

	part := map[string]interface{}{arms[0].name: arms[0].value}
	if source.Thought {
		part["thought"] = true
	}
	if source.ThoughtSignature != "" {
		part["thoughtSignature"] = source.ThoughtSignature
	}
	if mediaResolution, ok := canonicalGeminiMediaResolution(source.MediaResolution); ok &&
		(arms[0].name == "inlineData" || arms[0].name == "fileData") {
		part["mediaResolution"] = mediaResolution
	}
	if videoMetadata, ok := canonicalGeminiVideoMetadata(source.VideoMetadata); ok &&
		(arms[0].name == "inlineData" || arms[0].name == "fileData") {
		part["videoMetadata"] = videoMetadata
	}
	return part, true
}

func canonicalGeminiInlineData(source *model.InlineData) (map[string]interface{}, bool) {
	if source == nil || strings.TrimSpace(source.MimeType) == "" || source.Data == "" {
		return nil, false
	}
	result := map[string]interface{}{
		"mimeType": strings.TrimSpace(source.MimeType),
		"data":     source.Data,
	}
	return result, true
}

func canonicalGeminiFileData(source map[string]interface{}) (map[string]interface{}, bool) {
	mimeType := strings.TrimSpace(geminiMapString(source, "mimeType", "mime_type"))
	fileURI := strings.TrimSpace(geminiMapString(source, "fileUri", "file_uri"))
	if mimeType == "" || fileURI == "" {
		return nil, false
	}
	return map[string]interface{}{"mimeType": mimeType, "fileUri": fileURI}, true
}

func canonicalGeminiFunctionCall(source *model.FunctionCall) (map[string]interface{}, bool) {
	if source == nil {
		return nil, false
	}
	result := make(map[string]interface{}, 3)
	if id := strings.TrimSpace(source.ID); id != "" {
		result["id"] = id
	}
	if name := strings.TrimSpace(source.Name); name != "" {
		result["name"] = name
	}
	if source.Args != nil && (len(source.Args) > 0 || result["name"] != nil || result["id"] != nil) {
		result["args"] = source.Args
	}
	// partialArgs and willContinue are Vertex-only streaming fields. The
	// Gemini Developer API FunctionCall schema contains only id/name/args.
	meaningful := result["id"] != nil || result["name"] != nil || len(source.Args) > 0
	return result, meaningful
}

func canonicalGeminiFunctionResponse(source *model.FunctionResponse) (map[string]interface{}, bool) {
	if source == nil || strings.TrimSpace(source.Name) == "" || source.Response == nil {
		return nil, false
	}
	result := map[string]interface{}{
		"name":     strings.TrimSpace(source.Name),
		"response": source.Response,
	}
	if id := strings.TrimSpace(source.ID); id != "" {
		result["id"] = id
	}
	parts := canonicalGeminiFunctionResponseParts(source.Parts)
	if len(parts) > 0 {
		result["parts"] = parts
	}
	return result, true
}

func canonicalGeminiFunctionResponseParts(source []model.FunctionResponsePart) []map[string]interface{} {
	result := make([]map[string]interface{}, 0, len(source))
	for _, sourcePart := range source {
		// Gemini FunctionResponsePart supports inlineData; its fileData arm is
		// available only on Vertex AI and must not be exposed by this adapter.
		if inlineData, ok := canonicalGeminiInlineData(sourcePart.InlineData); ok {
			result = append(result, map[string]interface{}{"inlineData": inlineData})
		}
	}
	return result
}

func canonicalGeminiExecutableCode(source map[string]interface{}) (map[string]interface{}, bool) {
	language := strings.TrimSpace(geminiMapString(source, "language"))
	code := geminiMapString(source, "code")
	if language == "" || strings.EqualFold(language, "LANGUAGE_UNSPECIFIED") || code == "" {
		return nil, false
	}
	result := map[string]interface{}{"language": language, "code": code}
	if id := strings.TrimSpace(geminiMapString(source, "id")); id != "" {
		result["id"] = id
	}
	return result, true
}

func canonicalGeminiCodeExecutionResult(source map[string]interface{}) (map[string]interface{}, bool) {
	outcome := strings.TrimSpace(geminiMapString(source, "outcome"))
	if outcome == "" || strings.EqualFold(outcome, "OUTCOME_UNSPECIFIED") {
		return nil, false
	}
	result := map[string]interface{}{"outcome": outcome}
	if output, ok := geminiMapValue(source, "output"); ok {
		if text, valid := output.(string); valid {
			result["output"] = text
		}
	}
	if id := strings.TrimSpace(geminiMapString(source, "id")); id != "" {
		result["id"] = id
	}
	return result, true
}

func canonicalGeminiVideoMetadata(source map[string]interface{}) (map[string]interface{}, bool) {
	result := make(map[string]interface{}, 3)
	if startOffset := strings.TrimSpace(geminiMapString(source, "startOffset", "start_offset")); startOffset != "" {
		result["startOffset"] = startOffset
	}
	if endOffset := strings.TrimSpace(geminiMapString(source, "endOffset", "end_offset")); endOffset != "" {
		result["endOffset"] = endOffset
	}
	if value, ok := geminiMapValue(source, "fps"); ok {
		if fps, valid := geminiNumber(value); valid && fps > 0 && fps <= 24 {
			result["fps"] = fps
		}
	}
	return result, len(result) > 0
}

func canonicalGeminiMediaResolution(source map[string]interface{}) (map[string]interface{}, bool) {
	level := strings.TrimSpace(geminiMapString(source, "level"))
	if level == "" || strings.EqualFold(level, "MEDIA_RESOLUTION_UNSPECIFIED") {
		return nil, false
	}
	return map[string]interface{}{"level": level}, true
}

func geminiMapString(source map[string]interface{}, keys ...string) string {
	value, _ := geminiMapValue(source, keys...)
	text, _ := value.(string)
	return text
}

func geminiMapValue(source map[string]interface{}, keys ...string) (interface{}, bool) {
	for _, key := range keys {
		if value, ok := source[key]; ok {
			return value, true
		}
	}
	return nil, false
}

func geminiNumber(value interface{}) (float64, bool) {
	switch number := value.(type) {
	case float64:
		return number, true
	case float32:
		return float64(number), true
	case int:
		return float64(number), true
	case int8:
		return float64(number), true
	case int16:
		return float64(number), true
	case int32:
		return float64(number), true
	case int64:
		return float64(number), true
	case uint:
		return float64(number), true
	case uint8:
		return float64(number), true
	case uint16:
		return float64(number), true
	case uint32:
		return float64(number), true
	case uint64:
		return float64(number), true
	case json.Number:
		parsed, err := number.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

func coalesceCanonicalGeminiTextParts(parts []map[string]interface{}) []map[string]interface{} {
	if len(parts) <= 1 {
		return parts
	}
	merged := make([]map[string]interface{}, 0, len(parts))
	for _, part := range parts {
		last := len(merged) - 1
		if last >= 0 && canonicalGeminiTextPartIsMergeable(part) && canonicalGeminiTextPartIsMergeable(merged[last]) &&
			geminiPartThought(part) == geminiPartThought(merged[last]) {
			merged[last]["text"] = merged[last]["text"].(string) + part["text"].(string)
			continue
		}
		merged = append(merged, part)
	}
	return merged
}

func canonicalGeminiTextPartIsMergeable(part map[string]interface{}) bool {
	if _, ok := part["text"].(string); !ok {
		return false
	}
	if _, ok := part["thoughtSignature"]; ok {
		return false
	}
	for key := range part {
		if key != "text" && key != "thought" {
			return false
		}
	}
	return true
}

func geminiPartThought(part map[string]interface{}) bool {
	thought, _ := part["thought"].(bool)
	return thought
}

func streamGeminiResponse(
	w http.ResponseWriter,
	r *http.Request,
	modelName string,
	stream func(context.Context, func(*proxy.CallResult) error) error,
) {
	streamGeminiResponseWithProxy(w, r, modelName, nil, stream)
}

func streamGeminiResponseWithProxy(
	w http.ResponseWriter,
	r *http.Request,
	modelName string,
	vp *proxy.VertexProxy,
	stream func(context.Context, func(*proxy.CallResult) error) error,
) {
	streamGeminiResponseWithProxyOptions(w, r, modelName, vp, true, nil, stream)
}

func geminiRequestUsesSSE(r *http.Request, strictAltSSE bool) bool {
	if !strictAltSSE {
		return true
	}
	values, ok := r.URL.Query()["alt"]
	return ok && len(values) == 1 && values[0] == "sse"
}

type geminiStreamResponseState struct {
	includeThoughts     bool
	requestInput        interface{}
	aggregate           proxy.CallResult
	usageRevision       streamUsageRevision
	promptTokenEstimate int
	modelVersion        string
	responseID          string
	fallbackModel       string
	identityCommitted   bool
}

func (s *geminiStreamResponseState) buildChunk(result *proxy.CallResult) (map[string]interface{}, bool) {
	if result == nil {
		return nil, false
	}
	if s.requestInput != nil {
		if !s.identityCommitted {
			if s.modelVersion == "" && result.ModelVersion != "" {
				s.modelVersion = result.ModelVersion
			}
			if s.responseID == "" && result.ResponseID != "" {
				s.responseID = result.ResponseID
			}
		}
		s.usageRevision.observe(result)
		accumulateGeminiCallResult(&s.aggregate, result)
	}
	chunk := buildGeminiStreamResponseForRequest(result, s.includeThoughts)
	if len(chunk) == 0 {
		return nil, false
	}
	if s.requestInput == nil {
		return chunk, result.HasFinishReason()
	}

	// Keep usage at the response envelope level, but repeat the latest
	// cumulative value on every observable stream message for compatibility
	// with gateways that account for Gemini usage per chunk.
	delete(chunk, "usageMetadata")
	if !geminiStreamChunkHasMessage(chunk) {
		return nil, result.HasFinishReason()
	}
	return chunk, result.HasFinishReason()
}

func newGeminiStreamResponseID() string {
	data := make([]byte, 20)
	if _, err := rand.Read(data); err != nil {
		return fmt.Sprintf("response-%d", time.Now().UnixNano())
	}
	return base64.RawURLEncoding.EncodeToString(data)
}

func accumulateGeminiCallResult(dst, src *proxy.CallResult) {
	accumulateCallResult(dst, src)
	if dst == nil || src == nil {
		return
	}
	for _, incoming := range src.Candidates {
		matched := false
		for i := range dst.Candidates {
			candidate := &dst.Candidates[i]
			if candidate.Index != incoming.Index {
				continue
			}
			candidate.Parts = append(candidate.Parts, incoming.Parts...)
			candidate.TextParts = append(candidate.TextParts, incoming.TextParts...)
			candidate.ImageParts = append(candidate.ImageParts, incoming.ImageParts...)
			candidate.FunctionCalls = append(candidate.FunctionCalls, incoming.FunctionCalls...)
			matched = true
			break
		}
		if !matched {
			dst.Candidates = append(dst.Candidates, incoming)
		}
	}
}

type geminiStreamUsageSnapshot struct {
	metadata  map[string]interface{}
	estimated bool
}

func (s *geminiStreamResponseState) captureUsage() geminiStreamUsageSnapshot {
	if s == nil || s.requestInput == nil {
		return geminiStreamUsageSnapshot{}
	}
	if s.promptTokenEstimate == 0 {
		s.promptTokenEstimate = estimateJSONTokens(s.requestInput)
	}
	// Consume the shallow aggregate view synchronously and retain only computed
	// scalar metadata. No mutable CallResult data escapes this method.
	result := usageSnapshot(&s.aggregate, &s.usageRevision)
	metadata, estimated := geminiUsageMetadataWithPromptEstimate(s.promptTokenEstimate, &result)
	return geminiStreamUsageSnapshot{metadata: metadata, estimated: estimated}
}

type pendingGeminiStreamChunk struct {
	payload     map[string]interface{}
	usage       geminiStreamUsageSnapshot
	substantive bool
}

func (p *pendingGeminiStreamChunk) empty() bool {
	return p == nil || len(p.payload) == 0
}

func (p *pendingGeminiStreamChunk) set(payload map[string]interface{}, usage geminiStreamUsageSnapshot, substantive bool) {
	p.payload = payload
	p.usage = usage
	p.substantive = substantive
}

func (p *pendingGeminiStreamChunk) merge(payload map[string]interface{}, usage geminiStreamUsageSnapshot, substantive bool) {
	mergeGeminiStreamTail(p.payload, payload)
	p.usage = usage
	p.substantive = p.substantive || substantive
}

func (p *pendingGeminiStreamChunk) clear() {
	p.payload = nil
	p.usage = geminiStreamUsageSnapshot{}
	p.substantive = false
}

func (s *geminiStreamResponseState) prepareChunkForEmission(chunk map[string]interface{}, usage geminiStreamUsageSnapshot) bool {
	if chunk == nil || s.requestInput == nil {
		return false
	}
	if !s.identityCommitted {
		if s.modelVersion == "" {
			s.modelVersion = s.fallbackModel
		}
		if s.responseID == "" {
			s.responseID = newGeminiStreamResponseID()
		}
		s.identityCommitted = true
	}
	if s.responseID != "" {
		chunk["responseId"] = s.responseID
	}
	if s.modelVersion != "" {
		chunk["modelVersion"] = s.modelVersion
	}
	chunk["usageMetadata"] = usage.metadata
	return usage.estimated
}

func geminiStreamChunkHasMessage(chunk map[string]interface{}) bool {
	if chunk == nil {
		return false
	}
	for _, key := range []string{"candidates", "promptFeedback", "modelStatus"} {
		if _, ok := chunk[key]; ok {
			return true
		}
	}
	return false
}

func geminiStreamChunkHasSubstantiveCandidateContent(chunk map[string]interface{}) bool {
	candidates, _ := chunk["candidates"].([]map[string]interface{})
	for _, candidate := range candidates {
		content, _ := candidate["content"].(map[string]interface{})
		parts, _ := content["parts"].([]map[string]interface{})
		for _, part := range parts {
			if text, ok := part["text"].(string); ok && strings.TrimSpace(text) != "" {
				return true
			}
			for key, value := range part {
				if key != "text" && key != "thought" && key != "thoughtSignature" && value != nil {
					return true
				}
			}
		}
	}
	return false
}

func mergeGeminiStreamTail(dst, tail map[string]interface{}) {
	if dst == nil || tail == nil {
		return
	}
	for key, value := range tail {
		if key != "candidates" {
			dst[key] = value
		}
	}

	dstCandidates, _ := dst["candidates"].([]map[string]interface{})
	tailCandidates, _ := tail["candidates"].([]map[string]interface{})
	for _, tailCandidate := range tailCandidates {
		matched := false
		for _, dstCandidate := range dstCandidates {
			if fmt.Sprint(dstCandidate["index"]) != fmt.Sprint(tailCandidate["index"]) {
				continue
			}
			for key, value := range tailCandidate {
				if key == "content" {
					tailContent, _ := value.(map[string]interface{})
					if len(tailContent) == 0 {
						continue
					}
					dstContent, _ := dstCandidate["content"].(map[string]interface{})
					if dstContent == nil {
						dstContent = make(map[string]interface{}, len(tailContent))
						dstCandidate["content"] = dstContent
					}
					for contentKey, contentValue := range tailContent {
						if contentKey == "parts" {
							dstParts, _ := dstContent["parts"].([]map[string]interface{})
							tailParts, _ := contentValue.([]map[string]interface{})
							dstContent["parts"] = append(dstParts, tailParts...)
							continue
						}
						dstContent[contentKey] = contentValue
					}
					continue
				}
				dstCandidate[key] = value
			}
			matched = true
			break
		}
		if !matched {
			dstCandidates = append(dstCandidates, tailCandidate)
		}
	}
	if len(dstCandidates) > 0 {
		dst["candidates"] = dstCandidates
	}
}

func splitGeminiStreamFinishChunk(chunk map[string]interface{}) (map[string]interface{}, map[string]interface{}) {
	if chunk == nil {
		return nil, nil
	}
	contentChunk := make(map[string]interface{}, len(chunk))
	for key, value := range chunk {
		if key != "candidates" && key != "usageMetadata" && key != "modelVersion" && key != "responseId" {
			contentChunk[key] = value
		}
	}

	contentCandidates := make([]map[string]interface{}, 0)
	finishCandidates := make([]map[string]interface{}, 0)
	candidates, _ := chunk["candidates"].([]map[string]interface{})
	for _, candidate := range candidates {
		contentCandidate := make(map[string]interface{}, len(candidate))
		for key, value := range candidate {
			if key != "finishReason" && key != "finishMessage" {
				contentCandidate[key] = value
			}
		}
		if len(contentCandidate) > 1 || contentCandidate["content"] != nil {
			contentCandidates = append(contentCandidates, contentCandidate)
		}

		if finishReason, ok := candidate["finishReason"]; ok {
			finishCandidate := map[string]interface{}{
				"index":        candidate["index"],
				"finishReason": finishReason,
			}
			finishCandidates = append(finishCandidates, finishCandidate)
		}
	}
	if len(contentCandidates) > 0 {
		contentChunk["candidates"] = contentCandidates
	}
	if len(contentChunk) == 0 {
		contentChunk = nil
	}
	if len(finishCandidates) == 0 {
		return contentChunk, nil
	}
	return contentChunk, map[string]interface{}{"candidates": finishCandidates}
}

func streamGeminiResponseWithProxyOptions(
	w http.ResponseWriter,
	r *http.Request,
	modelName string,
	vp *proxy.VertexProxy,
	includeThoughts bool,
	requestInput interface{},
	stream func(context.Context, func(*proxy.CallResult) error) error,
) {
	ctx := r.Context()
	committed := false
	state := &geminiStreamResponseState{
		includeThoughts: includeThoughts,
		requestInput:    requestInput,
		fallbackModel:   modelName,
	}
	var pending pendingGeminiStreamChunk
	emit := func(chunk map[string]interface{}, usage geminiStreamUsageSnapshot) error {
		if len(chunk) == 0 {
			return nil
		}
		estimated := false
		if requestInput != nil {
			estimated = state.prepareChunkForEmission(chunk, usage)
			finishUsageEstimate(w, estimated)
		}
		if !committed {
			if requestInput != nil {
				declareUsageEstimateTrailer(w)
			}
			setSSEHeaders(w)
			committed = true
		}
		return writeGeminiStreamChunk(w, chunk)
	}
	err := stream(ctx, func(result *proxy.CallResult) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		chunk, tail := state.buildChunk(result)
		if len(chunk) == 0 {
			// Upstream may send a candidate shell containing only default values
			// and an unspecified promptFeedback. Normalize before deciding whether
			// this is an observable Gemini SSE event. It may still carry usage or a
			// hidden thought, so refresh the pending chunk's accounting snapshot.
			if !pending.empty() {
				pending.usage = state.captureUsage()
			}
			return nil
		}
		usageSnapshot := state.captureUsage()

		substantive := geminiStreamChunkHasSubstantiveCandidateContent(chunk)
		if tail {
			// The Vertex decoder normally delivers finish metadata separately.
			// If a caller supplies content and finishReason together, split them
			// so the Gemini compatibility surface still emits the requested final
			// candidate-only completion block.
			contentChunk, finishChunk := splitGeminiStreamFinishChunk(chunk)
			if len(contentChunk) > 0 {
				contentSubstantive := geminiStreamChunkHasSubstantiveCandidateContent(contentChunk)
				if !pending.empty() {
					if pending.substantive && contentSubstantive {
						if err := emit(pending.payload, pending.usage); err != nil {
							return err
						}
						pending.clear()
					} else {
						pending.merge(contentChunk, usageSnapshot, contentSubstantive)
					}
				}
				if pending.empty() {
					pending.set(contentChunk, usageSnapshot, contentSubstantive)
				}
			}
			if !pending.empty() {
				if pending.substantive {
					if err := emit(pending.payload, pending.usage); err != nil {
						return err
					}
				} else {
					// When the stream contains no actual output, retain the opaque
					// thoughtSignature by attaching its empty content part to the
					// terminal candidate instead of emitting an empty normal block.
					mergeGeminiStreamTail(finishChunk, pending.payload)
				}
				pending.clear()
			}
			return emit(finishChunk, usageSnapshot)
		}
		if !pending.empty() && pending.substantive && substantive {
			if err := emit(pending.payload, pending.usage); err != nil {
				return err
			}
			pending.clear()
		}
		if pending.empty() {
			pending.set(chunk, usageSnapshot, substantive)
		} else {
			// A thoughtSignature-only delta is transport metadata, not an actual
			// message. Fold it into the last visible message so the terminal
			// finishReason never lands on an empty text shell.
			pending.merge(chunk, usageSnapshot, substantive)
		}
		return nil
	})
	if err != nil {
		if requestContextCanceled(ctx, err) {
			log.Debug().Err(err).Str("model", modelName).Msg("Gemini stream client disconnected")
			return
		}
		if !proxy.IsUpstreamErrorLogged(err) {
			log.Error().Str("err", upstreamLogError(vp, err)).Str("model", modelName).Msg("Gemini Native stream failed")
		}
		if !pending.empty() {
			if writeErr := emit(pending.payload, pending.usage); writeErr != nil {
				return
			}
			pending.clear()
		}
		if !committed {
			writeUpstreamProtocolError(w, r, vp, err)
			return
		}
		_ = writeGeminiStreamError(w, vp, err)
		return
	}
	if !pending.empty() {
		if err := emit(pending.payload, pending.usage); err != nil {
			if requestContextCanceled(ctx, err) {
				log.Debug().Err(err).Str("model", modelName).Msg("Gemini stream client disconnected")
				return
			}
			log.Error().Err(err).Str("model", modelName).Msg("Gemini Native stream final chunk failed")
			return
		}
	}
	if !committed {
		setSSEHeaders(w)
	}
}

func writeGeminiStreamChunk(w http.ResponseWriter, chunk map[string]interface{}) error {
	data, _ := sonic.Marshal(chunk)
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

func writeGeminiStreamError(w http.ResponseWriter, vp *proxy.VertexProxy, streamErr error) error {
	data, _ := sonic.Marshal(geminiUpstreamErrorResponse(vp, streamErr))
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

func geminiErrorResponse(err error) map[string]interface{} {
	return geminiErrorResponseWithMessage(err, publicServerErrorMessageFor(err))
}

func geminiUpstreamErrorResponse(vp *proxy.VertexProxy, err error) map[string]interface{} {
	return geminiErrorResponseWithMessage(err, publicUpstreamErrorMessage(vp, err))
}

func geminiErrorResponseWithMessage(err error, message string) map[string]interface{} {
	status := proxy.HTTPStatusForError(err)
	return map[string]interface{}{
		"error": map[string]interface{}{
			"code":    status,
			"message": message,
			"status":  geminiErrorStatus(status),
		},
	}
}

// toSliceMap 将 interface{} 转换为 []map[string]interface{}
func toSliceMap(v interface{}) ([]map[string]interface{}, bool) {
	if v == nil {
		return nil, false
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil, false
	}
	result := make([]map[string]interface{}, 0, len(arr))
	for _, item := range arr {
		if m, ok := item.(map[string]interface{}); ok {
			result = append(result, m)
		}
	}
	return result, true
}

// toSafetySettings 转换 safetySettings
func toSafetySettings(v interface{}) []map[string]string {
	if v == nil {
		return nil
	}
	arr, ok := v.([]interface{})
	if !ok {
		return nil
	}
	result := make([]map[string]string, 0, len(arr))
	for _, item := range arr {
		if m, ok := item.(map[string]interface{}); ok {
			ss := make(map[string]string)
			for k, val := range m {
				ss[k] = fmt.Sprintf("%v", val)
			}
			result = append(result, ss)
		}
	}
	return result
}
