package handler

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"vertex2api/model"
	"vertex2api/proxy"
	schemanorm "vertex2api/schema"

	"github.com/bytedance/sonic"
	"github.com/rs/zerolog/log"
)

const chatLivenessProbeResponse = "Hello! How can I help you today?"

// ChatCompletions 处理 POST /v1/chat/completions。可选的防测活开关依次为“拒绝”和“构造正常响应”。
func ChatCompletions(vp *proxy.VertexProxy, allowCustomModelNames bool, livenessProbeOptions ...bool) http.Handler {
	rejectLivenessProbe := len(livenessProbeOptions) > 0 && livenessProbeOptions[0]
	respondLivenessProbe := len(livenessProbeOptions) > 1 && livenessProbeOptions[1]
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(proxy.WithCompatibilityLayer(r.Context(), proxy.CompatibilityLayerOpenAIChatCompletions))
		var req model.ChatCompletionRequest
		_, ok := readJSONRequest(w, r, &req)
		if !ok {
			return
		}

		if respondLivenessProbe && isWastefulLivenessProbe(req) {
			writeChatLivenessProbeResponse(w, r, req)
			return
		}

		if rejectLivenessProbe && isWastefulLivenessProbe(req) {
			WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{Error: &model.APIError{
				Message: `请勿使用模型推理请求进行验活。请改用 GET /health；发送 "hi" 等测试提示词会占用模型并发容量，挤占正常用户请求并浪费服务器资源。`,
				Type:    "invalid_request_error",
				Code:    "health_check_not_supported",
			}})
			return
		}

		if message := validateModelName(req.Model, allowCustomModelNames); message != "" {
			WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{Error: &model.APIError{
				Message: message,
				Type:    "invalid_request_error",
			}})
			return
		}

		if message := validateOpenAIRequest(req); message != "" {
			WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{Error: &model.APIError{
				Message: message,
				Type:    "invalid_request_error",
			}})
			return
		}

		// 转换 OpenAI messages → Vertex contents
		contents, systemInstruction := convertMessages(req.Model, req.Messages)

		// 构建 generationConfig
		genConfig := map[string]interface{}{}
		if req.Temperature != nil {
			genConfig["temperature"] = *req.Temperature
		}
		if req.TopP != nil {
			genConfig["topP"] = *req.TopP
		}
		var maxTokens int
		if req.MaxCompletionTokens != nil {
			maxTokens = *req.MaxCompletionTokens
		} else if req.MaxTokens != nil {
			maxTokens = *req.MaxTokens
		}
		if maxTokens > 0 {
			genConfig["maxOutputTokens"] = maxTokens
		}
		if thinkingConfig := openAIReasoningConfig(req.Model, req.ReasoningEffort); thinkingConfig != nil {
			genConfig["thinkingConfig"] = thinkingConfig
		}
		if stopSequences := openAIStopSequences(req.Stop); len(stopSequences) > 0 {
			genConfig["stopSequences"] = stopSequences
		}
		if req.FrequencyPenalty != nil {
			genConfig["frequencyPenalty"] = *req.FrequencyPenalty
		}
		if req.PresencePenalty != nil {
			genConfig["presencePenalty"] = *req.PresencePenalty
		}
		if req.Seed != nil {
			genConfig["seed"] = *req.Seed
		}
		if req.N != nil && *req.N > 1 {
			genConfig["candidateCount"] = *req.N
		}
		applyOpenAIResponseFormat(genConfig, req.ResponseFormat)
		options := buildOpenAIRequestOptions(req)

		if req.Stream {
			bodyJSON, tokenLease, err := vp.BuildBodyWithTokenWithOptionsContext(r.Context(), req.Model, contents, genConfig, nil, systemInstruction, options)
			if err != nil {
				if requestContextCanceled(r.Context(), err) {
					log.Debug().Err(err).Str("model", req.Model).Msg("OpenAI stream request canceled before upstream call")
					return
				}
				log.Error().Err(err).Str("model", req.Model).Msg("OpenAI Chat Completions request conversion failed")
				WriteJSON(w, http.StatusInternalServerError, publicServerErrorResponse(err))
				return
			}
			streamResponseForRequestWithProxy(w, r, req, vp, func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
				return vp.StreamWithTokenContext(ctx, bodyJSON, tokenLease, onChunk)
			})
			return
		}

		result, err := vp.CallWithTokenWithOptionsContext(r.Context(), req.Model, contents, genConfig, nil, systemInstruction, options)
		if err != nil {
			if requestContextCanceled(r.Context(), err) {
				log.Debug().Err(err).Str("model", req.Model).Msg("OpenAI request canceled")
				return
			}
			if !proxy.IsUpstreamErrorLogged(err) {
				log.Error().Str("err", vp.UpstreamLogError(err)).Str("model", req.Model).Msg("OpenAI Chat Completions call failed")
			}
			writeUpstreamProtocolError(w, r, vp, err)
			return
		}

		sendChatResponse(w, req, req.Model, result)
	})
}

func writeChatLivenessProbeResponse(w http.ResponseWriter, r *http.Request, req model.ChatCompletionRequest) {
	result := &proxy.CallResult{
		TextParts:    []model.TextPart{{Text: chatLivenessProbeResponse}},
		FinishReason: "STOP",
	}
	if req.Stream {
		streamResponseForRequest(w, r, req, func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
			return onChunk(result)
		})
		return
	}
	sendChatResponse(w, req, req.Model, result)
}

func isWastefulLivenessProbe(req model.ChatCompletionRequest) bool {
	if len(req.Messages) != 1 || len(req.Tools) != 0 || req.ToolChoice != nil {
		return false
	}

	message := req.Messages[0]
	if !strings.EqualFold(strings.TrimSpace(message.Role), "user") ||
		message.Name != "" || message.ToolCallID != "" || len(message.ToolCalls) != 0 ||
		strings.TrimSpace(message.ReasoningContent) != "" {
		return false
	}

	content, ok := message.Content.(string)
	return ok && strings.EqualFold(strings.TrimSpace(content), "hi")
}

func validateOpenAIRequest(req model.ChatCompletionRequest) string {
	if strings.TrimSpace(req.Model) == "" {
		return "model is required"
	}
	if len(req.Messages) == 0 {
		return "messages is required"
	}
	for _, message := range req.Messages {
		switch strings.ToLower(strings.TrimSpace(message.Role)) {
		case "user", "assistant", "system", "developer", "tool":
		default:
			return "messages roles must be user, assistant, system, developer, or tool"
		}
	}
	if req.N != nil && (*req.N < 1 || *req.N > 128) {
		return "n must be between 1 and 128"
	}
	if (req.MaxTokens != nil && *req.MaxTokens < 1) || (req.MaxCompletionTokens != nil && *req.MaxCompletionTokens < 1) {
		return "max_tokens and max_completion_tokens must be greater than zero"
	}
	if message := validateOpenAIReasoningEffort(req.Model, req.ReasoningEffort); message != "" {
		return message
	}
	if req.Stream && req.N != nil && *req.N > 1 {
		return "n greater than 1 is not supported for streaming"
	}
	if req.Temperature != nil && (*req.Temperature < 0 || *req.Temperature > 2) {
		return "temperature must be between 0 and 2"
	}
	if req.TopP != nil && (*req.TopP < 0 || *req.TopP > 1) {
		return "top_p must be between 0 and 1"
	}
	for name, value := range map[string]*float64{
		"frequency_penalty": req.FrequencyPenalty,
		"presence_penalty":  req.PresencePenalty,
	} {
		if value != nil && (*value < -2 || *value > 2) {
			return name + " must be between -2 and 2"
		}
	}
	if req.Stop != nil {
		stopSequences := openAIStopSequences(req.Stop)
		if len(stopSequences) == 0 || len(stopSequences) > 4 {
			return "stop must be a string or an array of up to 4 strings"
		}
	}
	if len(req.ResponseFormat) > 0 {
		formatType, _ := req.ResponseFormat["type"].(string)
		switch formatType {
		case "text", "json_object":
		case "json_schema":
			jsonSchema, ok := req.ResponseFormat["json_schema"].(map[string]interface{})
			if !ok || jsonSchema["schema"] == nil {
				return "response_format.json_schema.schema is required"
			}
		default:
			return "response_format.type must be text, json_object, or json_schema"
		}
	}
	for index, tool := range req.Tools {
		toolType, _ := tool["type"].(string)
		if toolType == "function" {
			function, ok := mapValue(tool, "function")
			name, _ := function["name"].(string)
			if !ok || strings.TrimSpace(name) == "" {
				return fmt.Sprintf("tools[%d].function.name is required", index)
			}
			continue
		}
		if _, ok := convertNativeGeminiTool(tool); !ok {
			return fmt.Sprintf("tools[%d].type %q has no Vertex equivalent", index, toolType)
		}
	}
	return ""
}

func validateOpenAIReasoningEffort(modelName string, effort *string) string {
	if effort == nil {
		return ""
	}

	normalized := strings.ToLower(strings.TrimSpace(*effort))
	switch normalized {
	case "minimal", "low", "medium", "high", "xhigh", "max":
	case "none":
		if strings.HasPrefix(strings.ToLower(modelName), "gemini-2.5-") && !strings.Contains(strings.ToLower(modelName), "pro") {
			return ""
		}
		return "reasoning_effort none is not supported by this model"
	default:
		return "reasoning_effort must be none, minimal, low, medium, high, xhigh, or max"
	}

	lowerModel := strings.ToLower(modelName)
	if !strings.HasPrefix(lowerModel, "gemini-3") && !strings.HasPrefix(lowerModel, "gemini-2.5-") {
		return "reasoning_effort is only supported by Gemini 3 and Gemini 2.5 models"
	}
	return ""
}

func openAIReasoningConfig(modelName string, effort *string) map[string]interface{} {
	if effort == nil {
		return nil
	}

	normalized := strings.ToLower(strings.TrimSpace(*effort))
	lowerModel := strings.ToLower(modelName)
	if strings.HasPrefix(lowerModel, "gemini-3") {
		level := strings.ToUpper(normalized)
		if normalized == "xhigh" || normalized == "max" {
			level = "HIGH"
		}
		// Gemini Pro models do not expose a minimal level; Google's OpenAI
		// compatibility mapping promotes that request to low.
		if normalized == "minimal" && strings.Contains(lowerModel, "pro") {
			level = "LOW"
		}
		return map[string]interface{}{"thinkingLevel": level}
	}

	if strings.HasPrefix(lowerModel, "gemini-2.5-") {
		budget := 0
		switch normalized {
		case "minimal", "low":
			budget = 1024
		case "medium":
			budget = 8192
		case "high", "xhigh", "max":
			budget = 24576
		}
		return map[string]interface{}{"thinkingBudget": budget}
	}
	return nil
}

func openAIStopSequences(value interface{}) []string {
	switch typed := value.(type) {
	case string:
		if typed != "" {
			return []string{typed}
		}
	case []interface{}:
		result := make([]string, 0, len(typed))
		for _, item := range typed {
			if text, ok := item.(string); ok && text != "" {
				result = append(result, text)
			}
		}
		return result
	case []string:
		return typed
	}
	return nil
}

func applyOpenAIResponseFormat(genConfig map[string]interface{}, responseFormat map[string]interface{}) {
	formatType, _ := responseFormat["type"].(string)
	switch formatType {
	case "json_object":
		genConfig["responseMimeType"] = "application/json"
	case "json_schema":
		genConfig["responseMimeType"] = "application/json"
		if jsonSchema, ok := responseFormat["json_schema"].(map[string]interface{}); ok {
			if schemaValue, ok := jsonSchema["schema"]; ok {
				genConfig["responseJsonSchema"] = schemanorm.Normalize(schemaValue)
			}
		}
	}
}

func sendChatResponse(w http.ResponseWriter, req model.ChatCompletionRequest, modelName string, result *proxy.CallResult) {
	usage, estimated := openAIUsage(req, result)
	if estimated {
		w.Header().Set("X-Usage-Estimated", "true")
	}
	choices := make([]model.ChatChoice, 0, maxInt(1, len(result.Candidates)))
	if len(result.Candidates) > 0 {
		for _, candidate := range result.Candidates {
			candidateResult := callResultFromCandidate(candidate)
			finishReason := openAIFinishReason(candidateResult)
			choices = append(choices, model.ChatChoice{
				Index:        candidate.Index,
				Message:      buildOpenAIMessage(candidateResult),
				FinishReason: &finishReason,
			})
		}
	} else {
		finishReason := openAIFinishReason(result)
		choices = append(choices, model.ChatChoice{
			Index:        0,
			Message:      buildOpenAIMessage(result),
			FinishReason: &finishReason,
		})
	}

	resp := model.ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: choices,
		Usage:   usage,
	}

	WriteJSON(w, http.StatusOK, resp)
}

func callResultFromCandidate(candidate proxy.CandidateResult) *proxy.CallResult {
	return &proxy.CallResult{
		Parts:             candidate.Parts,
		Role:              candidate.Role,
		TextParts:         candidate.TextParts,
		ImageParts:        candidate.ImageParts,
		FunctionCalls:     candidate.FunctionCalls,
		FinishReason:      candidate.FinishReason,
		GroundingMetadata: candidate.GroundingMetadata,
	}
}

func estimateOpenAIPromptTokens(req model.ChatCompletionRequest) int {
	payload := map[string]interface{}{"messages": req.Messages}
	if len(req.Tools) > 0 {
		payload["tools"] = req.Tools
	}
	if len(req.ResponseFormat) > 0 {
		payload["response_format"] = req.ResponseFormat
	}
	return estimateJSONTokens(payload)
}

func estimateOpenAICompletionTokens(result *proxy.CallResult) int {
	candidateTokens, reasoningTokens := estimateCompletionTokenBreakdown(result)
	return saturatingTokenAdd(candidateTokens, reasoningTokens)
}

func estimateCompletionTokenBreakdown(result *proxy.CallResult) (candidateTokens, reasoningTokens int) {
	if result != nil && len(result.Candidates) > 0 {
		for _, candidate := range result.Candidates {
			candidateEstimate, reasoningEstimate := estimateCompletionTokenBreakdown(callResultFromCandidate(candidate))
			candidateTokens = saturatingTokenAdd(candidateTokens, candidateEstimate)
			reasoningTokens = saturatingTokenAdd(reasoningTokens, reasoningEstimate)
		}
		return candidateTokens, reasoningTokens
	}
	if result == nil {
		return 0, 0
	}

	if len(result.Parts) > 0 {
		var candidateText, reasoningText strings.Builder
		for i := range result.Parts {
			part := result.Parts[i]
			if part.Thought {
				reasoningText.WriteString(part.Text)
			} else {
				candidateText.WriteString(part.Text)
			}
			part.Text = ""
			candidateEstimate, reasoningEstimate := estimateVertexPartTokens(part)
			candidateTokens = saturatingTokenAdd(candidateTokens, candidateEstimate)
			reasoningTokens = saturatingTokenAdd(reasoningTokens, reasoningEstimate)
		}
		candidateTokens = saturatingTokenAdd(candidateTokens, estimateTextTokens(candidateText.String()))
		reasoningTokens = saturatingTokenAdd(reasoningTokens, estimateTextTokens(reasoningText.String()))
		return candidateTokens, reasoningTokens
	}

	var candidateText, reasoningText strings.Builder
	for _, part := range result.TextParts {
		if part.Thought {
			reasoningText.WriteString(part.Text)
		} else {
			candidateText.WriteString(part.Text)
		}
	}
	candidateTokens = saturatingTokenAdd(candidateTokens, estimateTextTokens(candidateText.String()))
	reasoningTokens = saturatingTokenAdd(reasoningTokens, estimateTextTokens(reasoningText.String()))
	for i := range result.ImageParts {
		candidateTokens = saturatingTokenAdd(candidateTokens, estimateTokenValue(result.ImageParts[i], "inlineData"))
	}
	for i := range result.FunctionCalls {
		candidateTokens = saturatingTokenAdd(candidateTokens, estimateFunctionCallTokens(&result.FunctionCalls[i]))
	}
	return candidateTokens, reasoningTokens
}

func estimateVertexPartTokens(part model.VertexPart) (candidateTokens, reasoningTokens int) {
	if part.Text != "" {
		tokens := estimateTextTokens(part.Text)
		if part.Thought {
			reasoningTokens = saturatingTokenAdd(reasoningTokens, tokens)
		} else {
			candidateTokens = saturatingTokenAdd(candidateTokens, tokens)
		}
	}
	if part.InlineData != nil {
		candidateTokens = saturatingTokenAdd(candidateTokens, estimateTokenValue(*part.InlineData, "inlineData"))
	}
	for _, value := range []map[string]interface{}{
		part.FileData,
		part.ExecutableCode,
		part.CodeExecutionResult,
	} {
		if len(value) > 0 {
			candidateTokens = saturatingTokenAdd(candidateTokens, estimateTokenValue(value, ""))
		}
	}
	if part.FunctionCall != nil {
		candidateTokens = saturatingTokenAdd(candidateTokens, estimateFunctionCallTokens(part.FunctionCall))
	}
	if part.FunctionResponse != nil {
		candidateTokens = saturatingTokenAdd(candidateTokens, estimateTokenValue(part.FunctionResponse, ""))
	}
	return candidateTokens, reasoningTokens
}

func estimateFunctionCallTokens(call *model.FunctionCall) int {
	if call == nil {
		return 0
	}
	// IDs, partial argument deltas, continuation flags, and thought signatures
	// are transport state rather than generated semantic tool-call content.
	return estimateTokenValue(map[string]interface{}{
		"name": call.Name,
		"args": call.Args,
	}, "")
}

func streamResponse(
	w http.ResponseWriter,
	r *http.Request,
	modelName string,
	stream func(context.Context, func(*proxy.CallResult) error) error,
) {
	streamResponseForRequest(w, r, model.ChatCompletionRequest{Model: modelName, Stream: true}, stream)
}

func streamResponseForRequest(
	w http.ResponseWriter,
	r *http.Request,
	req model.ChatCompletionRequest,
	stream func(context.Context, func(*proxy.CallResult) error) error,
) {
	streamResponseForRequestWithProxy(w, r, req, nil, stream)
}

func streamResponseForRequestWithProxy(
	w http.ResponseWriter,
	r *http.Request,
	req model.ChatCompletionRequest,
	vp *proxy.VertexProxy,
	stream func(context.Context, func(*proxy.CallResult) error) error,
) {
	modelName := req.Model

	id := fmt.Sprintf("chatcmpl-%d", time.Now().UnixNano())
	streamState := &openAIStreamState{}
	aggregate := &proxy.CallResult{}
	usageRevision := &streamUsageRevision{}
	committed := false
	includeUsage := req.StreamOptions != nil && req.StreamOptions.IncludeUsage

	ctx := r.Context()
	err := stream(ctx, func(result *proxy.CallResult) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if result == nil || result.IsEmpty() {
			return nil
		}
		usageRevision.observe(result)
		accumulateCallResult(aggregate, result)
		if !result.HasContent() {
			return nil
		}
		if !committed {
			if includeUsage {
				declareUsageEstimateTrailer(w)
			}
			setSSEHeaders(w)
			committed = true
		}
		return writeOpenAIStreamChunk(w, id, modelName, result, streamState)
	})
	if err != nil {
		if requestContextCanceled(ctx, err) {
			log.Debug().Err(err).Str("model", modelName).Msg("OpenAI stream client disconnected")
			return
		}
		if !proxy.IsUpstreamErrorLogged(err) {
			log.Error().Str("err", upstreamLogError(vp, err)).Str("model", modelName).Msg("OpenAI Chat Completions stream failed")
		}
		if !committed {
			writeUpstreamProtocolError(w, r, vp, err)
			return
		}
		_ = writeOpenAIStreamError(w, vp, err)
		return
	}
	if err := ctx.Err(); err != nil {
		log.Debug().Err(err).Str("model", modelName).Msg("OpenAI stream client disconnected")
		return
	}
	if !committed {
		if includeUsage {
			declareUsageEstimateTrailer(w)
		}
		setSSEHeaders(w)
	}
	var usage *model.Usage
	if includeUsage {
		snapshot := usageSnapshot(aggregate, usageRevision)
		var estimated bool
		usage, estimated = openAIUsage(req, &snapshot)
		finishUsageEstimate(w, estimated)
	}
	// Emit the protocol terminator only after the upstream stream reaches EOF.
	// Vertex finishReason is completion metadata and may arrive before later
	// content chunks, so it must never end the downstream stream by itself.
	finishReason := openAIFinishReason(aggregate)
	if err := writeOpenAIStreamEnd(w, id, modelName, finishReason, usage); err != nil {
		if requestContextCanceled(ctx, err) {
			log.Debug().Err(err).Str("model", modelName).Msg("OpenAI stream client disconnected")
			return
		}
		log.Error().Err(err).Str("model", modelName).Msg("OpenAI stream end failed")
	}
}

func accumulateCallResult(dst, src *proxy.CallResult) {
	if dst == nil || src == nil {
		return
	}
	dst.Parts = append(dst.Parts, src.Parts...)
	dst.TextParts = append(dst.TextParts, src.TextParts...)
	dst.ImageParts = append(dst.ImageParts, src.ImageParts...)
	dst.FunctionCalls = append(dst.FunctionCalls, src.FunctionCalls...)
	if src.FinishReason != "" {
		dst.FinishReason = src.FinishReason
	}
	if len(src.UsageMetadata) > 0 {
		dst.UsageMetadata = src.UsageMetadata
	}
	if src.GroundingMetadata != nil {
		dst.GroundingMetadata = src.GroundingMetadata
	}
	if len(src.PromptFeedback) > 0 {
		dst.PromptFeedback = src.PromptFeedback
	}
	if len(src.ModelStatus) > 0 {
		dst.ModelStatus = src.ModelStatus
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
}

func writeOpenAIStreamChunk(w http.ResponseWriter, id, modelName string, result *proxy.CallResult, streamState *openAIStreamState) error {
	chunk := model.ChatCompletionResponse{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []model.ChatChoice{
			{
				Index: 0,
				Delta: buildOpenAIDelta(result, streamState),
			},
		},
	}

	chunkData, _ := sonic.Marshal(chunk)
	if _, err := io.WriteString(w, "data: "); err != nil {
		return err
	}
	if _, err := w.Write(chunkData); err != nil {
		return err
	}
	if _, err := io.WriteString(w, "\n\n"); err != nil {
		return err
	}
	return flushResponse(w)
}

func writeOpenAIStreamEnd(w http.ResponseWriter, id, modelName, finishReason string, usage *model.Usage) error {
	endChunk := model.ChatCompletionResponse{
		ID:      id,
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   modelName,
		Choices: []model.ChatChoice{
			{
				Index:        0,
				Delta:        &model.ChatMessage{},
				FinishReason: &finishReason,
			},
		},
	}
	endData, _ := sonic.Marshal(endChunk)
	if _, err := io.WriteString(w, "data: "); err != nil {
		return err
	}
	if _, err := w.Write(endData); err != nil {
		return err
	}
	if _, err := io.WriteString(w, "\n\n"); err != nil {
		return err
	}
	if usage != nil {
		usageChunk := model.ChatCompletionResponse{
			ID:      id,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   modelName,
			Choices: []model.ChatChoice{},
			Usage:   usage,
		}
		usageData, _ := sonic.Marshal(usageChunk)
		if _, err := io.WriteString(w, "data: "); err != nil {
			return err
		}
		if _, err := w.Write(usageData); err != nil {
			return err
		}
		if _, err := io.WriteString(w, "\n\n"); err != nil {
			return err
		}
	}
	if _, err := io.WriteString(w, "data: [DONE]\n\n"); err != nil {
		return err
	}
	return flushResponse(w)
}

func writeOpenAIStreamError(w http.ResponseWriter, vp *proxy.VertexProxy, streamErr error) error {
	status := proxy.HTTPStatusForError(streamErr)
	data, _ := sonic.Marshal(model.ErrorResponse{Error: &model.APIError{
		Message: publicUpstreamErrorMessage(vp, streamErr),
		Type:    openAIErrorType(status),
	}})
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

func buildOpenAIMessage(result *proxy.CallResult) *model.ChatMessage {
	content, reasoning := buildResponseContent(result)
	var toolCalls []model.ChatToolCall
	if result != nil {
		toolCalls = buildOpenAIToolCalls(result.FunctionCalls, false)
	}
	msg := &model.ChatMessage{
		Role:             "assistant",
		Content:          content,
		ReasoningContent: reasoning,
		ToolCalls:        toolCalls,
	}
	if result != nil && result.GroundingMetadata != nil {
		msg.Annotations = openAIAnnotationsFromGrounding(result.GroundingMetadata)
	}
	return msg
}

func buildOpenAIDelta(result *proxy.CallResult, streamState *openAIStreamState) *model.ChatMessage {
	content, reasoning := buildStreamResponseContent(result)
	role := ""
	if streamState != nil && !streamState.hasIndexedContent {
		streamState.hasIndexedContent = true
		role = "assistant"
	}
	var toolCalls []model.ChatToolCall
	if result != nil {
		toolCalls = buildOpenAIToolCalls(result.FunctionCalls, true)
	}
	msg := &model.ChatMessage{
		Role:             role,
		Content:          content,
		ReasoningContent: reasoning,
		ToolCalls:        toolCalls,
	}
	if result != nil && result.GroundingMetadata != nil {
		if streamState == nil || !streamState.emittedGrounding {
			if streamState != nil {
				streamState.emittedGrounding = true
			}
			msg.Annotations = openAIAnnotationsFromGrounding(result.GroundingMetadata)
		}
	}
	return msg
}

func openAIAnnotationsFromGrounding(gm *model.GroundingMetadata) []model.ChatAnnotation {
	if gm == nil || len(gm.GroundingChunks) == 0 || len(gm.GroundingSupports) == 0 {
		return nil
	}
	var annotations []model.ChatAnnotation
	for _, support := range gm.GroundingSupports {
		if support.Segment == nil {
			continue
		}
		for _, chunkIndex := range support.GroundingChunkIndices {
			if chunkIndex < 0 || chunkIndex >= len(gm.GroundingChunks) {
				continue
			}
			uri, title := groundingChunkLink(gm.GroundingChunks[chunkIndex])
			if uri == "" {
				continue
			}
			annotations = append(annotations, model.ChatAnnotation{
				Type: "url_citation",
				URLCitation: &model.ChatURLCitation{
					StartIndex: support.Segment.StartIndex,
					EndIndex:   support.Segment.EndIndex,
					URL:        uri,
					Title:      title,
				},
			})
		}
	}
	return annotations
}

func groundingChunkLink(chunk model.GroundingChunk) (string, string) {
	if chunk.Web != nil {
		return chunk.Web.URI, chunk.Web.Title
	}
	if len(chunk.RetrievedContext) > 0 {
		uri, _ := chunk.RetrievedContext["uri"].(string)
		title, _ := chunk.RetrievedContext["title"].(string)
		return uri, title
	}
	if len(chunk.Maps) > 0 {
		uri, _ := chunk.Maps["uri"].(string)
		if uri == "" {
			uri, _ = chunk.Maps["googleMapsUri"].(string)
		}
		title, _ := chunk.Maps["title"].(string)
		return uri, title
	}
	return "", ""
}

type openAIStreamState struct {
	hasIndexedContent bool
	emittedGrounding  bool
}

func generateOpenAIToolCallID(index int) string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return fmt.Sprintf("call_%x_%d", b, index)
}

func buildOpenAIToolCalls(functionCalls []model.FunctionCall, includeIndex bool) []model.ChatToolCall {
	if len(functionCalls) == 0 {
		return nil
	}

	toolCalls := make([]model.ChatToolCall, 0, len(functionCalls))
	for i, functionCall := range functionCalls {
		arguments := "{}"
		if functionCall.Args != nil {
			if data, err := sonic.Marshal(functionCall.Args); err == nil {
				arguments = string(data)
			}
		}
		toolCall := model.ChatToolCall{
			ID:   openAIToolCallID(functionCall, i),
			Type: "function",
			Function: &model.ChatFunctionCall{
				Name:      functionCall.Name,
				Arguments: arguments,
			},
		}
		if includeIndex {
			index := i
			toolCall.Index = &index
		}
		toolCalls = append(toolCalls, toolCall)
	}
	return toolCalls
}

func openAIFinishReason(result *proxy.CallResult) string {
	if result != nil && len(result.FunctionCalls) > 0 {
		return "tool_calls"
	}
	if result == nil {
		return "stop"
	}
	switch strings.ToUpper(result.FinishReason) {
	case "MAX_TOKENS":
		return "length"
	default:
		return "stop"
	}
}

const openAIToolSignatureMarker = "__vertex_sig_"

func openAIToolCallID(functionCall model.FunctionCall, index int) string {
	id := functionCall.ID
	if id == "" {
		id = generateOpenAIToolCallID(index)
	}
	if functionCall.ThoughtSignature == "" || strings.Contains(id, openAIToolSignatureMarker) {
		return id
	}
	return id + openAIToolSignatureMarker + base64.RawURLEncoding.EncodeToString([]byte(functionCall.ThoughtSignature))
}

func buildOpenAIRequestOptions(req model.ChatCompletionRequest) *proxy.VertexRequestOptions {
	tools := convertOpenAITools(req.Tools)
	toolConfig := convertOpenAIToolChoice(req.ToolChoice)
	if tools == nil && toolConfig == nil {
		return nil
	}
	return &proxy.VertexRequestOptions{
		Tools:      tools,
		ToolConfig: toolConfig,
	}
}

func convertOpenAITools(tools []map[string]interface{}) interface{} {
	var converted []interface{}
	var functionDeclarations []interface{}

	for _, tool := range tools {
		if function, ok := mapValue(tool, "function"); ok && isFunctionTool(tool) {
			functionDeclarations = append(functionDeclarations, openAIFunctionDeclaration(function))
			continue
		}
		if nativeTool, ok := convertNativeGeminiTool(tool); ok {
			converted = append(converted, nativeTool)
			continue
		}
		converted = append(converted, copyInterfaceMap(tool))
	}
	if len(functionDeclarations) > 0 {
		converted = append(converted, map[string]interface{}{"functionDeclarations": functionDeclarations})
	}
	if len(converted) == 0 {
		return nil
	}
	return converted
}

func isFunctionTool(tool map[string]interface{}) bool {
	toolType, _ := tool["type"].(string)
	return toolType == "function"
}

func openAIFunctionDeclaration(function map[string]interface{}) map[string]interface{} {
	declaration := copyInterfaceMap(function)
	delete(declaration, "strict")
	return declaration
}

func convertOpenAIToolChoice(choice interface{}) interface{} {
	switch v := choice.(type) {
	case nil:
		return nil
	case string:
		return functionCallingConfigForMode(openAIToolChoiceMode(v), nil)
	case map[string]interface{}:
		if config, ok := v["functionCallingConfig"]; ok {
			return map[string]interface{}{"functionCallingConfig": config}
		}
		if function, ok := mapValue(v, "function"); ok {
			if name, _ := function["name"].(string); name != "" {
				return functionCallingConfigForMode("ANY", []string{name})
			}
		}
	default:
		return nil
	}
	return nil
}

func openAIToolChoiceMode(choice string) string {
	switch strings.ToLower(strings.TrimSpace(choice)) {
	case "none":
		return "NONE"
	case "auto":
		return "AUTO"
	case "required":
		return "ANY"
	default:
		return ""
	}
}

func functionCallingConfigForMode(mode string, allowedFunctionNames []string) interface{} {
	if mode == "" {
		return nil
	}
	config := map[string]interface{}{"mode": mode}
	if len(allowedFunctionNames) > 0 {
		config["allowedFunctionNames"] = allowedFunctionNames
	}
	return map[string]interface{}{"functionCallingConfig": config}
}

func mapValue(m map[string]interface{}, key string) (map[string]interface{}, bool) {
	value, ok := m[key].(map[string]interface{})
	return value, ok
}

func copyInterfaceMap(src map[string]interface{}) map[string]interface{} {
	dst := make(map[string]interface{}, len(src))
	for key, value := range src {
		dst[key] = value
	}
	return dst
}

func buildResponseContent(result *proxy.CallResult) (content interface{}, reasoning string) {
	if result == nil {
		return "", ""
	}
	cStr, rStr := buildOpenAITextAndReasoning(result)
	if cStr == "" && len(result.FunctionCalls) > 0 {
		return nil, rStr
	}
	return cStr, rStr
}

func buildStreamResponseContent(result *proxy.CallResult) (content interface{}, reasoning string) {
	if result == nil {
		return nil, ""
	}
	cStr, rStr := buildOpenAITextAndReasoning(result)
	if cStr == "" {
		return nil, rStr
	}
	return cStr, rStr
}

func buildOpenAITextAndReasoning(result *proxy.CallResult) (string, string) {
	var contentBuilder, reasoningBuilder strings.Builder
	if len(result.Parts) > 0 {
		for _, part := range result.Parts {
			switch {
			case part.Text != "":
				if part.Thought {
					reasoningBuilder.WriteString(part.Text)
				} else {
					contentBuilder.WriteString(part.Text)
				}
			case part.InlineData != nil && part.InlineData.Data != "":
				appendMarkdownSeparator(&contentBuilder)
				contentBuilder.WriteString("![image](")
				contentBuilder.WriteString(imageDataURL(*part.InlineData))
				contentBuilder.WriteString(")")
			case len(part.FileData) > 0:
				uri, _ := part.FileData["fileUri"].(string)
				if uri != "" {
					appendMarkdownSeparator(&contentBuilder)
					contentBuilder.WriteString("[file](")
					contentBuilder.WriteString(uri)
					contentBuilder.WriteString(")")
				}
			case len(part.ExecutableCode) > 0:
				appendExecutableCodeMarkdown(&contentBuilder, part.ExecutableCode)
			case len(part.CodeExecutionResult) > 0:
				appendCodeExecutionResultMarkdown(&contentBuilder, part.CodeExecutionResult)
			}
		}
		return contentBuilder.String(), reasoningBuilder.String()
	}

	for _, part := range result.TextParts {
		if part.Thought {
			reasoningBuilder.WriteString(part.Text)
		} else {
			contentBuilder.WriteString(part.Text)
		}
	}
	for _, img := range result.ImageParts {
		appendMarkdownSeparator(&contentBuilder)
		contentBuilder.WriteString("![image](")
		contentBuilder.WriteString(imageDataURL(img))
		contentBuilder.WriteString(")")
	}
	return contentBuilder.String(), reasoningBuilder.String()
}

func appendMarkdownSeparator(builder *strings.Builder) {
	if builder.Len() > 0 {
		builder.WriteString("\n\n")
	}
}

func appendExecutableCodeMarkdown(builder *strings.Builder, value map[string]interface{}) {
	code, _ := value["code"].(string)
	if code == "" {
		return
	}
	language, _ := value["language"].(string)
	appendMarkdownSeparator(builder)
	builder.WriteString("```")
	builder.WriteString(strings.ToLower(language))
	builder.WriteByte('\n')
	builder.WriteString(code)
	builder.WriteString("\n```")
}

func appendCodeExecutionResultMarkdown(builder *strings.Builder, value map[string]interface{}) {
	output, _ := value["output"].(string)
	if output == "" {
		return
	}
	appendMarkdownSeparator(builder)
	builder.WriteString("```text\n")
	builder.WriteString(output)
	builder.WriteString("\n```")
}

func imageDataURL(img model.InlineData) string {
	mimeType := img.MimeType
	if mimeType == "" {
		mimeType = "image/png"
	}
	return fmt.Sprintf("data:%s;base64,%s", mimeType, img.Data)
}

func convertMessages(modelName string, messages []model.ChatMessage) ([]map[string]interface{}, interface{}) {
	var contents []map[string]interface{}
	var systemInstruction interface{}
	var systemParts []map[string]interface{}
	toolCallNames := make(map[string]string)

	scanOpenAIToolCallNames(messages, toolCallNames)

	var lastModelFunctionName string
	for _, msg := range messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		if role == "system" || role == "developer" {
			text := extractTextContent(msg.Content)
			if text != "" {
				systemParts = append(systemParts, map[string]interface{}{"text": text})
			}
			continue
		}

		var vertexRole string
		var parts []map[string]interface{}

		if role == "tool" {
			vertexRole = "user"
			name := msg.Name
			if name == "" && msg.ToolCallID != "" {
				name = toolCallNames[msg.ToolCallID]
			}
			if name == "" && !strings.HasPrefix(msg.ToolCallID, "call_") {
				name = msg.ToolCallID
			}
			if (name == "" || strings.HasPrefix(name, "call_")) && lastModelFunctionName != "" {
				name = lastModelFunctionName
			}
			if name == "" || strings.HasPrefix(name, "call_") {
				log.Warn().Str("tool_call_id", msg.ToolCallID).Msg("convertMessages: unable to resolve function name for OpenAI tool call, falling back to text part")
				parts = []map[string]interface{}{
					{"text": fmt.Sprintf("[Tool Result (%s)]: %s", msg.ToolCallID, extractTextContent(msg.Content))},
				}
			} else {
				functionResponse := map[string]interface{}{
					"name":     name,
					"response": parseFunctionResponseContent(msg.Content),
				}
				if id := vertexToolCallID(msg.ToolCallID); id != "" {
					functionResponse["id"] = id
				}
				parts = []map[string]interface{}{
					{
						"functionResponse": functionResponse,
					},
				}
			}
		} else {
			vertexRole = role
			if vertexRole == "assistant" {
				vertexRole = "model"
			}
			parts = convertContentToParts(msg.Content)
			if vertexRole == "model" && msg.ReasoningContent != "" {
				parts = append([]map[string]interface{}{{"text": msg.ReasoningContent, "thought": true}}, parts...)
			}
			for _, toolCall := range msg.ToolCalls {
				if toolCall.ID != "" && toolCall.Function != nil {
					toolCallNames[toolCall.ID] = toolCall.Function.Name
				}
				if functionPart := convertToolCallToFunctionCallPart(toolCall); functionPart != nil {
					parts = append(parts, functionPart)
				}
			}
			if vertexRole == "model" {
				for _, p := range parts {
					if fc, ok := p["functionCall"].(map[string]interface{}); ok {
						if fname, _ := fc["name"].(string); fname != "" {
							lastModelFunctionName = fname
						}
					}
				}
			}
		}

		if len(parts) == 0 {
			continue
		}

		if len(contents) > 0 && contents[len(contents)-1]["role"] == vertexRole {
			prevParts, _ := contents[len(contents)-1]["parts"].([]map[string]interface{})
			contents[len(contents)-1]["parts"] = append(prevParts, parts...)
		} else {
			contents = append(contents, map[string]interface{}{
				"role":  vertexRole,
				"parts": parts,
			})
		}
	}
	if len(systemParts) > 0 {
		systemInstruction = map[string]interface{}{"parts": systemParts}
	}

	return contents, systemInstruction
}

func scanOpenAIToolCallNames(messages []model.ChatMessage, toolCallNames map[string]string) {
	for _, msg := range messages {
		for _, toolCall := range msg.ToolCalls {
			if toolCall.ID != "" && toolCall.Function != nil && toolCall.Function.Name != "" {
				toolCallNames[toolCall.ID] = toolCall.Function.Name
			}
		}
	}
}

// extractTextContent 从 content 提取纯文本
func convertToolCallToFunctionCallPart(toolCall model.ChatToolCall) map[string]interface{} {
	if toolCall.Function == nil {
		return nil
	}
	part := functionCallPart(toolCall.Function.Name, toolCall.Function.Arguments)
	if id := vertexToolCallID(toolCall.ID); id != "" {
		part["functionCall"].(map[string]interface{})["id"] = id
	}
	if signature := extractToolCallThoughtSignature(toolCall); signature != "" {
		part["thoughtSignature"] = signature
	} else {
		part["thoughtSignature"] = anthropicDummyThoughtSignature
	}
	return part
}

func vertexToolCallID(id string) string {
	id = strings.TrimSpace(id)
	if marker := strings.LastIndex(id, openAIToolSignatureMarker); marker >= 0 {
		id = id[:marker]
	}
	return strings.TrimSpace(id)
}

func functionCallPart(name, arguments string) map[string]interface{} {
	return map[string]interface{}{
		"functionCall": map[string]interface{}{
			"name": name,
			"args": parseFunctionArguments(arguments),
		},
	}
}

func extractToolCallThoughtSignature(toolCall model.ChatToolCall) string {
	if signature := thoughtSignatureFromExtraContent(toolCall.ExtraContent); signature != "" {
		return signature
	}
	marker := strings.LastIndex(toolCall.ID, openAIToolSignatureMarker)
	if marker < 0 {
		return ""
	}
	encoded := toolCall.ID[marker+len(openAIToolSignatureMarker):]
	decoded, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		return ""
	}
	return string(decoded)
}

func thoughtSignatureFromExtraContent(extraContent map[string]interface{}) string {
	if extraContent == nil {
		return ""
	}
	if signature, ok := extraContent["thoughtSignature"].(string); ok {
		return signature
	}
	if google, ok := extraContent["google"].(map[string]interface{}); ok {
		if signature, ok := google["thoughtSignature"].(string); ok {
			return signature
		}
	}
	return ""
}

func parseFunctionArguments(arguments string) map[string]interface{} {
	arguments = strings.TrimSpace(arguments)
	if arguments == "" {
		return map[string]interface{}{}
	}

	var args map[string]interface{}
	if err := sonic.Unmarshal([]byte(arguments), &args); err == nil {
		return args
	}

	var value interface{}
	if err := sonic.Unmarshal([]byte(arguments), &value); err == nil {
		return map[string]interface{}{"value": value}
	}
	return map[string]interface{}{"arguments": arguments}
}

func parseFunctionResponseContent(content interface{}) map[string]interface{} {
	text := strings.TrimSpace(extractTextContent(content))
	if text == "" {
		return map[string]interface{}{}
	}

	var response map[string]interface{}
	if err := sonic.Unmarshal([]byte(text), &response); err == nil {
		return response
	}

	var value interface{}
	if err := sonic.Unmarshal([]byte(text), &value); err == nil {
		return map[string]interface{}{"result": value}
	}
	return map[string]interface{}{"result": text}
}

func extractTextContent(content interface{}) string {
	switch v := content.(type) {
	case nil:
		return ""
	case string:
		return v
	case []interface{}:
		var texts []string
		for _, item := range v {
			if m, ok := item.(map[string]interface{}); ok {
				if t, ok := m["text"].(string); ok {
					texts = append(texts, t)
				}
			}
		}
		return strings.Join(texts, "\n")
	}
	return fmt.Sprintf("%v", content)
}

// convertContentToParts 将 OpenAI content 转换为 Vertex parts
func convertContentToParts(content interface{}) []map[string]interface{} {
	switch v := content.(type) {
	case nil:
		return nil
	case string:
		if v == "" {
			return nil
		}
		return []map[string]interface{}{{"text": v}}
	case []interface{}:
		var parts []map[string]interface{}
		for _, item := range v {
			m, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			partType, _ := m["type"].(string)
			switch partType {
			case "text":
				if text, ok := m["text"].(string); ok {
					parts = append(parts, map[string]interface{}{"text": text})
				}
			case "image_url":
				if imgURL, ok := m["image_url"].(map[string]interface{}); ok {
					url, _ := imgURL["url"].(string)
					if strings.HasPrefix(url, "data:") {
						mimeType, b64Data := parseDataURL(url)
						parts = append(parts, map[string]interface{}{
							"inlineData": map[string]interface{}{
								"mimeType": mimeType,
								"data":     b64Data,
							},
						})
					} else if url != "" {
						parts = append(parts, map[string]interface{}{
							"fileData": map[string]interface{}{"fileUri": url},
						})
					}
				}
			case "file":
				file, _ := m["file"].(map[string]interface{})
				fileData, _ := file["file_data"].(string)
				fileURL, _ := file["file_url"].(string)
				if fileData != "" {
					mimeType, data := parseDataURL(fileData)
					if !strings.HasPrefix(fileData, "data:") {
						mimeType = "application/octet-stream"
					}
					parts = append(parts, map[string]interface{}{"inlineData": map[string]interface{}{"mimeType": mimeType, "data": data}})
				} else if fileURL != "" {
					parts = append(parts, map[string]interface{}{"fileData": map[string]interface{}{"fileUri": fileURL}})
				}
			case "input_audio":
				audio, _ := m["input_audio"].(map[string]interface{})
				data, _ := audio["data"].(string)
				format, _ := audio["format"].(string)
				if data != "" {
					parts = append(parts, map[string]interface{}{"inlineData": map[string]interface{}{
						"mimeType": audioFormatMimeType(format), "data": data,
					}})
				}
			}
		}
		return parts
	}
	return []map[string]interface{}{{"text": fmt.Sprintf("%v", content)}}
}

func audioFormatMimeType(format string) string {
	switch strings.ToLower(strings.TrimSpace(format)) {
	case "wav", "wave":
		return "audio/wav"
	case "mp3", "mpeg":
		return "audio/mpeg"
	case "flac":
		return "audio/flac"
	case "m4a", "mp4":
		return "audio/mp4"
	case "ogg", "opus":
		return "audio/ogg"
	default:
		return "application/octet-stream"
	}
}

// parseDataURL 解析 data:mime;base64,DATA 格式的 URL
func parseDataURL(dataURL string) (mimeType string, data string) {
	if !strings.HasPrefix(dataURL, "data:") {
		return "image/png", dataURL // 假设纯 base64
	}
	// data:image/png;base64,xxxxx
	dataURL = strings.TrimPrefix(dataURL, "data:")
	parts := strings.SplitN(dataURL, ";base64,", 2)
	if len(parts) == 2 {
		return parts[0], parts[1]
	}
	return "image/png", dataURL
}
