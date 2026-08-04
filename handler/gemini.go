package handler

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"

	"vertex2api/model"
	"vertex2api/proxy"

	"github.com/bytedance/sonic"
	"github.com/rs/zerolog/log"
)

// GeminiGenerate 处理 Gemini 原生端点
// POST /v1beta1/models/{model}:generateContent
// POST /v1beta1/models/{model}:streamGenerateContent
func GeminiGenerate(vp *proxy.VertexProxy, allowCustomModelNames bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
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
		safetySettings := toSafetySettings(geminiReq["safetySettings"])
		safetySettings = model.SanitizeSafetySettings(modelName, safetySettings)
		systemInstruction := geminiReq["systemInstruction"]
		options := buildGeminiRequestOptions(geminiReq)

		// 确保 responseModalities 默认值
		if genConfig == nil {
			genConfig = map[string]interface{}{}
		}

		if isStream {
			bodyJSON, tokenLease, err := vp.BuildBodyWithTokenWithOptionsContext(r.Context(), modelName, contents, genConfig, safetySettings, systemInstruction, options)
			if err != nil {
				if requestContextCanceled(r.Context(), err) {
					log.Debug().Err(err).Str("model", modelName).Msg("Gemini stream request canceled before upstream call")
					return
				}
				log.Error().Err(err).Str("model", modelName).Msg("Build Gemini stream request failed")
				WriteJSON(w, http.StatusInternalServerError, geminiErrorResponse(err))
				return
			}
			streamGeminiResponseWithProxy(w, r, modelName, vp, func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
				return vp.StreamWithTokenContext(ctx, bodyJSON, tokenLease, onChunk)
			})
			return
		}

		result, err := vp.CallWithTokenWithOptionsContext(r.Context(), modelName, contents, genConfig, safetySettings, systemInstruction, options)
		if err != nil {
			if requestContextCanceled(r.Context(), err) {
				log.Debug().Err(err).Str("model", modelName).Msg("Gemini request canceled")
				return
			}
			writeUpstreamProtocolError(w, r, err)
			return
		}

		// 构建 Gemini 原生格式响应
		geminiResp := buildGeminiResponse(result)
		usage, estimated := geminiUsageMetadata(geminiReq, result)
		geminiResp["usageMetadata"] = usage
		if estimated {
			w.Header().Set("X-Usage-Estimated", "true")
		}

		WriteJSON(w, http.StatusOK, geminiResp)
	})
}

func buildGeminiResponse(result *proxy.CallResult) map[string]interface{} {
	return buildGeminiResponseWithFinish(result, true)
}

func buildGeminiStreamResponse(result *proxy.CallResult) map[string]interface{} {
	return buildGeminiResponseWithFinish(result, false)
}

func buildGeminiRequestOptions(req map[string]interface{}) *proxy.VertexRequestOptions {
	options := &proxy.VertexRequestOptions{}
	if tools, ok := req["tools"]; ok {
		options.Tools = tools
	}
	if toolConfig, ok := req["toolConfig"]; ok {
		options.ToolConfig = toolConfig
	}
	if options.Tools == nil && options.ToolConfig == nil {
		return nil
	}
	return options
}

func buildGeminiResponseWithFinish(result *proxy.CallResult, defaultStop bool) map[string]interface{} {
	response := make(map[string]interface{})
	if result == nil {
		return response
	}
	candidates := make([]map[string]interface{}, 0, maxInt(1, len(result.Candidates)))
	if len(result.Candidates) > 0 {
		for _, candidate := range result.Candidates {
			candidateResult := callResultFromCandidate(candidate)
			candidateMap := buildGeminiCandidate(candidateResult, defaultStop)
			candidateMap["index"] = candidate.Index
			if candidate.FinishMessage != "" {
				candidateMap["finishMessage"] = candidate.FinishMessage
			}
			if len(candidate.SafetyRatings) > 0 {
				candidateMap["safetyRatings"] = candidate.SafetyRatings
			}
			if len(candidate.CitationMetadata) > 0 {
				candidateMap["citationMetadata"] = candidate.CitationMetadata
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
			candidates = append(candidates, candidateMap)
		}
	} else if result.HasContent() || result.FinishReason != "" {
		candidates = append(candidates, buildGeminiCandidate(result, defaultStop))
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
	if result.CreateTime != "" {
		response["createTime"] = result.CreateTime
	}
	if len(result.PromptFeedback) > 0 {
		response["promptFeedback"] = result.PromptFeedback
	}
	if len(result.ModelStatus) > 0 {
		response["modelStatus"] = result.ModelStatus
	}
	return response
}

func buildGeminiCandidate(result *proxy.CallResult, defaultStop bool) map[string]interface{} {
	var parts []map[string]interface{}
	textParts := result.TextParts
	if defaultStop {
		textParts = coalesceGeminiTextParts(textParts)
	}

	for _, textPart := range textParts {
		part := map[string]interface{}{"text": textPart.Text}
		if textPart.Thought {
			part["thought"] = true
		}
		if textPart.ThoughtSignature != "" {
			part["thoughtSignature"] = textPart.ThoughtSignature
		}
		parts = append(parts, part)
	}
	for _, img := range result.ImageParts {
		parts = append(parts, map[string]interface{}{
			"inlineData": map[string]interface{}{
				"mimeType": img.MimeType,
				"data":     img.Data,
			},
		})
	}
	for _, functionCall := range result.FunctionCalls {
		call := map[string]interface{}{"name": functionCall.Name}
		if functionCall.ID != "" {
			call["id"] = functionCall.ID
		}
		if functionCall.Args != nil {
			call["args"] = functionCall.Args
		}
		part := map[string]interface{}{"functionCall": call}
		if functionCall.ThoughtSignature != "" {
			part["thoughtSignature"] = functionCall.ThoughtSignature
		}
		parts = append(parts, part)
	}

	candidate := map[string]interface{}{
		"content": map[string]interface{}{
			"parts": parts,
			"role":  "model",
		},
	}
	finishReason := result.FinishReason
	if defaultStop && (finishReason == "" || finishReason == "FINISH_REASON_UNSPECIFIED") {
		finishReason = "STOP"
	} else if !defaultStop && finishReason == "FINISH_REASON_UNSPECIFIED" {
		finishReason = ""
	}
	if finishReason != "" {
		candidate["finishReason"] = finishReason
	}
	if result.GroundingMetadata != nil {
		candidate["groundingMetadata"] = result.GroundingMetadata
	}

	return candidate
}

func coalesceGeminiTextParts(parts []model.TextPart) []model.TextPart {
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
	ctx := r.Context()
	committed := false
	err := stream(ctx, func(result *proxy.CallResult) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if result == nil || result.IsEmpty() {
			return nil
		}
		if !committed {
			setSSEHeaders(w)
			committed = true
		}
		return writeGeminiStreamChunk(w, result)
	})
	if err != nil {
		if requestContextCanceled(ctx, err) {
			log.Debug().Err(err).Str("model", modelName).Msg("Gemini stream client disconnected")
			return
		}
		log.Error().Str("err", upstreamLogError(vp, err)).Str("model", modelName).Msg("Gemini API stream failed")
		if !committed {
			writeUpstreamProtocolError(w, r, err)
			return
		}
		_ = writeGeminiStreamError(w, err)
		return
	}
	if !committed {
		setSSEHeaders(w)
	}
}

func writeGeminiStreamChunk(w http.ResponseWriter, result *proxy.CallResult) error {
	data, _ := sonic.Marshal(buildGeminiStreamResponse(result))
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

func writeGeminiStreamError(w http.ResponseWriter, streamErr error) error {
	data, _ := sonic.Marshal(geminiErrorResponse(streamErr))
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
	status := proxy.HTTPStatusForError(err)
	return map[string]interface{}{
		"error": map[string]interface{}{
			"code":    status,
			"message": publicServerErrorMessageFor(err),
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
