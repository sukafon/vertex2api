package handler

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"errors"
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

const (
	defaultAnthropicModel          = "gemini-3.5-flash"
	anthropicDummyThoughtSignature = "skip_thought_signature_validator"
)

func AnthropicMessages(vp *proxy.VertexProxy, allowCustomModelNames bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req model.AnthropicMessageRequest
		_, ok := readAnthropicJSONRequest(w, r, &req)
		if !ok {
			return
		}

		if message := validateModelName(req.Model, allowCustomModelNames); message != "" {
			writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", message)
			return
		}

		if message := validateAnthropicRequest(req); message != "" {
			writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", message)
			return
		}

		if req.Stream {
			streamAnthropicResponseWithProxy(w, r, req, vp, func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
				return streamAnthropicWithSignatureFallback(ctx, vp, req, onChunk)
			})
			return
		}

		result, err := callAnthropicWithSignatureFallback(r.Context(), vp, req)
		if err != nil {
			if requestContextCanceled(r.Context(), err) {
				log.Debug().Err(err).Str("model", req.Model).Msg("Anthropic request canceled")
				return
			}
			log.Error().Str("err", vp.UpstreamLogError(err)).Str("model", req.Model).Msg("Vertex API call for Anthropic failed")
			writeUpstreamProtocolError(w, r, err)
			return
		}

		sendAnthropicResponse(w, req, req.Model, result)
	})
}

func validateAnthropicRequest(req model.AnthropicMessageRequest) string {
	if strings.TrimSpace(req.Model) == "" {
		return "model is required"
	}
	if req.MaxTokens == nil {
		return "max_tokens is required"
	}
	if *req.MaxTokens < 1 {
		return "max_tokens must be greater than zero"
	}
	if len(req.Messages) == 0 {
		return "messages is required"
	}
	if req.Temperature != nil && (*req.Temperature < 0 || *req.Temperature > 1) {
		return "temperature must be between 0 and 1"
	}
	if req.TopP != nil && (*req.TopP < 0 || *req.TopP > 1) {
		return "top_p must be between 0 and 1"
	}
	for _, message := range req.Messages {
		role := strings.ToLower(strings.TrimSpace(message.Role))
		if role != "user" && role != "assistant" && role != "system" {
			return "messages roles must be user, assistant, or system"
		}
	}
	return ""
}

func callAnthropicWithSignatureFallback(ctx context.Context, vp *proxy.VertexProxy, req model.AnthropicMessageRequest) (*proxy.CallResult, error) {
	var lastErr error
	for stage := 0; stage <= 2; stage++ {
		stageReq := anthropicRequestForSignatureRetry(req, stage)
		contents, systemInstruction := convertAnthropicMessages(stageReq)
		genConfig := buildAnthropicGenerationConfig(stageReq)
		options := buildAnthropicRequestOptions(stageReq)

		result, err := vp.CallWithTokenWithOptionsContext(ctx, stageReq.Model, contents, genConfig, nil, systemInstruction, options)
		if err == nil {
			return result, nil
		}
		lastErr = err
		if !isAnthropicSignatureRelatedError(err) || stage == 2 {
			return nil, err
		}
		log.Info().Str("err", vp.UpstreamLogError(err)).Int("stage", stage+1).Str("model", req.Model).Msg("Anthropic signature-related error, retrying with downgraded history")
	}
	return nil, lastErr
}

func streamAnthropicWithSignatureFallback(
	ctx context.Context,
	vp *proxy.VertexProxy,
	req model.AnthropicMessageRequest,
	onChunk func(*proxy.CallResult) error,
) error {
	var lastErr error
	for stage := 0; stage <= 2; stage++ {
		stageReq := anthropicRequestForSignatureRetry(req, stage)
		contents, systemInstruction := convertAnthropicMessages(stageReq)
		genConfig := buildAnthropicGenerationConfig(stageReq)
		options := buildAnthropicRequestOptions(stageReq)

		bodyJSON, tokenLease, err := vp.BuildBodyWithTokenWithOptionsContext(ctx, stageReq.Model, contents, genConfig, nil, systemInstruction, options)
		if err != nil {
			return err
		}

		emitted := false
		err = vp.StreamWithTokenContext(ctx, bodyJSON, tokenLease, func(result *proxy.CallResult) error {
			emitted = true
			return onChunk(result)
		})
		if err == nil {
			return nil
		}
		lastErr = err
		if emitted || !isAnthropicSignatureRelatedError(err) || stage == 2 {
			return err
		}
		log.Info().Str("err", vp.UpstreamLogError(err)).Int("stage", stage+1).Str("model", req.Model).Msg("Anthropic stream signature-related error, retrying with downgraded history")
	}
	return lastErr
}

func AnthropicCountTokens(allowCustomModelNames bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req model.AnthropicMessageRequest
		if _, ok := readAnthropicJSONRequest(w, r, &req); !ok {
			return
		}
		if message := validateModelName(req.Model, allowCustomModelNames); message != "" {
			writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", message)
			return
		}
		if len(req.Messages) == 0 {
			writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "model and messages are required")
			return
		}
		w.Header().Set("X-Usage-Estimated", "true")
		WriteJSON(w, http.StatusOK, model.AnthropicCountTokensResponse{
			InputTokens: estimateAnthropicInputTokens(req),
		})
	})
}

func sendAnthropicResponse(w http.ResponseWriter, req model.AnthropicMessageRequest, modelName string, result *proxy.CallResult) {
	stopReason := anthropicStopReason(result)
	content := buildAnthropicContent(result)
	usage, estimated := anthropicUsage(req, result)
	if estimated {
		w.Header().Set("X-Usage-Estimated", "true")
	}
	resp := model.AnthropicMessageResponse{
		ID:           anthropicMessageID(),
		Type:         "message",
		Role:         "assistant",
		Model:        modelName,
		Content:      content,
		StopReason:   &stopReason,
		StopSequence: nil,
		Usage:        usage,
	}
	WriteJSON(w, http.StatusOK, resp)
}

func estimateAnthropicOutputTokens(result *proxy.CallResult) int {
	if result == nil || (len(result.TextParts) == 0 && len(result.ImageParts) == 0 && len(result.FunctionCalls) == 0) {
		return 0
	}
	return estimateTokenValue(map[string]interface{}{
		"content":    result.TextParts,
		"images":     result.ImageParts,
		"tool_calls": result.FunctionCalls,
	}, "")
}

func streamAnthropicResponse(
	w http.ResponseWriter,
	r *http.Request,
	req model.AnthropicMessageRequest,
	stream func(context.Context, func(*proxy.CallResult) error) error,
) {
	streamAnthropicResponseWithProxy(w, r, req, nil, stream)
}

func streamAnthropicResponseWithProxy(
	w http.ResponseWriter,
	r *http.Request,
	req model.AnthropicMessageRequest,
	vp *proxy.VertexProxy,
	stream func(context.Context, func(*proxy.CallResult) error) error,
) {
	id := anthropicMessageID()
	ctx := r.Context()
	streamState := &anthropicStreamState{w: w, openTextIndex: -1, openThinkingIndex: -1}
	stopReason := "end_turn"
	aggregate := &proxy.CallResult{}
	committed := false
	err := stream(ctx, func(result *proxy.CallResult) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if result == nil || result.IsEmpty() {
			return nil
		}
		accumulateCallResult(aggregate, result)
		if !committed {
			setSSEHeaders(w)
			initialUsage, _ := anthropicUsage(req, result)
			if err := writeAnthropicMessageStart(w, id, req.Model, initialUsage); err != nil {
				return err
			}
			committed = true
		}
		if len(result.FunctionCalls) > 0 {
			stopReason = "tool_use"
		} else if result.FinishReason != "" {
			stopReason = anthropicStopReason(result)
		}
		for _, block := range buildAnthropicContent(result) {
			if err := streamState.writeBlock(block); err != nil {
				return err
			}
		}
		return nil
	})
	if err != nil {
		if requestContextCanceled(ctx, err) {
			log.Debug().Err(err).Str("model", req.Model).Msg("Anthropic stream client disconnected")
			return
		}
		log.Error().Str("err", upstreamLogError(vp, err)).Str("model", req.Model).Msg("Vertex API stream for Anthropic failed")
		if !committed {
			writeUpstreamProtocolError(w, r, err)
			return
		}
		_ = writeAnthropicStreamError(w, err)
		return
	}
	if err := ctx.Err(); err != nil {
		log.Debug().Err(err).Str("model", req.Model).Msg("Anthropic stream client disconnected")
		return
	}
	if !committed {
		setSSEHeaders(w)
		initialUsage, _ := anthropicUsage(req, aggregate)
		if err := writeAnthropicMessageStart(w, id, req.Model, initialUsage); err != nil {
			return
		}
	}
	if err := streamState.closeOpenBlocks(); err != nil {
		log.Error().Err(err).Str("model", req.Model).Msg("Anthropic stream content block stop failed")
		return
	}
	finalUsage, _ := anthropicUsage(req, aggregate)
	if err := writeAnthropicMessageDeltaWithUsage(w, stopReason, finalUsage); err != nil {
		log.Error().Err(err).Str("model", req.Model).Msg("Anthropic stream message_delta failed")
		return
	}
	if err := writeAnthropicSSE(w, "message_stop", map[string]interface{}{"type": "message_stop"}); err != nil {
		log.Error().Err(err).Str("model", req.Model).Msg("Anthropic stream message_stop failed")
	}
}

type anthropicStreamState struct {
	w                 http.ResponseWriter
	nextIndex         int
	openTextIndex     int
	openThinkingIndex int
	seenText          string
	seenThinking      string
	emittedSignature  bool
	outputTokens      int
}

func (s *anthropicStreamState) writeBlock(block map[string]interface{}) error {
	blockType, _ := block["type"].(string)
	switch blockType {
	case "tool_use":
		if err := s.closeOpenBlocks(); err != nil {
			return err
		}
		index := s.nextIndex
		s.nextIndex++
		inputJSON, _ := sonic.Marshal(block["input"])
		name, _ := block["name"].(string)
		s.outputTokens += (len(inputJSON)+len(name))/4 + 1
		return writeAnthropicToolUseBlock(s.w, index, block)
	case "thinking":
		thinking, _ := block["thinking"].(string)
		signature, _ := block["signature"].(string)
		return s.writeThinkingDelta(thinking, signature)
	default:
		if text, ok := block["text"].(string); ok && text != "" {
			return s.writeTextDelta(text)
		}
		return nil
	}
}

func (s *anthropicStreamState) writeThinkingDelta(thinking, signature string) error {
	delta, seen := anthropicTextDelta(s.seenThinking, thinking)
	s.seenThinking = seen
	if delta == "" && (signature == "" || s.emittedSignature) {
		return nil
	}

	if s.openTextIndex >= 0 {
		if err := s.closeOpenBlocks(); err != nil {
			return err
		}
	}

	if s.openThinkingIndex < 0 {
		index := s.nextIndex
		s.nextIndex++
		s.openThinkingIndex = index
		if err := writeAnthropicSSE(s.w, "content_block_start", map[string]interface{}{
			"type":          "content_block_start",
			"index":         index,
			"content_block": map[string]interface{}{"type": "thinking", "thinking": "", "signature": ""},
		}); err != nil {
			return err
		}
	}

	if delta != "" {
		s.outputTokens += len(delta)/4 + 1
		if err := writeAnthropicSSE(s.w, "content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": s.openThinkingIndex,
			"delta": map[string]interface{}{"type": "thinking_delta", "thinking": delta},
		}); err != nil {
			return err
		}
	}

	if signature != "" && !s.emittedSignature {
		s.emittedSignature = true
		s.outputTokens += len(signature)/4 + 1
		if err := writeAnthropicSSE(s.w, "content_block_delta", map[string]interface{}{
			"type":  "content_block_delta",
			"index": s.openThinkingIndex,
			"delta": map[string]interface{}{"type": "signature_delta", "signature": signature},
		}); err != nil {
			return err
		}
	}
	return nil
}

func (s *anthropicStreamState) writeTextDelta(text string) error {
	delta, seen := anthropicTextDelta(s.seenText, text)
	s.seenText = seen
	if delta == "" {
		return nil
	}

	if s.openThinkingIndex >= 0 {
		if err := s.closeOpenBlocks(); err != nil {
			return err
		}
	}

	s.outputTokens += len(delta)/4 + 1
	if s.openTextIndex < 0 {
		index := s.nextIndex
		s.nextIndex++
		s.openTextIndex = index
		if err := writeAnthropicSSE(s.w, "content_block_start", map[string]interface{}{
			"type":          "content_block_start",
			"index":         index,
			"content_block": map[string]interface{}{"type": "text", "text": ""},
		}); err != nil {
			return err
		}
	}
	return writeAnthropicSSE(s.w, "content_block_delta", map[string]interface{}{
		"type":  "content_block_delta",
		"index": s.openTextIndex,
		"delta": map[string]interface{}{"type": "text_delta", "text": delta},
	})
}

func (s *anthropicStreamState) closeOpenBlocks() error {
	if s.openThinkingIndex >= 0 {
		index := s.openThinkingIndex
		s.openThinkingIndex = -1
		if err := writeAnthropicSSE(s.w, "content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": index,
		}); err != nil {
			return err
		}
	}
	if s.openTextIndex >= 0 {
		index := s.openTextIndex
		s.openTextIndex = -1
		if err := writeAnthropicSSE(s.w, "content_block_stop", map[string]interface{}{
			"type":  "content_block_stop",
			"index": index,
		}); err != nil {
			return err
		}
	}
	return nil
}

func anthropicTextDelta(seen, incoming string) (string, string) {
	incoming = strings.TrimSuffix(incoming, "\u0000")
	if incoming == "" {
		return "", seen
	}
	if strings.HasPrefix(incoming, seen) {
		return strings.TrimPrefix(incoming, seen), incoming
	}
	if strings.HasPrefix(seen, incoming) {
		return "", seen
	}
	return incoming, seen + incoming
}

func convertAnthropicMessages(req model.AnthropicMessageRequest) ([]map[string]interface{}, interface{}) {
	contents := make([]map[string]interface{}, 0, len(req.Messages))
	systemTexts := anthropicSystemTexts(req.System)
	toolCallNames := make(map[string]string)

	for _, msg := range req.Messages {
		scanAnthropicToolCallNames(msg.Content, toolCallNames)
	}

	var lastModelFunctionName string
	for _, msg := range req.Messages {
		role := strings.ToLower(strings.TrimSpace(msg.Role))
		if role == "system" {
			if text := strings.TrimSpace(extractAnthropicText(msg.Content)); text != "" {
				systemTexts = append(systemTexts, text)
			}
			continue
		}

		vertexRole := role
		if vertexRole == "assistant" {
			vertexRole = "model"
		}
		parts := convertAnthropicContentToParts(req.Model, msg.Content, role, toolCallNames, lastModelFunctionName)
		if len(parts) == 0 {
			continue
		}

		if vertexRole == "model" {
			for _, p := range parts {
				if fc, ok := p["functionCall"].(map[string]interface{}); ok {
					if name, _ := fc["name"].(string); name != "" {
						lastModelFunctionName = name
					}
				}
			}
		}

		if vertexRole == "user" && anthropicNeedsStandaloneFunctionResponse(req.Model) && hasFunctionResponsePart(parts) {
			contents = appendAnthropicPartsWithStandaloneFunctionResponse(contents, vertexRole, parts)
		} else if len(contents) > 0 && contents[len(contents)-1]["role"] == vertexRole {
			prevParts, _ := contents[len(contents)-1]["parts"].([]map[string]interface{})
			contents[len(contents)-1]["parts"] = append(prevParts, parts...)
		} else {
			contents = append(contents, map[string]interface{}{
				"role":  vertexRole,
				"parts": parts,
			})
		}
	}

	return contents, buildAnthropicSystemInstruction(systemTexts)
}

func anthropicNeedsStandaloneFunctionResponse(modelName string) bool {
	modelName = strings.ToLower(strings.TrimSpace(modelName))
	return strings.HasPrefix(modelName, "gemini-3.6")
}

func hasFunctionResponsePart(parts []map[string]interface{}) bool {
	for _, part := range parts {
		if _, ok := part["functionResponse"]; ok {
			return true
		}
	}
	return false
}

func appendAnthropicPartsWithStandaloneFunctionResponse(
	contents []map[string]interface{},
	role string,
	parts []map[string]interface{},
) []map[string]interface{} {
	appendParts := func(current []map[string]interface{}, segment []map[string]interface{}, merge bool) []map[string]interface{} {
		if len(segment) == 0 {
			return current
		}
		if merge && len(current) > 0 && current[len(current)-1]["role"] == role && !contentHasFunctionResponse(current[len(current)-1]) {
			previous, _ := current[len(current)-1]["parts"].([]map[string]interface{})
			current[len(current)-1]["parts"] = append(previous, segment...)
			return current
		}
		return append(current, map[string]interface{}{"role": role, "parts": segment})
	}

	segment := make([]map[string]interface{}, 0, len(parts))
	for i := 0; i < len(parts); {
		if _, isFunctionResponse := parts[i]["functionResponse"]; !isFunctionResponse {
			segment = append(segment, parts[i])
			i++
			continue
		}

		contents = appendParts(contents, segment, true)
		segment = segment[:0]

		responseParts := make([]map[string]interface{}, 0, 1)
		for i < len(parts) {
			if _, isFunctionResponse := parts[i]["functionResponse"]; !isFunctionResponse {
				break
			}
			responseParts = append(responseParts, parts[i])
			i++
		}
		contents = appendParts(contents, responseParts, false)
	}

	return appendParts(contents, segment, true)
}

func contentHasFunctionResponse(content map[string]interface{}) bool {
	parts, ok := content["parts"].([]map[string]interface{})
	if !ok {
		return false
	}
	return hasFunctionResponsePart(parts)
}

func scanAnthropicToolCallNames(content interface{}, toolCallNames map[string]string) {
	switch v := content.(type) {
	case []interface{}:
		for _, item := range v {
			scanSingleAnthropicToolCallName(item, toolCallNames)
		}
	case map[string]interface{}:
		scanSingleAnthropicToolCallName(v, toolCallNames)
	}
}

func scanSingleAnthropicToolCallName(item interface{}, toolCallNames map[string]string) {
	block, ok := item.(map[string]interface{})
	if !ok {
		return
	}
	blockType, _ := block["type"].(string)
	if blockType == "tool_use" || blockType == "server_tool_use" {
		name, _ := block["name"].(string)
		id, _ := block["id"].(string)
		if id != "" && name != "" {
			toolCallNames[id] = name
		}
	}
}

func anthropicRequestForSignatureRetry(req model.AnthropicMessageRequest, stage int) model.AnthropicMessageRequest {
	if stage <= 0 {
		return req
	}

	retryReq := cloneAnthropicRequest(req)
	retryReq.Thinking = nil
	downgradeTools := stage >= 2
	for i := range retryReq.Messages {
		retryReq.Messages[i].Content = downgradeAnthropicContentForSignatureRetry(
			retryReq.Messages[i].Content,
			retryReq.Messages[i].Role,
			downgradeTools,
		)
	}
	return retryReq
}

func cloneAnthropicRequest(req model.AnthropicMessageRequest) model.AnthropicMessageRequest {
	data, err := sonic.Marshal(req)
	if err != nil {
		return req
	}
	var cloned model.AnthropicMessageRequest
	if err := sonic.Unmarshal(data, &cloned); err != nil {
		return req
	}
	return cloned
}

func downgradeAnthropicContentForSignatureRetry(content interface{}, role string, downgradeTools bool) interface{} {
	blocks, ok := content.([]interface{})
	if !ok {
		return content
	}

	newBlocks := make([]interface{}, 0, len(blocks))
	modified := false
	for _, item := range blocks {
		block, ok := item.(map[string]interface{})
		if !ok {
			newBlocks = append(newBlocks, item)
			continue
		}

		blockType, _ := block["type"].(string)
		switch blockType {
		case "text":
			if text, _ := block["text"].(string); text == "" {
				modified = true
				continue
			}
		case "thinking":
			modified = true
			if text, _ := block["thinking"].(string); text != "" {
				newBlocks = append(newBlocks, map[string]interface{}{"type": "text", "text": text})
			}
			continue
		case "redacted_thinking":
			modified = true
			continue
		case "tool_use", "server_tool_use":
			if downgradeTools {
				modified = true
				newBlocks = append(newBlocks, map[string]interface{}{"type": "text", "text": anthropicToolUseText(block)})
				continue
			}
		case "tool_result", "web_search_tool_result":
			if downgradeTools {
				modified = true
				newBlocks = append(newBlocks, map[string]interface{}{"type": "text", "text": anthropicToolResultText(block)})
				continue
			}
		}

		if blockType == "" {
			if rawThinking, ok := block["thinking"]; ok {
				modified = true
				if text := stringifyAnthropicRetryValue(rawThinking); text != "" {
					newBlocks = append(newBlocks, map[string]interface{}{"type": "text", "text": text})
				}
				continue
			}
		}

		newBlocks = append(newBlocks, item)
	}

	if !modified {
		return content
	}
	if len(newBlocks) == 0 {
		placeholder := "(content removed)"
		if strings.EqualFold(strings.TrimSpace(role), "assistant") {
			placeholder = "(assistant content removed)"
		}
		newBlocks = append(newBlocks, map[string]interface{}{"type": "text", "text": placeholder})
	}
	return newBlocks
}

func anthropicToolUseText(block map[string]interface{}) string {
	text := "(tool_use)"
	if name, _ := block["name"].(string); name != "" {
		text += " name=" + name
	}
	if id, _ := block["id"].(string); id != "" {
		text += " id=" + id
	}
	if inputText := stringifyAnthropicRetryValue(block["input"]); inputText != "" && inputText != "null" {
		text += " input=" + inputText
	}
	return text
}

func anthropicToolResultText(block map[string]interface{}) string {
	text := "(tool_result)"
	if toolUseID, _ := block["tool_use_id"].(string); toolUseID != "" {
		text += " tool_use_id=" + toolUseID
	}
	if isError, _ := block["is_error"].(bool); isError {
		text += " is_error=true"
	}
	if contentText := stringifyAnthropicRetryValue(block["content"]); contentText != "" && contentText != "null" {
		text += "\n" + contentText
	}
	return text
}

func stringifyAnthropicRetryValue(value interface{}) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	default:
		data, err := sonic.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(data)
	}
}

func isAnthropicSignatureRelatedError(err error) bool {
	if err == nil {
		return false
	}
	message := strings.ToLower(err.Error())
	return strings.Contains(message, "thought_signature") ||
		strings.Contains(message, "thought signature") ||
		strings.Contains(message, "thoughtsignature") ||
		strings.Contains(message, "thought_signature_validator")
}

func anthropicSystemTexts(system interface{}) []string {
	text := strings.TrimSpace(extractAnthropicText(system))
	if text == "" {
		return nil
	}
	return []string{text}
}

func buildAnthropicSystemInstruction(texts []string) interface{} {
	text := strings.TrimSpace(strings.Join(texts, "\n\n"))
	if text == "" {
		return nil
	}
	return map[string]interface{}{
		"parts": []map[string]interface{}{{"text": text}},
	}
}

func convertAnthropicContentToParts(modelName string, content interface{}, role string, toolCallNames map[string]string, lastModelFunctionName string) []map[string]interface{} {
	switch v := content.(type) {
	case nil:
		return nil
	case string:
		if v == "" {
			return nil
		}
		return []map[string]interface{}{{"text": v}}
	case []interface{}:
		parts := make([]map[string]interface{}, 0, len(v))
		for _, item := range v {
			block, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			blockType, _ := block["type"].(string)
			switch blockType {
			case "text":
				if text, _ := block["text"].(string); text != "" {
					parts = append(parts, map[string]interface{}{"text": text})
				}
			case "image":
				if part := anthropicImagePart(block); part != nil {
					parts = append(parts, part)
				}
			case "tool_use", "server_tool_use":
				if part := anthropicToolUsePart(modelName, block, toolCallNames); part != nil {
					parts = append(parts, part)
				}
			case "tool_result", "web_search_tool_result":
				if part := anthropicToolResultPart(block, toolCallNames, lastModelFunctionName); part != nil {
					parts = append(parts, part)
				}
			case "thinking":
				if text, _ := block["thinking"].(string); text != "" {
					part := map[string]interface{}{"text": text, "thought": true}
					if signature := anthropicBlockSignature(block, anthropicShouldInjectDummyThoughtSignature(modelName)); signature != "" {
						part["thoughtSignature"] = signature
					}
					parts = append(parts, part)
				}
			}
		}
		return parts
	default:
		return []map[string]interface{}{{"text": fmt.Sprintf("%v", content)}}
	}
}

func anthropicImagePart(block map[string]interface{}) map[string]interface{} {
	source, ok := block["source"].(map[string]interface{})
	if !ok {
		return nil
	}

	sourceType, _ := source["type"].(string)
	if sourceType != "base64" {
		return nil
	}
	data, _ := source["data"].(string)
	if data == "" {
		return nil
	}
	mimeType, _ := source["media_type"].(string)
	if mimeType == "" {
		mimeType = "image/png"
	}
	return map[string]interface{}{
		"inlineData": map[string]interface{}{
			"mimeType": mimeType,
			"data":     data,
		},
	}
}

func anthropicToolUsePart(modelName string, block map[string]interface{}, toolCallNames map[string]string) map[string]interface{} {
	name, _ := block["name"].(string)
	if name == "" {
		return nil
	}
	toolUseID, _ := block["id"].(string)
	if toolUseID != "" {
		toolCallNames[toolUseID] = name
	}

	functionCall := map[string]interface{}{
		"name": name,
		"args": anthropicToolInput(block["input"]),
	}
	if toolUseID != "" {
		functionCall["id"] = toolUseID
	}
	part := map[string]interface{}{"functionCall": functionCall}
	if signature := anthropicToolSignatureFromID(toolUseID); signature != "" {
		part["thoughtSignature"] = signature
	} else if signature, _ := block["signature"].(string); strings.TrimSpace(signature) != "" {
		part["thoughtSignature"] = strings.TrimSpace(signature)
	} else if signature := thoughtSignatureFromExtraContent(extractAnthropicGoogleExtra(block)); signature != "" {
		part["thoughtSignature"] = signature
	} else if anthropicShouldInjectDummyThoughtSignature(modelName) {
		part["thoughtSignature"] = anthropicDummyThoughtSignature
	}
	return part
}

func anthropicToolInput(input interface{}) map[string]interface{} {
	switch v := input.(type) {
	case nil:
		return map[string]interface{}{}
	case map[string]interface{}:
		return v
	default:
		return map[string]interface{}{"input": v}
	}
}

func anthropicToolResultPart(block map[string]interface{}, toolCallNames map[string]string, lastModelFunctionName string) map[string]interface{} {
	toolUseID, _ := block["tool_use_id"].(string)
	name := toolCallNames[toolUseID]
	if name == "" {
		if !strings.HasPrefix(toolUseID, "toolu_") && toolUseID != "" {
			name = toolUseID
		}
	}
	if name == "" || strings.HasPrefix(name, "toolu_") {
		if lastModelFunctionName != "" {
			name = lastModelFunctionName
		}
	}
	if name == "" || strings.HasPrefix(name, "toolu_") {
		log.Warn().Str("tool_use_id", toolUseID).Msg("anthropicToolResultPart: unable to resolve function name for tool_use_id, falling back to text part")
		text := anthropicToolResultText(block)
		return map[string]interface{}{"text": text}
	}

	response := parseFunctionResponseContent(extractAnthropicText(block["content"]))
	if isError, _ := block["is_error"].(bool); isError {
		response["is_error"] = true
	}
	functionResponse := map[string]interface{}{
		"name":     name,
		"response": response,
	}
	if toolUseID != "" {
		functionResponse["id"] = toolUseID
	}
	return map[string]interface{}{"functionResponse": functionResponse}
}

func extractAnthropicGoogleExtra(block map[string]interface{}) map[string]interface{} {
	if google, ok := block["google"].(map[string]interface{}); ok {
		return map[string]interface{}{"google": google}
	}
	return nil
}

func anthropicBlockSignature(block map[string]interface{}, useDummy bool) string {
	if signature, _ := block["signature"].(string); strings.TrimSpace(signature) != "" {
		return strings.TrimSpace(signature)
	}
	if useDummy {
		return anthropicDummyThoughtSignature
	}
	return ""
}

func anthropicShouldInjectDummyThoughtSignature(modelName string) bool {
	modelName = strings.TrimSpace(modelName)
	if modelName == "" {
		modelName = defaultAnthropicModel
	}
	modelLower := strings.ToLower(modelName)
	return strings.Contains(modelLower, "gemini-3") || strings.Contains(modelLower, "claude")
}

func buildAnthropicGenerationConfig(req model.AnthropicMessageRequest) map[string]interface{} {
	genConfig := map[string]interface{}{}
	if req.Temperature != nil {
		genConfig["temperature"] = *req.Temperature
	}
	if req.TopP != nil {
		genConfig["topP"] = *req.TopP
	}
	if topK := intFromInterface(req.TopK); topK > 0 {
		genConfig["topK"] = topK
	}
	if req.MaxTokens != nil {
		genConfig["maxOutputTokens"] = *req.MaxTokens
	}
	if len(req.StopSequences) > 0 {
		genConfig["stopSequences"] = req.StopSequences
	}
	if thinkingConfig := convertAnthropicThinking(req.Thinking); thinkingConfig != nil {
		genConfig["thinkingConfig"] = thinkingConfig
	}
	return genConfig
}

func convertAnthropicThinking(thinking map[string]interface{}) map[string]interface{} {
	if len(thinking) == 0 {
		return nil
	}
	thinkingType, _ := thinking["type"].(string)
	if strings.EqualFold(thinkingType, "disabled") {
		return nil
	}
	config := map[string]interface{}{"includeThoughts": true}
	if budget := intFromInterface(thinking["budget_tokens"]); budget > 0 {
		config["thinkingBudget"] = budget
	}
	if level, _ := thinking["effort"].(string); level != "" {
		config["thinkingLevel"] = level
	}
	if len(config) == 1 && thinkingType == "" {
		return nil
	}
	return config
}

func buildAnthropicRequestOptions(req model.AnthropicMessageRequest) *proxy.VertexRequestOptions {
	tools := convertAnthropicTools(req.Tools)
	toolConfig := convertAnthropicToolChoice(req.ToolChoice)
	if tools == nil && toolConfig == nil {
		return nil
	}
	return &proxy.VertexRequestOptions{
		Tools:      tools,
		ToolConfig: toolConfig,
	}
}

func convertAnthropicTools(tools []map[string]interface{}) interface{} {
	if len(tools) == 0 {
		return nil
	}

	converted := make([]interface{}, 0, 2)
	functionDeclarations := make([]interface{}, 0, len(tools))
	var nativeTools []interface{}
	for _, tool := range tools {
		if nativeTool, ok := convertNativeGeminiTool(tool); ok {
			nativeTools = append(nativeTools, nativeTool)
			continue
		}
		name, _ := tool["name"].(string)
		if name == "" {
			continue
		}
		declaration := map[string]interface{}{"name": name}
		if description := anthropicToolDescription(tool); description != "" {
			declaration["description"] = description
		}
		declaration["parameters"] = anthropicToolSchema(tool)
		functionDeclarations = append(functionDeclarations, declaration)
	}
	if len(functionDeclarations) > 0 {
		converted = append(converted, map[string]interface{}{"functionDeclarations": functionDeclarations})
	}
	converted = append(converted, nativeTools...)
	if len(converted) == 0 {
		return nil
	}
	return converted
}

func anthropicToolDescription(tool map[string]interface{}) string {
	if description, _ := tool["description"].(string); description != "" {
		return description
	}
	return ""
}

func anthropicToolSchema(tool map[string]interface{}) interface{} {
	if schema, ok := tool["input_schema"]; ok && schema != nil {
		return cleanAnthropicToolSchema(schema)
	}
	return cleanAnthropicToolSchema(map[string]interface{}{
		"type":                 "object",
		"additionalProperties": true,
	})
}

func cleanAnthropicToolSchema(schema interface{}) interface{} {
	return schemanorm.Normalize(schema)
}

func convertAnthropicToolChoice(choice interface{}) interface{} {
	switch v := choice.(type) {
	case nil:
		return nil
	case string:
		return functionCallingConfigForMode(anthropicToolChoiceMode(v), nil)
	case map[string]interface{}:
		choiceType, _ := v["type"].(string)
		if strings.EqualFold(choiceType, "tool") {
			if name, _ := v["name"].(string); name != "" {
				return functionCallingConfigForMode("ANY", []string{name})
			}
		}
		return functionCallingConfigForMode(anthropicToolChoiceMode(choiceType), nil)
	default:
		return nil
	}
}

func anthropicToolChoiceMode(choice string) string {
	switch strings.ToLower(strings.TrimSpace(choice)) {
	case "none":
		return "NONE"
	case "auto":
		return "AUTO"
	case "any":
		return "ANY"
	default:
		return ""
	}
}

func buildAnthropicContent(result *proxy.CallResult) []map[string]interface{} {
	if result == nil {
		return []map[string]interface{}{}
	}

	var content []map[string]interface{}
	var text strings.Builder
	for _, part := range result.TextParts {
		if part.Thought {
			if text.Len() > 0 {
				content = append(content, map[string]interface{}{"type": "text", "text": text.String()})
				text.Reset()
			}
			block := map[string]interface{}{"type": "thinking", "thinking": part.Text}
			if part.ThoughtSignature != "" {
				block["signature"] = part.ThoughtSignature
			}
			content = append(content, block)
			continue
		}
		text.WriteString(part.Text)
	}
	if text.Len() > 0 {
		content = append(content, map[string]interface{}{"type": "text", "text": text.String()})
	}

	for _, img := range result.ImageParts {
		content = append(content, map[string]interface{}{
			"type": "image",
			"source": map[string]interface{}{
				"type":       "base64",
				"media_type": img.MimeType,
				"data":       img.Data,
			},
		})
	}

	for i, functionCall := range result.FunctionCalls {
		block := map[string]interface{}{
			"type":  "tool_use",
			"id":    anthropicToolUseID(functionCall, i),
			"name":  functionCall.Name,
			"input": functionCall.Args,
		}
		if block["input"] == nil {
			block["input"] = map[string]interface{}{}
		}
		content = append(content, block)
	}

	return content
}

func generateAnthropicToolUseID(index int) string {
	b := make([]byte, 6)
	_, _ = rand.Read(b)
	return fmt.Sprintf("toolu_%x_%d", b, index)
}

const anthropicToolSignatureMarker = "__vertex_sig_"

func anthropicToolUseID(functionCall model.FunctionCall, index int) string {
	id := functionCall.ID
	if id == "" {
		id = generateAnthropicToolUseID(index)
	}
	if functionCall.ThoughtSignature == "" || strings.Contains(id, anthropicToolSignatureMarker) {
		return id
	}
	return id + anthropicToolSignatureMarker + base64.RawURLEncoding.EncodeToString([]byte(functionCall.ThoughtSignature))
}

func anthropicToolSignatureFromID(id string) string {
	marker := strings.LastIndex(id, anthropicToolSignatureMarker)
	if marker < 0 {
		return ""
	}
	encoded := id[marker+len(anthropicToolSignatureMarker):]
	decoded, err := base64.RawURLEncoding.DecodeString(encoded)
	if err != nil {
		return ""
	}
	return string(decoded)
}

func anthropicStopReason(result *proxy.CallResult) string {
	if result != nil && len(result.FunctionCalls) > 0 {
		return "tool_use"
	}
	if result == nil {
		return "end_turn"
	}
	switch strings.ToUpper(strings.TrimSpace(result.FinishReason)) {
	case "MAX_TOKENS":
		return "max_tokens"
	default:
		return "end_turn"
	}
}

func writeAnthropicMessageStart(w http.ResponseWriter, id, modelName string, usage *model.AnthropicUsage) error {
	if usage == nil {
		usage = &model.AnthropicUsage{}
	}
	startUsage := *usage
	startUsage.OutputTokens = 0
	message := model.AnthropicMessageResponse{
		ID:           id,
		Type:         "message",
		Role:         "assistant",
		Model:        modelName,
		Content:      []map[string]interface{}{},
		StopReason:   nil,
		StopSequence: nil,
		Usage:        &startUsage,
	}
	return writeAnthropicSSE(w, "message_start", map[string]interface{}{
		"type":    "message_start",
		"message": message,
	})
}

func writeAnthropicToolUseBlock(w http.ResponseWriter, index int, block map[string]interface{}) error {
	inputJSON, _ := sonic.Marshal(block["input"])
	id, _ := block["id"].(string)
	if id == "" {
		id = generateAnthropicToolUseID(index)
	}
	contentBlock := map[string]interface{}{
		"type":  "tool_use",
		"id":    id,
		"name":  block["name"],
		"input": map[string]interface{}{},
	}
	if err := writeAnthropicSSE(w, "content_block_start", map[string]interface{}{
		"type":          "content_block_start",
		"index":         index,
		"content_block": contentBlock,
	}); err != nil {
		return err
	}
	if err := writeAnthropicSSE(w, "content_block_delta", map[string]interface{}{
		"type":  "content_block_delta",
		"index": index,
		"delta": map[string]interface{}{"type": "input_json_delta", "partial_json": string(inputJSON)},
	}); err != nil {
		return err
	}
	return writeAnthropicSSE(w, "content_block_stop", map[string]interface{}{
		"type":  "content_block_stop",
		"index": index,
	})
}

func writeAnthropicMessageDelta(w http.ResponseWriter, stopReason string, outputTokens int) error {
	return writeAnthropicMessageDeltaWithUsage(w, stopReason, &model.AnthropicUsage{OutputTokens: outputTokens})
}

func writeAnthropicMessageDeltaWithUsage(w http.ResponseWriter, stopReason string, usage *model.AnthropicUsage) error {
	usagePayload := map[string]interface{}{"output_tokens": 0}
	if usage != nil {
		usagePayload["output_tokens"] = usage.OutputTokens
		if usage.OutputTokensDetails != nil {
			usagePayload["output_tokens_details"] = usage.OutputTokensDetails
		}
	}
	return writeAnthropicSSE(w, "message_delta", map[string]interface{}{
		"type": "message_delta",
		"delta": map[string]interface{}{
			"stop_reason":   stopReason,
			"stop_sequence": nil,
		},
		"usage": usagePayload,
	})
}

func writeAnthropicStreamError(w http.ResponseWriter, streamErr error) error {
	status := proxy.HTTPStatusForError(streamErr)
	return writeAnthropicSSE(w, "error", map[string]interface{}{
		"type": "error",
		"error": map[string]interface{}{
			"type":    anthropicErrorType(status, ""),
			"message": publicServerErrorMessageFor(streamErr),
		},
	})
}

func writeAnthropicSSE(w http.ResponseWriter, event string, payload interface{}) error {
	data, _ := sonic.Marshal(payload)
	if _, err := io.WriteString(w, "event: "+event+"\n"); err != nil {
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

func writeAnthropicError(w http.ResponseWriter, status int, errType, message string) {
	WriteJSON(w, status, map[string]interface{}{
		"type":       "error",
		"request_id": fmt.Sprintf("req_%d", time.Now().UnixNano()),
		"error": map[string]interface{}{
			"type":    errType,
			"message": message,
		},
	})
}

func readAnthropicJSONRequest(w http.ResponseWriter, r *http.Request, dst interface{}) ([]byte, bool) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			writeAnthropicError(w, http.StatusRequestEntityTooLarge, "request_too_large", "Request body too large")
			return nil, false
		}
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "Failed to read request body: "+err.Error())
		return nil, false
	}
	if err := sonic.Unmarshal(body, dst); err != nil {
		writeAnthropicError(w, http.StatusBadRequest, "invalid_request_error", "Invalid request body: "+err.Error())
		return nil, false
	}
	return body, true
}

func extractAnthropicText(content interface{}) string {
	switch v := content.(type) {
	case nil:
		return ""
	case string:
		return v
	case []interface{}:
		var texts []string
		for _, item := range v {
			switch block := item.(type) {
			case string:
				if block != "" {
					texts = append(texts, block)
				}
			case map[string]interface{}:
				blockType, _ := block["type"].(string)
				if text, ok := block["text"].(string); ok && text != "" {
					texts = append(texts, text)
				} else if blockType == "tool_result" {
					if text := extractAnthropicText(block["content"]); text != "" {
						texts = append(texts, text)
					}
				} else if contentVal, ok := block["content"]; ok && contentVal != nil {
					if text := extractAnthropicText(contentVal); text != "" {
						texts = append(texts, text)
					}
				} else if blockType == "" || blockType == "text" {
					if text := stringifyAnthropicRetryValue(block); text != "" && text != "{}" {
						texts = append(texts, text)
					}
				}
			}
		}
		return strings.Join(texts, "\n")
	case map[string]interface{}:
		if text, ok := v["text"].(string); ok && text != "" {
			return text
		}
		if contentVal, ok := v["content"]; ok && contentVal != nil {
			if text := extractAnthropicText(contentVal); text != "" {
				return text
			}
		}
		data, err := sonic.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(data)
	default:
		return fmt.Sprintf("%v", content)
	}
}

func intFromInterface(value interface{}) int {
	switch v := value.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	case jsonNumber:
		n, _ := v.Int64()
		return int(n)
	default:
		return 0
	}
}

type jsonNumber interface {
	Int64() (int64, error)
}

func estimateAnthropicInputTokens(req model.AnthropicMessageRequest) int {
	return estimateJSONTokens(map[string]interface{}{
		"system":   req.System,
		"messages": req.Messages,
		"tools":    req.Tools,
	})
}

func anthropicMessageID() string {
	return fmt.Sprintf("msg_%d", time.Now().UnixNano())
}

func isAnthropicRequest(r *http.Request) bool {
	if r == nil {
		return false
	}
	if strings.HasPrefix(r.URL.Path, "/v1/messages") {
		return true
	}
	if r.Header.Get("anthropic-version") != "" || r.Header.Get("anthropic-beta") != "" {
		return true
	}
	ua := strings.ToLower(r.Header.Get("User-Agent"))
	return strings.Contains(ua, "claude") || strings.Contains(ua, "anthropic")
}
