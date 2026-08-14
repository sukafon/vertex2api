package handler

import (
	"context"
	"encoding/base64"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vertex2api/model"
	"vertex2api/proxy"

	"github.com/bytedance/sonic"
)

func TestChatCompletionsRejectsWastefulLivenessProbeBeforeUpstreamCall(t *testing.T) {
	handler := ChatCompletions(nil, true, true)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"model":"gemini-3-flash-preview",
		"messages":[{"role":"user","content":"hi"}],
		"stream":false
	}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"type":"invalid_request_error"`) ||
		!strings.Contains(rec.Body.String(), `"code":"health_check_not_supported"`) ||
		!strings.Contains(rec.Body.String(), "GET /health") {
		t.Fatalf("unexpected liveness-probe error: %s", rec.Body.String())
	}
}

func TestChatCompletionsConstructsNormalLivenessProbeResponseBeforeUpstreamCall(t *testing.T) {
	// Enable both actions to verify that constructing a normal response takes precedence.
	handler := ChatCompletions(nil, true, true, true)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"model":"gemini-3-flash-preview",
		"messages":[{"role":"user","content":"hi"}],
		"stream":false
	}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	var response model.ChatCompletionResponse
	if err := sonic.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatalf("unmarshal response: %v", err)
	}
	if response.Object != "chat.completion" || response.Model != "gemini-3-flash-preview" {
		t.Fatalf("unexpected completion envelope: %#v", response)
	}
	if len(response.Choices) != 1 || response.Choices[0].Message == nil ||
		response.Choices[0].Message.Role != "assistant" || response.Choices[0].Message.Content != chatLivenessProbeResponse {
		t.Fatalf("unexpected constructed choice: %#v", response.Choices)
	}
	if response.Choices[0].FinishReason == nil || *response.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason = %v, want stop", response.Choices[0].FinishReason)
	}
	if response.Usage == nil || response.Usage.TotalTokens <= 0 {
		t.Fatalf("usage = %#v, want positive estimated usage", response.Usage)
	}
	if strings.Contains(rec.Body.String(), `"error"`) {
		t.Fatalf("constructed response contains an error: %s", rec.Body.String())
	}
}

func TestChatCompletionsConstructsStreamingLivenessProbeResponse(t *testing.T) {
	handler := ChatCompletions(nil, true, false, true)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"model":"gemini-3-flash-preview",
		"messages":[{"role":"user","content":"HI"}],
		"stream":true,
		"stream_options":{"include_usage":true}
	}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	if got := rec.Header().Get("Content-Type"); !strings.HasPrefix(got, "text/event-stream") {
		t.Fatalf("Content-Type = %q, want text/event-stream", got)
	}
	if !strings.Contains(rec.Body.String(), "data: [DONE]\n\n") {
		t.Fatalf("stream is missing terminator: %s", rec.Body.String())
	}
	objects := decodeStreamDataObjects(t, rec.Body.String())
	if len(objects) != 2 {
		t.Fatalf("stream objects = %d, want content and final usage-bearing chunks: %s", len(objects), rec.Body.String())
	}
	delta := deltaFromChunk(t, objects[0])
	if delta["role"] != "assistant" || delta["content"] != chatLivenessProbeResponse {
		t.Fatalf("unexpected constructed stream delta: %#v", delta)
	}
	if _, ok := objects[0]["usage"]; ok {
		t.Fatalf("content chunk must omit usage: %#v", objects[0])
	}
	if usage, ok := objects[1]["usage"].(map[string]interface{}); !ok || usage["total_tokens"].(float64) <= 0 {
		t.Fatalf("unexpected final stream usage: %#v", objects[1])
	}
}

func TestChatCompletionsAllowsLivenessProbeByDefault(t *testing.T) {
	handler := ChatCompletions(nil, true)
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{
		"messages":[{"role":"user","content":"hi"}]
	}`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if strings.Contains(rec.Body.String(), `"code":"health_check_not_supported"`) {
		t.Fatalf("default-off liveness probe rejection was active: %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "model is required") {
		t.Fatalf("request did not continue through normal validation: status=%d body=%s", rec.Code, rec.Body.String())
	}
}

func TestWastefulLivenessProbeDetectionAvoidsRealRequests(t *testing.T) {
	tests := []struct {
		name string
		req  model.ChatCompletionRequest
	}{
		{name: "normal prompt", req: model.ChatCompletionRequest{Messages: []model.ChatMessage{{Role: "user", Content: "hi there"}}}},
		{name: "conversation history", req: model.ChatCompletionRequest{Messages: []model.ChatMessage{{Role: "assistant", Content: "hello"}, {Role: "user", Content: "hi"}}}},
		{name: "tool request", req: model.ChatCompletionRequest{Messages: []model.ChatMessage{{Role: "user", Content: "hi"}}, Tools: []map[string]interface{}{{"type": "function"}}}},
		{name: "structured content", req: model.ChatCompletionRequest{Messages: []model.ChatMessage{{Role: "user", Content: []interface{}{map[string]interface{}{"type": "text", "text": "hi"}}}}}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if isWastefulLivenessProbe(tt.req) {
				t.Fatal("real request was classified as a liveness probe")
			}
		})
	}
}

func TestBuildResponseContentSeparatesReasoning(t *testing.T) {
	content, reasoning := buildResponseContent(&proxy.CallResult{
		TextParts: []model.TextPart{
			{Text: "thinking", Thought: true},
			{Text: "answer"},
		},
	})

	if got, want := content, "answer"; got != want {
		t.Fatalf("content = %v, want %q", got, want)
	}
	if got, want := reasoning, "thinking"; got != want {
		t.Fatalf("reasoning = %q, want %q", got, want)
	}
}

func TestWriteOpenAIStreamChunkSeparatesReasoning(t *testing.T) {
	rec := httptest.NewRecorder()
	state := &openAIStreamState{}

	chunks := []*proxy.CallResult{
		{TextParts: []model.TextPart{{Text: "thinking", Thought: true}}},
		{TextParts: []model.TextPart{{Text: " more", Thought: true}}},
		{TextParts: []model.TextPart{{Text: "answer"}}},
	}
	for _, chunk := range chunks {
		if err := writeOpenAIStreamChunk(rec, "chatcmpl-test", "gemini-test", chunk, state); err != nil {
			t.Fatalf("writeOpenAIStreamChunk returned error: %v", err)
		}
	}

	objects := decodeStreamDataObjects(t, rec.Body.String())
	if len(objects) != 3 {
		t.Fatalf("stream chunks = %d, want 3", len(objects))
	}

	if got, want := deltaFromChunk(t, objects[0])["reasoning_content"], "thinking"; got != want {
		t.Fatalf("first reasoning_content = %v, want %q", got, want)
	}
	if got, want := deltaFromChunk(t, objects[1])["reasoning_content"], " more"; got != want {
		t.Fatalf("second reasoning_content = %v, want %q", got, want)
	}
	if got, want := deltaFromChunk(t, objects[2])["content"], "answer"; got != want {
		t.Fatalf("third content = %v, want %q", got, want)
	}
	for index, object := range objects {
		delta := deltaFromChunk(t, object)
		if got := delta["role"]; got != "assistant" {
			t.Fatalf("chunk %d role = %v, want assistant", index, got)
		}
		if _, ok := delta["reasoning_content"]; !ok {
			t.Fatalf("chunk %d omitted reasoning_content: %#v", index, delta)
		}
		if toolCalls, ok := delta["tool_calls"]; !ok || toolCalls != nil {
			t.Fatalf("chunk %d tool_calls = %#v present=%v, want explicit null", index, toolCalls, ok)
		}
	}
	if got := deltaFromChunk(t, objects[2])["reasoning_content"]; got != nil {
		t.Fatalf("content-only chunk reasoning_content = %#v, want null", got)
	}
}

func TestBuildResponseContentFormatsImagesAsMarkdown(t *testing.T) {
	content, _ := buildResponseContent(&proxy.CallResult{
		TextParts: []model.TextPart{{Text: "done"}},
		ImageParts: []model.InlineData{
			{MimeType: "image/png", Data: "abc123"},
		},
	})

	text, ok := content.(string)
	if !ok {
		t.Fatalf("content type = %T, want string", content)
	}
	if got, want := text, "done\n\n![image](data:image/png;base64,abc123)"; got != want {
		t.Fatalf("content = %q, want %q", got, want)
	}
}

func TestSendChatResponseIncludesUsage(t *testing.T) {
	rec := httptest.NewRecorder()

	sendChatResponse(rec, model.ChatCompletionRequest{Model: "gemini-test"}, "gemini-test", &proxy.CallResult{
		TextParts:    []model.TextPart{{Text: "done"}},
		FinishReason: "STOP",
	})

	if !strings.Contains(rec.Body.String(), `"usage"`) {
		t.Fatalf("response should include usage, got %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"prompt_tokens"`) || !strings.Contains(rec.Body.String(), `"completion_tokens"`) {
		t.Fatalf("response usage missing token counts, got %s", rec.Body.String())
	}
}

func TestSendChatResponseIncludesReasoningTokens(t *testing.T) {
	rec := httptest.NewRecorder()

	sendChatResponse(rec, model.ChatCompletionRequest{Model: "gemini-test"}, "gemini-test", &proxy.CallResult{
		TextParts: []model.TextPart{
			{Text: "let me think about this for a moment...", Thought: true},
			{Text: "here is the final answer"},
		},
		FinishReason: "STOP",
	})

	body := rec.Body.String()
	if !strings.Contains(body, `"completion_tokens_details"`) {
		t.Fatalf("response usage should include completion_tokens_details, got %s", body)
	}
	if !strings.Contains(body, `"reasoning_tokens"`) {
		t.Fatalf("response usage should include reasoning_tokens, got %s", body)
	}
}

func TestBuildOpenAIRequestOptionsConvertsToolsAndToolChoice(t *testing.T) {
	options := buildOpenAIRequestOptions(model.ChatCompletionRequest{
		Tools: []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name":        "lookup",
					"description": "look up data",
					"parameters": map[string]interface{}{
						"type": "object",
					},
					"strict": true,
				},
			},
		},
		ToolChoice: map[string]interface{}{
			"type":     "function",
			"function": map[string]interface{}{"name": "lookup"},
		},
	})

	if options == nil {
		t.Fatal("options should not be nil")
	}
	tools := options.Tools.([]interface{})
	functionDeclarations := tools[0].(map[string]interface{})["functionDeclarations"].([]interface{})
	declaration := functionDeclarations[0].(map[string]interface{})
	if got := declaration["name"]; got != "lookup" {
		t.Fatalf("function declaration name = %v, want lookup", got)
	}
	if _, ok := declaration["strict"]; ok {
		t.Fatal("OpenAI strict field should not be passed to Gemini function declaration")
	}

	toolConfig := options.ToolConfig.(map[string]interface{})
	functionCallingConfig := toolConfig["functionCallingConfig"].(map[string]interface{})
	if got := functionCallingConfig["mode"]; got != "ANY" {
		t.Fatalf("tool choice mode = %v, want ANY", got)
	}
	allowed := functionCallingConfig["allowedFunctionNames"].([]string)
	if got := allowed[0]; got != "lookup" {
		t.Fatalf("allowed function = %v, want lookup", got)
	}
}

func TestBuildOpenAIRequestOptionsTranslatesNativeTools(t *testing.T) {
	options := buildOpenAIRequestOptions(model.ChatCompletionRequest{
		Tools: []map[string]interface{}{
			{"type": "web_search_preview"},
			{"type": "code_interpreter"},
			{"type": "url_context"},
			{"type": "google_maps"},
			{"type": "parallel_ai_search"},
			{"type": "computer_use"},
		},
	})

	if options == nil {
		t.Fatal("options should not be nil")
	}
	tools := options.Tools.([]interface{})
	if _, ok := tools[0].(map[string]interface{})["googleSearch"]; !ok {
		t.Fatal("web_search_preview should translate to googleSearch")
	}
	if _, ok := tools[1].(map[string]interface{})["codeExecution"]; !ok {
		t.Fatal("code_interpreter should translate to codeExecution")
	}
	if _, ok := tools[2].(map[string]interface{})["urlContext"]; !ok {
		t.Fatal("url_context should translate to urlContext")
	}
	if _, ok := tools[3].(map[string]interface{})["googleMaps"]; !ok {
		t.Fatal("google_maps should translate to googleMaps")
	}
	if _, ok := tools[4].(map[string]interface{})["parallelAiSearch"]; !ok {
		t.Fatal("parallel_ai_search should translate to parallelAiSearch")
	}
	if _, ok := tools[5].(map[string]interface{})["computerUse"]; !ok {
		t.Fatal("computer_use should translate to computerUse")
	}
}

func TestValidateOpenAIRequestRejectsUnknownToolInsteadOfDroppingIt(t *testing.T) {
	req := model.ChatCompletionRequest{
		Model: "gemini-test", Messages: []model.ChatMessage{{Role: "user", Content: "hello"}},
		Tools: []map[string]interface{}{{"type": "vendor_tool"}},
	}
	if got := validateOpenAIRequest(req); !strings.Contains(got, "no Vertex equivalent") {
		t.Fatalf("validation = %q", got)
	}
}

func TestConvertMessagesConvertsToolCallsAndResponses(t *testing.T) {
	contents, _ := convertMessages("gemini-2.0-flash-exp", []model.ChatMessage{
		{
			Role:    "assistant",
			Content: nil,
			ToolCalls: []model.ChatToolCall{
				{
					ID:   "call_1",
					Type: "function",
					Function: &model.ChatFunctionCall{
						Name:      "lookup",
						Arguments: `{"query":"weather"}`,
					},
					ExtraContent: map[string]interface{}{
						"google": map[string]interface{}{
							"thoughtSignature": "sig",
						},
					},
				},
			},
		},
		{
			Role:       "tool",
			ToolCallID: "call_1",
			Content:    `{"temperature":72}`,
		},
	})

	if len(contents) != 2 {
		t.Fatalf("contents length = %d, want 2", len(contents))
	}
	modelPart := contents[0]["parts"].([]map[string]interface{})[0]
	functionCall := modelPart["functionCall"].(map[string]interface{})
	if got := functionCall["name"]; got != "lookup" {
		t.Fatalf("functionCall name = %v, want lookup", got)
	}
	if got := functionCall["id"]; got != "call_1" {
		t.Fatalf("functionCall id = %v, want call_1", got)
	}
	args := functionCall["args"].(map[string]interface{})
	if got := args["query"]; got != "weather" {
		t.Fatalf("functionCall query = %v, want weather", got)
	}
	if got := modelPart["thoughtSignature"]; got != "sig" {
		t.Fatalf("functionCall thoughtSignature = %v, want sig", got)
	}

	userPart := contents[1]["parts"].([]map[string]interface{})[0]
	functionResponse := userPart["functionResponse"].(map[string]interface{})
	if got := functionResponse["name"]; got != "lookup" {
		t.Fatalf("functionResponse name = %v, want lookup", got)
	}
	if got := functionResponse["id"]; got != "call_1" {
		t.Fatalf("functionResponse id = %v, want call_1", got)
	}
	response := functionResponse["response"].(map[string]interface{})
	if got := response["temperature"]; got != float64(72) {
		t.Fatalf("functionResponse temperature = %v, want 72", got)
	}
}

func TestBuildOpenAIMessageIncludesToolCalls(t *testing.T) {
	message := buildOpenAIMessage(&proxy.CallResult{
		FunctionCalls: []model.FunctionCall{
			{
				Name:             "lookup",
				Args:             map[string]interface{}{"query": "weather"},
				ThoughtSignature: "sig",
			},
		},
	})

	if message.Content != nil {
		t.Fatalf("message content = %v, want nil", message.Content)
	}
	if len(message.ToolCalls) != 1 {
		t.Fatalf("tool calls length = %d, want 1", len(message.ToolCalls))
	}
	if got := message.ToolCalls[0].Function.Name; got != "lookup" {
		t.Fatalf("tool call function name = %v, want lookup", got)
	}
	if message.ToolCalls[0].ExtraContent != nil {
		t.Fatalf("tool call should not expose non-standard extra_content: %v", message.ToolCalls[0].ExtraContent)
	}
	if got := extractToolCallThoughtSignature(message.ToolCalls[0]); got != "sig" {
		t.Fatalf("opaque tool call signature = %v, want sig", got)
	}
	if got := openAIFinishReason(&proxy.CallResult{FunctionCalls: []model.FunctionCall{{Name: "lookup"}}}); got != "tool_calls" {
		t.Fatalf("finish reason = %v, want tool_calls", got)
	}
}

func TestConvertMessagesRestoresVertexToolCallIDFromOpaqueOpenAIID(t *testing.T) {
	opaqueID := "call_7" + openAIToolSignatureMarker + base64.RawURLEncoding.EncodeToString([]byte("sig"))
	contents, _ := convertMessages("gemini-3.1-pro-preview", []model.ChatMessage{
		{
			Role: "assistant",
			ToolCalls: []model.ChatToolCall{{
				ID: opaqueID,
				Function: &model.ChatFunctionCall{
					Name:      "lookup",
					Arguments: `{}`,
				},
			}},
		},
		{Role: "tool", ToolCallID: opaqueID, Content: "ok"},
	})

	call := contents[0]["parts"].([]map[string]interface{})[0]["functionCall"].(map[string]interface{})
	response := contents[1]["parts"].([]map[string]interface{})[0]["functionResponse"].(map[string]interface{})
	if call["id"] != "call_7" || response["id"] != "call_7" {
		t.Fatalf("Vertex IDs were not restored: call=%#v response=%#v", call, response)
	}
}

func TestStreamResponseUsesRequestContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil).WithContext(ctx)
	rec := httptest.NewRecorder()
	called := false

	streamResponse(rec, req, "gemini-test", func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
		called = true
		return ctx.Err()
	})

	if !called {
		t.Fatal("stream function was not called")
	}
	if rec.Body.Len() != 0 {
		t.Fatalf("response body length = %d, want 0 for canceled request", rec.Body.Len())
	}
}

func TestStreamResponseKeepsToolCallsAcrossTrailingFinishMetadata(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()

	streamResponse(rec, req, "gemini-test", func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
		if err := onChunk(&proxy.CallResult{
			FunctionCalls: []model.FunctionCall{{Name: "echo", Args: map[string]interface{}{"text": "hello"}}},
		}); err != nil {
			return err
		}
		return onChunk(&proxy.CallResult{FinishReason: "SAFETY"})
	})

	objects := decodeStreamDataObjects(t, rec.Body.String())
	if len(objects) == 0 {
		t.Fatal("stream should contain response chunks")
	}
	toolCalls, ok := deltaFromChunk(t, objects[0])["tool_calls"].([]interface{})
	if !ok || len(toolCalls) != 1 {
		t.Fatalf("tool call delta = %#v, want one actual tool call", deltaFromChunk(t, objects[0])["tool_calls"])
	}
	choices, ok := objects[len(objects)-1]["choices"].([]interface{})
	if !ok || len(choices) != 1 {
		t.Fatalf("final chunk choices = %v, want one choice", objects[len(objects)-1]["choices"])
	}
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		t.Fatalf("final choice type = %T, want object", choices[0])
	}
	if got, want := choice["finish_reason"], "tool_calls"; got != want {
		t.Fatalf("final finish_reason = %v, want %q; body=%s", got, want, rec.Body.String())
	}
	if _, ok := choice["native_finish_reason"]; ok {
		t.Fatalf("final choice must omit native_finish_reason: %#v", choice)
	}
}

func TestOpenAIFinishReasonMapsSafetyToContentFilter(t *testing.T) {
	result := &proxy.CallResult{
		FinishReason:   "SAFETY",
		PromptFeedback: map[string]interface{}{"blockReason": "SAFETY", "blockReasonMessage": "blocked"},
	}
	if got := openAIFinishReason(result); got != "content_filter" {
		t.Fatalf("finish reason = %q, want content_filter", got)
	}
}

func TestOpenAIFinishReasonIgnoresUnspecifiedBlockReason(t *testing.T) {
	result := &proxy.CallResult{PromptFeedback: map[string]interface{}{"blockReason": "BLOCKED_REASON_UNSPECIFIED"}}
	if got := openAIFinishReason(result); got != "stop" {
		t.Fatalf("finish reason = %q, want stop", got)
	}
}

func TestOpenAIStreamProtocolOptionsAndStableMetadata(t *testing.T) {
	tests := []struct {
		name            string
		options         *model.ChatStreamOptions
		wantUsage       bool
		wantObfuscation bool
	}{
		{name: "defaults", wantObfuscation: true},
		{name: "usage", options: &model.ChatStreamOptions{IncludeUsage: true}, wantUsage: true, wantObfuscation: true},
		{name: "obfuscation disabled", options: &model.ChatStreamOptions{IncludeObfuscation: boolPointer(false)}},
		{name: "obfuscation enabled", options: &model.ChatStreamOptions{IncludeObfuscation: boolPointer(true)}, wantObfuscation: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reqModel := model.ChatCompletionRequest{
				Model:         "gemini-test",
				Messages:      []model.ChatMessage{{Role: "user", Content: "hello"}},
				Stream:        true,
				StreamOptions: tt.options,
			}
			rec := httptest.NewRecorder()
			httpReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
			streamResponseForRequest(rec, httpReq, reqModel, func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
				if err := onChunk(&proxy.CallResult{TextParts: []model.TextPart{{Text: "hello"}}}); err != nil {
					return err
				}
				return onChunk(&proxy.CallResult{FinishReason: "STOP", UsageMetadata: map[string]interface{}{
					"promptTokenCount": float64(2), "candidatesTokenCount": float64(1), "totalTokenCount": float64(3),
				}})
			})

			objects := decodeStreamDataObjects(t, rec.Body.String())
			if len(objects) != 2 {
				t.Fatalf("stream chunks = %d, want 2: %s", len(objects), rec.Body.String())
			}
			firstID, firstCreated := objects[0]["id"], objects[0]["created"]
			for index, object := range objects {
				if object["id"] != firstID || object["created"] != firstCreated {
					t.Fatalf("chunk %d metadata changed: first=%#v current=%#v", index, objects[0], object)
				}
				_, hasObfuscation := object["obfuscation"].(string)
				if hasObfuscation != tt.wantObfuscation {
					t.Fatalf("chunk %d obfuscation presence = %v, want %v: %#v", index, hasObfuscation, tt.wantObfuscation, object)
				}
				choices, ok := object["choices"].([]interface{})
				if !ok || len(choices) != 1 {
					t.Fatalf("chunk %d choices = %#v, want one choice", index, object["choices"])
				}
				choice := choices[0].(map[string]interface{})
				if _, ok := choice["native_finish_reason"]; ok {
					t.Fatalf("chunk %d unexpectedly included native_finish_reason: %#v", index, choice)
				}
				delta := choice["delta"].(map[string]interface{})
				if delta["role"] != "assistant" {
					t.Fatalf("chunk %d role = %#v, want assistant", index, delta["role"])
				}
				if _, ok := delta["reasoning_content"]; !ok {
					t.Fatalf("chunk %d omitted reasoning_content: %#v", index, delta)
				}
				if _, ok := delta["tool_calls"]; !ok {
					t.Fatalf("chunk %d omitted tool_calls: %#v", index, delta)
				}
			}
			if _, hasUsage := objects[0]["usage"]; hasUsage {
				t.Fatalf("intermediate chunk must omit usage: %#v", objects[0])
			}
			usage, hasUsage := objects[1]["usage"]
			if hasUsage != tt.wantUsage {
				t.Fatalf("final usage presence = %v, want %v: %#v", hasUsage, tt.wantUsage, objects[1])
			}
			if tt.wantUsage {
				usageObject, ok := usage.(map[string]interface{})
				if !ok || usageObject["prompt_tokens"] != float64(2) || usageObject["completion_tokens"] != float64(1) || usageObject["total_tokens"] != float64(3) {
					t.Fatalf("final usage = %#v, want complete 2/1/3 counters", usage)
				}
			}
			firstChoice := objects[0]["choices"].([]interface{})[0].(map[string]interface{})
			if firstChoice["finish_reason"] != nil {
				t.Fatalf("intermediate finish_reason = %#v, want null", firstChoice["finish_reason"])
			}
			finalChoice := objects[1]["choices"].([]interface{})[0].(map[string]interface{})
			if finalChoice["finish_reason"] != "stop" {
				t.Fatalf("final finish_reason = %#v, want stop", finalChoice["finish_reason"])
			}
			if !strings.HasSuffix(rec.Body.String(), "data: [DONE]\n\n") {
				t.Fatalf("stream does not end in [DONE]: %s", rec.Body.String())
			}
		})
	}
}

func TestValidateOpenAIRequestRejectsStreamOptionsWithoutStreaming(t *testing.T) {
	message := validateOpenAIRequest(model.ChatCompletionRequest{
		Model:         "gemini-test",
		Messages:      []model.ChatMessage{{Role: "user", Content: "hello"}},
		StreamOptions: &model.ChatStreamOptions{IncludeUsage: true},
	})
	if message != "stream_options may only be set when stream is true" {
		t.Fatalf("validation message = %q", message)
	}
}

func boolPointer(value bool) *bool {
	return &value
}

func TestStreamResponseIgnoresUnspecifiedPromptFeedback(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()

	streamResponse(rec, req, "gemini-test", func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
		return onChunk(&proxy.CallResult{
			TextParts:      []model.TextPart{{Text: "complete answer"}},
			FinishReason:   "STOP",
			PromptFeedback: map[string]interface{}{"blockReason": "BLOCKED_REASON_UNSPECIFIED"},
		})
	})

	objects := decodeStreamDataObjects(t, rec.Body.String())
	if len(objects) == 0 {
		t.Fatal("stream should contain response chunks")
	}
	choices, ok := objects[len(objects)-1]["choices"].([]interface{})
	if !ok || len(choices) != 1 {
		t.Fatalf("final chunk choices = %v, want one choice", objects[len(objects)-1]["choices"])
	}
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		t.Fatalf("final choice type = %T, want object", choices[0])
	}
	if got, want := choice["finish_reason"], "stop"; got != want {
		t.Fatalf("final finish_reason = %v, want %q; body=%s", got, want, rec.Body.String())
	}
}

func TestStreamResponseInvalidatesStaleOutputUsage(t *testing.T) {
	reqModel := model.ChatCompletionRequest{
		Model:         "gemini-test",
		Messages:      []model.ChatMessage{{Role: "user", Content: "hello"}},
		Stream:        true,
		StreamOptions: &model.ChatStreamOptions{IncludeUsage: true},
	}
	httpReq := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()

	streamResponseForRequest(rec, httpReq, reqModel, func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
		if err := onChunk(&proxy.CallResult{
			TextParts: []model.TextPart{{Text: "a"}},
			UsageMetadata: map[string]interface{}{
				"promptTokenCount": float64(7), "candidatesTokenCount": float64(1), "totalTokenCount": float64(8),
			},
		}); err != nil {
			return err
		}
		return onChunk(&proxy.CallResult{TextParts: []model.TextPart{{Text: strings.Repeat("later output ", 20)}}, FinishReason: "STOP"})
	})

	objects := decodeStreamDataObjects(t, rec.Body.String())
	usageObject := objects[len(objects)-1]["usage"].(map[string]interface{})
	if got := int(usageObject["prompt_tokens"].(float64)); got != 7 {
		t.Fatalf("prompt_tokens = %d, want invariant upstream value 7", got)
	}
	if got := int(usageObject["completion_tokens"].(float64)); got <= 1 {
		t.Fatalf("completion_tokens = %d, stale upstream value was retained", got)
	}
	if got := rec.Header().Get("X-Usage-Estimated"); got != "true" {
		t.Fatalf("X-Usage-Estimated = %q, want true", got)
	}
	if got := rec.Result().Trailer.Get("X-Usage-Estimated"); got != "true" {
		t.Fatalf("X-Usage-Estimated trailer = %q, want true", got)
	}
}

func TestWriteOpenAIStreamErrorPassthroughUpstreamMessage(t *testing.T) {
	rec := httptest.NewRecorder()

	errMsg := "api error (code 5): model not found"
	if err := writeOpenAIStreamError(rec, nil, errors.New(errMsg)); err != nil {
		t.Fatalf("writeOpenAIStreamError returned error: %v", err)
	}

	body := rec.Body.String()
	if !strings.Contains(body, errMsg) {
		t.Fatalf("stream error body = %q, want %q", body, errMsg)
	}
}

func TestWriteOpenAIStreamErrorPreservesGenerationFinishReason(t *testing.T) {
	rec := httptest.NewRecorder()
	message := "generation failed: finishReason=SAFETY"

	if err := writeOpenAIStreamError(rec, nil, errors.New(message)); err != nil {
		t.Fatalf("writeOpenAIStreamError returned error: %v", err)
	}

	body := rec.Body.String()
	if !strings.Contains(body, message) {
		t.Fatalf("stream error body = %q, want finish reason message", body)
	}
}

func TestConvertMessagesMergesParallelToolResponses(t *testing.T) {
	messages := []model.ChatMessage{
		{
			Role: "assistant",
			ToolCalls: []model.ChatToolCall{
				{ID: "call_1", Function: &model.ChatFunctionCall{Name: "read_file", Arguments: `{"path":"a"}`}},
				{ID: "call_2", Function: &model.ChatFunctionCall{Name: "read_file", Arguments: `{"path":"b"}`}},
			},
		},
		{
			Role:       "tool",
			ToolCallID: "call_1",
			Content:    "content of a",
		},
		{
			Role:       "tool",
			ToolCallID: "call_2",
			Content:    "content of b",
		},
	}

	contents, _ := convertMessages("gemini-2.0-flash-exp", messages)
	if len(contents) != 2 {
		t.Fatalf("contents length = %d, want 2 (assistant + merged user tool responses)", len(contents))
	}

	if got := contents[1]["role"]; got != "user" {
		t.Fatalf("merged tool response role = %v, want user", got)
	}

	parts := contents[1]["parts"].([]map[string]interface{})
	if len(parts) != 2 {
		t.Fatalf("merged tool response parts length = %d, want 2", len(parts))
	}

	fnResp1 := parts[0]["functionResponse"].(map[string]interface{})
	if got := fnResp1["name"]; got != "read_file" {
		t.Fatalf("fnResp1 name = %v, want read_file", got)
	}
	fnResp2 := parts[1]["functionResponse"].(map[string]interface{})
	if got := fnResp2["name"]; got != "read_file" {
		t.Fatalf("fnResp2 name = %v, want read_file", got)
	}
}

func TestConvertMessagesCombinesSystemAndDeveloperInstructions(t *testing.T) {
	contents, system := convertMessages("gemini-test", []model.ChatMessage{
		{Role: "system", Content: "system rule"},
		{Role: "developer", Content: "developer rule"},
		{Role: "user", Content: "hello"},
	})
	if len(contents) != 1 || contents[0]["role"] != "user" {
		t.Fatalf("contents = %v, want one user turn", contents)
	}
	parts := system.(map[string]interface{})["parts"].([]map[string]interface{})
	if len(parts) != 2 || parts[0]["text"] != "system rule" || parts[1]["text"] != "developer rule" {
		t.Fatalf("system parts = %v", parts)
	}
}

func TestApplyOpenAIResponseFormatNormalizesJSONSchema(t *testing.T) {
	config := map[string]interface{}{}
	applyOpenAIResponseFormat(config, map[string]interface{}{
		"type": "json_schema",
		"json_schema": map[string]interface{}{
			"schema": map[string]interface{}{
				"$defs": map[string]interface{}{"value": map[string]interface{}{"type": "string"}},
				"$ref":  "#/$defs/value",
			},
		},
	})
	if config["responseMimeType"] != "application/json" {
		t.Fatalf("responseMimeType = %v", config["responseMimeType"])
	}
	schema := config["responseJsonSchema"].(map[string]interface{})
	if schema["type"] != "string" || schema["$ref"] != nil || schema["$defs"] != nil {
		t.Fatalf("response schema was not normalized: %v", schema)
	}
}

func TestBuildOpenAIToolCallsGeneratesUniqueIDs(t *testing.T) {
	functionCalls := []model.FunctionCall{
		{Name: "search", Args: map[string]interface{}{"q": "test"}},
	}

	calls1 := buildOpenAIToolCalls(functionCalls, false)
	calls2 := buildOpenAIToolCalls(functionCalls, false)

	if calls1[0].ID == calls2[0].ID {
		t.Fatalf("tool call IDs should be unique, got duplicate: %s", calls1[0].ID)
	}
}

func decodeStreamDataObjects(t *testing.T, body string) []map[string]interface{} {
	t.Helper()

	var objects []map[string]interface{}
	for _, block := range strings.Split(body, "\n\n") {
		line := strings.TrimSpace(block)
		if line == "" || !strings.HasPrefix(line, "data: ") {
			continue
		}
		payload := strings.TrimPrefix(line, "data: ")
		if payload == "[DONE]" {
			continue
		}
		var object map[string]interface{}
		if err := sonic.Unmarshal([]byte(payload), &object); err != nil {
			t.Fatalf("unmarshal stream payload %q: %v", payload, err)
		}
		objects = append(objects, object)
	}
	return objects
}

func decodeSSEEventObjects(t *testing.T, body, eventType string) []map[string]interface{} {
	t.Helper()
	var objects []map[string]interface{}
	for _, block := range strings.Split(body, "\n\n") {
		lines := strings.Split(strings.TrimSpace(block), "\n")
		matched := eventType == ""
		payload := ""
		for _, line := range lines {
			if strings.TrimSpace(line) == "event: "+eventType {
				matched = true
			}
			if strings.HasPrefix(strings.TrimSpace(line), "data: ") {
				payload = strings.TrimPrefix(strings.TrimSpace(line), "data: ")
			}
		}
		if !matched || payload == "" || payload == "[DONE]" {
			continue
		}
		var object map[string]interface{}
		if err := sonic.Unmarshal([]byte(payload), &object); err != nil {
			t.Fatalf("unmarshal %s payload %q: %v", eventType, payload, err)
		}
		objects = append(objects, object)
	}
	return objects
}

func deltaFromChunk(t *testing.T, chunk map[string]interface{}) map[string]interface{} {
	t.Helper()

	choices, ok := chunk["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Fatalf("chunk choices = %v, want non-empty list", chunk["choices"])
	}
	choice, ok := choices[0].(map[string]interface{})
	if !ok {
		t.Fatalf("choice type = %T, want object", choices[0])
	}
	delta, ok := choice["delta"].(map[string]interface{})
	if !ok {
		t.Fatalf("delta type = %T, want object", choice["delta"])
	}
	return delta
}

func TestBuildOpenAIMessageUsesOfficialURLAnnotations(t *testing.T) {
	gm := &model.GroundingMetadata{
		WebSearchQueries: []string{"test"},
		GroundingChunks: []model.GroundingChunk{
			{Web: &model.WebChunk{Title: "Test Page", URI: "https://example.com"}},
		},
		GroundingSupports: []model.GroundingSupport{{
			GroundingChunkIndices: []int{0},
			Segment:               &model.Segment{StartIndex: 0, EndIndex: 6},
		}},
	}
	msg := buildOpenAIMessage(&proxy.CallResult{
		TextParts:         []model.TextPart{{Text: "result text"}},
		GroundingMetadata: gm,
	})

	if msg.GroundingMetadata != nil || msg.Citations != nil {
		t.Fatalf("non-standard grounding fields should be omitted: %+v", msg)
	}
	if len(msg.Annotations) != 1 || msg.Annotations[0].URLCitation == nil {
		t.Fatalf("Annotations = %+v, want one url_citation", msg.Annotations)
	}
	if got := msg.Annotations[0].URLCitation.URL; got != "https://example.com" {
		t.Fatalf("citation url = %v, want https://example.com", got)
	}
}

func TestConvertMessagesDoesNotInjectHiddenTurn(t *testing.T) {
	messages := []model.ChatMessage{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there"},
	}

	contents, _ := convertMessages("gemini-3.6", messages)
	if len(contents) != 2 {
		t.Fatalf("contents length = %d, want 2", len(contents))
	}
	if got := contents[1]["role"]; got != "model" {
		t.Fatalf("last role = %v, want model", got)
	}
}

func TestValidateOpenAIRequestDistinguishesOmittedAndInvalidZeroValues(t *testing.T) {
	base := model.ChatCompletionRequest{
		Model:    "gemini-test",
		Messages: []model.ChatMessage{{Role: "user", Content: "hello"}},
	}
	if got := validateOpenAIRequest(base); got != "" {
		t.Fatalf("omitted optional values rejected: %s", got)
	}

	zero := 0
	base.N = &zero
	if got := validateOpenAIRequest(base); got == "" {
		t.Fatal("explicit n=0 should be rejected")
	}
	base.N = nil
	base.MaxCompletionTokens = &zero
	if got := validateOpenAIRequest(base); got == "" {
		t.Fatal("explicit max_completion_tokens=0 should be rejected")
	}
}

func TestOpenAIReasoningConfigMapsExplicitEffort(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
		effort    string
		key       string
		want      interface{}
	}{
		{name: "Gemini 3.6 minimal", modelName: "gemini-3.6-flash", effort: "minimal", key: "thinkingLevel", want: "MINIMAL"},
		{name: "Gemini 3.6 xhigh degrades to high", modelName: "gemini-3.6-flash", effort: "xhigh", key: "thinkingLevel", want: "HIGH"},
		{name: "Gemini 3.6 max degrades to high", modelName: "gemini-3.6-flash", effort: "max", key: "thinkingLevel", want: "HIGH"},
		{name: "Gemini 3 medium", modelName: "gemini-3-flash-preview", effort: "medium", key: "thinkingLevel", want: "MEDIUM"},
		{name: "Gemini 3 Pro defers minimal capability to catalog", modelName: "gemini-3.1-pro-preview", effort: "minimal", key: "thinkingLevel", want: "MINIMAL"},
		{name: "Gemini 2.5 low", modelName: "gemini-2.5-flash", effort: "low", key: "thinkingBudget", want: 1024},
		{name: "Gemini 2.5 medium", modelName: "gemini-2.5-flash", effort: "medium", key: "thinkingBudget", want: 8192},
		{name: "Gemini 2.5 high", modelName: "gemini-2.5-flash", effort: "high", key: "thinkingBudget", want: 24576},
		{name: "Gemini 2.5 xhigh degrades to high", modelName: "gemini-2.5-flash", effort: "xhigh", key: "thinkingBudget", want: 24576},
		{name: "Gemini 2.5 max degrades to high", modelName: "gemini-2.5-flash", effort: "max", key: "thinkingBudget", want: 24576},
		{name: "Gemini 2.5 disables thinking", modelName: "gemini-2.5-flash", effort: "none", key: "thinkingBudget", want: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			effort := tt.effort
			if message := validateOpenAIReasoningEffort(tt.modelName, &effort); message != "" {
				t.Fatalf("reasoning_effort rejected: %s", message)
			}
			config := openAIReasoningConfig(tt.modelName, &effort)
			if len(config) != 1 {
				t.Fatalf("thinking config = %#v, want one field", config)
			}
			if got := config[tt.key]; got != tt.want {
				t.Fatalf("%s = %v, want %v", tt.key, got, tt.want)
			}
		})
	}
}

func TestOpenAIReasoningEffortValidation(t *testing.T) {
	if config := openAIReasoningConfig("gemini-3.6-flash", nil); config != nil {
		t.Fatalf("omitted reasoning_effort produced config: %#v", config)
	}

	tests := []struct {
		modelName string
		effort    string
	}{
		{modelName: "gemini-3.6-flash", effort: "none"},
		{modelName: "gemini-2.5-pro", effort: "none"},
		{modelName: "gemini-3.6-flash", effort: "ultra"},
		{modelName: "gemini-2.0-flash", effort: "high"},
	}
	for _, tt := range tests {
		effort := tt.effort
		if message := validateOpenAIReasoningEffort(tt.modelName, &effort); message == "" {
			t.Fatalf("model=%s effort=%s should be rejected", tt.modelName, tt.effort)
		}
	}
}

func TestValidateOpenAIRequestRejectsUnsupportedMessageRole(t *testing.T) {
	req := model.ChatCompletionRequest{
		Model:    "gemini-test",
		Messages: []model.ChatMessage{{Role: "function", Content: "legacy tool result"}},
	}
	if got := validateOpenAIRequest(req); got == "" {
		t.Fatal("legacy function role should be rejected")
	}
}

func TestBuildResponseContentKeepsThoughtOutOfImageMarkdown(t *testing.T) {
	result := &proxy.CallResult{Parts: []model.VertexPart{
		{Text: "private reasoning", Thought: true},
		{Text: "visible answer"},
		{InlineData: &model.InlineData{MimeType: "image/png", Data: "aW1n"}},
	}}
	content, reasoning := buildResponseContent(result)
	text, _ := content.(string)
	if reasoning != "private reasoning" {
		t.Fatalf("reasoning = %q", reasoning)
	}
	if strings.Contains(text, "private reasoning") || !strings.Contains(text, "visible answer") || !strings.Contains(text, "data:image/png;base64,aW1n") {
		t.Fatalf("visible content = %q", text)
	}
}

func TestConvertContentToPartsPreservesRemoteImageAndFiles(t *testing.T) {
	parts := convertContentToParts([]interface{}{
		map[string]interface{}{"type": "image_url", "image_url": map[string]interface{}{"url": "https://example.com/a.png"}},
		map[string]interface{}{"type": "file", "file": map[string]interface{}{"file_data": "data:application/pdf;base64,cGRm"}},
	})
	if len(parts) != 2 {
		t.Fatalf("parts = %#v", parts)
	}
	remote, _ := parts[0]["fileData"].(map[string]interface{})
	if remote["fileUri"] != "https://example.com/a.png" {
		t.Fatalf("remote image part = %#v", parts[0])
	}
	inline, _ := parts[1]["inlineData"].(map[string]interface{})
	if inline["mimeType"] != "application/pdf" || inline["data"] != "cGRm" {
		t.Fatalf("inline file part = %#v", parts[1])
	}
}
