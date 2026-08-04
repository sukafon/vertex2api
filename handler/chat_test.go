package handler

import (
	"context"
	"errors"
	"net/http/httptest"
	"strings"
	"testing"

	"vertex2api/model"
	"vertex2api/proxy"

	"github.com/bytedance/sonic"
)

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

func TestBuildOpenAIRequestOptionsTranslatesNativeToolsAndPassesThroughUnknown(t *testing.T) {
	options := buildOpenAIRequestOptions(model.ChatCompletionRequest{
		Tools: []map[string]interface{}{
			{"type": "web_search_preview"},
			{"type": "code_interpreter"},
			{"type": "url_context"},
			{"type": "vendor_tool", "name": "google_search"},
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
	unknown := tools[3].(map[string]interface{})
	if got := unknown["type"]; got != "vendor_tool" {
		t.Fatalf("unknown tool should pass through, got type %v", got)
	}
	if got := unknown["name"]; got != "google_search" {
		t.Fatalf("unknown tool name should be preserved, got %v", got)
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

func TestWriteOpenAIStreamErrorPassthroughUpstreamMessage(t *testing.T) {
	rec := httptest.NewRecorder()

	errMsg := "api error (code 5): model not found"
	if err := writeOpenAIStreamError(rec, errors.New(errMsg)); err != nil {
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

	if err := writeOpenAIStreamError(rec, errors.New(message)); err != nil {
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

func TestValidateOpenAIRequestRejectsUnsupportedMessageRole(t *testing.T) {
	req := model.ChatCompletionRequest{
		Model:    "gemini-test",
		Messages: []model.ChatMessage{{Role: "function", Content: "legacy tool result"}},
	}
	if got := validateOpenAIRequest(req); got == "" {
		t.Fatal("legacy function role should be rejected")
	}
}
