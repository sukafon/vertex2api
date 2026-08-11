package handler

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vertex2api/model"
	"vertex2api/proxy"
)

func TestConvertAnthropicMessagesConvertsToolsAndResults(t *testing.T) {
	req := model.AnthropicMessageRequest{
		System: "be concise",
		Messages: []model.AnthropicInputMessage{
			{
				Role: "assistant",
				Content: []interface{}{
					map[string]interface{}{
						"type":  "tool_use",
						"id":    "toolu_1",
						"name":  "lookup",
						"input": map[string]interface{}{"query": "weather"},
						"google": map[string]interface{}{
							"thoughtSignature": "sig",
						},
					},
				},
			},
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type":        "tool_result",
						"tool_use_id": "toolu_1",
						"content":     `{"temperature":72}`,
					},
				},
			},
		},
	}

	contents, system := convertAnthropicMessages(req)
	if system == nil {
		t.Fatal("system instruction should be converted")
	}
	if len(contents) != 2 {
		t.Fatalf("contents length = %d, want 2", len(contents))
	}

	assistantPart := contents[0]["parts"].([]map[string]interface{})[0]
	functionCall := assistantPart["functionCall"].(map[string]interface{})
	if got := functionCall["name"]; got != "lookup" {
		t.Fatalf("functionCall name = %v, want lookup", got)
	}
	args := functionCall["args"].(map[string]interface{})
	if got := args["query"]; got != "weather" {
		t.Fatalf("functionCall query = %v, want weather", got)
	}
	if got := assistantPart["thoughtSignature"]; got != "sig" {
		t.Fatalf("thoughtSignature = %v, want sig", got)
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

func TestConvertAnthropicMessagesAddsDummySignatureForToolUse(t *testing.T) {
	contents, _ := convertAnthropicMessages(model.AnthropicMessageRequest{
		Messages: []model.AnthropicInputMessage{
			{
				Role: "assistant",
				Content: []interface{}{
					map[string]interface{}{
						"type":  "tool_use",
						"id":    "toolu_1",
						"name":  "lookup",
						"input": map[string]interface{}{"query": "weather"},
					},
				},
			},
		},
	})

	part := contents[0]["parts"].([]map[string]interface{})[0]
	if got := part["thoughtSignature"]; got != anthropicDummyThoughtSignature {
		t.Fatalf("thoughtSignature = %v, want dummy", got)
	}
}

func TestConvertAnthropicMessagesDoesNotAddDummySignatureForNonGemini3(t *testing.T) {
	contents, _ := convertAnthropicMessages(model.AnthropicMessageRequest{
		Model: "gemini-2.5-flash",
		Messages: []model.AnthropicInputMessage{
			{
				Role: "assistant",
				Content: []interface{}{
					map[string]interface{}{
						"type": "tool_use",
						"id":   "toolu_1",
						"name": "lookup",
					},
				},
			},
		},
	})

	part := contents[0]["parts"].([]map[string]interface{})[0]
	if got, ok := part["thoughtSignature"]; ok {
		t.Fatalf("thoughtSignature should not be added for non-Gemini 3 models, got %v", got)
	}
}

func TestSendAnthropicResponseIncludesUsage(t *testing.T) {
	rec := httptest.NewRecorder()

	sendAnthropicResponse(rec, model.AnthropicMessageRequest{Model: "gemini-test"}, "gemini-test", &proxy.CallResult{
		TextParts:    []model.TextPart{{Text: "done"}},
		FinishReason: "STOP",
	})

	if !strings.Contains(rec.Body.String(), `"usage"`) {
		t.Fatalf("response should include usage, got %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"input_tokens"`) || !strings.Contains(rec.Body.String(), `"output_tokens"`) {
		t.Fatalf("response usage missing token counts, got %s", rec.Body.String())
	}
}

func TestAnthropicStreamEventsIncludeUsage(t *testing.T) {
	rec := httptest.NewRecorder()

	if err := writeAnthropicMessageStart(rec, "msg_test", "gemini-test", &model.AnthropicUsage{InputTokens: 10}); err != nil {
		t.Fatalf("writeAnthropicMessageStart returned error: %v", err)
	}
	if err := writeAnthropicMessageDelta(rec, "end_turn", 5); err != nil {
		t.Fatalf("writeAnthropicMessageDelta returned error: %v", err)
	}

	if !strings.Contains(rec.Body.String(), `"usage"`) {
		t.Fatalf("stream events should include usage, got %s", rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"input_tokens"`) || !strings.Contains(rec.Body.String(), `"output_tokens"`) {
		t.Fatalf("stream events usage missing token counts, got %s", rec.Body.String())
	}
}

func TestConvertAnthropicMessagesFoldsSystemRoleIntoSingleInstruction(t *testing.T) {
	req := model.AnthropicMessageRequest{
		System: []interface{}{
			map[string]interface{}{"type": "text", "text": "top level system"},
		},
		Messages: []model.AnthropicInputMessage{
			{
				Role:    "system",
				Content: "message system",
			},
			{
				Role:    "user",
				Content: "hello",
			},
		},
	}

	contents, system := convertAnthropicMessages(req)
	if len(contents) != 1 {
		t.Fatalf("contents length = %d, want 1", len(contents))
	}
	if got := contents[0]["role"]; got != "user" {
		t.Fatalf("content role = %v, want user", got)
	}

	systemMap := system.(map[string]interface{})
	parts := systemMap["parts"].([]map[string]interface{})
	if len(parts) != 1 {
		t.Fatalf("system parts length = %d, want 1", len(parts))
	}
	text := parts[0]["text"].(string)
	if !strings.Contains(text, "top level system") || !strings.Contains(text, "message system") {
		t.Fatalf("system text = %q, want both system inputs", text)
	}
}

func TestAnthropicRequestForSignatureRetryDowngradesThinkingAndTools(t *testing.T) {
	req := model.AnthropicMessageRequest{
		Model:    "gemini-3.1-pro-preview",
		Thinking: map[string]interface{}{"type": "enabled", "budget_tokens": float64(1024)},
		Messages: []model.AnthropicInputMessage{
			{
				Role: "assistant",
				Content: []interface{}{
					map[string]interface{}{"type": "thinking", "thinking": "private chain", "signature": "bad"},
					map[string]interface{}{"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": map[string]interface{}{"command": "ls"}},
					map[string]interface{}{"type": "text", "text": "visible"},
				},
			},
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
				},
			},
		},
	}

	stage1 := anthropicRequestForSignatureRetry(req, 1)
	if stage1.Thinking != nil {
		t.Fatal("stage 1 should disable top-level thinking")
	}
	stage1Content := stage1.Messages[0].Content.([]interface{})
	first := stage1Content[0].(map[string]interface{})
	if got := first["type"]; got != "text" {
		t.Fatalf("stage 1 first block type = %v, want text", got)
	}
	if got := first["text"]; got != "private chain" {
		t.Fatalf("stage 1 thinking text = %v, want private chain", got)
	}
	if got := stage1Content[1].(map[string]interface{})["type"]; got != "tool_use" {
		t.Fatalf("stage 1 should keep tool_use, got %v", got)
	}

	stage2 := anthropicRequestForSignatureRetry(req, 2)
	stage2Assistant := stage2.Messages[0].Content.([]interface{})
	if got := stage2Assistant[1].(map[string]interface{})["type"]; got != "text" {
		t.Fatalf("stage 2 should downgrade tool_use to text, got %v", got)
	}
	if text := stage2Assistant[1].(map[string]interface{})["text"].(string); !strings.Contains(text, "tool_use") || !strings.Contains(text, "Bash") {
		t.Fatalf("stage 2 tool text = %q, want tool_use details", text)
	}
	stage2User := stage2.Messages[1].Content.([]interface{})
	if text := stage2User[0].(map[string]interface{})["text"].(string); !strings.Contains(text, "tool_result") || !strings.Contains(text, "toolu_1") {
		t.Fatalf("stage 2 tool result text = %q, want tool_result details", text)
	}
}

func TestIsAnthropicSignatureRelatedError(t *testing.T) {
	if !isAnthropicSignatureRelatedError(errors.New("invalid thought_signature in request")) {
		t.Fatal("thought_signature error should be treated as signature related")
	}
	if isAnthropicSignatureRelatedError(errors.New("request signature mismatch")) {
		t.Fatal("generic signature error should not be treated as thought signature related")
	}
	if isAnthropicSignatureRelatedError(errors.New("model not found")) {
		t.Fatal("unrelated error should not be treated as signature related")
	}
}

func TestBuildAnthropicRequestOptionsConvertsToolsAndChoice(t *testing.T) {
	options := buildAnthropicRequestOptions(model.AnthropicMessageRequest{
		Tools: []map[string]interface{}{
			{
				"name":        "lookup",
				"description": "look up data",
				"input_schema": map[string]interface{}{
					"$schema":              "https://json-schema.org/draft/2020-12/schema",
					"type":                 "object",
					"additionalProperties": false,
					"properties": map[string]interface{}{
						"query": map[string]interface{}{
							"type":      "string",
							"minLength": 1,
						},
					},
				},
			},
			{
				"type": "web_search_20250305",
				"name": "web_search",
			},
			{
				"type": "code_execution",
			},
		},
		ToolChoice: map[string]interface{}{
			"type": "tool",
			"name": "lookup",
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
	parameters := declaration["parameters"].(map[string]interface{})
	if got := parameters["type"]; got != "object" {
		t.Fatalf("parameters type = %v, want object", got)
	}
	if _, ok := parameters["$schema"]; ok {
		t.Fatal("unsupported $schema field should be removed")
	}
	if got := parameters["additionalProperties"]; got != false {
		t.Fatalf("additionalProperties = %v, want false", got)
	}
	properties := parameters["properties"].(map[string]interface{})
	query := properties["query"].(map[string]interface{})
	if got := query["minLength"]; got != float64(1) && got != 1 {
		t.Fatalf("minLength = %v, want 1", got)
	}
	if _, ok := tools[1].(map[string]interface{})["googleSearch"]; !ok {
		t.Fatal("web search tool should be converted to googleSearch")
	}
	if _, ok := tools[2].(map[string]interface{})["codeExecution"]; !ok {
		t.Fatal("code execution tool should be converted to codeExecution")
	}

	toolConfig := options.ToolConfig.(map[string]interface{})
	functionCallingConfig := toolConfig["functionCallingConfig"].(map[string]interface{})
	if got := functionCallingConfig["mode"]; got != "ANY" {
		t.Fatalf("tool choice mode = %v, want ANY", got)
	}
}

func TestBuildAnthropicContentIncludesToolUse(t *testing.T) {
	content := buildAnthropicContent(&proxy.CallResult{
		TextParts: []model.TextPart{{Text: "checking"}},
		FunctionCalls: []model.FunctionCall{
			{
				Name:             "lookup",
				Args:             map[string]interface{}{"query": "weather"},
				ThoughtSignature: "sig",
			},
		},
	})

	if len(content) != 2 {
		t.Fatalf("content length = %d, want 2", len(content))
	}
	if got := content[0]["type"]; got != "text" {
		t.Fatalf("first content type = %v, want text", got)
	}
	if got := content[1]["type"]; got != "tool_use" {
		t.Fatalf("second content type = %v, want tool_use", got)
	}
	if _, ok := content[1]["google"]; ok {
		t.Fatalf("tool_use should not expose non-standard google metadata: %v", content[1])
	}
	if _, ok := content[1]["signature"]; ok {
		t.Fatalf("tool_use should not expose non-standard signature: %v", content[1])
	}
	if got := anthropicToolSignatureFromID(content[1]["id"].(string)); got != "sig" {
		t.Fatalf("opaque tool signature = %v, want sig", got)
	}
	if got := anthropicStopReason(&proxy.CallResult{FunctionCalls: []model.FunctionCall{{Name: "lookup"}}}); got != "tool_use" {
		t.Fatalf("stop reason = %v, want tool_use", got)
	}
}

func TestStreamAnthropicResponseWritesMessageEvents(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/messages", nil)
	rec := httptest.NewRecorder()

	streamAnthropicResponse(rec, req, model.AnthropicMessageRequest{Model: "gemini-test"}, func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
		return onChunk(&proxy.CallResult{TextParts: []model.TextPart{{Text: "hello"}}, FinishReason: "STOP"})
	})

	body := rec.Body.String()
	for _, want := range []string{"event: message_start", "event: content_block_start", "event: content_block_delta", "event: message_delta", "event: message_stop"} {
		if !strings.Contains(body, want) {
			t.Fatalf("stream body missing %q: %s", want, body)
		}
	}
	if !strings.Contains(body, `"text":"hello"`) {
		t.Fatalf("stream body missing text delta: %s", body)
	}
}

func TestStreamAnthropicResponseKeepsTextInOneBlockAcrossChunks(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/messages", nil)
	rec := httptest.NewRecorder()

	streamAnthropicResponse(rec, req, model.AnthropicMessageRequest{Model: "gemini-test"}, func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
		chunks := []string{"hello ", "from ", "stream"}
		for _, chunk := range chunks {
			if err := onChunk(&proxy.CallResult{TextParts: []model.TextPart{{Text: chunk}}}); err != nil {
				return err
			}
		}
		return nil
	})

	body := rec.Body.String()
	if got := strings.Count(body, "event: content_block_start"); got != 1 {
		t.Fatalf("content_block_start count = %d, want 1: %s", got, body)
	}
	if got := strings.Count(body, "event: content_block_stop"); got != 1 {
		t.Fatalf("content_block_stop count = %d, want 1: %s", got, body)
	}
	for _, want := range []string{`"text":"hello "`, `"text":"from "`, `"text":"stream"`} {
		if !strings.Contains(body, want) {
			t.Fatalf("stream body missing %q: %s", want, body)
		}
	}
}

func TestStreamAnthropicResponseHandlesCumulativeTextChunks(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/messages", nil)
	rec := httptest.NewRecorder()

	streamAnthropicResponse(rec, req, model.AnthropicMessageRequest{Model: "gemini-test"}, func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
		chunks := []string{"hello", "hello world", "hello world"}
		for _, chunk := range chunks {
			if err := onChunk(&proxy.CallResult{TextParts: []model.TextPart{{Text: chunk}}}); err != nil {
				return err
			}
		}
		return nil
	})

	body := rec.Body.String()
	if got := strings.Count(body, "event: content_block_start"); got != 1 {
		t.Fatalf("content_block_start count = %d, want 1: %s", got, body)
	}
	if !strings.Contains(body, `"text":"hello"`) || !strings.Contains(body, `"text":" world"`) {
		t.Fatalf("stream body should emit only cumulative deltas: %s", body)
	}
	if got := strings.Count(body, `"text":"hello world"`); got != 0 {
		t.Fatalf("stream body repeated cumulative text: %s", body)
	}
}

func TestStreamAnthropicResponseEmitsDetachedVertexThoughtSignatureBeforeStop(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/messages", nil)
	rec := httptest.NewRecorder()

	streamAnthropicResponse(rec, req, model.AnthropicMessageRequest{Model: "gemini-test"}, func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
		if err := onChunk(&proxy.CallResult{Parts: []model.VertexPart{{Text: "planning", Thought: true}}}); err != nil {
			return err
		}
		return onChunk(&proxy.CallResult{
			Parts:        []model.VertexPart{{Thought: true, ThoughtSignature: "vertex-signature"}},
			FinishReason: "STOP",
		})
	})

	body := rec.Body.String()
	signature := strings.Index(body, `"signature":"vertex-signature"`)
	blockStop := strings.Index(body, "event: content_block_stop\n")
	messageDelta := strings.Index(body, "event: message_delta\n")
	messageStop := strings.Index(body, "event: message_stop\n")
	if signature < 0 || blockStop < signature || messageDelta < blockStop || messageStop < messageDelta {
		t.Fatalf("invalid thinking termination order:\n%s", body)
	}
	if got := strings.Count(body, "event: content_block_start\n"); got != 1 {
		t.Fatalf("thinking block starts = %d, want 1:\n%s", got, body)
	}
}

func TestBuildAnthropicContentMergesDetachedThoughtSignature(t *testing.T) {
	content := buildAnthropicContent(&proxy.CallResult{Parts: []model.VertexPart{
		{Text: "planning", Thought: true},
		{Thought: true, ThoughtSignature: "vertex-signature"},
	}})
	if len(content) != 1 || content[0]["type"] != "thinking" || content[0]["signature"] != "vertex-signature" {
		t.Fatalf("content = %#v", content)
	}
}

func TestStreamAnthropicResponseKeepsCompletedResponseAsEndTurnWithPromptFeedback(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/messages", nil)
	rec := httptest.NewRecorder()

	streamAnthropicResponse(rec, req, model.AnthropicMessageRequest{Model: "gemini-test"}, func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
		if err := onChunk(&proxy.CallResult{Parts: []model.VertexPart{{Text: "completed response"}}}); err != nil {
			return err
		}
		return onChunk(&proxy.CallResult{
			FinishReason:   "STOP",
			PromptFeedback: map[string]interface{}{"safetyRatings": []interface{}{}},
		})
	})

	body := rec.Body.String()
	if !strings.Contains(body, `"text":"completed response"`) {
		t.Fatalf("completed response missing:\n%s", body)
	}
	if !strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("final stop reason is not end_turn:\n%s", body)
	}
	if strings.Contains(body, `"stop_reason":"refusal"`) || strings.Contains(body, "event: ping\n") {
		t.Fatalf("completed third-party response was rewritten as refusal or heartbeat:\n%s", body)
	}
	if got := strings.Count(body, "event: message_stop\n"); got != 1 {
		t.Fatalf("message_stop events = %d, want 1:\n%s", got, body)
	}
}

func TestStreamAnthropicResponsePreservesToolUseAcrossTrailingFinishChunk(t *testing.T) {
	req := httptest.NewRequest("POST", "/v1/messages", nil)
	rec := httptest.NewRecorder()

	streamAnthropicResponse(rec, req, model.AnthropicMessageRequest{Model: "gemini-test"}, func(ctx context.Context, onChunk func(*proxy.CallResult) error) error {
		if err := onChunk(&proxy.CallResult{Parts: []model.VertexPart{{
			FunctionCall: &model.FunctionCall{Name: "lookup", Args: map[string]interface{}{"query": "weather"}},
		}}}); err != nil {
			return err
		}
		return onChunk(&proxy.CallResult{FinishReason: "STOP"})
	})

	body := rec.Body.String()
	if !strings.Contains(body, `"stop_reason":"tool_use"`) {
		t.Fatalf("trailing finish chunk overwrote tool_use:\n%s", body)
	}
	if strings.Contains(body, `"stop_reason":"end_turn"`) {
		t.Fatalf("tool call was incorrectly finalized as end_turn:\n%s", body)
	}
}

func TestAnthropicTextDelta(t *testing.T) {
	tests := []struct {
		name     string
		seen     string
		incoming string
		delta    string
		newSeen  string
	}{
		{name: "cumulative", seen: "hi", incoming: "hi there", delta: " there", newSeen: "hi there"},
		{name: "duplicate", seen: "hi there", incoming: "hi", delta: "", newSeen: "hi there"},
		{name: "incremental", seen: "hi", incoming: " there", delta: " there", newSeen: "hi there"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			delta, newSeen := anthropicTextDelta(tt.seen, tt.incoming)
			if delta != tt.delta || newSeen != tt.newSeen {
				t.Fatalf("anthropicTextDelta() = (%q, %q), want (%q, %q)", delta, newSeen, tt.delta, tt.newSeen)
			}
		})
	}
}

func TestModelsListUsesAnthropicFormatForAnthropicRequests(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("anthropic-version", "2023-06-01")
	rec := httptest.NewRecorder()

	ModelsList().ServeHTTP(rec, req)

	body := rec.Body.String()
	if !strings.Contains(body, `"has_more"`) {
		t.Fatalf("Anthropic model list missing has_more: %s", body)
	}
	if strings.Contains(body, `"object":"list"`) {
		t.Fatalf("Anthropic model list should not use OpenAI list shape: %s", body)
	}
}

func TestConvertAnthropicMessagesToolResultArrayContent(t *testing.T) {
	req := model.AnthropicMessageRequest{
		Messages: []model.AnthropicInputMessage{
			{
				Role: "assistant",
				Content: []interface{}{
					map[string]interface{}{
						"type":  "tool_use",
						"id":    "toolu_array_test",
						"name":  "read_file",
						"input": map[string]interface{}{"path": "go.mod"},
					},
				},
			},
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type":        "tool_result",
						"tool_use_id": "toolu_array_test",
						"content": []interface{}{
							map[string]interface{}{"type": "text", "text": "module vertex2api"},
						},
					},
				},
			},
		},
	}

	contents, _ := convertAnthropicMessages(req)
	if len(contents) != 2 {
		t.Fatalf("contents length = %d, want 2", len(contents))
	}

	modelPart := contents[0]["parts"].([]map[string]interface{})[0]
	functionCall := modelPart["functionCall"].(map[string]interface{})
	if got := functionCall["id"]; got != "toolu_array_test" {
		t.Fatalf("functionCall id = %v, want toolu_array_test", got)
	}

	userPart := contents[1]["parts"].([]map[string]interface{})[0]
	functionResponse := userPart["functionResponse"].(map[string]interface{})
	if got := functionResponse["name"]; got != "read_file" {
		t.Fatalf("functionResponse name = %v, want read_file", got)
	}
	if got := functionResponse["id"]; got != "toolu_array_test" {
		t.Fatalf("functionResponse id = %v, want toolu_array_test", got)
	}
	response := functionResponse["response"].(map[string]interface{})
	if got := response["result"]; got != "module vertex2api" {
		t.Fatalf("functionResponse result = %v, want 'module vertex2api'", got)
	}
}

func TestConvertAnthropicMessagesIsolatesGemini36FunctionResponse(t *testing.T) {
	req := model.AnthropicMessageRequest{
		Model: "gemini-3.6-flash",
		Messages: []model.AnthropicInputMessage{
			{
				Role: "assistant",
				Content: []interface{}{map[string]interface{}{
					"type": "tool_use", "id": "toolu_isolated", "name": "read_file",
					"input": map[string]interface{}{"path": "App.tsx"},
				}},
			},
			{
				Role: "user",
				Content: []interface{}{
					map[string]interface{}{
						"type": "tool_result", "tool_use_id": "toolu_isolated", "content": "file contents",
					},
					map[string]interface{}{"type": "text", "text": "Continue analyzing."},
				},
			},
		},
	}

	contents, _ := convertAnthropicMessages(req)
	if len(contents) != 3 {
		t.Fatalf("contents length = %d, want 3", len(contents))
	}
	if got := contents[1]["role"]; got != "user" || !contentHasFunctionResponse(contents[1]) {
		t.Fatalf("function response content = %#v, want standalone user function response", contents[1])
	}
	if parts := contents[1]["parts"].([]map[string]interface{}); len(parts) != 1 {
		t.Fatalf("function response parts = %d, want 1", len(parts))
	}
	if got := contents[2]["role"]; got != "user" || contentHasFunctionResponse(contents[2]) {
		t.Fatalf("text content = %#v, want standalone user text", contents[2])
	}
}

func TestConvertAnthropicMessagesMergesConsecutiveRoles(t *testing.T) {
	req := model.AnthropicMessageRequest{
		Messages: []model.AnthropicInputMessage{
			{
				Role:    "user",
				Content: "Hello",
			},
			{
				Role:    "user",
				Content: "World",
			},
		},
	}

	contents, _ := convertAnthropicMessages(req)
	if len(contents) != 1 {
		t.Fatalf("contents length = %d, want 1 after merging consecutive user messages", len(contents))
	}
	parts := contents[0]["parts"].([]map[string]interface{})
	if len(parts) != 2 {
		t.Fatalf("merged user message parts length = %d, want 2", len(parts))
	}
}

func TestBuildAnthropicContentGeneratesUniqueToolIDs(t *testing.T) {
	res1 := &proxy.CallResult{
		FunctionCalls: []model.FunctionCall{{Name: "test_tool", Args: map[string]interface{}{}}},
	}
	res2 := &proxy.CallResult{
		FunctionCalls: []model.FunctionCall{{Name: "test_tool", Args: map[string]interface{}{}}},
	}

	c1 := buildAnthropicContent(res1)
	c2 := buildAnthropicContent(res2)

	id1 := c1[0]["id"].(string)
	id2 := c2[0]["id"].(string)

	if id1 == id2 {
		t.Fatalf("tool IDs should be unique across calls, got duplicate: %s", id1)
	}
}

func TestBuildAnthropicContentDegradesGeneratedImageToStandardTextBlock(t *testing.T) {
	content := buildAnthropicContent(&proxy.CallResult{ImageParts: []model.InlineData{{MimeType: "image/png", Data: "aGVsbG8="}}})
	if len(content) != 1 || content[0]["type"] != "text" {
		t.Fatalf("content = %v, want text block", content)
	}
	text, _ := content[0]["text"].(string)
	if !strings.Contains(text, "data:image/png;base64,aGVsbG8=") {
		t.Fatalf("image text = %q", text)
	}
}

func TestBuildAnthropicContentOmitsUnrepresentableGroundingExtension(t *testing.T) {
	gm := &model.GroundingMetadata{
		WebSearchQueries: []string{"anthropic"},
		GroundingChunks: []model.GroundingChunk{
			{Web: &model.WebChunk{Title: "Anthropic", URI: "https://anthropic.com"}},
		},
	}
	res := &proxy.CallResult{
		TextParts:         []model.TextPart{{Text: "anthropic info"}},
		GroundingMetadata: gm,
	}

	content := buildAnthropicContent(res)
	if len(content) != 1 || content[0]["type"] != "text" {
		t.Fatalf("content = %v, want only the official text block", content)
	}
}

func TestConvertAnthropicMessagesDoesNotInjectHiddenTurn(t *testing.T) {
	req := model.AnthropicMessageRequest{
		Model: "gemini-3.6-flash",
		Messages: []model.AnthropicInputMessage{
			{
				Role:    "user",
				Content: "Hello",
			},
			{
				Role:    "assistant",
				Content: "Here is the result:",
			},
		},
	}

	contents, _ := convertAnthropicMessages(req)
	if len(contents) != 2 {
		t.Fatalf("contents length = %d, want 2", len(contents))
	}
	if got := contents[1]["role"]; got != "model" {
		t.Fatalf("last role = %v, want model", got)
	}
}

func TestValidateAnthropicRequestRejectsZeroMaxTokens(t *testing.T) {
	zero := 0
	req := model.AnthropicMessageRequest{
		Model:     "gemini-test",
		MaxTokens: &zero,
		Messages:  []model.AnthropicInputMessage{{Role: "user", Content: "hello"}},
	}
	if got := validateAnthropicRequest(req); got == "" {
		t.Fatal("explicit max_tokens=0 should be rejected")
	}
}

func TestValidateAnthropicRequestAcceptsSystemMessage(t *testing.T) {
	maxTokens := 32
	req := model.AnthropicMessageRequest{
		Model:     "gemini-test",
		MaxTokens: &maxTokens,
		Messages: []model.AnthropicInputMessage{
			{Role: "system", Content: "You are helpful."},
			{Role: "user", Content: "hello"},
		},
	}

	if got := validateAnthropicRequest(req); got != "" {
		t.Fatalf("system message should be accepted, got %q", got)
	}

	contents, systemInstruction := convertAnthropicMessages(req)
	if len(contents) != 1 || contents[0]["role"] != "user" {
		t.Fatalf("system message should be folded out of contents, got %#v", contents)
	}
	if systemInstruction == nil || !strings.Contains(fmt.Sprintf("%v", systemInstruction), "You are helpful.") {
		t.Fatalf("system message was not preserved in system instruction: %#v", systemInstruction)
	}
}

func TestValidateAnthropicRequestRejectsUnknownNamelessToolInsteadOfDroppingIt(t *testing.T) {
	maxTokens := 32
	req := model.AnthropicMessageRequest{
		Model: "gemini-test", MaxTokens: &maxTokens,
		Messages: []model.AnthropicInputMessage{{Role: "user", Content: "hello"}},
		Tools:    []map[string]interface{}{{"type": "unknown_server_tool"}},
	}
	if got := validateAnthropicRequest(req); !strings.Contains(got, "no Vertex equivalent") {
		t.Fatalf("validation = %q", got)
	}
}

func TestAnthropicStructuredOutputMapsToVertexJSONSchema(t *testing.T) {
	maxTokens := 64
	req := model.AnthropicMessageRequest{
		Model: "gemini-test", MaxTokens: &maxTokens,
		Messages: []model.AnthropicInputMessage{{Role: "user", Content: "extract"}},
		OutputConfig: map[string]interface{}{"format": map[string]interface{}{
			"type": "json_schema", "schema": map[string]interface{}{"type": "object", "properties": map[string]interface{}{"name": map[string]interface{}{"type": "string"}}},
		}},
	}
	if got := validateAnthropicRequest(req); got != "" {
		t.Fatalf("structured output rejected: %s", got)
	}
	config := buildAnthropicGenerationConfig(req)
	if config["responseMimeType"] != "application/json" || config["responseJsonSchema"] == nil {
		t.Fatalf("generation config = %#v", config)
	}
}

func TestAnthropicOutputEffortMapsToVertexThinking(t *testing.T) {
	maxTokens := 64
	req := model.AnthropicMessageRequest{
		Model: "gemini-3.6-flash", MaxTokens: &maxTokens,
		Messages:     []model.AnthropicInputMessage{{Role: "user", Content: "think"}},
		OutputConfig: map[string]interface{}{"effort": "max"},
	}
	if got := validateAnthropicRequest(req); got != "" {
		t.Fatalf("output effort rejected: %s", got)
	}
	thinking := buildAnthropicGenerationConfig(req)["thinkingConfig"].(map[string]interface{})
	if thinking["thinkingLevel"] != "HIGH" || thinking["includeThoughts"] != true {
		t.Fatalf("thinking config = %#v", thinking)
	}
}

func TestAnthropicDocumentsAndRemoteImagesMapToVertexMedia(t *testing.T) {
	content := []interface{}{
		map[string]interface{}{"type": "image", "source": map[string]interface{}{"type": "url", "url": "https://example.com/a.png", "media_type": "image/png"}},
		map[string]interface{}{"type": "document", "source": map[string]interface{}{"type": "base64", "media_type": "application/pdf", "data": "cGRm"}},
		map[string]interface{}{"type": "document", "title": "Notes", "source": map[string]interface{}{"type": "text", "data": "hello"}},
	}
	parts := convertAnthropicContentToParts("gemini-test", content, "user", map[string]string{}, "")
	if len(parts) != 3 || parts[0]["fileData"] == nil || parts[1]["inlineData"] == nil || !strings.Contains(fmt.Sprintf("%v", parts[2]["text"]), "Notes") {
		t.Fatalf("document/image conversion = %#v", parts)
	}
}

func TestBuildAnthropicContentPreservesMixedPartOrder(t *testing.T) {
	content := buildAnthropicContent(&proxy.CallResult{Parts: []model.VertexPart{
		{Text: "think", Thought: true, ThoughtSignature: "sig"},
		{Text: "before"},
		{FunctionCall: &model.FunctionCall{ID: "c1", Name: "lookup", Args: map[string]interface{}{"q": "x"}}},
		{Text: "after"},
	}})
	if len(content) != 4 || content[0]["type"] != "thinking" || content[1]["text"] != "before" || content[2]["type"] != "tool_use" || content[3]["text"] != "after" {
		t.Fatalf("content order = %#v", content)
	}
}
