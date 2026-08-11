package handler

import (
	"context"
	"net/http/httptest"
	"strings"
	"testing"

	"vertex2api/model"
	"vertex2api/proxy"
)

func TestResponseCompactCodecRoundTripAndAuthentication(t *testing.T) {
	codec := newResponseCompactCodec("test-secret")
	want := compactPayload{Version: 1, Model: "gemini-3-pro", Summary: "preserve this state"}
	sealed, err := codec.seal(want)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.HasPrefix(sealed, compactContentPrefix) {
		t.Fatalf("sealed value = %q", sealed)
	}
	got, err := codec.open(sealed)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("payload = %+v, want %+v", got, want)
	}

	tampered := sealed[:len(sealed)-1] + "A"
	if _, err := codec.open(tampered); err == nil {
		t.Fatal("tampered compact payload was accepted")
	}
	if _, err := newResponseCompactCodec("another-secret").open(sealed); err == nil {
		t.Fatal("compact payload encrypted with another secret was accepted")
	}
}

func TestResponseInputRestoresCompactionAndToolHistory(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	compactItem, err := api.newCompactionItem("gemini-3-pro", "goal, decisions, and current code state")
	if err != nil {
		t.Fatal(err)
	}
	input := []interface{}{
		compactItem,
		map[string]interface{}{
			"type": "function_call", "call_id": "call_1", "name": "lookup", "arguments": `{"q":"koi"}`,
		},
		map[string]interface{}{
			"type": "function_call_output", "call_id": "call_1", "output": `{"result":"found"}`,
		},
		map[string]interface{}{
			"type": "message", "role": "user", "content": []interface{}{
				map[string]interface{}{"type": "input_text", "text": "continue"},
				map[string]interface{}{"type": "input_image", "image_url": "data:image/png;base64,AA=="},
			},
		},
	}
	messages, err := api.responseInputToMessages(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 4 {
		t.Fatalf("messages length = %d, want 4: %+v", len(messages), messages)
	}
	if messages[0].Role != "user" || !strings.Contains(messages[0].Content.(string), "goal, decisions") {
		t.Fatalf("compaction message = %+v", messages[0])
	}
	if got := messages[1].ToolCalls[0].Function.Name; got != "lookup" {
		t.Fatalf("tool call name = %q", got)
	}
	if messages[2].Role != "tool" || messages[2].ToolCallID != "call_1" {
		t.Fatalf("tool output = %+v", messages[2])
	}
	parts, ok := messages[3].Content.([]interface{})
	if !ok || len(parts) != 2 {
		t.Fatalf("multimodal content = %#v", messages[3].Content)
	}
}

func TestResponseInputRejectsForeignCompaction(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	_, err := api.responseInputToMessages([]interface{}{
		map[string]interface{}{"type": "compaction", "encrypted_content": "foreign-token"},
	})
	if err == nil || !strings.Contains(err.Error(), "not created") {
		t.Fatalf("error = %v", err)
	}
}

func TestCompactionPreservesInstructionHierarchy(t *testing.T) {
	messages := []model.ChatMessage{
		{Role: "developer", Content: "trusted policy"},
		{Role: "user", Content: "untrusted request"},
		{Role: "assistant", Content: "work so far"},
		{Role: "user", Content: "latest request"},
	}
	instructions := retainResponseInstructions(messages)
	if len(instructions) != 1 || instructions[0].Role != "developer" {
		t.Fatalf("instructions = %+v", instructions)
	}
	latest := retainLatestResponseTurn(messages)
	if len(latest) != 1 || latest[0].Role != "user" || latest[0].Content != "latest request" {
		t.Fatalf("latest turn = %+v", latest)
	}
}

func TestResponseToolsConvertFunctionCustomAndLocalShell(t *testing.T) {
	tools, kinds, err := responseToolsToChat([]map[string]interface{}{
		{"type": "function", "name": "lookup", "description": "find", "parameters": map[string]interface{}{"type": "object"}},
		{"type": "custom", "name": "raw_patch", "description": "patch files"},
		{"type": "local_shell"},
		{"type": "shell"},
		{"type": "apply_patch"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 5 {
		t.Fatalf("tools length = %d", len(tools))
	}
	if kinds["lookup"] != "function" || kinds["raw_patch"] != "custom" || kinds["apply_patch"] != "apply_patch" || kinds["local_shell"] != "local_shell" || kinds["shell"] != "shell" {
		t.Fatalf("tool kinds = %#v", kinds)
	}
	if _, _, err := responseToolsToChat([]map[string]interface{}{{"type": "web_search_preview"}}); err == nil {
		t.Fatal("unsupported built-in tool was accepted")
	}
}

func TestResponsesReasoningEffortUsesModelMapping(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	req := model.ResponseRequest{
		Model:     "gemini-3.6-flash",
		Input:     "hello",
		Reasoning: map[string]interface{}{"effort": "minimal"},
	}
	if message := api.validateResponseRequest(req); message != "" {
		t.Fatalf("request rejected: %s", message)
	}
	chatReq, _, err := api.responseRequestToChat(req)
	if err != nil {
		t.Fatal(err)
	}
	if chatReq.ReasoningEffort == nil || *chatReq.ReasoningEffort != "minimal" {
		t.Fatalf("chat reasoning effort = %v", chatReq.ReasoningEffort)
	}
	config := buildResponsesGenerationConfig(req, chatReq)
	thinkingConfig, ok := config["thinkingConfig"].(map[string]interface{})
	if !ok {
		t.Fatalf("thinking config = %#v", config["thinkingConfig"])
	}
	if got := thinkingConfig["thinkingLevel"]; got != "MINIMAL" {
		t.Fatalf("thinkingLevel = %v, want MINIMAL", got)
	}
	if got := thinkingConfig["includeThoughts"]; got != true {
		t.Fatalf("includeThoughts = %v, want true", got)
	}
}

func TestResponsesReasoningEffortRejectsUnsupportedValues(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	tests := []interface{}{"none", "max", 3}
	for _, effort := range tests {
		req := model.ResponseRequest{
			Model:     "gemini-3.6-flash",
			Input:     "hello",
			Reasoning: map[string]interface{}{"effort": effort},
		}
		if message := api.validateResponseRequest(req); message == "" {
			t.Fatalf("effort=%v should be rejected", effort)
		}
	}
}

func TestResponsesRejectsStatefulStorage(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	store := true
	req := model.ResponseRequest{
		Model: "gemini-3.6-flash",
		Input: "hello",
		Store: &store,
	}
	if message := api.validateResponseRequest(req); !strings.Contains(message, "store=true is not supported") {
		t.Fatalf("validation message = %q", message)
	}
}

func TestBuildResponseOutputPreservesTextReasoningAndToolKinds(t *testing.T) {
	result := &proxy.CallResult{
		TextParts: []model.TextPart{
			{Text: "private thought", Thought: true},
			{Text: "answer"},
		},
		FunctionCalls: []model.FunctionCall{
			{ID: "call_1", Name: "lookup", Args: map[string]interface{}{"q": "x"}},
			{ID: "call_2", Name: "apply_patch", Args: map[string]interface{}{"input": "*** patch"}},
			{ID: "call_3", Name: "local_shell", Args: map[string]interface{}{"command": []interface{}{"go test ./..."}}},
		},
	}
	output := buildResponseOutput(result, map[string]string{
		"lookup": "function", "apply_patch": "custom", "local_shell": "local_shell",
	})
	if len(output) != 5 {
		t.Fatalf("output length = %d, want 5: %#v", len(output), output)
	}
	wantTypes := []string{"reasoning", "message", "function_call", "custom_tool_call", "local_shell_call"}
	for i, want := range wantTypes {
		if got := output[i]["type"]; got != want {
			t.Fatalf("output[%d].type = %v, want %q", i, got, want)
		}
	}
}

func TestResponsesStreamEmitsLifecycleTextAndFunctionEvents(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	req := model.ResponseRequest{Model: "gemini-3-pro", Input: "hello", Stream: true}
	chatReq := model.ChatCompletionRequest{Model: req.Model, Messages: []model.ChatMessage{{Role: "user", Content: "hello"}}, Stream: true}
	recorder := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/responses", nil)

	api.streamResponse(recorder, httpReq, req, chatReq, nil, map[string]string{"lookup": "function"}, func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
		if err := onChunk(&proxy.CallResult{TextParts: []model.TextPart{{Text: "hel"}}}); err != nil {
			return err
		}
		return onChunk(&proxy.CallResult{
			TextParts:     []model.TextPart{{Text: "lo"}},
			FunctionCalls: []model.FunctionCall{{ID: "call_1", Name: "lookup", Args: map[string]interface{}{"q": "x"}}},
			FinishReason:  "STOP",
			UsageMetadata: map[string]interface{}{"promptTokenCount": float64(3), "candidatesTokenCount": float64(2), "totalTokenCount": float64(5)},
		})
	})

	if got := recorder.Header().Get("Content-Type"); got != "text/event-stream" {
		t.Fatalf("content type = %q", got)
	}
	body := recorder.Body.String()
	for _, eventType := range []string{
		"response.created", "response.in_progress", "response.output_item.added",
		"response.output_text.delta", "response.output_text.done",
		"response.function_call_arguments.done", "response.completed",
	} {
		if !strings.Contains(body, "event: "+eventType+"\n") {
			t.Fatalf("missing %s in stream:\n%s", eventType, body)
		}
	}
	if !strings.Contains(body, `"text":"hello"`) {
		t.Fatalf("completed text missing from stream: %s", body)
	}
}
