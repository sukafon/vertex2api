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

func TestResponseInputReplaysHistoryContainingWebSearchCall(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	input := []interface{}{
		map[string]interface{}{
			"type": "function_call", "call_id": "call_plan", "name": "update_plan", "arguments": `{"plan":[]}`,
		},
		map[string]interface{}{
			"type": "web_search_call", "id": "ws_1", "status": "completed",
			"action": map[string]interface{}{"type": "search", "query": "Codex OpenAI features 2026"},
		},
		map[string]interface{}{
			"type": "function_call_output", "call_id": "call_plan", "output": "Plan updated",
		},
		map[string]interface{}{
			"type": "message", "role": "assistant", "phase": "final_answer",
			"content": []interface{}{map[string]interface{}{
				"type": "output_text", "text": "Search-grounded answer",
				"annotations": []interface{}{map[string]interface{}{
					"type": "url_citation", "url": "https://example.com", "title": "Example",
				}},
			}},
		},
		map[string]interface{}{
			"type": "message", "role": "user",
			"content": []interface{}{map[string]interface{}{"type": "input_text", "text": "continue"}},
		},
	}

	messages, err := api.responseInputToMessages(input)
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 4 {
		t.Fatalf("messages length = %d, want 4 after omitting web_search_call metadata: %#v", len(messages), messages)
	}
	if len(messages[0].ToolCalls) != 1 || messages[0].ToolCalls[0].Function.Name != "update_plan" {
		t.Fatalf("tool call = %#v", messages[0])
	}
	if messages[1].Role != "tool" || messages[1].ToolCallID != "call_plan" {
		t.Fatalf("tool output = %#v", messages[1])
	}
	if messages[2].Role != "assistant" || extractTextContent(messages[2].Content) != "Search-grounded answer" {
		t.Fatalf("assistant answer = %#v", messages[2])
	}
	if messages[3].Role != "user" || extractTextContent(messages[3].Content) != "continue" {
		t.Fatalf("follow-up message = %#v", messages[3])
	}

	contents, _ := convertMessages("gemini-3.6-flash", messages)
	if len(contents) != 4 {
		t.Fatalf("Vertex contents length = %d, want 4: %#v", len(contents), contents)
	}
	modelParts := contents[0]["parts"].([]map[string]interface{})
	toolParts := contents[1]["parts"].([]map[string]interface{})
	if got := modelParts[0]["functionCall"].(map[string]interface{})["name"]; got != "update_plan" {
		t.Fatalf("Vertex functionCall name = %v", got)
	}
	if got := toolParts[0]["functionResponse"].(map[string]interface{})["name"]; got != "update_plan" {
		t.Fatalf("Vertex functionResponse name = %v", got)
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
	if kinds["lookup"].Kind != "function" || kinds["raw_patch"].Kind != "custom" || kinds["apply_patch"].Kind != "apply_patch" || kinds["local_shell"].Kind != "local_shell" || kinds["shell"].Kind != "shell" {
		t.Fatalf("tool kinds = %#v", kinds)
	}
	native, _, err := responseToolsToChat([]map[string]interface{}{{"type": "web_search_preview"}, {"type": "code_interpreter"}, {"type": "url_context"}})
	if err != nil || len(native) != 3 {
		t.Fatalf("native tools = %#v, err = %v", native, err)
	}
	vertexTools, _ := convertOpenAITools(native).([]interface{})
	if len(vertexTools) != 3 || vertexTools[0].(map[string]interface{})["googleSearch"] == nil || vertexTools[1].(map[string]interface{})["codeExecution"] == nil || vertexTools[2].(map[string]interface{})["urlContext"] == nil {
		t.Fatalf("Vertex native tools = %#v", vertexTools)
	}
	if _, _, err := responseToolsToChat([]map[string]interface{}{{"type": "file_search"}}); err == nil {
		t.Fatal("non-equivalent hosted tool was accepted")
	}
}

func TestResponseWebSearchExternalAccessMapping(t *testing.T) {
	tools, _, err := responseToolsToChat([]map[string]interface{}{
		{"type": "web_search", "external_web_access": true},
		{"type": "web_search", "external_web_access": true, "indexed_web_access": true},
		{"type": "web_search_preview", "external_web_access": false},
	})
	if err != nil {
		t.Fatal(err)
	}
	vertexTools, _ := convertOpenAITools(tools).([]interface{})
	if len(vertexTools) != 3 {
		t.Fatalf("Vertex tools length = %d, want 3", len(vertexTools))
	}
	for index, tool := range vertexTools {
		if _, ok := tool.(map[string]interface{})["googleSearch"]; !ok {
			t.Fatalf("Vertex tool %d = %#v, want googleSearch", index, tool)
		}
	}
}

func TestResponseWebSearchDropsOptionalCacheOnlyMode(t *testing.T) {
	tools, _, err := responseToolsToChatWithChoice([]map[string]interface{}{
		{"type": "web_search", "external_web_access": false},
	}, "auto")
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 0 {
		t.Fatalf("tools = %#v, want cache-only search removed", tools)
	}
}

func TestResponseWebSearchRejectsRequiredCacheOnlyModeInChinese(t *testing.T) {
	_, _, err := responseToolsToChatWithChoice([]map[string]interface{}{
		{"type": "web_search", "external_web_access": false},
	}, "required")
	if err == nil || !strings.Contains(err.Error(), "移除网页搜索工具") || !strings.Contains(err.Error(), "配置 → 网页搜索") || !strings.Contains(err.Error(), "已索引") || !strings.Contains(err.Error(), "实时") || !strings.Contains(err.Error(), "已禁用") {
		t.Fatalf("error = %v, want actionable Chinese cache-only incompatibility", err)
	}

	_, _, err = responseToolsToChatWithChoice([]map[string]interface{}{
		{"type": "web_search", "external_web_access": false},
	}, map[string]interface{}{"type": "web_search"})
	if err == nil || !strings.Contains(err.Error(), "已索引") || !strings.Contains(err.Error(), "实时") || !strings.Contains(err.Error(), "已禁用") {
		t.Fatalf("error = %v, want required web search incompatibility", err)
	}
}

func TestResponseWebSearchValidatesIndexedAccess(t *testing.T) {
	_, _, err := responseToolsToChat([]map[string]interface{}{
		{"type": "web_search", "external_web_access": true, "indexed_web_access": "yes"},
	})
	if err == nil || !strings.Contains(err.Error(), "indexed_web_access must be a boolean") {
		t.Fatalf("error = %v, want indexed_web_access type error", err)
	}

	_, _, err = responseToolsToChat([]map[string]interface{}{
		{"type": "web_search", "external_web_access": false, "indexed_web_access": true},
	})
	if err == nil || !strings.Contains(err.Error(), "requires external_web_access=true") {
		t.Fatalf("error = %v, want contradictory indexed access error", err)
	}
}

func TestResponseToolsExpandNamespaceFunctions(t *testing.T) {
	tools, kinds, err := responseToolsToChat([]map[string]interface{}{
		{
			"type":        "namespace",
			"name":        "codex_app",
			"description": "Tools provided by the Codex app.",
			"tools": []interface{}{
				map[string]interface{}{
					"type":         "function",
					"name":         "list_threads",
					"description":  "List Codex tasks.",
					"inputSchema":  map[string]interface{}{"type": "object", "properties": map[string]interface{}{"limit": map[string]interface{}{"type": "integer"}}},
					"deferLoading": true,
				},
			},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(tools) != 1 {
		t.Fatalf("tools length = %d, want 1", len(tools))
	}
	if got := kinds["codex_app__list_threads"]; got.Kind != "function" || got.Namespace != "codex_app" || got.Name != "list_threads" {
		t.Fatalf("tool binding = %#v, want reversible codex_app namespace binding", got)
	}
	function := tools[0]["function"].(map[string]interface{})
	if got := function["name"]; got != "codex_app__list_threads" {
		t.Fatalf("function name = %v", got)
	}
	parameters := function["parameters"].(map[string]interface{})
	if _, ok := parameters["properties"].(map[string]interface{})["limit"]; !ok {
		t.Fatalf("namespace inputSchema was not preserved: %#v", parameters)
	}
	if _, ok := function["deferLoading"]; ok {
		t.Fatal("namespace loading metadata must not be sent to Vertex")
	}

	vertexTools, _ := convertOpenAITools(tools).([]interface{})
	declarations := vertexTools[0].(map[string]interface{})["functionDeclarations"].([]interface{})
	if got := declarations[0].(map[string]interface{})["name"]; got != "codex_app__list_threads" {
		t.Fatalf("Vertex function name = %v", got)
	}
}

func TestResponseNamespaceFunctionCallRoundTripsThroughCodex(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	_, bindings, err := responseToolsToChat([]map[string]interface{}{
		{
			"type": "namespace", "name": "codex_app",
			"tools": []interface{}{map[string]interface{}{
				"type": "function", "name": "load_workspace_dependencies",
				"inputSchema": map[string]interface{}{"type": "object"},
			}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}

	output := buildResponseOutput(&proxy.CallResult{FunctionCalls: []model.FunctionCall{{
		ID: "call_1", Name: "codex_app__load_workspace_dependencies", Args: map[string]interface{}{},
	}}}, bindings)
	if len(output) != 1 {
		t.Fatalf("output = %#v", output)
	}
	call := output[0]
	if call["type"] != "function_call" || call["namespace"] != "codex_app" || call["name"] != "load_workspace_dependencies" {
		t.Fatalf("Codex namespace call = %#v", call)
	}

	messages, err := api.responseInputToMessages([]interface{}{
		call,
		map[string]interface{}{"type": "function_call_output", "call_id": call["call_id"], "output": "ok"},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 2 || len(messages[0].ToolCalls) != 1 {
		t.Fatalf("round-trip messages = %#v", messages)
	}
	if got := messages[0].ToolCalls[0].Function.Name; got != "codex_app__load_workspace_dependencies" {
		t.Fatalf("restored Vertex tool name = %q", got)
	}
	contents, _ := convertMessages("gemini-3-pro", messages)
	modelParts := contents[0]["parts"].([]map[string]interface{})
	userParts := contents[1]["parts"].([]map[string]interface{})
	if got := modelParts[0]["functionCall"].(map[string]interface{})["name"]; got != "codex_app__load_workspace_dependencies" {
		t.Fatalf("Vertex functionCall name = %v", got)
	}
	if got := userParts[0]["functionResponse"].(map[string]interface{})["name"]; got != "codex_app__load_workspace_dependencies" {
		t.Fatalf("Vertex functionResponse name = %v", got)
	}
}

func TestResponseCommentaryInputRestoresVertexThought(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	messages, err := api.responseInputToMessages([]interface{}{map[string]interface{}{
		"type": "message", "role": "assistant", "phase": "commentary",
		"content": []interface{}{map[string]interface{}{"type": "output_text", "text": "checking tools"}},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 1 || messages[0].ReasoningContent != "checking tools" || messages[0].Content != nil {
		t.Fatalf("commentary message = %#v, want Vertex thought", messages)
	}
}

func TestResponseNamespaceToolChoiceMapsToVertexName(t *testing.T) {
	choice := responseToolChoiceToChat(map[string]interface{}{
		"type": "function", "namespace": "mcp__node_repl", "name": "js",
	})
	function := choice.(map[string]interface{})["function"].(map[string]interface{})
	if got := function["name"]; got != "mcp__node_repl__js" {
		t.Fatalf("Vertex namespace tool choice = %v", got)
	}
}

func TestResponseToolsRejectInvalidNamespaceFunctions(t *testing.T) {
	_, _, err := responseToolsToChat([]map[string]interface{}{
		{"type": "namespace", "name": "codex_app", "tools": []interface{}{map[string]interface{}{"type": "custom", "name": "bad"}}},
	})
	if err == nil {
		t.Fatal("unsupported namespace member was accepted")
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

func TestResponsesReasoningEffortMaxDegradesToHigh(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	req := model.ResponseRequest{
		Model:     "gemini-3.6-flash",
		Input:     "hello",
		Reasoning: map[string]interface{}{"effort": "max"},
	}
	if message := api.validateResponseRequest(req); message != "" {
		t.Fatalf("request rejected: %s", message)
	}
	chatReq, _, err := api.responseRequestToChat(req)
	if err != nil {
		t.Fatal(err)
	}
	config := buildResponsesGenerationConfig(req, chatReq)
	thinkingConfig := config["thinkingConfig"].(map[string]interface{})
	if got := thinkingConfig["thinkingLevel"]; got != "HIGH" {
		t.Fatalf("thinkingLevel = %v, want HIGH", got)
	}
}

func TestResponsesReasoningEffortRejectsUnsupportedValues(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	tests := []interface{}{"none", "ultra", 3}
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

func TestBuildResponseOutputPreservesTextThoughtAndToolKinds(t *testing.T) {
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
	output := buildResponseOutput(result, responseToolBindings{
		"lookup": {Kind: "function"}, "apply_patch": {Kind: "custom"}, "local_shell": {Kind: "local_shell"},
	})
	if len(output) != 5 {
		t.Fatalf("output length = %d, want 5: %#v", len(output), output)
	}
	wantTypes := []string{"message", "message", "function_call", "custom_tool_call", "local_shell_call"}
	for i, want := range wantTypes {
		if got := output[i]["type"]; got != want {
			t.Fatalf("output[%d].type = %v, want %q", i, got, want)
		}
	}
	if got := output[0]["phase"]; got != "commentary" {
		t.Fatalf("thought phase = %v, want commentary", got)
	}
	if got := output[1]["phase"]; got != "final_answer" {
		t.Fatalf("answer phase = %v, want final_answer", got)
	}
	thoughtContent := output[0]["content"].([]map[string]interface{})
	if len(thoughtContent) != 1 || thoughtContent[0]["type"] != "output_text" || thoughtContent[0]["text"] != "private thought" {
		t.Fatalf("commentary content = %#v", thoughtContent)
	}
}

func TestResponsesStreamEmitsLifecycleTextAndFunctionEvents(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	req := model.ResponseRequest{Model: "gemini-3-pro", Input: "hello", Stream: true}
	chatReq := model.ChatCompletionRequest{Model: req.Model, Messages: []model.ChatMessage{{Role: "user", Content: "hello"}}, Stream: true}
	recorder := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/responses", nil)

	api.streamResponse(recorder, httpReq, req, chatReq, nil, responseToolBindings{"lookup": {Kind: "function"}}, func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
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
	functionLifecycle := []struct {
		name     string
		position int
	}{
		{"output_item.added", strings.LastIndex(body, "event: response.output_item.added\n")},
		{"arguments.delta", strings.Index(body, "event: response.function_call_arguments.delta\n")},
		{"arguments.done", strings.Index(body, "event: response.function_call_arguments.done\n")},
		{"output_item.done", strings.LastIndex(body, "event: response.output_item.done\n")},
		{"response.completed", strings.Index(body, "event: response.completed\n")},
	}
	previous := -1
	for _, stage := range functionLifecycle {
		if stage.position < 0 || stage.position <= previous {
			t.Fatalf("function-call lifecycle is incomplete or out of order at %s: %#v\n%s", stage.name, functionLifecycle, body)
		}
		previous = stage.position
	}
}

func TestResponsesStreamEmitsThoughtAsCommentaryTextDeltas(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	req := model.ResponseRequest{Model: "gemini-3-pro", Input: "hello", Stream: true}
	chatReq := model.ChatCompletionRequest{Model: req.Model, Messages: []model.ChatMessage{{Role: "user", Content: "hello"}}, Stream: true}
	recorder := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/responses", nil)

	api.streamResponse(recorder, httpReq, req, chatReq, nil, nil, func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
		if err := onChunk(&proxy.CallResult{TextParts: []model.TextPart{{Text: "checking", Thought: true}}}); err != nil {
			return err
		}
		if err := onChunk(&proxy.CallResult{TextParts: []model.TextPart{{Text: " tools", Thought: true}}}); err != nil {
			return err
		}
		return onChunk(&proxy.CallResult{TextParts: []model.TextPart{{Text: "done"}}, FinishReason: "STOP"})
	})

	body := recorder.Body.String()
	for _, eventType := range []string{
		"response.content_part.added",
		"response.output_text.delta",
		"response.output_text.done",
		"response.content_part.done",
		"response.completed",
	} {
		if !strings.Contains(body, "event: "+eventType+"\n") {
			t.Fatalf("missing %s in stream:\n%s", eventType, body)
		}
	}
	if !strings.Contains(body, `"delta":"checking"`) || !strings.Contains(body, `"delta":" tools"`) {
		t.Fatalf("commentary text deltas missing from stream: %s", body)
	}
	if !strings.Contains(body, `"text":"checking tools"`) {
		t.Fatalf("completed commentary text missing from stream: %s", body)
	}
	if !strings.Contains(body, `"phase":"commentary"`) || strings.Contains(body, `response.reasoning_text`) {
		t.Fatalf("thought was not emitted exclusively as commentary: %s", body)
	}
	commentaryDelta := strings.Index(body, `"delta":"checking"`)
	finalDelta := strings.Index(body, `"delta":"done"`)
	if commentaryDelta < 0 || finalDelta < 0 || commentaryDelta > finalDelta {
		t.Fatalf("commentary text was not streamed before final output:\n%s", body)
	}
}

func TestResponsesStreamFinishesCommentaryBeforeFunctionCall(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	req := model.ResponseRequest{Model: "gemini-3-pro", Input: "inspect", Stream: true}
	chatReq := model.ChatCompletionRequest{Model: req.Model, Messages: []model.ChatMessage{{Role: "user", Content: "inspect"}}, Stream: true}
	recorder := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/responses", nil)

	api.streamResponse(recorder, httpReq, req, chatReq, nil, responseToolBindings{"lookup": {Kind: "function"}}, func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
		return onChunk(&proxy.CallResult{
			TextParts:     []model.TextPart{{Text: "I should inspect it.", Thought: true}},
			FunctionCalls: []model.FunctionCall{{ID: "call_1", Name: "lookup", Args: map[string]interface{}{"q": "state"}}},
			FinishReason:  "STOP",
		})
	})

	body := recorder.Body.String()
	commentaryDone := strings.Index(body, "event: response.output_text.done\n")
	functionDelta := strings.Index(body, "event: response.function_call_arguments.delta\n")
	if commentaryDone < 0 || functionDelta < 0 || commentaryDone > functionDelta {
		t.Fatalf("commentary text did not finish before the function call:\n%s", body)
	}
}

func TestResponsesStreamReportsFinishOnlyResultAsFailure(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	req := model.ResponseRequest{Model: "gemini-3-pro", Input: "hello", Stream: true}
	chatReq := model.ChatCompletionRequest{Model: req.Model, Messages: []model.ChatMessage{{Role: "user", Content: "hello"}}, Stream: true}
	recorder := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/responses", nil)

	api.streamResponse(recorder, httpReq, req, chatReq, nil, nil, func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
		return onChunk(&proxy.CallResult{FinishReason: "STOP"})
	})

	body := recorder.Body.String()
	if !strings.Contains(body, "event: response.failed\n") {
		t.Fatalf("missing response.failed event:\n%s", body)
	}
	if strings.Contains(body, "event: response.completed\n") {
		t.Fatalf("finish-only result was reported as completed:\n%s", body)
	}
	if !strings.Contains(body, `"status":"failed"`) || !strings.Contains(body, proxy.ErrNoAssistantOutput.Error()) {
		t.Fatalf("failure details missing from stream: %s", body)
	}
}

func TestResponsesStreamEmitsErrorWhenEmptyOutputRetriesAreExhausted(t *testing.T) {
	api := NewResponsesAPI(nil, true, "test-secret")
	req := model.ResponseRequest{Model: "gemini-3-pro", Input: "hello", Stream: true}
	chatReq := model.ChatCompletionRequest{Model: req.Model, Messages: []model.ChatMessage{{Role: "user", Content: "hello"}}, Stream: true}
	recorder := httptest.NewRecorder()
	httpReq := httptest.NewRequest("POST", "/v1/responses", nil)

	api.streamResponse(recorder, httpReq, req, chatReq, nil, nil, func(_ context.Context, _ func(*proxy.CallResult) error) error {
		return proxy.ErrNoAssistantOutput
	})

	body := recorder.Body.String()
	if !strings.Contains(body, "event: error\n") {
		t.Fatalf("missing error event:\n%s", body)
	}
	if strings.Contains(body, "event: response.completed\n") {
		t.Fatalf("exhausted empty response was reported as completed:\n%s", body)
	}
	if !strings.Contains(body, proxy.ErrNoAssistantOutput.Error()) {
		t.Fatalf("error details missing from stream: %s", body)
	}
}

func TestBuildResponseOutputPreservesGroundingAsURLCitation(t *testing.T) {
	result := &proxy.CallResult{
		TextParts: []model.TextPart{{Text: "answer"}},
		GroundingMetadata: &model.GroundingMetadata{
			GroundingChunks:   []model.GroundingChunk{{RetrievedContext: map[string]interface{}{"uri": "https://example.com/doc", "title": "Doc"}}},
			GroundingSupports: []model.GroundingSupport{{GroundingChunkIndices: []int{0}, Segment: &model.Segment{StartIndex: 0, EndIndex: 6, Text: "answer"}}},
		},
	}
	output := buildResponseOutput(result, nil)
	content := output[0]["content"].([]map[string]interface{})
	annotations := content[0]["annotations"].([]map[string]interface{})
	if len(annotations) != 1 || annotations[0]["url"] != "https://example.com/doc" {
		t.Fatalf("annotations = %#v", annotations)
	}
}

func TestBuildResponseOutputDoesNotInferRefusalFromVertexMetadata(t *testing.T) {
	result := &proxy.CallResult{PromptFeedback: map[string]interface{}{
		"blockReason": "SAFETY", "blockReasonMessage": "blocked by safety policy",
	}}
	output := buildResponseOutput(result, nil)
	if len(output) != 0 {
		t.Fatalf("metadata-only result must not be synthesized into a refusal: %#v", output)
	}
}

func TestBuildResponseOutputIncludesStandardWebSearchCall(t *testing.T) {
	result := &proxy.CallResult{
		TextParts: []model.TextPart{{Text: "answer"}},
		GroundingMetadata: &model.GroundingMetadata{
			WebSearchQueries: []string{"vertex ai", "gemini api"},
		},
	}
	output := buildResponseOutput(result, nil)
	if len(output) != 3 || output[0]["type"] != "web_search_call" || output[1]["type"] != "web_search_call" {
		t.Fatalf("output = %#v", output)
	}
	action := output[0]["action"].(map[string]interface{})
	if action["query"] != "vertex ai" {
		t.Fatalf("search action = %#v", action)
	}
	if _, exists := action["queries"]; exists {
		t.Fatalf("non-standard queries field leaked: %#v", action)
	}
}

func TestResponseInputFileMapsToVertexCompatibleChatFile(t *testing.T) {
	content, err := responseContentToChat([]interface{}{
		map[string]interface{}{"type": "input_file", "file_data": "data:application/pdf;base64,cGRm", "filename": "a.pdf"},
	})
	if err != nil {
		t.Fatal(err)
	}
	parts := convertContentToParts(content)
	if len(parts) != 1 || parts[0]["inlineData"] == nil {
		t.Fatalf("parts = %#v", parts)
	}
}
