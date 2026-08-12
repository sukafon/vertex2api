package handler

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vertex2api/model"
	"vertex2api/proxy"
)

func TestGeminiMalformedJSONUsesGeminiErrorEnvelope(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-test:generateContent", strings.NewReader("{"))
	req.SetPathValue("modelAction", "gemini-test:generateContent")
	rec := httptest.NewRecorder()
	GeminiGenerate(nil, true).ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), `"status":"INVALID_ARGUMENT"`) {
		t.Fatalf("response = status %d body %s", rec.Code, rec.Body.String())
	}
}

func TestGeminiCountTokensReturnsEstimatedOfficialShape(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-test:countTokens", strings.NewReader(`{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}`))
	req.SetPathValue("modelAction", "gemini-test:countTokens")
	rec := httptest.NewRecorder()
	GeminiGenerate(nil, true).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK || !strings.Contains(rec.Body.String(), `"totalTokens":`) {
		t.Fatalf("response = status %d body %s", rec.Code, rec.Body.String())
	}
	if rec.Header().Get("X-Usage-Estimated") != "true" {
		t.Fatal("estimated count should be identified by X-Usage-Estimated")
	}
}

func TestBuildGeminiResponseMarksThoughtParts(t *testing.T) {
	resp := buildGeminiResponse(&proxy.CallResult{
		TextParts: []model.TextPart{
			{Text: "thinking", Thought: true},
			{Text: "answer"},
		},
	})

	candidates := resp["candidates"].([]map[string]interface{})
	content := candidates[0]["content"].(map[string]interface{})
	parts := content["parts"].([]map[string]interface{})

	if got := parts[0]["thought"]; got != true {
		t.Fatalf("first part thought = %v, want true", got)
	}
	if _, ok := parts[1]["thought"]; ok {
		t.Fatal("second part should not include thought")
	}
}

func TestBuildGeminiResponsesDropBareEmptyThoughtAndPreserveDetachedSignature(t *testing.T) {
	result := &proxy.CallResult{Candidates: []proxy.CandidateResult{{
		Index: 0,
		Parts: []model.VertexPart{
			{Thought: true},
			{Thought: true, ThoughtSignature: "c2ln"},
		},
	}}}

	for name, build := range map[string]func(*proxy.CallResult) map[string]interface{}{
		"non-stream": buildGeminiResponse,
		"stream":     buildGeminiStreamResponse,
	} {
		t.Run(name, func(t *testing.T) {
			candidates := build(result)["candidates"].([]map[string]interface{})
			if candidates[0]["index"] != 0 {
				t.Fatalf("thought candidate index = %#v", candidates[0]["index"])
			}
			parts := candidates[0]["content"].(map[string]interface{})["parts"].([]map[string]interface{})
			if len(parts) != 1 || parts[0]["text"] != "" || parts[0]["thought"] != true || parts[0]["thoughtSignature"] != "c2ln" {
				t.Fatalf("thought parts were not normalized like v1.0.3: %#v", parts)
			}
		})
	}
}

func TestBuildGeminiResponseCoalescesNonStreamTextParts(t *testing.T) {
	resp := buildGeminiResponse(&proxy.CallResult{
		TextParts: []model.TextPart{
			{Text: "hello "},
			{Text: "world"},
		},
	})

	candidates := resp["candidates"].([]map[string]interface{})
	content := candidates[0]["content"].(map[string]interface{})
	parts := content["parts"].([]map[string]interface{})

	if len(parts) != 1 {
		t.Fatalf("parts length = %d, want 1", len(parts))
	}
	if got := parts[0]["text"]; got != "hello world" {
		t.Fatalf("text = %v, want hello world", got)
	}
}

func TestBuildGeminiResponseOmitsUsageMetadataWhenUnknown(t *testing.T) {
	resp := buildGeminiResponse(&proxy.CallResult{
		TextParts: []model.TextPart{{Text: "done"}},
	})

	if got, ok := resp["usageMetadata"]; ok {
		t.Fatalf("response should omit unknown usageMetadata, got %v", got)
	}
}

func TestBuildGeminiStreamResponseOmitsUnspecifiedFinishReason(t *testing.T) {
	resp := buildGeminiStreamResponse(&proxy.CallResult{
		TextParts:    []model.TextPart{{Text: "hello"}},
		FinishReason: "FINISH_REASON_UNSPECIFIED",
	})

	candidates := resp["candidates"].([]map[string]interface{})
	if _, ok := candidates[0]["finishReason"]; ok {
		t.Fatal("stream chunk should not include FINISH_REASON_UNSPECIFIED")
	}
}

func TestBuildGeminiResponsesIncludeIndexOnFinishOnlyCandidate(t *testing.T) {
	result := &proxy.CallResult{Candidates: []proxy.CandidateResult{{
		Index:        0,
		FinishReason: "STOP",
	}}}

	for name, build := range map[string]func(*proxy.CallResult) map[string]interface{}{
		"non-stream": buildGeminiResponse,
		"stream":     buildGeminiStreamResponse,
	} {
		t.Run(name, func(t *testing.T) {
			candidates := build(result)["candidates"].([]map[string]interface{})
			if got := candidates[0]["finishReason"]; got != "STOP" {
				t.Fatalf("finishReason = %v, want STOP", got)
			}
			if got := candidates[0]["index"]; got != 0 {
				t.Fatalf("index = %v, want 0", got)
			}
		})
	}
}

func TestBuildGeminiResponsesFallbackDoesNotInventCandidateIndex(t *testing.T) {
	result := &proxy.CallResult{
		TextParts:    []model.TextPart{{Text: "done"}},
		FinishReason: "STOP",
	}

	for name, build := range map[string]func(*proxy.CallResult) map[string]interface{}{
		"non-stream": buildGeminiResponse,
		"stream":     buildGeminiStreamResponse,
	} {
		t.Run(name, func(t *testing.T) {
			candidates := build(result)["candidates"].([]map[string]interface{})
			if _, ok := candidates[0]["index"]; ok {
				t.Fatalf("fallback candidate invented an upstream index: %#v", candidates[0])
			}
		})
	}
}

func TestBuildGeminiResponsesUseUpstreamCandidateIndex(t *testing.T) {
	result := &proxy.CallResult{Candidates: []proxy.CandidateResult{{
		Index: 2,
		Parts: []model.VertexPart{{Text: "candidate"}},
	}}}
	for name, build := range map[string]func(*proxy.CallResult) map[string]interface{}{
		"non-stream": buildGeminiResponse,
		"stream":     buildGeminiStreamResponse,
	} {
		t.Run(name, func(t *testing.T) {
			candidates := build(result)["candidates"].([]map[string]interface{})
			if got := candidates[0]["index"]; got != 2 {
				t.Fatalf("index = %v, want 2", got)
			}
		})
	}
}

func TestBuildGeminiResponsesDropIndexOnlyEmptyCandidate(t *testing.T) {
	result := &proxy.CallResult{
		PromptFeedback: map[string]interface{}{"blockReason": "BLOCKED_REASON_UNSPECIFIED"},
		Candidates: []proxy.CandidateResult{{
			Index: 0,
			Parts: []model.VertexPart{{
				InlineData:          &model.InlineData{},
				FileData:            map[string]interface{}{"mimeType": "", "fileUri": ""},
				FunctionCall:        &model.FunctionCall{},
				FunctionResponse:    &model.FunctionResponse{},
				ExecutableCode:      map[string]interface{}{"language": "", "code": ""},
				CodeExecutionResult: map[string]interface{}{"outcome": "", "output": ""},
				VideoMetadata:       map[string]interface{}{"startOffset": "", "endOffset": ""},
			}},
		}},
	}

	for name, build := range map[string]func(*proxy.CallResult) map[string]interface{}{
		"non-stream": buildGeminiResponse,
		"stream":     buildGeminiStreamResponse,
	} {
		t.Run(name, func(t *testing.T) {
			response := build(result)
			if candidates, ok := response["candidates"]; ok {
				t.Fatalf("index-only candidate leaked into response: %#v", candidates)
			}
			if promptFeedback, ok := response["promptFeedback"]; ok {
				t.Fatalf("unspecified promptFeedback leaked into response: %#v", promptFeedback)
			}
		})
	}
}

func TestStreamGeminiResponseSkipsEmptyCandidateShellAndKeepsFinishIndex(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-test:streamGenerateContent", nil)
	rec := httptest.NewRecorder()

	streamGeminiResponse(rec, req, "gemini-test", func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
		if err := onChunk(&proxy.CallResult{
			PromptFeedback: map[string]interface{}{"blockReason": "BLOCKED_REASON_UNSPECIFIED"},
			Candidates: []proxy.CandidateResult{{
				Index: 0,
				Parts: []model.VertexPart{{InlineData: &model.InlineData{}}, {Thought: true}},
			}},
		}); err != nil {
			return err
		}
		return onChunk(&proxy.CallResult{Candidates: []proxy.CandidateResult{{
			Index:        0,
			FinishReason: "STOP",
		}}})
	})

	body := rec.Body.String()
	if chunks := strings.Count(body, "data: "); chunks != 1 {
		t.Fatalf("SSE chunk count = %d, want 1; body=%q", chunks, body)
	}
	if !strings.Contains(body, `"index":0`) || !strings.Contains(body, `"finishReason":"STOP"`) {
		t.Fatalf("finish chunk lost index or finishReason: %q", body)
	}
}

func TestBuildGeminiResponseDropsIncompleteDecodedParts(t *testing.T) {
	resp := buildGeminiStreamResponse(&proxy.CallResult{
		TextParts:     []model.TextPart{{}},
		ImageParts:    []model.InlineData{{MimeType: "image/png"}, {Data: "aW1hZ2U="}},
		FunctionCalls: []model.FunctionCall{{Name: "  "}},
	})
	if candidates, ok := resp["candidates"]; ok {
		t.Fatalf("incomplete decoded fields became Gemini candidates: %#v", candidates)
	}
}

func TestBuildGeminiResponseNormalizesDefaultFilledUnionAroundValidText(t *testing.T) {
	response := buildGeminiStreamResponse(&proxy.CallResult{Parts: []model.VertexPart{{
		Text:                "answer",
		InlineData:          &model.InlineData{},
		FileData:            map[string]interface{}{"mimeType": "", "fileUri": ""},
		FunctionCall:        &model.FunctionCall{},
		FunctionResponse:    &model.FunctionResponse{},
		ExecutableCode:      map[string]interface{}{"language": "", "code": ""},
		CodeExecutionResult: map[string]interface{}{"outcome": "", "output": ""},
		VideoMetadata:       map[string]interface{}{"startOffset": "", "endOffset": ""},
		MediaResolution:     map[string]interface{}{"level": "MEDIA_RESOLUTION_UNSPECIFIED"},
	}}})
	candidates := response["candidates"].([]map[string]interface{})
	parts := candidates[0]["content"].(map[string]interface{})["parts"].([]map[string]interface{})
	if len(parts) != 1 || len(parts[0]) != 1 || parts[0]["text"] != "answer" {
		t.Fatalf("default-filled union was not normalized to one text arm: %#v", parts)
	}
}

func TestBuildGeminiRequestOptionsPassesToolsAndToolConfig(t *testing.T) {
	req := map[string]interface{}{
		"tools": []interface{}{
			map[string]interface{}{
				"functionDeclarations": []interface{}{
					map[string]interface{}{"name": "lookup"},
				},
			},
		},
		"toolConfig": map[string]interface{}{
			"functionCallingConfig": map[string]interface{}{"mode": "ANY"},
		},
		"serviceTier": "PRIORITY",
		"store":       false,
	}

	options := buildGeminiRequestOptions(req)
	if options == nil {
		t.Fatal("options should not be nil")
	}
	if options.Tools == nil {
		t.Fatal("tools should be passed through")
	}
	if options.ToolConfig == nil {
		t.Fatal("toolConfig should be passed through")
	}
	if got := options.AdditionalVariables["serviceTier"]; got != "PRIORITY" {
		t.Fatalf("serviceTier = %v, want PRIORITY", got)
	}
	if got := options.AdditionalVariables["store"]; got != false {
		t.Fatalf("store = %v, want false", got)
	}
}

func TestGeminiErrorResponsePassthroughUpstreamMessage(t *testing.T) {
	rawErr := "api error (code 503): HTTP 503 content-aiplatform.googleapis.com"
	resp := geminiErrorResponse(errors.New(rawErr))

	errObj := resp["error"].(map[string]interface{})
	if got := errObj["message"]; got != rawErr {
		t.Fatalf("error message = %v, want %q", got, rawErr)
	}
}

func TestGeminiErrorResponsePreservesGenerationFinishReason(t *testing.T) {
	message := "generation failed: finishReason=SAFETY"
	resp := geminiErrorResponse(errors.New(message))

	errObj := resp["error"].(map[string]interface{})
	if got := errObj["message"]; got != message {
		t.Fatalf("error message = %v, want %q", got, message)
	}
}

func TestBuildGeminiResponseIncludesFunctionCalls(t *testing.T) {
	resp := buildGeminiResponse(&proxy.CallResult{
		FunctionCalls: []model.FunctionCall{
			{
				Name:             "lookup",
				Args:             map[string]interface{}{"query": "weather"},
				ThoughtSignature: "sig",
			},
		},
	})

	candidates := resp["candidates"].([]map[string]interface{})
	content := candidates[0]["content"].(map[string]interface{})
	parts := content["parts"].([]map[string]interface{})
	functionCall := parts[0]["functionCall"].(map[string]interface{})

	if got := functionCall["name"]; got != "lookup" {
		t.Fatalf("function call name = %v, want lookup", got)
	}
	args := functionCall["args"].(map[string]interface{})
	if got := args["query"]; got != "weather" {
		t.Fatalf("function call query = %v, want weather", got)
	}
	if got := parts[0]["thoughtSignature"]; got != "sig" {
		t.Fatalf("function call thoughtSignature = %v, want sig", got)
	}
}

func TestBuildGeminiResponseIncludesGroundingMetadata(t *testing.T) {
	gm := &model.GroundingMetadata{
		WebSearchQueries: []string{"golang"},
		GroundingChunks: []model.GroundingChunk{
			{Web: &model.WebChunk{Title: "Go", URI: "https://golang.org"}},
		},
	}
	resp := buildGeminiResponse(&proxy.CallResult{
		TextParts:         []model.TextPart{{Text: "Go programming"}},
		GroundingMetadata: gm,
	})

	candidates := resp["candidates"].([]map[string]interface{})
	gotGM, ok := candidates[0]["groundingMetadata"].(*model.GroundingMetadata)
	if !ok || gotGM == nil {
		t.Fatal("groundingMetadata should be preserved in candidate")
	}
	if len(gotGM.GroundingChunks) != 1 || gotGM.GroundingChunks[0].Web.URI != "https://golang.org" {
		t.Fatalf("unexpected grounding metadata content: %+v", gotGM)
	}
}

func TestBuildGeminiResponseOmitsSemanticallyEmptyMetadata(t *testing.T) {
	resp := buildGeminiResponse(&proxy.CallResult{Candidates: []proxy.CandidateResult{{
		Index:             0,
		Parts:             []model.VertexPart{{Text: "answer"}},
		GroundingMetadata: &model.GroundingMetadata{SearchEntryPoint: &model.SearchEntryPoint{}},
		CitationMetadata: map[string]interface{}{
			"citations": []interface{}{},
		},
	}}})
	candidates := resp["candidates"].([]map[string]interface{})
	if _, ok := candidates[0]["groundingMetadata"]; ok {
		t.Fatalf("empty groundingMetadata leaked: %#v", candidates[0])
	}
	if _, ok := candidates[0]["citationMetadata"]; ok {
		t.Fatalf("empty citationMetadata leaked: %#v", candidates[0])
	}
}

func TestBuildGeminiResponsePreservesValidCitationMetadata(t *testing.T) {
	resp := buildGeminiResponse(&proxy.CallResult{Candidates: []proxy.CandidateResult{{
		Index: 0,
		Parts: []model.VertexPart{{Text: "answer"}},
		CitationMetadata: map[string]interface{}{
			"citations": []interface{}{
				map[string]interface{}{
					"startIndex": float64(0), "endIndex": float64(6), "uri": "https://example.com",
					"title": "Vertex-only title", "publicationDate": map[string]interface{}{"year": float64(2026)},
				},
				map[string]interface{}{},
			},
		},
	}}})
	candidates := resp["candidates"].([]map[string]interface{})
	metadata := candidates[0]["citationMetadata"].(map[string]interface{})
	citations := metadata["citationSources"].([]interface{})
	if len(citations) != 1 {
		t.Fatalf("citationMetadata = %#v, want one valid citation", metadata)
	}
	citation := citations[0].(map[string]interface{})
	if _, ok := citation["title"]; ok {
		t.Fatalf("Vertex-only citation field leaked: %#v", citation)
	}
	if _, ok := metadata["citations"]; ok {
		t.Fatalf("Vertex citation field name leaked: %#v", metadata)
	}
}

func TestGeminiGenerateRejectsModelArmorWithSafetySettings(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/models/gemini-test:generateContent", strings.NewReader(`{
		"contents":[{"role":"user","parts":[{"text":"hello"}]}],
		"modelArmorConfig":{},
		"safetySettings":[{"category":"HARM_CATEGORY_HATE_SPEECH","threshold":"OFF"}]
	}`))
	req.SetPathValue("modelAction", "gemini-test:generateContent")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	GeminiGenerate(nil, true).ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "cannot be used together") {
		t.Fatalf("status = %d body = %s", rec.Code, rec.Body.String())
	}
}

func TestBuildGeminiResponsePreservesOrderedCanonicalParts(t *testing.T) {
	willContinue := true
	result := &proxy.CallResult{Role: "model", Parts: []model.VertexPart{
		{Text: "before"},
		{FunctionCall: &model.FunctionCall{
			ID: "c1", Name: "lookup", Args: map[string]interface{}{"query": "weather"}, PartialArgs: []map[string]interface{}{{
				"jsonPath": "$.query", "stringValue": "wea", "willContinue": true,
			}}, WillContinue: &willContinue,
		}, ThoughtSignature: "call-signature"},
		{Text: "after"},
		{ExecutableCode: map[string]interface{}{"language": "PYTHON", "code": "print(1)", "id": "code-1", "vertexOnly": true}},
		{CodeExecutionResult: map[string]interface{}{"outcome": "OUTCOME_OK", "output": "1", "id": "code-1", "vertexOnly": true}},
		{InlineData: &model.InlineData{MimeType: "image/png", Data: "aW1n", DisplayName: "chart.png"},
			VideoMetadata:   map[string]interface{}{"startOffset": "1s", "endOffset": "2s", "fps": 2, "vertexOnly": true},
			MediaResolution: map[string]interface{}{"level": "MEDIA_RESOLUTION_HIGH", "vertexOnly": true}},
		{FileData: map[string]interface{}{"mimeType": "application/pdf", "fileUri": "gs://bucket/a.pdf", "displayName": "a.pdf", "vertexOnly": true}},
		{FunctionResponse: &model.FunctionResponse{
			ID: "c1", Name: "lookup", Response: map[string]interface{}{"ok": true},
			Parts: []model.FunctionResponsePart{
				{InlineData: &model.InlineData{MimeType: "image/png", Data: "cmVzdWx0", DisplayName: "result.png"}},
				{FileData: map[string]interface{}{"mimeType": "image/png", "fileUri": "gs://bucket/result.png"}},
			},
		}},
	}}
	response := buildGeminiResponse(result)
	candidates := response["candidates"].([]map[string]interface{})
	content := candidates[0]["content"].(map[string]interface{})
	parts := content["parts"].([]map[string]interface{})
	wantKinds := []string{"text", "functionCall", "text", "executableCode", "codeExecutionResult", "inlineData", "fileData", "functionResponse"}
	if len(parts) != len(wantKinds) {
		t.Fatalf("parts length = %d, want %d: %#v", len(parts), len(wantKinds), parts)
	}
	for i, wantKind := range wantKinds {
		if _, ok := parts[i][wantKind]; !ok {
			t.Fatalf("part[%d] kind = %#v, want %s", i, parts[i], wantKind)
		}
	}
	functionCall := parts[1]["functionCall"].(map[string]interface{})
	if functionCall["name"] != "lookup" || functionCall["id"] != "c1" || parts[1]["thoughtSignature"] != "call-signature" {
		t.Fatalf("function call was not preserved canonically: %#v", parts[1])
	}
	if _, ok := functionCall["partialArgs"]; ok {
		t.Fatalf("Vertex partialArgs leaked: %#v", functionCall)
	}
	if _, ok := functionCall["willContinue"]; ok {
		t.Fatalf("Vertex willContinue leaked: %#v", functionCall)
	}
	if _, ok := parts[3]["executableCode"].(map[string]interface{})["vertexOnly"]; ok {
		t.Fatalf("executableCode leaked an unknown field: %#v", parts[3])
	}
	if _, ok := parts[4]["codeExecutionResult"].(map[string]interface{})["vertexOnly"]; ok {
		t.Fatalf("codeExecutionResult leaked an unknown field: %#v", parts[4])
	}
	if parts[5]["videoMetadata"] == nil || parts[5]["mediaResolution"] == nil {
		t.Fatalf("media metadata was dropped: %#v", parts[5])
	}
	if _, ok := parts[5]["inlineData"].(map[string]interface{})["displayName"]; ok {
		t.Fatalf("Vertex inlineData displayName leaked: %#v", parts[5])
	}
	if _, ok := parts[6]["fileData"].(map[string]interface{})["displayName"]; ok {
		t.Fatalf("Vertex fileData displayName leaked: %#v", parts[6])
	}
	functionResponse := parts[7]["functionResponse"].(map[string]interface{})
	if len(functionResponse["parts"].([]map[string]interface{})) != 1 {
		t.Fatalf("function response parts = %#v", functionResponse["parts"])
	}
	responseInlineData := functionResponse["parts"].([]map[string]interface{})[0]["inlineData"].(map[string]interface{})
	if _, ok := responseInlineData["displayName"]; ok {
		t.Fatalf("Vertex function response displayName leaked: %#v", responseInlineData)
	}
}

func TestBuildGeminiResponseOmitsVertexOnlyTopLevelCreateTime(t *testing.T) {
	response := buildGeminiResponse(&proxy.CallResult{
		CreateTime: "2026-08-13T00:00:00Z",
		Parts:      []model.VertexPart{{Text: "answer"}},
	})
	if _, ok := response["createTime"]; ok {
		t.Fatalf("Vertex createTime leaked: %#v", response)
	}
}

func TestBuildGeminiResponseDropsMalformedPartOneOfAndRequiredFields(t *testing.T) {
	result := &proxy.CallResult{Parts: []model.VertexPart{
		{Text: "invalid", InlineData: &model.InlineData{MimeType: "image/png", Data: "aW1n"}},
		{InlineData: &model.InlineData{Data: "aW1n"}},
		{FileData: map[string]interface{}{"fileUri": "gs://bucket/a.pdf"}},
		{FunctionResponse: &model.FunctionResponse{Name: "lookup"}},
		{ExecutableCode: map[string]interface{}{"code": "print(1)"}},
		{CodeExecutionResult: map[string]interface{}{"output": "1"}},
	}}
	response := buildGeminiStreamResponse(result)
	if candidates, ok := response["candidates"]; ok {
		t.Fatalf("malformed Parts became Gemini candidates: %#v", candidates)
	}
}

func TestBuildGeminiResponseSuppressesCandidatesForBlockedPrompt(t *testing.T) {
	response := buildGeminiResponse(&proxy.CallResult{
		PromptFeedback: map[string]interface{}{"blockReason": "SAFETY"},
		Candidates: []proxy.CandidateResult{{
			Index: 0, Parts: []model.VertexPart{{Text: "must not leak"}}, FinishReason: "STOP",
		}},
	})
	if candidates, ok := response["candidates"]; ok {
		t.Fatalf("blocked prompt returned candidates: %#v", candidates)
	}
	if response["promptFeedback"] == nil {
		t.Fatal("promptFeedback should be preserved")
	}
}

func TestBuildGeminiResponseDropsUnspecifiedPromptFeedback(t *testing.T) {
	for _, blockReason := range []string{"BLOCKED_REASON_UNSPECIFIED", "BLOCK_REASON_UNSPECIFIED"} {
		t.Run(blockReason, func(t *testing.T) {
			response := buildGeminiResponse(&proxy.CallResult{
				PromptFeedback: map[string]interface{}{
					"blockReason":        blockReason,
					"blockReasonMessage": "default-initialized Vertex value",
					"safetyRatings":      []interface{}{map[string]interface{}{}},
				},
				Parts: []model.VertexPart{{Text: "answer"}},
			})

			if promptFeedback, ok := response["promptFeedback"]; ok {
				t.Fatalf("unspecified promptFeedback was not omitted: %#v", promptFeedback)
			}
			if _, ok := response["candidates"]; !ok {
				t.Fatal("an unspecified block reason suppressed a valid candidate")
			}
		})
	}
}

func TestBuildGeminiStreamResponseMapsVertexOnlyPromptBlockReason(t *testing.T) {
	for _, blockReason := range []string{"MODEL_ARMOR", "JAILBREAK", "FUTURE_VERTEX_REASON"} {
		t.Run(blockReason, func(t *testing.T) {
			source := map[string]interface{}{
				"blockReason":        blockReason,
				"blockReasonMessage": "Vertex-only field",
			}
			response := buildGeminiStreamResponse(&proxy.CallResult{
				PromptFeedback: source,
				Parts:          []model.VertexPart{{Text: "must not leak"}},
			})

			if _, ok := response["candidates"]; ok {
				t.Fatal("a blocked prompt returned candidates")
			}
			promptFeedback := response["promptFeedback"].(map[string]interface{})
			if promptFeedback["blockReason"] != "OTHER" || len(promptFeedback) != 1 {
				t.Fatalf("Vertex block reason was not converted to Gemini: %#v", promptFeedback)
			}
			if source["blockReason"] != blockReason || source["blockReasonMessage"] == nil {
				t.Fatalf("normalization mutated the upstream object: %#v", source)
			}
		})
	}
}

func TestBuildGeminiResponseCanonicalizesPromptSafetyRatings(t *testing.T) {
	response := buildGeminiResponse(&proxy.CallResult{
		PromptFeedback: map[string]interface{}{
			"safetyRatings": []interface{}{
				map[string]interface{}{
					"category":         "HARM_CATEGORY_HARASSMENT",
					"probability":      "LOW",
					"blocked":          false,
					"probabilityScore": 0.1,
				},
				map[string]interface{}{
					"category":    "HARM_CATEGORY_UNSPECIFIED",
					"probability": "HARM_PROBABILITY_UNSPECIFIED",
					"blocked":     false,
				},
				map[string]interface{}{
					"category":    "HARM_CATEGORY_FUTURE_GEMINI_VALUE",
					"probability": "FUTURE_GEMINI_PROBABILITY",
					"blocked":     true,
				},
			},
		},
		Parts: []model.VertexPart{{Text: "answer"}},
	})

	promptFeedback := response["promptFeedback"].(map[string]interface{})
	ratings := promptFeedback["safetyRatings"].([]map[string]interface{})
	if len(ratings) != 2 || ratings[0]["category"] != "HARM_CATEGORY_HARASSMENT" || ratings[0]["probability"] != "LOW" {
		t.Fatalf("safety ratings were not normalized: %#v", ratings)
	}
	if ratings[1]["category"] != "HARM_CATEGORY_FUTURE_GEMINI_VALUE" ||
		ratings[1]["probability"] != "FUTURE_GEMINI_PROBABILITY" || ratings[1]["blocked"] != true {
		t.Fatalf("unknown future safety enum was dropped: %#v", ratings[1])
	}
	if _, ok := ratings[0]["blocked"]; ok {
		t.Fatalf("default false value was retained: %#v", ratings[0])
	}
	if _, ok := ratings[0]["probabilityScore"]; ok {
		t.Fatalf("Vertex-only safety field leaked into Gemini response: %#v", ratings[0])
	}
	if _, ok := response["candidates"]; !ok {
		t.Fatal("safety ratings without a block reason suppressed the candidate")
	}
}

func TestBuildGeminiResponseCanonicalizesCandidateSafetyRatings(t *testing.T) {
	response := buildGeminiResponse(&proxy.CallResult{Candidates: []proxy.CandidateResult{{
		Index: 0,
		Parts: []model.VertexPart{{Text: "answer"}},
		SafetyRatings: []map[string]interface{}{
			{"category": "HARM_CATEGORY_JAILBREAK", "probability": "HIGH", "blocked": true, "severity": "HIGH"},
			{"category": "HARM_CATEGORY_FUTURE_GEMINI_VALUE", "probability": "LOW"},
		},
	}}})

	candidates := response["candidates"].([]map[string]interface{})
	ratings := candidates[0]["safetyRatings"].([]map[string]interface{})
	if len(ratings) != 2 || ratings[0]["category"] != "HARM_CATEGORY_JAILBREAK" || ratings[0]["blocked"] != true {
		t.Fatalf("candidate safety ratings were not normalized: %#v", ratings)
	}
	if ratings[1]["category"] != "HARM_CATEGORY_FUTURE_GEMINI_VALUE" {
		t.Fatalf("unknown future candidate safety enum was dropped: %#v", ratings[1])
	}
	if _, ok := ratings[0]["severity"]; ok {
		t.Fatalf("Vertex-only candidate safety field leaked: %#v", ratings[0])
	}
}
