package handler

import (
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

func TestBuildGeminiStreamResponseIncludesStopFinishReason(t *testing.T) {
	resp := buildGeminiStreamResponse(&proxy.CallResult{
		TextParts:    []model.TextPart{{Text: "done"}},
		FinishReason: "STOP",
	})

	candidates := resp["candidates"].([]map[string]interface{})
	if got := candidates[0]["finishReason"]; got != "STOP" {
		t.Fatalf("finishReason = %v, want STOP", got)
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

func TestBuildGeminiResponsePreservesOrderedNativeParts(t *testing.T) {
	result := &proxy.CallResult{Role: "model", Parts: []model.VertexPart{
		{Text: "before"},
		{FunctionCall: &model.FunctionCall{ID: "c1", Name: "lookup", Args: map[string]interface{}{"q": "x"}}, ThoughtSignature: "sig"},
		{Text: "after"},
		{ExecutableCode: map[string]interface{}{"language": "PYTHON", "code": "print(1)"}},
		{CodeExecutionResult: map[string]interface{}{"outcome": "OUTCOME_OK", "output": "1"}},
		{FileData: map[string]interface{}{"fileUri": "gs://bucket/a.pdf", "mimeType": "application/pdf"}},
	}}
	response := buildGeminiResponse(result)
	candidates := response["candidates"].([]map[string]interface{})
	content := candidates[0]["content"].(map[string]interface{})
	parts := content["parts"].([]map[string]interface{})
	if len(parts) != 6 || parts[0]["text"] != "before" || parts[1]["functionCall"] == nil || parts[2]["text"] != "after" || parts[3]["executableCode"] == nil || parts[4]["codeExecutionResult"] == nil || parts[5]["fileData"] == nil {
		t.Fatalf("native parts were reordered or dropped: %#v", parts)
	}
	if got := parts[1]["thoughtSignature"]; got != "sig" {
		t.Fatalf("thoughtSignature = %v", got)
	}
}
