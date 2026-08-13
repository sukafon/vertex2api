package handler

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vertex2api/config"
	"vertex2api/model"
	"vertex2api/proxy"
)

func TestUpstreamResponseRedactionDoesNotExposeDetail(t *testing.T) {
	vp := proxy.NewVertexProxy(nil, nil, &config.Config{RedactUpstreamResponses: true})
	err := errors.New("sensitive upstream response body")
	if got := publicUpstreamErrorMessage(vp, err); got != publicServerErrorMessage {
		t.Fatalf("public message = %q, want %q", got, publicServerErrorMessage)
	}
	if got := vp.UpstreamLogError(err); got != "sensitive upstream response body" {
		t.Fatalf("server log message = %q, want original detail", got)
	}
}

func TestUpstreamResponseRedactionCoversAllStreamingProtocols(t *testing.T) {
	const secret = "sensitive upstream response body"
	vp := proxy.NewVertexProxy(nil, nil, &config.Config{RedactUpstreamResponses: true})
	stream := func(context.Context, func(*proxy.CallResult) error) error {
		return errors.New(secret)
	}

	tests := []struct {
		name string
		run  func(*httptest.ResponseRecorder)
	}{
		{
			name: "openai_chat",
			run: func(rec *httptest.ResponseRecorder) {
				req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
				streamResponseForRequestWithProxy(rec, req, model.ChatCompletionRequest{Model: "gemini-test", Stream: true}, vp, stream)
			},
		},
		{
			name: "gemini",
			run: func(rec *httptest.ResponseRecorder) {
				req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-test:streamGenerateContent", nil)
				streamGeminiResponseWithProxy(rec, req, "gemini-test", vp, stream)
			},
		},
		{
			name: "anthropic",
			run: func(rec *httptest.ResponseRecorder) {
				req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
				streamAnthropicResponseWithProxy(rec, req, model.AnthropicMessageRequest{Model: "gemini-test", Stream: true}, vp, stream)
			},
		},
		{
			name: "openai_responses",
			run: func(rec *httptest.ResponseRecorder) {
				req := httptest.NewRequest(http.MethodPost, "/v1/responses", nil)
				api := &ResponsesAPI{vp: vp}
				api.streamResponse(rec, req, model.ResponseRequest{Model: "gemini-test", Stream: true}, model.ChatCompletionRequest{Model: "gemini-test", Stream: true}, nil, nil, nil, stream)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			rec := httptest.NewRecorder()
			test.run(rec)
			body := rec.Body.String()
			if strings.Contains(body, secret) {
				t.Fatalf("response leaked upstream detail: %s", body)
			}
			if !strings.Contains(body, publicServerErrorMessage) {
				t.Fatalf("response does not contain generic error: %s", body)
			}
		})
	}
}

func TestOpenAIStreamFirstErrorUsesHTTPJSON(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()
	streamResponse(rec, req, "gemini-test", func(context.Context, func(*proxy.CallResult) error) error {
		return errors.New("upstream unavailable")
	})
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
	if got := rec.Header().Get("Content-Type"); !strings.HasPrefix(got, "application/json") {
		t.Fatalf("Content-Type = %q, want application/json", got)
	}
	if strings.Contains(rec.Body.String(), "data:") {
		t.Fatalf("first error must not be wrapped as SSE: %s", rec.Body.String())
	}
}

func TestGeminiStreamFirstErrorUsesHTTPJSON(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1beta/models/gemini-test:streamGenerateContent", nil)
	rec := httptest.NewRecorder()
	streamGeminiResponse(rec, req, "gemini-test", func(context.Context, func(*proxy.CallResult) error) error {
		return errors.New("upstream unavailable")
	})
	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), `"status":"INTERNAL"`) {
		t.Fatalf("response = status %d body %s", rec.Code, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "data:") {
		t.Fatalf("first error must not be wrapped as SSE: %s", rec.Body.String())
	}
}

func TestAnthropicStreamFirstErrorUsesHTTPJSON(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	rec := httptest.NewRecorder()
	streamAnthropicResponse(rec, req, model.AnthropicMessageRequest{Model: "gemini-test"}, func(context.Context, func(*proxy.CallResult) error) error {
		return errors.New("upstream unavailable")
	})
	if rec.Code != http.StatusInternalServerError || !strings.Contains(rec.Body.String(), `"type":"error"`) {
		t.Fatalf("response = status %d body %s", rec.Code, rec.Body.String())
	}
	if strings.Contains(rec.Body.String(), "message_start") {
		t.Fatalf("message_start must not precede a first error: %s", rec.Body.String())
	}
}

func TestOpenAIStreamIncludeUsageWritesFinalUsageChunk(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	rec := httptest.NewRecorder()
	streamResponseForRequest(rec, req, model.ChatCompletionRequest{
		Model:         "gemini-test",
		Messages:      []model.ChatMessage{{Role: "user", Content: "hello"}},
		Stream:        true,
		StreamOptions: &model.ChatStreamOptions{IncludeUsage: true},
	}, func(_ context.Context, onChunk func(*proxy.CallResult) error) error {
		if err := onChunk(&proxy.CallResult{TextParts: []model.TextPart{{Text: "hi"}}}); err != nil {
			return err
		}
		return onChunk(&proxy.CallResult{FinishReason: "STOP", UsageMetadata: map[string]interface{}{
			"promptTokenCount":     float64(5),
			"candidatesTokenCount": float64(2),
			"totalTokenCount":      float64(7),
		}})
	})
	body := rec.Body.String()
	if !strings.Contains(body, `"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7`) {
		t.Fatalf("missing final usage chunk: %s", body)
	}
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Fatalf("stream does not end with [DONE]: %s", body)
	}
}
