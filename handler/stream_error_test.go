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
