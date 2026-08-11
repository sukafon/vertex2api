package handler

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestImageGenerationsRejectsInvalidNBeforeUpstreamCall(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/images/generations", strings.NewReader(`{
        "prompt":"draw a cat",
        "n":0
    }`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	ImageGenerations(nil, false).ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body.String())
	}
}

func TestImageGenerationsRejectsUnsupportedResponseFormat(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/v1/images/generations", strings.NewReader(`{
        "prompt":"draw a cat",
        "response_format":"url"
    }`))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	ImageGenerations(nil, false).ServeHTTP(rec, req)
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body.String())
	}
}

func TestOpenAIImageAspectRatio(t *testing.T) {
	tests := map[string]string{
		"1024x1024": "1:1",
		"1536x1024": "3:2",
		"1024x1536": "2:3",
		"1920x1080": "16:9",
		"1000x700":  "",
		"auto":      "",
	}
	for size, want := range tests {
		if got := openAIImageAspectRatio(size); got != want {
			t.Errorf("openAIImageAspectRatio(%q) = %q, want %q", size, got, want)
		}
	}
}

func TestDefaultOpenAIImageModelIsStableCatalogEntry(t *testing.T) {
	if defaultOpenAIImageModel != "gemini-3-pro-image" {
		t.Fatalf("default image model = %q", defaultOpenAIImageModel)
	}
}
