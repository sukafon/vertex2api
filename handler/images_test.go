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
