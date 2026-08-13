package middleware

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"vertex2api/config"
)

func TestAuthAcceptsAnthropicAPIKeyHeader(t *testing.T) {
	cfg := &config.Config{APIKeys: []string{"secret"}}
	called := false
	handler := Auth(cfg)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusNoContent)
	}))

	req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
	req.Header.Set("x-api-key", "secret")
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if !called {
		t.Fatal("handler should be called for valid x-api-key")
	}
	if rec.Code != http.StatusNoContent {
		t.Fatalf("status = %d, want 204", rec.Code)
	}
}

func TestAuthHealthIsPublicButAPIRoutesFailClosed(t *testing.T) {
	cfg := &config.Config{}
	handler := Auth(cfg)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	health := httptest.NewRecorder()
	handler.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/health", nil))
	if health.Code != http.StatusNoContent {
		t.Fatalf("health status = %d, want 204", health.Code)
	}

	api := httptest.NewRecorder()
	handler.ServeHTTP(api, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if api.Code != http.StatusUnauthorized {
		t.Fatalf("API status = %d, want 401", api.Code)
	}
}

func TestAuthLeavesStatsKeyRoutesToTheirHandlers(t *testing.T) {
	cfg := &config.Config{APIKeys: []string{"api-secret"}}
	handler := Auth(cfg)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))

	for _, path := range []string{"/v1/stats", "/v1/models/refresh"} {
		rec := httptest.NewRecorder()
		handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, path, nil))
		if rec.Code != http.StatusNoContent {
			t.Fatalf("%s status = %d, want 204", path, rec.Code)
		}
	}
}

func TestAuthUsesNativeGeminiEnvelopeForV1Route(t *testing.T) {
	cfg := &config.Config{APIKeys: []string{"secret"}}
	handler := Auth(cfg)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	}))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/models/gemini:test", nil))

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401", rec.Code)
	}
	var body struct {
		Error struct {
			Status string `json:"status"`
		} `json:"error"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if body.Error.Status != "UNAUTHENTICATED" {
		t.Fatalf("Gemini status = %q, want UNAUTHENTICATED", body.Error.Status)
	}
}

func TestAuthAnthropicErrorIncludesRequestID(t *testing.T) {
	cfg := &config.Config{APIKeys: []string{"secret"}}
	handler := Auth(cfg)(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	rec := httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/messages", nil))

	var body map[string]interface{}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if body["request_id"] == "" || body["request_id"] == nil {
		t.Fatalf("request_id missing from Anthropic error: %v", body)
	}
}
