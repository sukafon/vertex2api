package main

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vertex2api/config"
)

func TestApplicationHealthAuthAndLocalCountTokens(t *testing.T) {
	cfg := &config.Config{APIKeys: []string{"secret"}}
	app := newApplication(cfg, nil)

	health := httptest.NewRecorder()
	app.ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/health", nil))
	if health.Code != http.StatusOK {
		t.Fatalf("health status = %d, want 200", health.Code)
	}

	unauthorized := httptest.NewRecorder()
	app.ServeHTTP(unauthorized, httptest.NewRequest(http.MethodGet, "/v1/models", nil))
	if unauthorized.Code != http.StatusUnauthorized {
		t.Fatalf("unauthorized models status = %d, want 401", unauthorized.Code)
	}

	modelsRequest := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	modelsRequest.Header.Set("Authorization", "bearer secret")
	models := httptest.NewRecorder()
	app.ServeHTTP(models, modelsRequest)
	if models.Code != http.StatusOK {
		t.Fatalf("authorized models status = %d, want 200: %s", models.Code, models.Body.String())
	}
	if strings.Contains(models.Body.String(), "test-model") {
		t.Fatal("public model catalog contains test-model")
	}

	countRequest := httptest.NewRequest(http.MethodPost, "/v1/messages/count_tokens", strings.NewReader(`{
        "model":"gemini-3.5-flash",
        "messages":[{"role":"user","content":"hello"}]
    }`))
	countRequest.Header.Set("Content-Type", "application/json")
	countRequest.Header.Set("x-api-key", "secret")
	count := httptest.NewRecorder()
	app.ServeHTTP(count, countRequest)
	if count.Code != http.StatusOK {
		t.Fatalf("count_tokens status = %d, want 200: %s", count.Code, count.Body.String())
	}
	if count.Header().Get("X-Usage-Estimated") != "true" {
		t.Fatal("count_tokens response did not identify estimated usage")
	}
	var countBody struct {
		InputTokens int `json:"input_tokens"`
	}
	if err := json.Unmarshal(count.Body.Bytes(), &countBody); err != nil {
		t.Fatalf("decode count_tokens response: %v", err)
	}
	if countBody.InputTokens < 1 {
		t.Fatalf("input_tokens = %d, want positive estimate", countBody.InputTokens)
	}
}

func TestApplicationCORSIsOptInAndOriginBound(t *testing.T) {
	cfg := &config.Config{APIKeys: []string{"secret"}, CORSAllowOrigin: "https://app.example"}
	app := newApplication(cfg, nil)

	allowedRequest := httptest.NewRequest(http.MethodOptions, "/v1/models", nil)
	allowedRequest.Header.Set("Origin", "https://app.example")
	allowed := httptest.NewRecorder()
	app.ServeHTTP(allowed, allowedRequest)
	if got := allowed.Header().Get("Access-Control-Allow-Origin"); got != "https://app.example" {
		t.Fatalf("allowed origin = %q", got)
	}

	deniedRequest := httptest.NewRequest(http.MethodOptions, "/v1/models", nil)
	deniedRequest.Header.Set("Origin", "https://other.example")
	denied := httptest.NewRecorder()
	app.ServeHTTP(denied, deniedRequest)
	if got := denied.Header().Get("Access-Control-Allow-Origin"); got != "" {
		t.Fatalf("unexpected CORS grant for other origin: %q", got)
	}
}

func TestApplicationRejectsTraversalBeforeAuthentication(t *testing.T) {
	cfg := &config.Config{APIKeys: []string{"1234567890123456"}}
	app := newApplication(cfg, nil)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	req.URL.Path = "/v1/models/../../health"
	req.Header.Set("Authorization", "Bearer 1234567890123456")
	rec := httptest.NewRecorder()
	app.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body.String())
	}
}
