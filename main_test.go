package main

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"vertex2api/config"
)

func TestApplicationHealthAuthAndLocalCountTokens(t *testing.T) {
	cfg := &config.Config{APIKeys: []string{"secret"}}
	app := newApplication(cfg, nil, nil)

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
	app := newApplication(cfg, nil, nil)

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
	app := newApplication(cfg, nil, nil)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	req.URL.Path = "/v1/models/../../health"
	req.Header.Set("Authorization", "Bearer 1234567890123456")
	rec := httptest.NewRecorder()
	app.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400: %s", rec.Code, rec.Body.String())
	}
}

func TestApplicationRegistersResponsesRoutes(t *testing.T) {
	cfg := &config.Config{APIKeys: []string{"1234567890123456"}, AllowCustomModelNames: true}
	app := newApplication(cfg, nil, nil)

	for _, path := range []string{"/v1/responses", "/v1/responses/compact"} {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(`{"model":"custom-model"}`))
		req.Header.Set("Authorization", "Bearer 1234567890123456")
		req.Header.Set("Content-Type", "application/json")
		recorder := httptest.NewRecorder()
		app.ServeHTTP(recorder, req)
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("%s status = %d, want 400: %s", path, recorder.Code, recorder.Body.String())
		}
		if !strings.Contains(recorder.Body.String(), "input is required") {
			t.Fatalf("%s did not reach Responses handler: %s", path, recorder.Body.String())
		}
	}
}

func TestApplicationModelRefreshUsesStatsKey(t *testing.T) {
	cfg := &config.Config{
		APIKeys:  []string{"api-secret"},
		StatsKey: "stats-secret",
	}
	called := 0
	app := newApplication(cfg, nil, func(context.Context) (int, error) {
		called++
		return 3, nil
	})

	apiKeyRequest := httptest.NewRequest(http.MethodPost, "/v1/models/refresh", nil)
	apiKeyRequest.Header.Set("Authorization", "Bearer api-secret")
	apiKeyResponse := httptest.NewRecorder()
	app.ServeHTTP(apiKeyResponse, apiKeyRequest)
	if apiKeyResponse.Code != http.StatusUnauthorized {
		t.Fatalf("API_KEY status = %d, want 401: %s", apiKeyResponse.Code, apiKeyResponse.Body.String())
	}
	if called != 0 {
		t.Fatalf("refresher called %d times with API_KEY, want 0", called)
	}

	statsKeyRequest := httptest.NewRequest(http.MethodPost, "/v1/models/refresh", nil)
	statsKeyRequest.Header.Set("Authorization", "Bearer stats-secret")
	statsKeyResponse := httptest.NewRecorder()
	app.ServeHTTP(statsKeyResponse, statsKeyRequest)
	if statsKeyResponse.Code != http.StatusOK {
		t.Fatalf("STATS_KEY status = %d, want 200: %s", statsKeyResponse.Code, statsKeyResponse.Body.String())
	}
	if called != 1 {
		t.Fatalf("refresher called %d times with STATS_KEY, want 1", called)
	}
}
