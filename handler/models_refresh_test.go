package handler

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"vertex2api/config"
)

func TestRefreshModelsRequiresConfiguredStatsKey(t *testing.T) {
	called := false
	refresh := func(context.Context) (int, error) {
		called = true
		return 1, nil
	}
	rec := httptest.NewRecorder()
	RefreshModels(&config.Config{}, refresh).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, "/v1/models/refresh", nil))

	if rec.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403: %s", rec.Code, rec.Body.String())
	}
	if called {
		t.Fatal("refresher was called while STATS_KEY was disabled")
	}
}

func TestRefreshModelsRejectsInvalidStatsKey(t *testing.T) {
	called := false
	refresh := func(context.Context) (int, error) {
		called = true
		return 1, nil
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/models/refresh", nil)
	req.Header.Set("Authorization", "Bearer wrong-key")
	rec := httptest.NewRecorder()
	RefreshModels(&config.Config{StatsKey: "stats-secret"}, refresh).ServeHTTP(rec, req)

	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d, want 401: %s", rec.Code, rec.Body.String())
	}
	if called {
		t.Fatal("refresher was called with an invalid STATS_KEY")
	}
}

func TestRefreshModelsUpdatesCatalog(t *testing.T) {
	refresh := func(ctx context.Context) (int, error) {
		if ctx == nil {
			t.Fatal("request context was not forwarded")
		}
		return 7, nil
	}
	req := httptest.NewRequest(http.MethodPost, "/v1/models/refresh", nil)
	req.Header.Set("x-api-key", "stats-secret")
	rec := httptest.NewRecorder()
	RefreshModels(&config.Config{StatsKey: "stats-secret"}, refresh).ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
	var body struct {
		Success    bool `json:"success"`
		Updated    bool `json:"updated"`
		ModelCount int  `json:"model_count"`
	}
	if err := json.Unmarshal(rec.Body.Bytes(), &body); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !body.Success || !body.Updated || body.ModelCount != 7 {
		t.Fatalf("response = %#v", body)
	}
}

func TestRefreshModelsKeepsCatalogOnUpstreamFailureOrEmptyResult(t *testing.T) {
	tests := []struct {
		name    string
		refresh ModelCatalogRefreshFunc
	}{
		{name: "upstream failure", refresh: func(context.Context) (int, error) { return 0, errors.New("upstream unavailable") }},
		{name: "empty result", refresh: func(context.Context) (int, error) { return 0, nil }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/models/refresh?key=stats-secret", nil)
			rec := httptest.NewRecorder()
			RefreshModels(&config.Config{StatsKey: "stats-secret"}, tt.refresh).ServeHTTP(rec, req)
			if rec.Code != http.StatusBadGateway {
				t.Fatalf("status = %d, want 502: %s", rec.Code, rec.Body.String())
			}
		})
	}
}
