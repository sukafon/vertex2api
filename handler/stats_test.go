package handler

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"vertex2api/config"
)

func TestStatsEndpointIsDisabledWithoutDedicatedKey(t *testing.T) {
	rec := httptest.NewRecorder()
	Stats(&config.Config{}).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/v1/stats", nil))
	if rec.Code != http.StatusForbidden {
		t.Fatalf("status = %d, want 403", rec.Code)
	}
}

func TestStatsEndpointAcceptsCaseInsensitiveBearerScheme(t *testing.T) {
	req := httptest.NewRequest(http.MethodGet, "/v1/stats", nil)
	req.Header.Set("Authorization", "bearer stats-secret")
	rec := httptest.NewRecorder()
	Stats(&config.Config{StatsKey: "stats-secret"}).ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200: %s", rec.Code, rec.Body.String())
	}
}
