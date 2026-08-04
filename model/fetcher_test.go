package model

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"vertex2api/client"
	"vertex2api/config"
)

func TestFilterGoogleGlobalModels(t *testing.T) {
	input := []rawCatalogModel{
		{
			ModelID:          "gemini-3.5-flash",
			PublisherID:      "google",
			ModelFamily:      "gemini",
			SupportedRegions: []string{"global", "us", "eu"},
		},
		{
			ModelID:          "claude-3-5-sonnet",
			PublisherID:      "anthropic",
			ModelFamily:      "anthropic",
			SupportedRegions: []string{"global"},
		},
		{
			ModelID:          "gemini-us-only",
			PublisherID:      "google",
			ModelFamily:      "gemini",
			SupportedRegions: []string{"us"},
		},
		{
			ModelID:          "gemma-3-12b",
			PublisherID:      "google",
			ModelFamily:      "gemma",
			SupportedRegions: []string{"global"},
		},
		{
			ModelID:          "adaptive-mt-translate--nmt",
			PublisherID:      "google",
			ModelFamily:      "gemini",
			SupportedRegions: []string{"global"},
		},
		{
			ModelID:          "gemini-3-pro",
			PublisherID:      "GOOGLE",
			ModelFamily:      "GEMINI",
			SupportedRegions: []string{"GLOBAL"},
		},
	}

	filtered := FilterGoogleGlobalModels(input)
	if len(filtered) != 2 {
		t.Fatalf("expected 2 filtered models, got %d", len(filtered))
	}

	if filtered[0].ModelID != "gemini-3.5-flash" || filtered[1].ModelID != "gemini-3-pro" {
		t.Errorf("unexpected filtered models: %+v", filtered)
	}
}

func TestExtractRawModelsDirectArray(t *testing.T) {
	jsonData := `[
		{
			"modelId": "gemini-3.5-flash",
			"publisherId": "google",
			"supportedRegions": ["global"]
		}
	]`

	models, err := ExtractRawModels([]byte(jsonData))
	if err != nil {
		t.Fatalf("ExtractRawModels returned error: %v", err)
	}
	if len(models) != 1 || models[0].ModelID != "gemini-3.5-flash" {
		t.Fatalf("unexpected extracted models: %+v", models)
	}
}

func TestExtractRawModelsBatchGraphQL(t *testing.T) {
	jsonData := `[
		{
			"results": [
				{
					"data": {
						"modelConfigs": [
							{
								"modelId": "gemini-3.5-flash",
								"publisherId": "google",
								"supportedRegions": ["global"]
							}
						]
					}
				}
			]
		}
	]`

	models, err := ExtractRawModels([]byte(jsonData))
	if err != nil {
		t.Fatalf("ExtractRawModels returned error: %v", err)
	}
	if len(models) != 1 || models[0].ModelID != "gemini-3.5-flash" {
		t.Fatalf("unexpected extracted models: %+v", models)
	}
}

func TestExtractRawModelsUIConfigs(t *testing.T) {
	jsonData := `[
		{
			"results": [
				{
					"data": {
						"ui": {
							"modelConfigs": {
								"configs": [
									{
										"modelId": "gemini-3.6-flash",
										"publisherId": "google",
										"supportedRegions": ["global"]
									}
								]
							}
						}
					}
				}
			]
		}
	]`

	models, err := ExtractRawModels([]byte(jsonData))
	if err != nil {
		t.Fatalf("ExtractRawModels returned error: %v", err)
	}
	if len(models) != 1 || models[0].ModelID != "gemini-3.6-flash" {
		t.Fatalf("unexpected extracted models: %+v", models)
	}
}

func TestNextScheduleTimeDefaultCron(t *testing.T) {
	loc := time.Local

	// Case 1: Time is 01:00 -> Next schedule should be 04:00 today
	now1 := time.Date(2026, 7, 22, 1, 0, 0, 0, loc)
	next1 := NextScheduleTimeWithCron("0 0,4 * * *", now1)
	expected1 := time.Date(2026, 7, 22, 4, 0, 0, 0, loc)
	if !next1.Equal(expected1) {
		t.Errorf("now = %v: expected next = %v, got %v", now1, expected1, next1)
	}

	// Case 2: Time is 05:00 -> Next schedule should be 00:00 tomorrow
	now2 := time.Date(2026, 7, 22, 5, 0, 0, 0, loc)
	next2 := NextScheduleTimeWithCron("0 0,4 * * *", now2)
	expected2 := time.Date(2026, 7, 23, 0, 0, 0, 0, loc)
	if !next2.Equal(expected2) {
		t.Errorf("now = %v: expected next = %v, got %v", now2, expected2, next2)
	}

	// Case 3: Time is 23:30 -> Next schedule should be 00:00 tomorrow
	now3 := time.Date(2026, 7, 22, 23, 30, 0, 0, loc)
	next3 := NextScheduleTimeWithCron("0 0,4 * * *", now3)
	expected3 := time.Date(2026, 7, 23, 0, 0, 0, 0, loc)
	if !next3.Equal(expected3) {
		t.Errorf("now = %v: expected next = %v, got %v", now3, expected3, next3)
	}
}

func TestNextScheduleTimeWithCron(t *testing.T) {
	loc := time.Local
	now := time.Date(2026, 7, 22, 10, 15, 30, 0, loc) // Wed Jul 22 10:15:30 2026

	// Custom Cron 1: "*/15 * * * *" -> Next at 10:30:00
	next1 := NextScheduleTimeWithCron("*/15 * * * *", now)
	expected1 := time.Date(2026, 7, 22, 10, 30, 0, 0, loc)
	if !next1.Equal(expected1) {
		t.Errorf("cron '*/15 * * * *': expected %v, got %v", expected1, next1)
	}

	// Custom Cron 2: "30 2 * * *" -> Next at 02:30:00 tomorrow
	next2 := NextScheduleTimeWithCron("30 2 * * *", now)
	expected2 := time.Date(2026, 7, 23, 2, 30, 0, 0, loc)
	if !next2.Equal(expected2) {
		t.Errorf("cron '30 2 * * *': expected %v, got %v", expected2, next2)
	}

	// Invalid Cron fallback: "invalid cron" -> falls back to "0 0,4 * * *" -> Next at 00:00 tomorrow
	next3 := NextScheduleTimeWithCron("invalid cron", now)
	expected3 := time.Date(2026, 7, 23, 0, 0, 0, 0, loc)
	if !next3.Equal(expected3) {
		t.Errorf("invalid cron fallback: expected %v, got %v", expected3, next3)
	}
}

func TestUpdateCatalogFromUpstream(t *testing.T) {
	catalogMu.RLock()
	originalCatalog := defaultCatalog
	catalogMu.RUnlock()
	t.Cleanup(func() { SetCatalog(originalCatalog) })

	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Referer") != "https://console.cloud.google.com/" {
			t.Errorf("expected Referer header, got %q", r.Header.Get("Referer"))
		}
		if got := r.URL.Query().Get("key"); got != "test-vertex-key" {
			t.Errorf("Vertex API key = %q, want test-vertex-key", got)
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`[
			{
				"modelId": "gemini-test-auto-fetched",
				"publisherId": "google",
				"modelFamily": "gemini",
				"supportedRegions": ["global"]
			}
		]`))
	}))
	defer ts.Close()

	cfg := &config.Config{HTTPTimeoutSeconds: 5}
	httpClient, err := client.New(cfg)
	if err != nil {
		t.Fatalf("client.New: %v", err)
	}

	models, err := UpdateCatalogFromUpstream(context.Background(), httpClient, ts.URL, "test-vertex-key")
	if err != nil {
		t.Fatalf("UpdateCatalogFromUpstream failed: %v", err)
	}
	if len(models) != 1 || models[0].ModelID != "gemini-test-auto-fetched" {
		t.Fatalf("unexpected fetched models: %+v", models)
	}

	// Verify catalog in memory was updated
	openAIList := OpenAIModelList()
	found := false
	for _, m := range openAIList.Data {
		if m.ID == "gemini-test-auto-fetched" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected gemini-test-auto-fetched to be present in OpenAIModelList after catalog update")
	}
}

func TestUpdateCatalogFromUpstreamFailureKeepsExistingCatalog(t *testing.T) {
	beforeList := OpenAIModelList()

	// Server returns HTTP 500 Internal Server Error
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		w.Write([]byte("internal error"))
	}))
	defer ts.Close()

	cfg := &config.Config{HTTPTimeoutSeconds: 5}
	httpClient, err := client.New(cfg)
	if err != nil {
		t.Fatalf("client.New: %v", err)
	}

	models, err := UpdateCatalogFromUpstream(context.Background(), httpClient, ts.URL, "test-vertex-key")
	if err == nil {
		t.Fatal("expected error on upstream HTTP 500, got nil")
	}
	if len(models) != 0 {
		t.Fatalf("expected empty models slice on error, got %d", len(models))
	}

	// Verify catalog in memory was NOT mutated
	afterList := OpenAIModelList()
	if len(afterList.Data) != len(beforeList.Data) {
		t.Fatalf("catalog in memory was modified on failure: before len = %d, after len = %d", len(beforeList.Data), len(afterList.Data))
	}
}
