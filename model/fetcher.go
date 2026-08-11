package model

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"

	"vertex2api/client"
	"vertex2api/config"

	"github.com/robfig/cron/v3"
	"github.com/rs/zerolog/log"
)

const (
	defaultFetchURLBase        = config.DefaultVertexBaseURL
	batchGraphqlFetchPath      = "/v3/entityServices/AiplatformEntityService/schemas/AIPLATFORM_GRAPHQL:batchGraphql"
	modelConfigsQuerySignature = "2/JtkEk+kqIKiFpvlAyssOl4zYIew7MD0pXLN7f9wau2Y="
	modelConfigsOperationName  = "ModelConfigs"
	modelConfigsDefaultReferer = "https://console.cloud.google.com/"
)

type ModelConfigsRequestBody struct {
	QuerySignature string                `json:"querySignature"`
	OperationName  string                `json:"operationName"`
	Variables      ModelConfigsVariables `json:"variables"`
}

type ModelConfigsVariables struct {
	LocationID    string `json:"locationId"`
	LanguageTag   string `json:"languageTag"`
	ProjectNumber string `json:"projectNumber"`
}

// FetchUpstreamModels requests upstream Vertex AI model catalog and filters models.
func FetchUpstreamModels(ctx context.Context, httpClient *client.HTTPClient, baseURL, apiKey string) ([]rawCatalogModel, error) {
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" || baseURL == "https://content-aiplatform.googleapis.com" {
		baseURL = defaultFetchURLBase
	}
	targetURL := strings.TrimRight(baseURL, "/") + batchGraphqlFetchPath + "?key=" + url.QueryEscape(apiKey) + "&prettyPrint=false"

	reqBodyObj := ModelConfigsRequestBody{
		QuerySignature: modelConfigsQuerySignature,
		OperationName:  modelConfigsOperationName,
		Variables: ModelConfigsVariables{
			LocationID:    "us-central1",
			LanguageTag:   "zh_CN",
			ProjectNumber: "",
		},
	}

	reqBodyBytes, err := json.Marshal(reqBodyObj)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, targetURL, bytes.NewReader(reqBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Referer", modelConfigsDefaultReferer)

	body, statusCode, err := httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http request failed: %w", err)
	}
	if statusCode != http.StatusOK {
		return nil, fmt.Errorf("upstream API returned status code %d: %s", statusCode, string(body))
	}

	allModels, err := ExtractRawModels(body)
	if err != nil {
		return nil, fmt.Errorf("failed to parse upstream models response: %w", err)
	}

	filtered := FilterGoogleGlobalModels(allModels)
	return filtered, nil
}

// UpdateCatalogFromUpstream fetches models from upstream and updates in-memory catalog.
func UpdateCatalogFromUpstream(ctx context.Context, httpClient *client.HTTPClient, baseURL, apiKey string) ([]rawCatalogModel, error) {
	models, err := FetchUpstreamModels(ctx, httpClient, baseURL, apiKey)
	if err != nil {
		return nil, err
	}

	if len(models) == 0 {
		log.Warn().Msg("Auto-fetched model list from upstream is empty, catalog not updated")
		return models, nil
	}

	newCatalog := buildModelCatalog(models)
	SetCatalog(newCatalog)
	log.Info().Int("count", len(models)).Msg("Successfully auto-fetched and updated model catalog in memory")
	return models, nil
}

// FilterGoogleGlobalModels filters models with publisherId == "google", modelFamily == "gemini",
// modelId starting with "gemini", and supportedRegions containing "global".
func FilterGoogleGlobalModels(models []rawCatalogModel) []rawCatalogModel {
	var filtered []rawCatalogModel
	for _, m := range models {
		modelID := strings.ToLower(strings.TrimSpace(m.ModelID))
		if !strings.HasPrefix(modelID, "gemini") {
			continue
		}
		if !strings.EqualFold(strings.TrimSpace(m.PublisherID), "google") {
			continue
		}
		if !strings.EqualFold(strings.TrimSpace(m.ModelFamily), "gemini") {
			continue
		}
		hasGlobal := false
		for _, region := range m.SupportedRegions {
			if strings.EqualFold(strings.TrimSpace(region), "global") {
				hasGlobal = true
				break
			}
		}
		if hasGlobal {
			filtered = append(filtered, m)
		}
	}
	return filtered
}

// ExtractRawModels parses response body into []rawCatalogModel.
func ExtractRawModels(body []byte) ([]rawCatalogModel, error) {
	// Try parsing directly as []rawCatalogModel
	decoder := json.NewDecoder(bytes.NewReader(body))
	decoder.UseNumber()
	var directModels []rawCatalogModel
	if err := decoder.Decode(&directModels); err == nil && hasValidModelIDs(directModels) {
		return directModels, nil
	}

	// Try parsing batchGraphql wrapper structure: [ { "results": [ { "data": { "modelConfigs": [...] } } ] } ]
	var generic []json.RawMessage
	if err := json.Unmarshal(body, &generic); err == nil {
		for _, item := range generic {
			if models := tryExtractFromRawMessage(item); len(models) > 0 {
				return models, nil
			}
		}
	}

	// Try parsing single object wrapper
	if models := tryExtractFromRawMessage(body); len(models) > 0 {
		return models, nil
	}

	return nil, fmt.Errorf("unable to locate model list in response")
}

func hasValidModelIDs(models []rawCatalogModel) bool {
	for _, m := range models {
		if strings.TrimSpace(m.ModelID) != "" {
			return true
		}
	}
	return false
}

func tryExtractFromRawMessage(raw json.RawMessage) []rawCatalogModel {
	var obj map[string]interface{}
	if err := json.Unmarshal(raw, &obj); err != nil {
		return nil
	}

	return searchModelsInMap(obj)
}

func searchModelsInMap(m map[string]interface{}) []rawCatalogModel {
	if configs, ok := m["configs"]; ok {
		if models := rawModelsFromInterface(configs); len(models) > 0 {
			return models
		}
	}

	if modelConfigs, ok := m["modelConfigs"]; ok {
		if models := rawModelsFromInterface(modelConfigs); len(models) > 0 {
			return models
		}
	}

	for _, v := range m {
		switch child := v.(type) {
		case map[string]interface{}:
			if models := searchModelsInMap(child); len(models) > 0 {
				return models
			}
		case []interface{}:
			for _, item := range child {
				if itemMap, ok := item.(map[string]interface{}); ok {
					if models := searchModelsInMap(itemMap); len(models) > 0 {
						return models
					}
				}
			}
		}
	}

	return nil
}

func rawModelsFromInterface(v interface{}) []rawCatalogModel {
	data, err := json.Marshal(v)
	if err != nil {
		return nil
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.UseNumber()
	var models []rawCatalogModel
	if err := decoder.Decode(&models); err == nil && hasValidModelIDs(models) {
		return models
	}
	return nil
}

// NextScheduleTimeWithCron returns the next scheduled run time for the given cron expression strictly after now.
func NextScheduleTimeWithCron(expr string, now time.Time) time.Time {
	sched, err := cron.ParseStandard(strings.TrimSpace(expr))
	if err != nil {
		log.Warn().Err(err).Str("cron", expr).Msg("Invalid cron expression, falling back to default '0 0,4 * * *'")
		sched, _ = cron.ParseStandard("0 0,4 * * *")
	}
	return sched.Next(now)
}

// StartAutoFetcher starts background automatic model fetching if enabled in config.
func StartAutoFetcher(ctx context.Context, httpClient *client.HTTPClient, cfg *config.Config) {
	if !cfg.AutoFetchModels {
		log.Info().Msg("Auto-fetch upstream models is disabled (AUTO_FETCH_MODELS=false)")
		return
	}

	cronExpr := strings.TrimSpace(cfg.AutoFetchCron)
	if cronExpr == "" {
		cronExpr = "0 0,4 * * *"
	}

	log.Info().Str("cron", cronExpr).Msg("Auto-fetch upstream models is enabled. Triggering initial fetch...")
	go func() {
		if _, err := UpdateCatalogFromUpstream(ctx, httpClient, cfg.VertexBaseURL, cfg.GraphQLAPIKey); err != nil && ctx.Err() == nil {
			log.Error().Str("err", config.UpstreamLogError(err, false, 512)).Msg("Initial auto-fetch of upstream models failed, keeping default in-memory catalog unchanged without retry")
		}

		for {
			now := time.Now()
			next := NextScheduleTimeWithCron(cronExpr, now)
			waitDuration := next.Sub(now)

			log.Info().
				Str("cron", cronExpr).
				Time("next_run", next).
				Dur("wait", waitDuration).
				Msg("Scheduled next upstream model fetch")

			timer := time.NewTimer(waitDuration)
			select {
			case <-ctx.Done():
				timer.Stop()
				log.Info().Msg("Auto-fetch scheduler stopped")
				return
			case <-timer.C:
				log.Info().Msg("Running scheduled upstream model fetch...")
				if _, err := UpdateCatalogFromUpstream(ctx, httpClient, cfg.VertexBaseURL, cfg.GraphQLAPIKey); err != nil && ctx.Err() == nil {
					log.Error().Str("err", config.UpstreamLogError(err, false, 512)).Msg("Scheduled auto-fetch of upstream models failed, keeping existing in-memory catalog unchanged without retry")
				}
			}
		}
	}()
}
