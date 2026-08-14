package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestLoadModelCatalogFromOptionalJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "model.json")
	data := []byte(`[{
        "modelId":"gemini-image-test",
        "publisherId":"google",
        "modelFamily":"gemini",
        "releaseDate":"2026-06-18",
		"parameters":[{"name":"thinking_level","enumValue":{"options":[{"value":"LOW"},{"value":"MEDIUM"},{"value":"HIGH"}]}}],
        "taskConfigs":[{"outputConfig":{"supportedContentTypes":[{"type":"text"},{"type":"image"}]}}]
    }]`)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatalf("write model catalog fixture: %v", err)
	}

	catalog, err := loadModelCatalogFromFile(path)
	if err != nil {
		t.Fatalf("loadModelCatalogFromFile returned error: %v", err)
	}

	var found bool
	for _, item := range catalog.openAIList.Data {
		if item.ID == "gemini-image-test" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("catalog loaded from model.json should include fixture model")
	}
	if !catalog.responseModalities["gemini-image-test"]["IMAGE"] {
		t.Fatal("fixture model should support IMAGE output")
	}
	levels := catalog.thinkingLevels["gemini-image-test"]
	if len(levels) != 3 || !levels["LOW"] || !levels["MEDIUM"] || !levels["HIGH"] || levels["MINIMAL"] {
		t.Fatalf("fixture thinking levels = %#v, want LOW/MEDIUM/HIGH", levels)
	}
}

func TestCatalogThinkingLevelSelectionAndSanitization(t *testing.T) {
	catalogMu.RLock()
	originalCatalog := defaultCatalog
	catalogMu.RUnlock()
	t.Cleanup(func() { SetCatalog(originalCatalog) })

	parameterJSON := json.RawMessage(`{"name":"thinkingLevel","enumValue":{"values":["THINKING_LEVEL_LOW","MEDIUM","HIGH"]}}`)
	SetCatalog(buildModelCatalog([]rawCatalogModel{{
		ModelID: "gemini-catalog-thinking",
		Parameters: []rawCatalogParameter{{
			Name: "thinkingLevel",
			raw:  parameterJSON,
		}},
	}}))

	levels, ok := SupportedThinkingLevels("models/gemini-catalog-thinking")
	if !ok || len(levels) != 3 || levels[0] != "LOW" || levels[1] != "MEDIUM" || levels[2] != "HIGH" {
		t.Fatalf("SupportedThinkingLevels = %v, %v", levels, ok)
	}
	if lowest, ok := LowestSupportedThinkingLevel("gemini-catalog-thinking"); !ok || lowest != "LOW" {
		t.Fatalf("LowestSupportedThinkingLevel = %q, %v, want LOW, true", lowest, ok)
	}

	config := map[string]interface{}{"thinkingConfig": map[string]interface{}{"thinkingLevel": "minimal"}}
	SanitizeGenerationConfigThinkingLevel("gemini-catalog-thinking", config)
	if got := config["thinkingConfig"].(map[string]interface{})["thinkingLevel"]; got != "LOW" {
		t.Fatalf("sanitized thinkingLevel = %v, want LOW", got)
	}

	unknown := map[string]interface{}{"thinkingConfig": map[string]interface{}{"thinkingLevel": "MINIMAL"}}
	SanitizeGenerationConfigThinkingLevel("gemini-not-in-catalog", unknown)
	if got := unknown["thinkingConfig"].(map[string]interface{})["thinkingLevel"]; got != "MINIMAL" {
		t.Fatalf("unknown model thinkingLevel = %v, want unchanged MINIMAL", got)
	}
}

func TestBuildModelCatalogBuildsListsFromRawModels(t *testing.T) {
	catalog := buildModelCatalog([]rawCatalogModel{
		{
			ModelID:               "gemini-test",
			PublisherID:           "google",
			ShortDescription:      "test model",
			ReleaseDate:           "2026-06-18",
			IsTokenCountSupported: true,
			Parameters: []rawCatalogParameter{
				{
					Name: "max_output_tokens",
					NumericRangeValue: &rawCatalogNumericRange{
						Max: json.Number("65535"),
					},
				},
			},
			TaskConfigs: []rawCatalogTaskConfig{
				{
					OutputConfig: &rawCatalogOutputConfig{
						SupportedContentTypes: []rawCatalogContentType{
							{Type: "text"},
						},
					},
				},
			},
		},
	})

	if len(catalog.openAIList.Data) != 1 {
		t.Fatalf("OpenAI models length = %d, want 1", len(catalog.openAIList.Data))
	}
	if got := catalog.openAIList.Data[0].ID; got != "gemini-test" {
		t.Fatalf("OpenAI model id = %q, want gemini-test", got)
	}
	if got := catalog.geminiList.Models[0].OutputTokenLimit; got != 65536 {
		t.Fatalf("output token limit = %d, want 65536", got)
	}
	if got := catalog.geminiList.Models[0].InputTokenLimit; got != 0 {
		t.Fatalf("input token limit = %d, want 0 so the field is omitted", got)
	}
	gotMethods := catalog.geminiList.Models[0].SupportedGenerationMethods
	if len(gotMethods) != 2 || gotMethods[0] != "generateContent" || gotMethods[1] != "streamGenerateContent" {
		t.Fatalf("supported methods = %v, want [generateContent streamGenerateContent]", gotMethods)
	}
	if catalog.responseModalities["gemini-test"]["IMAGE"] {
		t.Fatal("text-only model should not support IMAGE output")
	}
}

func TestSanitizeGenerationConfigResponseModalitiesRemovesUnsupportedImage(t *testing.T) {
	genConfig := map[string]interface{}{
		"responseModalities": []interface{}{"TEXT", "IMAGE"},
	}

	SanitizeGenerationConfigResponseModalities("gemini-2.5-pro", genConfig)

	got := genConfig["responseModalities"].([]string)
	if len(got) != 1 || got[0] != "TEXT" {
		t.Fatalf("responseModalities = %v, want [TEXT]", got)
	}
}

func TestSanitizeGenerationConfigResponseModalitiesKeepsSupportedImage(t *testing.T) {
	genConfig := map[string]interface{}{
		"responseModalities": []interface{}{"TEXT", "IMAGE"},
	}

	SanitizeGenerationConfigResponseModalities("gemini-3-pro-image", genConfig)

	got := genConfig["responseModalities"].([]string)
	if len(got) != 2 || got[0] != "TEXT" || got[1] != "IMAGE" {
		t.Fatalf("responseModalities = %v, want [TEXT IMAGE]", got)
	}
}

func TestSanitizeGenerationConfigResponseModalitiesAddsSupportedImage(t *testing.T) {
	genConfig := map[string]interface{}{
		"responseModalities": []interface{}{"TEXT"},
	}
	SanitizeGenerationConfigResponseModalities("gemini-3-pro-image", genConfig)
	got := genConfig["responseModalities"].([]string)
	if len(got) != 2 || got[0] != "TEXT" || got[1] != "IMAGE" {
		t.Fatalf("responseModalities = %v, want [TEXT IMAGE]", got)
	}
}

func TestSanitizeGenerationConfigResponseModalitiesFillsMissing(t *testing.T) {
	genConfig := map[string]interface{}{}
	SanitizeGenerationConfigResponseModalities("gemini-3-pro-image", genConfig)
	got := genConfig["responseModalities"].([]string)
	if len(got) != 2 || got[0] != "TEXT" || got[1] != "IMAGE" {
		t.Fatalf("responseModalities = %v, want [TEXT IMAGE]", got)
	}
}

func TestAcquireVertexRequestHandlesNilVariables(t *testing.T) {
	req := AcquireVertexRequest()
	req.Variables = nil
	ReleaseVertexRequest(req)

	req2 := AcquireVertexRequest()
	defer ReleaseVertexRequest(req2)
	if req2.Variables == nil {
		t.Fatal("AcquireVertexRequest should initialize req.Variables if nil")
	}
	req2.Variables["test"] = "val"
}
