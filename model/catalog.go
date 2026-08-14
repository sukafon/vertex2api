package model

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
	"unicode"
)

const (
	defaultModelJSONName = "model.json"
	defaultCreatedAt     = int64(1700000000)
)

type ModelCatalog struct {
	openAIList         OpenAIModelListResponse
	geminiList         GeminiModelListResponse
	anthropicList      AnthropicModelListResponse
	responseModalities map[string]map[string]bool
	thinkingLevels     map[string]map[string]bool
	safetySettings     map[string]map[string]map[string]bool
}

type rawCatalogModel struct {
	ModelID               string                          `json:"modelId"`
	PublisherID           string                          `json:"publisherId"`
	ModelFamily           string                          `json:"modelFamily"`
	DisplayName           string                          `json:"displayName"`
	ShortDescription      string                          `json:"shortDescription"`
	ReleaseDate           string                          `json:"releaseDate"`
	IsTokenCountSupported bool                            `json:"isTokenCountSupported"`
	Parameters            []rawCatalogParameter           `json:"parameters"`
	TaskConfigs           []rawCatalogTaskConfig          `json:"taskConfigs"`
	SafetyFilterSettings  *rawCatalogSafetyFilterSettings `json:"safetyFilterSettings"`
	SupportedRegions      []string                        `json:"supportedRegions"`
}

type rawCatalogParameter struct {
	Name              string                  `json:"name"`
	NumericRangeValue *rawCatalogNumericRange `json:"numericRangeValue"`
	raw               json.RawMessage
}

func (p *rawCatalogParameter) UnmarshalJSON(data []byte) error {
	type knownFields struct {
		Name              string                  `json:"name"`
		NumericRangeValue *rawCatalogNumericRange `json:"numericRangeValue"`
	}
	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.UseNumber()
	var known knownFields
	if err := decoder.Decode(&known); err != nil {
		return err
	}
	p.Name = known.Name
	p.NumericRangeValue = known.NumericRangeValue
	p.raw = append(p.raw[:0], data...)
	return nil
}

type rawCatalogNumericRange struct {
	Value json.Number `json:"value"`
	Max   json.Number `json:"max"`
}

type rawCatalogTaskConfig struct {
	OutputConfig *rawCatalogOutputConfig `json:"outputConfig"`
}

type rawCatalogOutputConfig struct {
	SupportedContentTypes []rawCatalogContentType `json:"supportedContentTypes"`
}

type rawCatalogContentType struct {
	Type string `json:"type"`
}

type rawCatalogSafetyFilterSettings struct {
	Settings []rawCatalogSafetySetting `json:"settings"`
}

type rawCatalogSafetySetting struct {
	Value   string                          `json:"value"`
	Options []rawCatalogSafetySettingOption `json:"options"`
}

type rawCatalogSafetySettingOption struct {
	Value string `json:"value"`
}

var (
	catalogMu      sync.RWMutex
	defaultCatalog = loadDefaultCatalog()
)

func SetCatalog(catalog ModelCatalog) {
	catalogMu.Lock()
	defer catalogMu.Unlock()
	defaultCatalog = catalog
}

func OpenAIModelList() OpenAIModelListResponse {
	catalogMu.RLock()
	defer catalogMu.RUnlock()
	return defaultCatalog.openAIList
}

func GeminiModelList() GeminiModelListResponse {
	catalogMu.RLock()
	defer catalogMu.RUnlock()
	return defaultCatalog.geminiList
}

func AnthropicModelList() AnthropicModelListResponse {
	catalogMu.RLock()
	defer catalogMu.RUnlock()
	return defaultCatalog.anthropicList
}

// IsKnownModel reports whether modelID is present in the current in-memory
// catalog. Request handlers use this as the default model authorization
// boundary unless custom model names are explicitly enabled.
func IsKnownModel(modelID string) bool {
	modelID = strings.TrimSpace(modelID)
	if modelID == "" || strings.ContainsAny(modelID, `/\\`) || strings.Contains(modelID, "..") {
		return false
	}

	catalogMu.RLock()
	defer catalogMu.RUnlock()
	for _, item := range defaultCatalog.openAIList.Data {
		if item.ID == modelID {
			return true
		}
	}
	return false
}

func SanitizeGenerationConfigResponseModalities(modelName string, genConfig map[string]interface{}) {
	if genConfig == nil {
		return
	}

	catalogMu.RLock()
	allowed, ok := defaultCatalog.responseModalities[normalizeCatalogModelID(modelName)]
	catalogMu.RUnlock()
	if !ok {
		allowed = map[string]bool{"TEXT": true}
	}

	value, hasValue := genConfig["responseModalities"]

	var modalities []string
	if !hasValue {
		if !allowed["IMAGE"] {
			return
		}
		if allowed["TEXT"] {
			modalities = append(modalities, "TEXT")
		}
		modalities = append(modalities, "IMAGE")
	} else {
		modalities = sanitizeResponseModalities(value, allowed)
		if len(modalities) == 0 {
			if allowed["TEXT"] {
				modalities = []string{"TEXT"}
			} else {
				modalities = firstAllowedModality(allowed)
			}
		}

		if allowed["IMAGE"] {
			hasImage := false
			for _, m := range modalities {
				if m == "IMAGE" {
					hasImage = true
					break
				}
			}
			if !hasImage {
				modalities = append(modalities, "IMAGE")
			}
		}
	}

	if len(modalities) == 0 {
		modalities = []string{"TEXT"}
	}
	genConfig["responseModalities"] = modalities
}

// SupportedThinkingLevels returns the ordered thinking levels advertised by
// the current upstream-backed model catalog. The bool is false when the model
// list did not publish a thinking-level capability for this model.
func SupportedThinkingLevels(modelName string) ([]string, bool) {
	catalogMu.RLock()
	allowed, ok := defaultCatalog.thinkingLevels[normalizeCatalogModelID(modelName)]
	catalogMu.RUnlock()
	if !ok {
		return nil, false
	}
	levels := make([]string, 0, len(allowed))
	for _, level := range orderedThinkingLevels {
		if allowed[level] {
			levels = append(levels, level)
		}
	}
	return levels, true
}

func LowestSupportedThinkingLevel(modelName string) (string, bool) {
	levels, ok := SupportedThinkingLevels(modelName)
	if !ok || len(levels) == 0 {
		return "", false
	}
	return levels[0], true
}

// SanitizeGenerationConfigThinkingLevel clamps an explicitly requested level
// to the nearest level published by the upstream model list. Prefer promoting
// the request so an unsupported lower level does not cause a Vertex Code 3.
func SanitizeGenerationConfigThinkingLevel(modelName string, genConfig map[string]interface{}) {
	if genConfig == nil {
		return
	}
	thinkingConfig, ok := genConfig["thinkingConfig"].(map[string]interface{})
	if !ok {
		return
	}
	requested, ok := thinkingConfig["thinkingLevel"].(string)
	if !ok {
		return
	}
	requested = normalizeThinkingLevel(requested)
	if requested == "" {
		return
	}

	catalogMu.RLock()
	allowed, known := defaultCatalog.thinkingLevels[normalizeCatalogModelID(modelName)]
	catalogMu.RUnlock()
	if !known || len(allowed) == 0 {
		return
	}
	if allowed[requested] {
		thinkingConfig["thinkingLevel"] = requested
		return
	}
	if replacement := closestSupportedThinkingLevel(requested, allowed); replacement != "" {
		thinkingConfig["thinkingLevel"] = replacement
	}
}

func loadDefaultCatalog() ModelCatalog {
	catalog, err := loadModelCatalogFromFile(defaultModelJSONName)
	if err == nil {
		return catalog
	}
	return fallbackModelCatalog()
}

func loadModelCatalogFromFile(name string) (ModelCatalog, error) {
	path, err := findCatalogFile(name)
	if err != nil {
		return ModelCatalog{}, err
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return ModelCatalog{}, err
	}

	decoder := json.NewDecoder(strings.NewReader(string(data)))
	decoder.UseNumber()
	var models []rawCatalogModel
	if err := decoder.Decode(&models); err != nil {
		return ModelCatalog{}, err
	}
	return buildModelCatalog(models), nil
}

func findCatalogFile(name string) (string, error) {
	if filepath.IsAbs(name) {
		_, err := os.Stat(name)
		return name, err
	}

	var roots []string
	if cwd, err := os.Getwd(); err == nil {
		roots = append(roots, cwd)
	}
	if exe, err := os.Executable(); err == nil {
		roots = append(roots, filepath.Dir(exe))
	}

	seen := make(map[string]bool, len(roots))
	for _, root := range roots {
		if seen[root] {
			continue
		}
		seen[root] = true
		path := filepath.Join(root, name)
		if _, err := os.Stat(path); err == nil {
			return path, nil
		}
	}

	return "", os.ErrNotExist
}

func buildModelCatalog(models []rawCatalogModel) ModelCatalog {
	openAIModels := make([]OpenAIModelInfo, 0, len(models))
	geminiModels := make([]GeminiModelInfo, 0, len(models))
	anthropicModels := make([]AnthropicModelInfo, 0, len(models))
	responseModalities := make(map[string]map[string]bool, len(models))
	thinkingLevels := make(map[string]map[string]bool, len(models))
	safetySettings := make(map[string]map[string]map[string]bool, len(models))

	for _, item := range models {
		modelID := strings.TrimSpace(item.ModelID)
		if modelID == "" {
			continue
		}

		publisher := strings.TrimSpace(item.PublisherID)
		if publisher == "" {
			publisher = "google"
		}

		openAIModels = append(openAIModels, OpenAIModelInfo{
			ID:      modelID,
			Object:  "model",
			Created: releaseDateUnix(item.ReleaseDate),
			OwnedBy: publisher,
		})
		anthropicModels = append(anthropicModels, AnthropicModelInfo{
			ID:          modelID,
			Type:        "model",
			DisplayName: displayNameForModel(item),
			CreatedAt:   releaseDateRFC3339(item.ReleaseDate),
		})

		geminiInfo := GeminiModelInfo{
			Name:                       "models/" + modelID,
			DisplayName:                displayNameForModel(item),
			Description:                descriptionForModel(item),
			SupportedGenerationMethods: generationMethodsForModel(),
			Version:                    versionForModelID(modelID),
		}
		if strings.HasPrefix(modelID, "gemini-") {
			geminiInfo.OutputTokenLimit = outputTokenLimitForModel(item)
		}
		geminiModels = append(geminiModels, geminiInfo)

		modalities := responseModalitiesForModel(item)
		responseModalities[normalizeCatalogModelID(modelID)] = modalities
		if levels, published := thinkingLevelsForModel(item); published {
			thinkingLevels[normalizeCatalogModelID(modelID)] = levels
		}

		sSettings := safetySettingsForModel(item)
		safetySettings[normalizeCatalogModelID(modelID)] = sSettings

	}

	return ModelCatalog{
		openAIList: OpenAIModelListResponse{
			Object: "list",
			Data:   openAIModels,
		},
		anthropicList: AnthropicModelListResponse{
			Data:    anthropicModels,
			HasMore: false,
			FirstID: firstAnthropicModelID(anthropicModels),
			LastID:  lastAnthropicModelID(anthropicModels),
		},
		geminiList: GeminiModelListResponse{
			Models: geminiModels,
		},
		responseModalities: responseModalities,
		thinkingLevels:     thinkingLevels,
		safetySettings:     safetySettings,
	}
}

func releaseDateUnix(value string) int64 {
	value = strings.TrimSpace(value)
	if value == "" {
		return defaultCreatedAt
	}
	t, err := time.Parse("2006-01-02", value)
	if err != nil {
		return defaultCreatedAt
	}
	return t.Unix()
}

func releaseDateRFC3339(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return time.Unix(defaultCreatedAt, 0).UTC().Format(time.RFC3339)
	}
	t, err := time.Parse("2006-01-02", value)
	if err != nil {
		return time.Unix(defaultCreatedAt, 0).UTC().Format(time.RFC3339)
	}
	return t.UTC().Format(time.RFC3339)
}

func firstAnthropicModelID(models []AnthropicModelInfo) string {
	if len(models) == 0 {
		return ""
	}
	return models[0].ID
}

func lastAnthropicModelID(models []AnthropicModelInfo) string {
	if len(models) == 0 {
		return ""
	}
	return models[len(models)-1].ID
}

func displayNameForModel(item rawCatalogModel) string {
	if strings.TrimSpace(item.DisplayName) != "" {
		return strings.TrimSpace(item.DisplayName)
	}
	return titleizeModelID(item.ModelID)
}

func descriptionForModel(item rawCatalogModel) string {
	if strings.TrimSpace(item.ShortDescription) != "" {
		return strings.TrimSpace(item.ShortDescription)
	}
	return displayNameForModel(item)
}

func generationMethodsForModel() []string {
	return []string{"generateContent", "streamGenerateContent"}
}

func outputTokenLimitForModel(item rawCatalogModel) int {
	for _, parameter := range item.Parameters {
		if parameter.Name != "max_output_tokens" || parameter.NumericRangeValue == nil {
			continue
		}
		if limit := numberToInt(parameter.NumericRangeValue.Max); limit > 0 {
			return normalizeTokenLimit(limit)
		}
		if limit := numberToInt(parameter.NumericRangeValue.Value); limit > 0 {
			return normalizeTokenLimit(limit)
		}
	}
	return 0
}

func numberToInt(number json.Number) int {
	if number == "" {
		return 0
	}
	if value, err := number.Int64(); err == nil {
		return int(value)
	}
	if value, err := strconv.ParseFloat(number.String(), 64); err == nil {
		return int(value)
	}
	return 0
}

func normalizeTokenLimit(limit int) int {
	if limit == 65535 {
		return 65536
	}
	return limit
}

func responseModalitiesForModel(item rawCatalogModel) map[string]bool {
	modalities := make(map[string]bool, 2)
	hasOutputConfig := false

	for _, task := range item.TaskConfigs {
		if task.OutputConfig == nil {
			continue
		}
		hasOutputConfig = true
		for _, contentType := range task.OutputConfig.SupportedContentTypes {
			switch strings.ToLower(strings.TrimSpace(contentType.Type)) {
			case "text":
				modalities["TEXT"] = true
			case "image":
				modalities["IMAGE"] = true
			}
		}
	}

	if !hasOutputConfig || len(modalities) == 0 {
		modalities["TEXT"] = true
	}
	return modalities
}

var orderedThinkingLevels = []string{"MINIMAL", "LOW", "MEDIUM", "HIGH"}

func thinkingLevelsForModel(item rawCatalogModel) (map[string]bool, bool) {
	for _, parameter := range item.Parameters {
		name := normalizeCatalogParameterName(parameter.Name)
		if name != "thinkinglevel" && name != "reasoninglevel" {
			continue
		}
		levels := make(map[string]bool, len(orderedThinkingLevels))
		if len(parameter.raw) > 0 {
			var value interface{}
			if err := json.Unmarshal(parameter.raw, &value); err == nil {
				collectThinkingLevels(value, levels)
			}
		}
		return levels, true
	}
	return nil, false
}

func collectThinkingLevels(value interface{}, levels map[string]bool) {
	switch typed := value.(type) {
	case string:
		if level := normalizeThinkingLevel(typed); level != "" {
			levels[level] = true
		}
	case []interface{}:
		for _, item := range typed {
			collectThinkingLevels(item, levels)
		}
	case map[string]interface{}:
		for _, item := range typed {
			collectThinkingLevels(item, levels)
		}
	}
}

func normalizeCatalogParameterName(value string) string {
	return strings.Map(func(r rune) rune {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			return unicode.ToLower(r)
		}
		return -1
	}, value)
}

func normalizeThinkingLevel(value string) string {
	value = strings.ToUpper(strings.TrimSpace(value))
	value = strings.TrimPrefix(value, "THINKING_LEVEL_")
	for _, level := range orderedThinkingLevels {
		if value == level {
			return level
		}
	}
	return ""
}

func closestSupportedThinkingLevel(requested string, allowed map[string]bool) string {
	requestedIndex := -1
	for index, level := range orderedThinkingLevels {
		if level == requested {
			requestedIndex = index
			break
		}
	}
	if requestedIndex < 0 {
		return ""
	}
	for index := requestedIndex + 1; index < len(orderedThinkingLevels); index++ {
		if allowed[orderedThinkingLevels[index]] {
			return orderedThinkingLevels[index]
		}
	}
	for index := requestedIndex - 1; index >= 0; index-- {
		if allowed[orderedThinkingLevels[index]] {
			return orderedThinkingLevels[index]
		}
	}
	return ""
}

func sanitizeResponseModalities(value interface{}, allowed map[string]bool) []string {
	requested := responseModalitiesFromValue(value)
	if len(requested) == 0 {
		return nil
	}

	seen := make(map[string]bool, len(requested))
	modalities := make([]string, 0, len(requested))
	for _, modality := range requested {
		modality = strings.ToUpper(strings.TrimSpace(modality))
		if modality == "" || seen[modality] || !allowed[modality] {
			continue
		}
		seen[modality] = true
		modalities = append(modalities, modality)
	}
	return modalities
}

func responseModalitiesFromValue(value interface{}) []string {
	switch v := value.(type) {
	case []string:
		return append([]string(nil), v...)
	case []interface{}:
		modalities := make([]string, 0, len(v))
		for _, item := range v {
			if modality, ok := item.(string); ok {
				modalities = append(modalities, modality)
			}
		}
		return modalities
	case string:
		return []string{v}
	default:
		return nil
	}
}

func firstAllowedModality(allowed map[string]bool) []string {
	modalities := make([]string, 0, len(allowed))
	for modality := range allowed {
		modalities = append(modalities, modality)
	}
	sort.Strings(modalities)
	if len(modalities) == 0 {
		return nil
	}
	return modalities[:1]
}

func normalizeCatalogModelID(modelID string) string {
	modelID = strings.TrimSpace(modelID)
	modelID = strings.TrimPrefix(modelID, "models/")
	if index := strings.LastIndex(modelID, "/"); index >= 0 {
		modelID = modelID[index+1:]
	}
	if index := strings.Index(modelID, ":"); index >= 0 {
		modelID = modelID[:index]
	}
	return modelID
}

func titleizeModelID(modelID string) string {
	words := strings.FieldsFunc(modelID, func(r rune) bool {
		return r == '-' || r == '_' || r == '/'
	})
	for i, word := range words {
		words[i] = titleizeWord(word)
	}
	return strings.Join(words, " ")
}

func titleizeWord(word string) string {
	if word == "" {
		return word
	}
	runes := []rune(word)
	if len(runes) == 0 {
		return word
	}
	if unicode.IsLetter(runes[0]) {
		runes[0] = unicode.ToUpper(runes[0])
	}
	return string(runes)
}

func versionForModelID(modelID string) string {
	parts := strings.Split(modelID, "-")
	if len(parts) < 2 || parts[0] != "gemini" {
		return modelID
	}

	version := parts[1]
	if _, err := strconv.Atoi(version); err == nil {
		return version + ".0"
	}
	return version
}

func fallbackModelCatalog() ModelCatalog {
	models := []rawCatalogModel{
		fallbackRawModel("gemini-3-pro-image-preview", true),
		fallbackRawModel("gemini-3-pro-image", true),
		fallbackRawModel("gemini-3.1-flash-image", true),
		fallbackRawModel("gemini-3.1-flash-image-preview", true),
		fallbackRawModel("gemini-3.1-pro-preview", false),
		fallbackRawModel("gemini-3.5-flash", false),
		fallbackRawModel("gemini-3-flash-preview", false),
		fallbackRawModel("gemini-2.5-flash-image", true),
	}
	catalog := buildModelCatalog(models)
	for i, version := range []string{"3.0-preview", "3.0", "3.1", "3.1-preview", "3.1-preview", "3.5", "3.0-preview", "2.5"} {
		catalog.geminiList.Models[i].Version = version
	}
	return catalog
}

func fallbackRawModel(modelID string, imageOutput bool) rawCatalogModel {
	outputConfig := (*rawCatalogOutputConfig)(nil)
	if imageOutput {
		outputConfig = &rawCatalogOutputConfig{
			SupportedContentTypes: []rawCatalogContentType{
				{Type: "image"},
				{Type: "text"},
			},
		}
	}
	return rawCatalogModel{
		ModelID:               modelID,
		PublisherID:           "google",
		ShortDescription:      titleizeModelID(modelID) + " Model",
		ReleaseDate:           "2023-11-14",
		IsTokenCountSupported: true,
		Parameters: []rawCatalogParameter{
			{
				Name: "max_output_tokens",
				NumericRangeValue: &rawCatalogNumericRange{
					Max: json.Number("8192"),
				},
			},
		},
		TaskConfigs: []rawCatalogTaskConfig{
			{OutputConfig: outputConfig},
		},
	}
}

func safetySettingsForModel(item rawCatalogModel) map[string]map[string]bool {
	if item.SafetyFilterSettings == nil {
		return nil
	}
	res := make(map[string]map[string]bool)
	for _, s := range item.SafetyFilterSettings.Settings {
		opts := make(map[string]bool)
		for _, o := range s.Options {
			opts[o.Value] = true
		}
		res[s.Value] = opts
	}
	return res
}

func SanitizeSafetySettings(modelName string, safetySettings []map[string]string) []map[string]string {
	if len(safetySettings) == 0 {
		return safetySettings
	}
	catalogMu.RLock()
	allowed, ok := defaultCatalog.safetySettings[normalizeCatalogModelID(modelName)]
	catalogMu.RUnlock()
	if !ok || allowed == nil {
		return safetySettings
	}

	var result []map[string]string
	for _, s := range safetySettings {
		cat, okCat := s["category"]
		threshold, okThresh := s["threshold"]
		if !okCat || !okThresh {
			result = append(result, s)
			continue
		}

		matchedCat := matchSafetyCategory(cat, allowed)
		if matchedCat == "" {
			matchedCat = cat
		}

		matchedThresh := matchSafetyThreshold(threshold, allowed[matchedCat])
		if matchedThresh == "" {
			matchedThresh = threshold
		}

		newSetting := make(map[string]string)
		for k, v := range s {
			newSetting[k] = v
		}
		newSetting["category"] = matchedCat
		newSetting["threshold"] = matchedThresh
		result = append(result, newSetting)
	}
	return result
}

func matchSafetyCategory(cat string, allowed map[string]map[string]bool) string {
	if _, exists := allowed[cat]; exists {
		return cat
	}
	normReq := normalizeSafetyValue(cat)
	for allowedCat := range allowed {
		if normalizeSafetyValue(allowedCat) == normReq {
			return allowedCat
		}
	}
	return ""
}

func matchSafetyThreshold(threshold string, allowedThresh map[string]bool) string {
	if allowedThresh == nil {
		return ""
	}
	if allowedThresh[threshold] {
		return threshold
	}
	if threshold == "BLOCK_NONE" && allowedThresh["OFF"] {
		return "OFF"
	}
	if threshold == "OFF" && allowedThresh["BLOCK_NONE"] {
		return "BLOCK_NONE"
	}
	normReq := normalizeSafetyValue(threshold)
	for allowedT := range allowedThresh {
		if normalizeSafetyValue(allowedT) == normReq {
			return allowedT
		}
	}
	return ""
}

func normalizeSafetyValue(s string) string {
	s = strings.ToUpper(s)
	s = strings.TrimPrefix(s, "HARM_CATEGORY_")
	s = strings.TrimSuffix(s, "_CONTENT")
	return s
}
