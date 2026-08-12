package model

import (
	"encoding/json"
	"strings"
)

// ==================== Vertex AI GraphQL Request ====================

type VertexRequest struct {
	QuerySignature string                 `json:"querySignature"`
	OperationName  string                 `json:"operationName"`
	Variables      map[string]interface{} `json:"variables"`
}

// ==================== Vertex AI Response ====================

// 顶层响应是一个数组: []VertexResponseElement
type VertexResponseElement struct {
	Results []VertexResult `json:"results"`
}

type VertexResult struct {
	Errors []VertexError `json:"errors,omitempty"`
	Data   *VertexData   `json:"data,omitempty"`
}

type VertexError struct {
	Message    string           `json:"message"`
	Extensions *VertexExtension `json:"extensions,omitempty"`
}

type VertexExtension struct {
	Status *VertexStatus `json:"status,omitempty"`
}

type VertexStatus struct {
	Code int `json:"code"`
}

type VertexData struct {
	Candidates     []VertexCandidate      `json:"candidates,omitempty"`
	UI             *VertexUI              `json:"ui,omitempty"`
	UsageMetadata  map[string]interface{} `json:"usageMetadata,omitempty"`
	ModelVersion   string                 `json:"modelVersion,omitempty"`
	ResponseID     string                 `json:"responseId,omitempty"`
	CreateTime     string                 `json:"createTime,omitempty"`
	PromptFeedback map[string]interface{} `json:"promptFeedback,omitempty"`
	ModelStatus    map[string]interface{} `json:"modelStatus,omitempty"`
}

// VertexUI covers the newer anonymous GraphQL response envelope. Google may
// return streamGenerateContentAnonymous as either one response object or an
// array of response objects, so it intentionally remains raw until the proxy
// expands every item.
type VertexUI struct {
	StreamGenerateContentAnonymous json.RawMessage `json:"streamGenerateContentAnonymous,omitempty"`
}

type VertexCandidate struct {
	Index              int                      `json:"index,omitempty"`
	FinishReason       string                   `json:"finishReason,omitempty"`
	FinishMessage      string                   `json:"finishMessage,omitempty"`
	Content            *VertexContent           `json:"content,omitempty"`
	GroundingMetadata  *GroundingMetadata       `json:"groundingMetadata,omitempty"`
	SafetyRatings      []map[string]interface{} `json:"safetyRatings,omitempty"`
	CitationMetadata   map[string]interface{}   `json:"citationMetadata,omitempty"`
	URLContextMetadata map[string]interface{}   `json:"urlContextMetadata,omitempty"`
	LogprobsResult     map[string]interface{}   `json:"logprobsResult,omitempty"`
	AvgLogprobs        *float64                 `json:"avgLogprobs,omitempty"`
}

type GroundingMetadata struct {
	WebSearchQueries             []string                 `json:"webSearchQueries,omitempty"`
	SearchEntryPoint             *SearchEntryPoint        `json:"searchEntryPoint,omitempty"`
	GroundingChunks              []GroundingChunk         `json:"groundingChunks,omitempty"`
	GroundingSupports            []GroundingSupport       `json:"groundingSupports,omitempty"`
	RetrievalQueries             []string                 `json:"retrievalQueries,omitempty"`
	SourceFlaggingURIs           []map[string]interface{} `json:"sourceFlaggingUris,omitempty"`
	RetrievalMetadata            map[string]interface{}   `json:"retrievalMetadata,omitempty"`
	GoogleMapsWidgetContextToken string                   `json:"googleMapsWidgetContextToken,omitempty"`
}

type SearchEntryPoint struct {
	RenderedContent string `json:"renderedContent,omitempty"`
	SDKBlob         string `json:"sdkBlob,omitempty"`
}

type GroundingChunk struct {
	Web              *WebChunk              `json:"web,omitempty"`
	RetrievedContext map[string]interface{} `json:"retrievedContext,omitempty"`
	Maps             map[string]interface{} `json:"maps,omitempty"`
}

type WebChunk struct {
	URI    string `json:"uri,omitempty"`
	Title  string `json:"title,omitempty"`
	Domain string `json:"domain,omitempty"`
}

type GroundingSupport struct {
	GroundingChunkIndices []int     `json:"groundingChunkIndices,omitempty"`
	ConfidenceScores      []float64 `json:"confidenceScores,omitempty"`
	Segment               *Segment  `json:"segment,omitempty"`
}

type Segment struct {
	PartIndex  int    `json:"partIndex,omitempty"`
	StartIndex int    `json:"startIndex,omitempty"`
	EndIndex   int    `json:"endIndex,omitempty"`
	Text       string `json:"text,omitempty"`
}

// NormalizeGroundingMetadata removes default-initialized protobuf fields while
// retaining metadata values that carry actual grounding information.
func NormalizeGroundingMetadata(source *GroundingMetadata) *GroundingMetadata {
	if source == nil {
		return nil
	}
	result := &GroundingMetadata{
		WebSearchQueries:             nonBlankMetadataStrings(source.WebSearchQueries),
		RetrievalQueries:             nonBlankMetadataStrings(source.RetrievalQueries),
		GoogleMapsWidgetContextToken: source.GoogleMapsWidgetContextToken,
	}
	if strings.TrimSpace(result.GoogleMapsWidgetContextToken) == "" {
		result.GoogleMapsWidgetContextToken = ""
	}
	if searchEntryPointHasContent(source.SearchEntryPoint) {
		entry := *source.SearchEntryPoint
		result.SearchEntryPoint = &entry
	}
	for _, chunk := range source.GroundingChunks {
		if normalized, ok := normalizeGroundingChunk(chunk); ok {
			result.GroundingChunks = append(result.GroundingChunks, normalized)
		}
	}
	for _, support := range source.GroundingSupports {
		if normalized, ok := normalizeGroundingSupport(support); ok {
			result.GroundingSupports = append(result.GroundingSupports, normalized)
		}
	}
	for _, sourceURI := range source.SourceFlaggingURIs {
		if metadataMapHasContent(sourceURI) {
			result.SourceFlaggingURIs = append(result.SourceFlaggingURIs, sourceURI)
		}
	}
	if metadataMapHasContent(source.RetrievalMetadata) {
		result.RetrievalMetadata = source.RetrievalMetadata
	}
	if len(result.WebSearchQueries) == 0 && result.SearchEntryPoint == nil &&
		len(result.GroundingChunks) == 0 && len(result.GroundingSupports) == 0 &&
		len(result.RetrievalQueries) == 0 && len(result.SourceFlaggingURIs) == 0 &&
		len(result.RetrievalMetadata) == 0 && result.GoogleMapsWidgetContextToken == "" {
		return nil
	}
	return result
}

func GroundingMetadataHasContent(source *GroundingMetadata) bool {
	return NormalizeGroundingMetadata(source) != nil
}

// NormalizeCitationMetadata translates Vertex's citations field to the Gemini
// Developer API citationSources field and removes default-initialized entries.
func NormalizeCitationMetadata(source map[string]interface{}) map[string]interface{} {
	if len(source) == 0 {
		return nil
	}
	value, ok := source["citationSources"]
	if !ok {
		value, ok = source["citations"]
	}
	if !ok {
		return nil
	}
	items := metadataSliceValues(value)
	citationSources := make([]interface{}, 0, len(items))
	for _, item := range items {
		citation, ok := item.(map[string]interface{})
		if !ok {
			continue
		}
		if normalized := normalizeCitationSource(citation); normalized != nil {
			citationSources = append(citationSources, normalized)
		}
	}
	if len(citationSources) == 0 {
		return nil
	}
	return map[string]interface{}{"citationSources": citationSources}
}

func CitationMetadataHasContent(source map[string]interface{}) bool {
	return len(NormalizeCitationMetadata(source)) > 0
}

func nonBlankMetadataStrings(source []string) []string {
	result := make([]string, 0, len(source))
	for _, value := range source {
		if strings.TrimSpace(value) != "" {
			result = append(result, value)
		}
	}
	return result
}

func searchEntryPointHasContent(source *SearchEntryPoint) bool {
	return source != nil &&
		(strings.TrimSpace(source.RenderedContent) != "" || strings.TrimSpace(source.SDKBlob) != "")
}

func normalizeGroundingChunk(source GroundingChunk) (GroundingChunk, bool) {
	result := GroundingChunk{}
	arms := 0
	if source.Web != nil && (strings.TrimSpace(source.Web.URI) != "" ||
		strings.TrimSpace(source.Web.Title) != "" || strings.TrimSpace(source.Web.Domain) != "") {
		web := *source.Web
		result.Web = &web
		arms++
	}
	if metadataMapHasContent(source.RetrievedContext) {
		result.RetrievedContext = source.RetrievedContext
		arms++
	}
	if metadataMapHasContent(source.Maps) {
		result.Maps = source.Maps
		arms++
	}
	return result, arms == 1
}

func normalizeGroundingSupport(source GroundingSupport) (GroundingSupport, bool) {
	result := GroundingSupport{
		GroundingChunkIndices: source.GroundingChunkIndices,
		ConfidenceScores:      source.ConfidenceScores,
	}
	if segmentHasContent(source.Segment) {
		segment := *source.Segment
		result.Segment = &segment
	}
	return result, len(result.GroundingChunkIndices) > 0 || len(result.ConfidenceScores) > 0 || result.Segment != nil
}

func segmentHasContent(source *Segment) bool {
	return source != nil && (source.PartIndex != 0 || source.StartIndex != 0 ||
		source.EndIndex != 0 || strings.TrimSpace(source.Text) != "")
}

func normalizeCitationSource(source map[string]interface{}) map[string]interface{} {
	result := make(map[string]interface{}, 4)
	for _, key := range []string{"uri", "license"} {
		if value, ok := source[key].(string); ok && strings.TrimSpace(value) != "" {
			result[key] = value
		}
	}
	start, hasStart := metadataNumber(source["startIndex"])
	end, hasEnd := metadataNumber(source["endIndex"])
	if hasEnd && end > 0 && (!hasStart || start >= 0 && end > start) {
		if hasStart && start > 0 {
			result["startIndex"] = source["startIndex"]
		}
		result["endIndex"] = source["endIndex"]
	}
	if len(result) == 0 {
		return nil
	}
	return result
}

func metadataNumber(value interface{}) (float64, bool) {
	switch number := value.(type) {
	case float64:
		return number, true
	case float32:
		return float64(number), true
	case int:
		return float64(number), true
	case int32:
		return float64(number), true
	case int64:
		return float64(number), true
	case uint:
		return float64(number), true
	case uint32:
		return float64(number), true
	case uint64:
		return float64(number), true
	case json.Number:
		parsed, err := number.Float64()
		return parsed, err == nil
	default:
		return 0, false
	}
}

func metadataSliceValues(value interface{}) []interface{} {
	switch items := value.(type) {
	case []interface{}:
		return items
	case []map[string]interface{}:
		result := make([]interface{}, len(items))
		for i := range items {
			result[i] = items[i]
		}
		return result
	default:
		return nil
	}
}

func metadataMapHasContent(source map[string]interface{}) bool {
	for _, value := range source {
		if metadataValueHasContent(value) {
			return true
		}
	}
	return false
}

func metadataValueHasContent(value interface{}) bool {
	switch typed := value.(type) {
	case nil:
		return false
	case string:
		return strings.TrimSpace(typed) != ""
	case map[string]interface{}:
		return metadataMapHasContent(typed)
	case []interface{}:
		for _, item := range typed {
			if metadataValueHasContent(item) {
				return true
			}
		}
		return false
	case []map[string]interface{}:
		for _, item := range typed {
			if metadataMapHasContent(item) {
				return true
			}
		}
		return false
	case []string:
		return len(nonBlankMetadataStrings(typed)) > 0
	default:
		// Numeric zero and false may be meaningful metadata values. Presence of a
		// typed scalar is therefore sufficient; only containers and strings use
		// semantic emptiness checks above.
		return true
	}
}

type VertexContent struct {
	Parts []VertexPart `json:"parts"`
	Role  string       `json:"role,omitempty"`
}

type VertexPart struct {
	Text                string                 `json:"text,omitempty"`
	InlineData          *InlineData            `json:"inlineData,omitempty"`
	FileData            map[string]interface{} `json:"fileData,omitempty"`
	FunctionCall        *FunctionCall          `json:"functionCall,omitempty"`
	FunctionResponse    *FunctionResponse      `json:"functionResponse,omitempty"`
	ExecutableCode      map[string]interface{} `json:"executableCode,omitempty"`
	CodeExecutionResult map[string]interface{} `json:"codeExecutionResult,omitempty"`
	VideoMetadata       map[string]interface{} `json:"videoMetadata,omitempty"`
	MediaResolution     map[string]interface{} `json:"mediaResolution,omitempty"`
	Thought             bool                   `json:"thought,omitempty"`
	ThoughtSignature    string                 `json:"thoughtSignature,omitempty"`
}

type InlineData struct {
	MimeType    string `json:"mimeType"`
	Data        string `json:"data"`
	DisplayName string `json:"displayName,omitempty"`
}

type TextPart struct {
	Text             string `json:"text"`
	Thought          bool   `json:"thought,omitempty"`
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
}

type FunctionCall struct {
	ID               string                   `json:"id,omitempty"`
	Name             string                   `json:"name,omitempty"`
	Args             map[string]interface{}   `json:"args,omitempty"`
	PartialArgs      []map[string]interface{} `json:"partialArgs,omitempty"`
	WillContinue     *bool                    `json:"willContinue,omitempty"`
	ThoughtSignature string                   `json:"thoughtSignature,omitempty"`
}

type FunctionResponse struct {
	ID       string                 `json:"id,omitempty"`
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response,omitempty"`
	Parts    []FunctionResponsePart `json:"parts,omitempty"`
}

type FunctionResponsePart struct {
	InlineData *InlineData            `json:"inlineData,omitempty"`
	FileData   map[string]interface{} `json:"fileData,omitempty"`
}
