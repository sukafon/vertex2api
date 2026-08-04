package model

import "encoding/json"

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
	WebSearchQueries  []string           `json:"webSearchQueries,omitempty"`
	SearchEntryPoint  *SearchEntryPoint  `json:"searchEntryPoint,omitempty"`
	GroundingChunks   []GroundingChunk   `json:"groundingChunks,omitempty"`
	GroundingSupports []GroundingSupport `json:"groundingSupports,omitempty"`
	RetrievalQueries  []string           `json:"retrievalQueries,omitempty"`
}

type SearchEntryPoint struct {
	RenderedContent string `json:"renderedContent,omitempty"`
}

type GroundingChunk struct {
	Web *WebChunk `json:"web,omitempty"`
}

type WebChunk struct {
	URI   string `json:"uri,omitempty"`
	Title string `json:"title,omitempty"`
}

type GroundingSupport struct {
	GroundingChunkIndices []int    `json:"groundingChunkIndices,omitempty"`
	Segment               *Segment `json:"segment,omitempty"`
}

type Segment struct {
	StartIndex int    `json:"startIndex,omitempty"`
	EndIndex   int    `json:"endIndex,omitempty"`
	Text       string `json:"text,omitempty"`
}

type VertexContent struct {
	Parts []VertexPart `json:"parts"`
	Role  string       `json:"role,omitempty"`
}

type VertexPart struct {
	Text             string            `json:"text,omitempty"`
	InlineData       *InlineData       `json:"inlineData,omitempty"`
	FunctionCall     *FunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *FunctionResponse `json:"functionResponse,omitempty"`
	Thought          bool              `json:"thought,omitempty"`
	ThoughtSignature string            `json:"thoughtSignature,omitempty"`
}

type InlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type TextPart struct {
	Text             string `json:"text"`
	Thought          bool   `json:"thought,omitempty"`
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
}

type FunctionCall struct {
	ID               string                 `json:"id,omitempty"`
	Name             string                 `json:"name"`
	Args             map[string]interface{} `json:"args,omitempty"`
	ThoughtSignature string                 `json:"thoughtSignature,omitempty"`
}

type FunctionResponse struct {
	Name     string                 `json:"name"`
	Response map[string]interface{} `json:"response,omitempty"`
}
