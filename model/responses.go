package model

// ResponseRequest is the subset of the OpenAI Responses create request that
// vertex2api can translate to Vertex generateContent.
type ResponseRequest struct {
	Model              string                      `json:"model"`
	Input              interface{}                 `json:"input"`
	Instructions       string                      `json:"instructions,omitempty"`
	Stream             bool                        `json:"stream,omitempty"`
	Store              *bool                       `json:"store,omitempty"`
	Background         bool                        `json:"background,omitempty"`
	PreviousResponseID string                      `json:"previous_response_id,omitempty"`
	Temperature        *float64                    `json:"temperature,omitempty"`
	TopP               *float64                    `json:"top_p,omitempty"`
	MaxOutputTokens    *int                        `json:"max_output_tokens,omitempty"`
	ParallelToolCalls  *bool                       `json:"parallel_tool_calls,omitempty"`
	Tools              []map[string]interface{}    `json:"tools,omitempty"`
	ToolChoice         interface{}                 `json:"tool_choice,omitempty"`
	Text               map[string]interface{}      `json:"text,omitempty"`
	Reasoning          map[string]interface{}      `json:"reasoning,omitempty"`
	Metadata           map[string]string           `json:"metadata,omitempty"`
	Include            []string                    `json:"include,omitempty"`
	Truncation         string                      `json:"truncation,omitempty"`
	ContextManagement  []ResponseContextManagement `json:"context_management,omitempty"`
	PromptCacheKey     string                      `json:"prompt_cache_key,omitempty"`
	SafetyIdentifier   string                      `json:"safety_identifier,omitempty"`
	ServiceTier        string                      `json:"service_tier,omitempty"`
}

type ResponseContextManagement struct {
	Type             string `json:"type"`
	CompactThreshold int    `json:"compact_threshold,omitempty"`
}

// ResponseCompactRequest is accepted by POST /v1/responses/compact.
type ResponseCompactRequest struct {
	Model              string      `json:"model"`
	Input              interface{} `json:"input"`
	Instructions       string      `json:"instructions,omitempty"`
	PreviousResponseID string      `json:"previous_response_id,omitempty"`
}

type ResponseUsage struct {
	InputTokens         int                         `json:"input_tokens"`
	InputTokensDetails  ResponseInputTokensDetails  `json:"input_tokens_details"`
	OutputTokens        int                         `json:"output_tokens"`
	OutputTokensDetails ResponseOutputTokensDetails `json:"output_tokens_details"`
	TotalTokens         int                         `json:"total_tokens"`
}

type ResponseInputTokensDetails struct {
	CachedTokens     int `json:"cached_tokens"`
	CacheWriteTokens int `json:"cache_write_tokens,omitempty"`
}

type ResponseOutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type CompactedResponse struct {
	ID        string                   `json:"id"`
	CreatedAt int64                    `json:"created_at"`
	Object    string                   `json:"object"`
	Output    []map[string]interface{} `json:"output"`
	Usage     ResponseUsage            `json:"usage"`
}
