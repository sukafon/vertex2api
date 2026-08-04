package model

type AnthropicMessageRequest struct {
	Model         string                   `json:"model"`
	MaxTokens     *int                     `json:"max_tokens"`
	Messages      []AnthropicInputMessage  `json:"messages"`
	System        interface{}              `json:"system,omitempty"`
	Stream        bool                     `json:"stream,omitempty"`
	Temperature   *float64                 `json:"temperature,omitempty"`
	TopP          *float64                 `json:"top_p,omitempty"`
	TopK          interface{}              `json:"top_k,omitempty"`
	StopSequences []string                 `json:"stop_sequences,omitempty"`
	Tools         []map[string]interface{} `json:"tools,omitempty"`
	ToolChoice    interface{}              `json:"tool_choice,omitempty"`
	Thinking      map[string]interface{}   `json:"thinking,omitempty"`
}

type AnthropicInputMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"`
}

type AnthropicMessageResponse struct {
	ID           string                   `json:"id"`
	Type         string                   `json:"type"`
	Role         string                   `json:"role"`
	Model        string                   `json:"model"`
	Content      []map[string]interface{} `json:"content"`
	StopReason   *string                  `json:"stop_reason"`
	StopSequence *string                  `json:"stop_sequence"`
	Usage        *AnthropicUsage          `json:"usage"`
}

type AnthropicUsage struct {
	InputTokens              int                           `json:"input_tokens"`
	OutputTokens             int                           `json:"output_tokens"`
	CacheCreationInputTokens int                           `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int                           `json:"cache_read_input_tokens,omitempty"`
	OutputTokensDetails      *AnthropicOutputTokensDetails `json:"output_tokens_details,omitempty"`
}

type AnthropicOutputTokensDetails struct {
	ThinkingTokens int `json:"thinking_tokens"`
}

type AnthropicCountTokensResponse struct {
	InputTokens int `json:"input_tokens"`
}

type AnthropicModelListResponse struct {
	Data    []AnthropicModelInfo `json:"data"`
	HasMore bool                 `json:"has_more"`
	FirstID string               `json:"first_id,omitempty"`
	LastID  string               `json:"last_id,omitempty"`
}

type AnthropicModelInfo struct {
	ID          string `json:"id"`
	Type        string `json:"type"`
	DisplayName string `json:"display_name"`
	CreatedAt   string `json:"created_at"`
}
