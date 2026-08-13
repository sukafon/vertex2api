package model

// ==================== Chat Completions ====================

type ChatCompletionRequest struct {
	Model               string                   `json:"model"`
	Messages            []ChatMessage            `json:"messages"`
	Stream              bool                     `json:"stream,omitempty"`
	StreamOptions       *ChatStreamOptions       `json:"stream_options,omitempty"`
	Temperature         *float64                 `json:"temperature,omitempty"`
	TopP                *float64                 `json:"top_p,omitempty"`
	MaxTokens           *int                     `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                     `json:"max_completion_tokens,omitempty"`
	ReasoningEffort     *string                  `json:"reasoning_effort,omitempty"`
	Stop                interface{}              `json:"stop,omitempty"`
	FrequencyPenalty    *float64                 `json:"frequency_penalty,omitempty"`
	PresencePenalty     *float64                 `json:"presence_penalty,omitempty"`
	N                   *int                     `json:"n,omitempty"`
	Seed                *int64                   `json:"seed,omitempty"`
	ResponseFormat      map[string]interface{}   `json:"response_format,omitempty"`
	Tools               []map[string]interface{} `json:"tools,omitempty"`
	ToolChoice          interface{}              `json:"tool_choice,omitempty"`
}

type ChatStreamOptions struct {
	IncludeUsage       bool  `json:"include_usage,omitempty"`
	IncludeObfuscation *bool `json:"include_obfuscation,omitempty"`
}

type ChatMessage struct {
	Role              string           `json:"role,omitempty"`
	Content           interface{}      `json:"content"` // string 或 []ContentPart
	ReasoningContent  string           `json:"reasoning_content,omitempty"`
	Name              string           `json:"name,omitempty"`
	ToolCallID        string           `json:"tool_call_id,omitempty"`
	ToolCalls         []ChatToolCall   `json:"tool_calls,omitempty"`
	Annotations       []ChatAnnotation `json:"annotations,omitempty"`
	GroundingMetadata interface{}      `json:"grounding_metadata,omitempty"`
	Citations         interface{}      `json:"citations,omitempty"`
}

type ChatAnnotation struct {
	Type        string           `json:"type"`
	URLCitation *ChatURLCitation `json:"url_citation,omitempty"`
}

type ChatURLCitation struct {
	StartIndex int    `json:"start_index"`
	EndIndex   int    `json:"end_index"`
	URL        string `json:"url"`
	Title      string `json:"title"`
}

type ChatCompletionResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []ChatChoice `json:"choices"`
	Usage   *Usage       `json:"usage,omitempty"`
}

// ChatCompletionChunk is kept separate from ChatCompletionResponse so the
// streaming compatibility shape can expose explicit nullable delta fields and
// attach usage to the final choice-bearing chunk.
type ChatCompletionChunk struct {
	ID          string                      `json:"id"`
	Object      string                      `json:"object"`
	Created     int64                       `json:"created"`
	Model       string                      `json:"model"`
	Choices     []ChatCompletionChunkChoice `json:"choices"`
	Usage       *Usage                      `json:"usage,omitempty"`
	Obfuscation string                      `json:"obfuscation,omitempty"`
}

type ChatCompletionChunkChoice struct {
	Index        int                  `json:"index"`
	Delta        *ChatCompletionDelta `json:"delta"`
	FinishReason *string              `json:"finish_reason"`
}

type ChatCompletionDelta struct {
	Role             string           `json:"role"`
	Content          interface{}      `json:"content"`
	ReasoningContent *string          `json:"reasoning_content"`
	ToolCalls        []ChatToolCall   `json:"tool_calls"`
	Annotations      []ChatAnnotation `json:"annotations,omitempty"`
}

type ChatChoice struct {
	Index        int          `json:"index"`
	Message      *ChatMessage `json:"message,omitempty"`
	FinishReason *string      `json:"finish_reason"`
}

type ChatToolCall struct {
	Index        *int                   `json:"index,omitempty"`
	ID           string                 `json:"id,omitempty"`
	Type         string                 `json:"type"`
	Function     *ChatFunctionCall      `json:"function,omitempty"`
	ExtraContent map[string]interface{} `json:"extra_content,omitempty"`
}

type ChatFunctionCall struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type Usage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	PromptTokensDetails     *PromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *CompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type PromptTokensDetails struct {
	CachedTokens int `json:"cached_tokens,omitempty"`
}

type CompletionTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens,omitempty"`
}

// ==================== Images ====================

type ImageGenerationRequest struct {
	Model          string `json:"model,omitempty"`
	Prompt         string `json:"prompt"`
	N              *int   `json:"n,omitempty"`
	Size           string `json:"size,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"` // "b64_json" | "url"
	Image          string `json:"image,omitempty"`           // 可选 base64 图片输入（图生图）
	Quality        string `json:"quality,omitempty"`
}

type ImageGenerationResponse struct {
	Created int64       `json:"created"`
	Data    []ImageData `json:"data"`
}

type ImageData struct {
	B64JSON       string `json:"b64_json,omitempty"`
	URL           string `json:"url,omitempty"`
	RevisedPrompt string `json:"revised_prompt,omitempty"`
}

// ==================== Models ====================

type OpenAIModelListResponse struct {
	Object string            `json:"object"`
	Data   []OpenAIModelInfo `json:"data"`
}

type OpenAIModelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type GeminiModelListResponse struct {
	Models []GeminiModelInfo `json:"models"`
}

type GeminiModelInfo struct {
	Description                string   `json:"description"`
	DisplayName                string   `json:"displayName"`
	InputTokenLimit            int      `json:"inputTokenLimit,omitempty"` // omitempty: 为 0 时不输出该字段
	Name                       string   `json:"name"`
	OutputTokenLimit           int      `json:"outputTokenLimit,omitempty"` // omitempty: 为 0 时不输出该字段
	SupportedGenerationMethods []string `json:"supportedGenerationMethods"`
	Version                    string   `json:"version"`
}

// ==================== Error ====================

type ErrorResponse struct {
	Error *APIError `json:"error"`
}

type APIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}
