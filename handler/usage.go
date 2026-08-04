package handler

import (
	"encoding/base64"
	"math"
	"strings"
	"unicode/utf8"

	"vertex2api/model"
	"vertex2api/proxy"

	"github.com/bytedance/sonic"
)

const estimatedMediaTokens = 258

func estimateJSONTokens(value interface{}) int {
	data, err := sonic.Marshal(value)
	if err != nil {
		return 1
	}
	var normalized interface{}
	if err := sonic.Unmarshal(data, &normalized); err != nil {
		return maxInt(1, estimateTextTokens(string(data)))
	}
	return maxInt(1, estimateTokenValue(normalized, ""))
}

func estimateTokenValue(value interface{}, key string) int {
	switch typed := value.(type) {
	case nil:
		return 0
	case string:
		if isMediaPayload(key, typed) {
			return estimatedMediaTokens
		}
		return estimateTextTokens(typed)
	case []interface{}:
		total := len(typed)
		for _, item := range typed {
			total += estimateTokenValue(item, key)
		}
		return total
	case map[string]interface{}:
		total := 2
		for childKey, item := range typed {
			total += estimateTextTokens(childKey) + estimateTokenValue(item, childKey)
		}
		return total
	case bool, float64, float32, int, int32, int64, uint, uint32, uint64:
		return 1
	default:
		return estimateJSONTokens(typed)
	}
}

func estimateTextTokens(text string) int {
	if text == "" {
		return 0
	}
	var asciiBytes, nonASCII int
	for len(text) > 0 {
		r, size := utf8.DecodeRuneInString(text)
		text = text[size:]
		if r <= 0x7f {
			asciiBytes++
		} else {
			nonASCII++
		}
	}
	return int(math.Ceil(float64(asciiBytes)/4.0)) + nonASCII
}

func isMediaPayload(key, value string) bool {
	key = strings.ToLower(strings.ReplaceAll(key, "_", ""))
	if key != "data" && key != "filedata" && key != "image" && key != "source" {
		return false
	}
	if strings.HasPrefix(value, "data:image/") || strings.HasPrefix(value, "data:application/pdf") {
		return true
	}
	if len(value) < 64 {
		return false
	}
	_, err := base64.StdEncoding.DecodeString(strings.TrimSpace(value))
	if err == nil {
		return true
	}
	_, err = base64.RawStdEncoding.DecodeString(strings.TrimSpace(value))
	return err == nil
}

func usageMetadataInt(metadata map[string]interface{}, key string) int {
	value, ok := metadata[key]
	if !ok {
		return 0
	}
	switch typed := value.(type) {
	case float64:
		return clampUsageFloat(typed)
	case float32:
		return clampUsageFloat(float64(typed))
	case int:
		return maxInt(0, typed)
	case int32:
		return clampUsageSigned(int64(typed))
	case int64:
		return clampUsageSigned(typed)
	case uint:
		return clampUsageUnsigned(uint64(typed))
	case uint32:
		return clampUsageUnsigned(uint64(typed))
	case uint64:
		return clampUsageUnsigned(typed)
	default:
		return 0
	}
}

func clampUsageFloat(value float64) int {
	if math.IsNaN(value) || value <= 0 {
		return 0
	}
	max := maxIntValue()
	if value >= float64(max) {
		return max
	}
	return int(value)
}

func clampUsageSigned(value int64) int {
	if value <= 0 {
		return 0
	}
	max := int64(maxIntValue())
	if value > max {
		return int(max)
	}
	return int(value)
}

func clampUsageUnsigned(value uint64) int {
	max := uint64(maxIntValue()) // #nosec G115 -- maxIntValue is the platform int maximum.
	if value > max {
		return maxIntValue()
	}
	return int(value) // #nosec G115 -- value is bounded by the platform int maximum.
}

func maxIntValue() int {
	return int(^uint(0) >> 1)
}

func saturatingTokenAdd(a, b int) int {
	max := maxIntValue()
	if a < 0 {
		a = 0
	}
	if b < 0 {
		b = 0
	}
	if a >= max || b >= max-a {
		return max
	}
	return a + b
}

func openAIUsage(req model.ChatCompletionRequest, result *proxy.CallResult) (*model.Usage, bool) {
	promptTokens := estimateOpenAIPromptTokens(req)
	completionTokens := estimateOpenAICompletionTokens(result)
	reasoningTokens := estimateReasoningTokens(result)
	estimated := true

	var cachedTokens int
	if result != nil && len(result.UsageMetadata) > 0 {
		metadata := result.UsageMetadata
		if value := usageMetadataInt(metadata, "promptTokenCount"); value > 0 {
			promptTokens = value
			estimated = false
		}
		candidateTokens := usageMetadataInt(metadata, "candidatesTokenCount")
		thoughtTokens := usageMetadataInt(metadata, "thoughtsTokenCount")
		if candidateTokens > 0 || thoughtTokens > 0 {
			completionTokens = saturatingTokenAdd(candidateTokens, thoughtTokens)
			reasoningTokens = thoughtTokens
			estimated = false
		}
		cachedTokens = usageMetadataInt(metadata, "cachedContentTokenCount")
	}

	totalTokens := saturatingTokenAdd(promptTokens, completionTokens)
	if result != nil {
		if upstreamTotal := usageMetadataInt(result.UsageMetadata, "totalTokenCount"); upstreamTotal > 0 {
			totalTokens = upstreamTotal
			estimated = false
		}
	}
	usage := &model.Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      totalTokens,
	}
	if cachedTokens > 0 {
		usage.PromptTokensDetails = &model.PromptTokensDetails{CachedTokens: cachedTokens}
	}
	if reasoningTokens > 0 {
		usage.CompletionTokensDetails = &model.CompletionTokensDetails{ReasoningTokens: reasoningTokens}
	}
	return usage, estimated
}

func anthropicUsage(req model.AnthropicMessageRequest, result *proxy.CallResult) (*model.AnthropicUsage, bool) {
	inputTokens := estimateAnthropicInputTokens(req)
	outputTokens := estimateAnthropicOutputTokens(result)
	thinkingTokens := estimateReasoningTokens(result)
	estimated := true
	cachedTokens := 0

	if result != nil && len(result.UsageMetadata) > 0 {
		if value := usageMetadataInt(result.UsageMetadata, "promptTokenCount"); value > 0 {
			inputTokens = value
			estimated = false
		}
		candidateTokens := usageMetadataInt(result.UsageMetadata, "candidatesTokenCount")
		thinkingTokens = usageMetadataInt(result.UsageMetadata, "thoughtsTokenCount")
		if candidateTokens > 0 || thinkingTokens > 0 {
			outputTokens = saturatingTokenAdd(candidateTokens, thinkingTokens)
			estimated = false
		}
		cachedTokens = usageMetadataInt(result.UsageMetadata, "cachedContentTokenCount")
	}

	usage := &model.AnthropicUsage{InputTokens: inputTokens, OutputTokens: outputTokens}
	if usage.OutputTokens == 0 && result != nil && (req.MaxTokens == nil || *req.MaxTokens != 0) {
		usage.OutputTokens = 1
	}
	if cachedTokens > 0 {
		usage.InputTokens = maxInt(0, inputTokens-cachedTokens)
		usage.CacheReadInputTokens = cachedTokens
	}
	if thinkingTokens > 0 {
		usage.OutputTokensDetails = &model.AnthropicOutputTokensDetails{ThinkingTokens: thinkingTokens}
	}
	return usage, estimated
}

func geminiUsageMetadata(input interface{}, result *proxy.CallResult) (map[string]interface{}, bool) {
	if result != nil && len(result.UsageMetadata) > 0 {
		metadata := make(map[string]interface{}, len(result.UsageMetadata))
		for key, value := range result.UsageMetadata {
			metadata[key] = value
		}
		return metadata, false
	}
	promptTokens := estimateJSONTokens(input)
	thoughtTokens := estimateReasoningTokens(result)
	candidateTokens := maxInt(0, estimateOpenAICompletionTokens(result)-thoughtTokens)
	metadata := map[string]interface{}{
		"promptTokenCount":     promptTokens,
		"candidatesTokenCount": candidateTokens,
		"totalTokenCount":      saturatingTokenAdd(promptTokens, candidateTokens),
	}
	if thoughtTokens > 0 {
		metadata["thoughtsTokenCount"] = thoughtTokens
	}
	return metadata, true
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
