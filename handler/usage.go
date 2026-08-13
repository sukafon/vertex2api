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
			if isImageURLField(childKey) {
				total += estimateTextTokens(childKey) + estimateImageURLValue(item)
				continue
			}
			total += estimateTextTokens(childKey) + estimateTokenValue(item, childKey)
		}
		return total
	case bool, float64, float32, int, int32, int64, uint, uint32, uint64:
		return 1
	default:
		return estimateJSONTokens(typed)
	}
}

func isImageURLField(key string) bool {
	key = strings.ToLower(strings.ReplaceAll(key, "_", ""))
	return key == "imageurl"
}

func estimateImageURLValue(value interface{}) int {
	switch typed := value.(type) {
	case string:
		if strings.TrimSpace(typed) == "" {
			return 0
		}
		return estimatedMediaTokens
	case map[string]interface{}:
		total := 0
		if imageURL, _ := typed["url"].(string); strings.TrimSpace(imageURL) != "" {
			total = estimatedMediaTokens
		}
		for key, item := range typed {
			if key == "url" {
				continue
			}
			total += estimateTextTokens(key) + estimateTokenValue(item, key)
		}
		return total
	default:
		return estimateTokenValue(value, "image_url")
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
	value = strings.TrimSpace(value)
	prefix := value
	if len(prefix) > 32 {
		prefix = prefix[:32]
	}
	prefix = strings.ToLower(prefix)
	if strings.HasPrefix(prefix, "data:image/") || strings.HasPrefix(prefix, "data:application/pdf") {
		return true
	}

	key = strings.ToLower(strings.ReplaceAll(key, "_", ""))
	if key != "data" && key != "filedata" && key != "fileuri" && key != "image" && key != "source" {
		return false
	}
	if key == "fileuri" {
		return value != ""
	}
	if len(value) < 64 {
		return false
	}
	_, err := base64.StdEncoding.DecodeString(value)
	if err == nil {
		return true
	}
	_, err = base64.RawStdEncoding.DecodeString(value)
	return err == nil
}

func usageMetadataInt(metadata map[string]interface{}, key string) int {
	value, _ := usageMetadataIntOK(metadata, key)
	return value
}

func usageMetadataIntOK(metadata map[string]interface{}, key string) (int, bool) {
	value, ok := metadata[key]
	if !ok {
		return 0, false
	}
	switch typed := value.(type) {
	case float64:
		return clampUsageFloat(typed), !math.IsNaN(typed) && !math.IsInf(typed, 0) && typed >= 0
	case float32:
		value := float64(typed)
		return clampUsageFloat(value), !math.IsNaN(value) && !math.IsInf(value, 0) && value >= 0
	case int:
		return maxInt(0, typed), typed >= 0
	case int32:
		return clampUsageSigned(int64(typed)), typed >= 0
	case int64:
		return clampUsageSigned(typed), typed >= 0
	case uint:
		return clampUsageUnsigned(uint64(typed)), true
	case uint32:
		return clampUsageUnsigned(uint64(typed)), true
	case uint64:
		return clampUsageUnsigned(typed), true
	default:
		return 0, false
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

// streamUsageRevision keeps an upstream cumulative usage measurement valid
// only while no subsequently received assistant output can change it.
type streamUsageRevision struct {
	outputRevision uint64
	metadataAt     uint64
	metadata       map[string]interface{}
}

func (s *streamUsageRevision) observe(result *proxy.CallResult) {
	if s == nil || result == nil {
		return
	}
	if estimateOpenAICompletionTokens(result) > 0 {
		s.outputRevision++
	}
	if len(result.UsageMetadata) > 0 {
		s.metadata = cloneUsageMetadata(result.UsageMetadata)
		s.metadataAt = s.outputRevision
	}
}

func (s *streamUsageRevision) currentMetadata() map[string]interface{} {
	if s == nil || len(s.metadata) == 0 {
		return nil
	}
	if s.metadataAt != s.outputRevision {
		// Prompt and cache measurements are invariant for the lifetime of one
		// generation. Output and total counters become stale after another delta.
		invariant := make(map[string]interface{})
		for _, key := range []string{
			"promptTokenCount", "cachedContentTokenCount",
			"promptTokensDetails", "cacheTokensDetails", "serviceTier",
		} {
			if value, ok := s.metadata[key]; ok {
				invariant[key] = value
			}
		}
		return invariant
	}
	return cloneUsageMetadata(s.metadata)
}

func cloneUsageMetadata(metadata map[string]interface{}) map[string]interface{} {
	if len(metadata) == 0 {
		return nil
	}
	cloned := make(map[string]interface{}, len(metadata))
	for key, value := range metadata {
		cloned[key] = value
	}
	return cloned
}

func usageSnapshot(aggregate *proxy.CallResult, revision *streamUsageRevision) proxy.CallResult {
	if aggregate == nil {
		return proxy.CallResult{}
	}
	snapshot := *aggregate
	snapshot.UsageMetadata = revision.currentMetadata()
	return snapshot
}

type resolvedVertexUsage struct {
	prompt, candidates, thoughts, toolUse, total                          int
	promptExact, candidatesExact, thoughtsExact, toolUseExact, totalExact bool
}

func usageMetadata(result *proxy.CallResult) map[string]interface{} {
	if result == nil {
		return nil
	}
	return result.UsageMetadata
}

func resolveVertexUsage(estimatedPrompt, estimatedCandidates, estimatedThoughts int, metadata map[string]interface{}) resolvedVertexUsage {
	resolved := resolvedVertexUsage{
		prompt:       estimatedPrompt,
		candidates:   estimatedCandidates,
		thoughts:     estimatedThoughts,
		toolUseExact: true,
	}
	if value, ok := usageMetadataIntOK(metadata, "promptTokenCount"); ok {
		resolved.prompt, resolved.promptExact = value, true
	}
	if value, ok := usageMetadataIntOK(metadata, "candidatesTokenCount"); ok {
		resolved.candidates, resolved.candidatesExact = value, true
	}
	if value, ok := usageMetadataIntOK(metadata, "thoughtsTokenCount"); ok {
		resolved.thoughts, resolved.thoughtsExact = value, true
	}
	if value, ok := usageMetadataIntOK(metadata, "toolUsePromptTokenCount"); ok {
		resolved.toolUse = value
	} else if _, reported := metadata["toolUsePromptTokenCount"]; reported {
		resolved.toolUseExact = false
	}
	if value, ok := usageMetadataIntOK(metadata, "totalTokenCount"); ok {
		resolved.total, resolved.totalExact = value, true
	}

	if !resolved.totalExact {
		if _, reported := metadata["thoughtsTokenCount"]; !reported && resolved.thoughts == 0 {
			// Vertex omits thoughtsTokenCount when no reasoning tokens were used.
			resolved.thoughtsExact = true
		}
		resolved.total = saturatingTokenAdd(
			saturatingTokenAdd(resolved.prompt, resolved.toolUse),
			saturatingTokenAdd(resolved.candidates, resolved.thoughts),
		)
		resolved.totalExact = resolved.promptExact && resolved.candidatesExact && resolved.thoughtsExact && resolved.toolUseExact
		return resolved
	}

	if resolved.promptExact {
		outputTotal := nonNegativeTokenDifference(
			nonNegativeTokenDifference(resolved.total, resolved.prompt),
			resolved.toolUse,
		)
		switch {
		case resolved.candidatesExact && resolved.thoughtsExact:
		case resolved.candidatesExact && !resolved.thoughtsExact:
			resolved.thoughts = nonNegativeTokenDifference(outputTotal, resolved.candidates)
			resolved.thoughtsExact = true
		case !resolved.candidatesExact && resolved.thoughtsExact:
			resolved.candidates = nonNegativeTokenDifference(outputTotal, resolved.thoughts)
			resolved.candidatesExact = true
		case !resolved.candidatesExact && !resolved.thoughtsExact:
			resolved.thoughts = minInt(resolved.thoughts, outputTotal)
			resolved.candidates = outputTotal - resolved.thoughts
		}

		componentInput := saturatingTokenAdd(resolved.prompt, resolved.toolUse)
		componentOutput := saturatingTokenAdd(resolved.candidates, resolved.thoughts)
		componentTotal := saturatingTokenAdd(componentInput, componentOutput)
		switch {
		case resolved.total > componentTotal:
			// Preserve unexplained upstream billable tokens instead of silently
			// shrinking total_tokens. Unknown buckets are input-side accounting:
			// putting them in completion would make deterministic output usage vary.
			resolved.toolUse = saturatingTokenAdd(resolved.toolUse, resolved.total-componentTotal)
			resolved.toolUseExact = false
		case resolved.total < componentTotal:
			// Preserve the larger complete component breakdown when the reported
			// total is impossible, and make the inconsistency observable as estimated.
			resolved.total = componentTotal
			resolved.totalExact = false
		}
		return resolved
	}

	nonPrompt := saturatingTokenAdd(
		resolved.toolUse,
		saturatingTokenAdd(resolved.candidates, resolved.thoughts),
	)
	if resolved.total < nonPrompt {
		resolved.prompt = 0
		resolved.total = nonPrompt
		resolved.totalExact = false
		resolved.promptExact = false
		return resolved
	}
	resolved.prompt = resolved.total - nonPrompt
	if resolved.candidatesExact && resolved.thoughtsExact && resolved.toolUseExact {
		resolved.promptExact = true
	}
	return resolved
}

func nonNegativeTokenDifference(total, used int) int {
	if total <= used {
		return 0
	}
	return total - used
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func openAIUsage(req model.ChatCompletionRequest, result *proxy.CallResult) (*model.Usage, bool) {
	estimatedCandidates, estimatedReasoning := estimateCompletionTokenBreakdown(result)
	resolved := resolveVertexUsage(estimateOpenAIPromptTokens(req), estimatedCandidates, estimatedReasoning, usageMetadata(result))
	var cachedTokens int
	if result != nil && len(result.UsageMetadata) > 0 {
		cachedTokens = usageMetadataInt(result.UsageMetadata, "cachedContentTokenCount")
	}
	cacheExact := true
	if cachedTokens > resolved.prompt {
		cachedTokens = resolved.prompt
		cacheExact = false
	}

	promptTokens := saturatingTokenAdd(resolved.prompt, resolved.toolUse)
	completionTokens := saturatingTokenAdd(resolved.candidates, resolved.thoughts)
	reasoningBudgetExhausted := result != nil &&
		strings.EqualFold(result.FinishReason, "MAX_TOKENS") &&
		estimatedCandidates == 0 && resolved.thoughts > 0
	reasoningTokens, reasoningExact := openAIReasoningTokenDetails(
		resolved, estimatedCandidates, estimatedReasoning, reasoningBudgetExhausted,
	)
	estimated := !(resolved.promptExact && resolved.candidatesExact && resolved.thoughtsExact && resolved.toolUseExact && resolved.totalExact && cacheExact && reasoningExact)
	usage := &model.Usage{
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
		TotalTokens:      resolved.total,
	}
	if cachedTokens > 0 {
		usage.PromptTokensDetails = &model.PromptTokensDetails{CachedTokens: cachedTokens}
	}
	if reasoningTokens > 0 {
		usage.CompletionTokensDetails = &model.CompletionTokensDetails{ReasoningTokens: reasoningTokens}
	}
	return usage, estimated
}

// openAIReasoningTokenDetails maps both internal Vertex thinking tokens and
// candidate tokens returned as thought parts to OpenAI reasoning_tokens.
// candidatesTokenCount covers every returned candidate part, so an all-thought
// response can be split exactly. Mixed visible/thought output needs a
// proportional estimate because Vertex does not expose that candidate split.
func openAIReasoningTokenDetails(
	resolved resolvedVertexUsage,
	estimatedCandidates, estimatedReasoning int,
	reasoningBudgetExhausted bool,
) (int, bool) {
	completionTokens := saturatingTokenAdd(resolved.candidates, resolved.thoughts)
	reasoningTokens := minInt(resolved.thoughts, completionTokens)
	breakdownExact := resolved.thoughtsExact
	if (estimatedReasoning <= 0 && !reasoningBudgetExhausted) || resolved.candidates <= 0 {
		return reasoningTokens, breakdownExact
	}
	if !resolved.thoughtsExact || !resolved.candidatesExact {
		return reasoningTokens, false
	}

	returnedReasoning := 0
	if estimatedCandidates <= 0 {
		// Every returned candidate token is reasoning when the response contains
		// only thought parts, or when thinking exhausts the output budget before
		// any visible candidate is produced.
		returnedReasoning = resolved.candidates
	} else {
		estimatedTotal := saturatingTokenAdd(estimatedCandidates, estimatedReasoning)
		if estimatedTotal > 0 {
			share := float64(resolved.candidates) * float64(estimatedReasoning) / float64(estimatedTotal)
			returnedReasoning = int(math.Round(share))
			returnedReasoning = maxInt(1, minInt(returnedReasoning, resolved.candidates))
		}
		breakdownExact = false
	}

	reasoningTokens = saturatingTokenAdd(reasoningTokens, returnedReasoning)
	return minInt(reasoningTokens, completionTokens), breakdownExact
}

type openAIUsageContribution struct {
	usage     *model.Usage
	estimated bool
}

func combineOpenAIUsage(usages ...*model.Usage) *model.Usage {
	combined := &model.Usage{}
	for _, usage := range usages {
		if usage == nil {
			continue
		}
		combined.PromptTokens = saturatingTokenAdd(combined.PromptTokens, usage.PromptTokens)
		combined.CompletionTokens = saturatingTokenAdd(combined.CompletionTokens, usage.CompletionTokens)
		combined.TotalTokens = saturatingTokenAdd(combined.TotalTokens, usage.TotalTokens)
		if usage.PromptTokensDetails != nil {
			if combined.PromptTokensDetails == nil {
				combined.PromptTokensDetails = &model.PromptTokensDetails{}
			}
			combined.PromptTokensDetails.CachedTokens = saturatingTokenAdd(
				combined.PromptTokensDetails.CachedTokens,
				usage.PromptTokensDetails.CachedTokens,
			)
		}
		if usage.CompletionTokensDetails != nil {
			if combined.CompletionTokensDetails == nil {
				combined.CompletionTokensDetails = &model.CompletionTokensDetails{}
			}
			combined.CompletionTokensDetails.ReasoningTokens = saturatingTokenAdd(
				combined.CompletionTokensDetails.ReasoningTokens,
				usage.CompletionTokensDetails.ReasoningTokens,
			)
		}
	}
	return combined
}

func anthropicUsage(req model.AnthropicMessageRequest, result *proxy.CallResult) (*model.AnthropicUsage, bool) {
	estimatedCandidates, estimatedThinking := estimateCompletionTokenBreakdown(result)
	resolved := resolveVertexUsage(estimateAnthropicInputTokens(req), estimatedCandidates, estimatedThinking, usageMetadata(result))
	cachedTokens := 0

	if result != nil && len(result.UsageMetadata) > 0 {
		cachedTokens = usageMetadataInt(result.UsageMetadata, "cachedContentTokenCount")
	}

	inputTokens := saturatingTokenAdd(resolved.prompt, resolved.toolUse)
	outputTokens := saturatingTokenAdd(resolved.candidates, resolved.thoughts)
	estimated := !(resolved.promptExact && resolved.candidatesExact && resolved.thoughtsExact && resolved.toolUseExact)
	usage := &model.AnthropicUsage{InputTokens: inputTokens, OutputTokens: outputTokens}
	if usage.OutputTokens == 0 && !resolved.candidatesExact && result != nil && (req.MaxTokens == nil || *req.MaxTokens != 0) {
		usage.OutputTokens = 1
	}
	if cachedTokens > 0 {
		usage.InputTokens = maxInt(0, inputTokens-cachedTokens)
		usage.CacheReadInputTokens = cachedTokens
	}
	if resolved.thoughts > 0 {
		usage.OutputTokensDetails = &model.AnthropicOutputTokensDetails{ThinkingTokens: resolved.thoughts}
	}
	return usage, estimated
}

func geminiUsageMetadata(input interface{}, result *proxy.CallResult) (map[string]interface{}, bool) {
	return geminiUsageMetadataWithPromptEstimate(estimateJSONTokens(input), result)
}

func geminiUsageMetadataWithPromptEstimate(estimatedPrompt int, result *proxy.CallResult) (map[string]interface{}, bool) {
	estimatedCandidates, estimatedThoughts := estimateCompletionTokenBreakdown(result)
	upstream := usageMetadata(result)
	resolved := resolveVertexUsage(estimatedPrompt, estimatedCandidates, estimatedThoughts, upstream)
	metadata := cloneUsageMetadata(upstream)
	if metadata == nil {
		metadata = make(map[string]interface{})
	}
	setResolvedUsageField(metadata, upstream, "promptTokenCount", resolved.prompt)
	setResolvedUsageField(metadata, upstream, "candidatesTokenCount", resolved.candidates)
	if resolved.toolUse > 0 || upstreamHasNumericUsageField(upstream, "toolUsePromptTokenCount") {
		setResolvedUsageField(metadata, upstream, "toolUsePromptTokenCount", resolved.toolUse)
	} else {
		delete(metadata, "toolUsePromptTokenCount")
	}
	if resolved.thoughts > 0 || upstreamHasNumericUsageField(upstream, "thoughtsTokenCount") {
		setResolvedUsageField(metadata, upstream, "thoughtsTokenCount", resolved.thoughts)
	} else {
		delete(metadata, "thoughtsTokenCount")
	}
	setResolvedUsageField(metadata, upstream, "totalTokenCount", resolved.total)
	estimated := !(resolved.promptExact && resolved.candidatesExact && resolved.thoughtsExact && resolved.toolUseExact && resolved.totalExact)
	return metadata, estimated
}

func estimateGeminiUsageMetadata(input interface{}, result *proxy.CallResult) map[string]interface{} {
	candidateTokens, thoughtTokens := estimateCompletionTokenBreakdown(result)
	promptTokens := estimateJSONTokens(input)
	metadata := map[string]interface{}{
		"promptTokenCount":     promptTokens,
		"candidatesTokenCount": candidateTokens,
		"totalTokenCount":      saturatingTokenAdd(saturatingTokenAdd(promptTokens, candidateTokens), thoughtTokens),
	}
	if thoughtTokens > 0 {
		metadata["thoughtsTokenCount"] = thoughtTokens
	}
	return metadata
}

func upstreamHasNumericUsageField(metadata map[string]interface{}, key string) bool {
	_, ok := usageMetadataIntOK(metadata, key)
	return ok
}

func setResolvedUsageField(dst, upstream map[string]interface{}, key string, value int) {
	if upstreamValue, ok := usageMetadataIntOK(upstream, key); ok && upstreamValue == value {
		dst[key] = upstream[key]
		return
	}
	dst[key] = value
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
