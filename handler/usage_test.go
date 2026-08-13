package handler

import (
	"math"
	"strings"
	"testing"

	"vertex2api/model"
	"vertex2api/proxy"
)

func TestUsageMetadataIntClampsOutOfRangeValues(t *testing.T) {
	max := maxIntValue()
	tests := []struct {
		name  string
		value interface{}
		want  int
	}{
		{name: "uint64 overflow", value: ^uint64(0), want: max},
		{name: "int64 negative", value: int64(-1), want: 0},
		{name: "float infinity", value: math.Inf(1), want: max},
		{name: "float negative", value: -1.5, want: 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := usageMetadataInt(map[string]interface{}{"tokens": tt.value}, "tokens"); got != tt.want {
				t.Fatalf("usageMetadataInt() = %d, want %d", got, tt.want)
			}
		})
	}
}

func TestSaturatingTokenAddPreventsOverflow(t *testing.T) {
	max := maxIntValue()
	if got := saturatingTokenAdd(max-1, 2); got != max {
		t.Fatalf("saturatingTokenAdd() = %d, want %d", got, max)
	}
}

func TestEstimateJSONTokensDoesNotCountBase64AsText(t *testing.T) {
	small := map[string]interface{}{"inlineData": map[string]interface{}{"data": strings.Repeat("A", 400)}}
	large := map[string]interface{}{"inlineData": map[string]interface{}{"data": strings.Repeat("A", 40000)}}
	if gotSmall, gotLarge := estimateJSONTokens(small), estimateJSONTokens(large); gotSmall != gotLarge {
		t.Fatalf("base64 estimate changed with payload length: small=%d large=%d", gotSmall, gotLarge)
	}
}

func TestEstimateJSONTokensRecognizesOpenAIImageDataURL(t *testing.T) {
	small := map[string]interface{}{"image_url": map[string]interface{}{"url": "data:image/png;base64," + strings.Repeat("A", 400)}}
	large := map[string]interface{}{"image_url": map[string]interface{}{"url": "data:image/png;base64," + strings.Repeat("A", 40000)}}
	if gotSmall, gotLarge := estimateJSONTokens(small), estimateJSONTokens(large); gotSmall != gotLarge {
		t.Fatalf("OpenAI image estimate changed with data URL length: small=%d large=%d", gotSmall, gotLarge)
	}
}

func TestEstimateJSONTokensDoesNotChargeOpenAIImageURLByURLLength(t *testing.T) {
	short := map[string]interface{}{"image_url": map[string]interface{}{"url": "https://example.com/a"}}
	long := map[string]interface{}{"image_url": map[string]interface{}{"url": "https://example.com/" + strings.Repeat("path/", 1000)}}
	if gotShort, gotLong := estimateJSONTokens(short), estimateJSONTokens(long); gotShort != gotLong {
		t.Fatalf("OpenAI remote image estimate changed with URL length: short=%d long=%d", gotShort, gotLong)
	}
}

func TestEstimateJSONTokensDoesNotChargeGeminiFileURIByURLLength(t *testing.T) {
	short := map[string]interface{}{"fileData": map[string]interface{}{"mimeType": "application/pdf", "fileUri": "gs://bucket/a.pdf"}}
	long := map[string]interface{}{"fileData": map[string]interface{}{"mimeType": "application/pdf", "fileUri": "gs://bucket/" + strings.Repeat("path/", 1000) + "a.pdf"}}
	if gotShort, gotLong := estimateJSONTokens(short), estimateJSONTokens(long); gotShort != gotLong {
		t.Fatalf("Gemini file estimate changed with URI length: short=%d long=%d", gotShort, gotLong)
	}
}

func TestOpenAIUsagePrefersUpstreamMetadata(t *testing.T) {
	req := model.ChatCompletionRequest{Messages: []model.ChatMessage{{Role: "user", Content: "hello"}}}
	result := &proxy.CallResult{UsageMetadata: map[string]interface{}{
		"promptTokenCount":        float64(10),
		"candidatesTokenCount":    float64(4),
		"thoughtsTokenCount":      float64(2),
		"cachedContentTokenCount": float64(3),
		"totalTokenCount":         float64(16),
	}}
	usage, estimated := openAIUsage(req, result)
	if estimated {
		t.Fatal("usage should be marked as upstream, not estimated")
	}
	if usage.PromptTokens != 10 || usage.CompletionTokens != 6 || usage.TotalTokens != 16 {
		t.Fatalf("usage = %+v, want prompt=10 completion=6 total=16", usage)
	}
	if usage.PromptTokensDetails == nil || usage.PromptTokensDetails.CachedTokens != 3 {
		t.Fatalf("prompt details = %+v, want cached_tokens=3", usage.PromptTokensDetails)
	}
	if usage.CompletionTokensDetails == nil || usage.CompletionTokensDetails.ReasoningTokens != 2 {
		t.Fatalf("completion details = %+v, want reasoning_tokens=2", usage.CompletionTokensDetails)
	}
}

func TestGeminiEstimatedUsageIncludesThoughtTokensInTotal(t *testing.T) {
	result := &proxy.CallResult{TextParts: []model.TextPart{
		{Text: "reasoning", Thought: true},
		{Text: "answer"},
	}}
	metadata, estimated := geminiUsageMetadata(map[string]interface{}{"contents": []interface{}{"hello"}}, result)
	if !estimated {
		t.Fatal("usage without upstream metadata should be estimated")
	}
	prompt := usageMetadataInt(metadata, "promptTokenCount")
	candidates := usageMetadataInt(metadata, "candidatesTokenCount")
	thoughts := usageMetadataInt(metadata, "thoughtsTokenCount")
	total := usageMetadataInt(metadata, "totalTokenCount")
	if thoughts == 0 {
		t.Fatalf("thoughtsTokenCount = %d, want positive", thoughts)
	}
	if want := saturatingTokenAdd(saturatingTokenAdd(prompt, candidates), thoughts); total != want {
		t.Fatalf("totalTokenCount = %d, want %d", total, want)
	}
}

func TestGeminiUsageCompletesPartialUpstreamMetadata(t *testing.T) {
	result := &proxy.CallResult{
		TextParts: []model.TextPart{{Text: "estimated candidate output"}},
		UsageMetadata: map[string]interface{}{
			"promptTokenCount": float64(10),
		},
	}
	metadata, estimated := geminiUsageMetadata(map[string]interface{}{"contents": []interface{}{"hello"}}, result)
	if !estimated {
		t.Fatal("partially upstream-derived usage should be marked estimated")
	}
	if got := usageMetadataInt(metadata, "promptTokenCount"); got != 10 {
		t.Fatalf("promptTokenCount = %d, want upstream value 10", got)
	}
	candidates := usageMetadataInt(metadata, "candidatesTokenCount")
	if candidates <= 0 {
		t.Fatalf("candidatesTokenCount = %d, want a positive estimate", candidates)
	}
	if got, want := usageMetadataInt(metadata, "totalTokenCount"), 10+candidates; got != want {
		t.Fatalf("totalTokenCount = %d, want recomputed value %d", got, want)
	}
}

func TestAnthropicUsageSeparatesCachedInputTokens(t *testing.T) {
	result := &proxy.CallResult{UsageMetadata: map[string]interface{}{
		"promptTokenCount":        float64(10),
		"cachedContentTokenCount": float64(3),
		"candidatesTokenCount":    float64(4),
	}}
	usage, estimated := anthropicUsage(model.AnthropicMessageRequest{}, result)
	if estimated {
		t.Fatal("usage should be marked as upstream, not estimated")
	}
	if usage.InputTokens != 7 || usage.CacheReadInputTokens != 3 || usage.OutputTokens != 4 {
		t.Fatalf("usage = %+v, want input=7 cache_read=3 output=4", usage)
	}
}

func TestOpenAIUsageKeepsPartialMetadataEstimated(t *testing.T) {
	req := model.ChatCompletionRequest{Messages: []model.ChatMessage{{Role: "user", Content: "hello"}}}
	result := &proxy.CallResult{
		TextParts:     []model.TextPart{{Text: "estimated candidate output"}},
		UsageMetadata: map[string]interface{}{"promptTokenCount": float64(10)},
	}
	usage, estimated := openAIUsage(req, result)
	if !estimated {
		t.Fatal("partial upstream usage was incorrectly marked exact")
	}
	if usage.PromptTokens != 10 || usage.CompletionTokens <= 0 || usage.TotalTokens != usage.PromptTokens+usage.CompletionTokens {
		t.Fatalf("usage = %+v, want exact prompt plus estimated completion", usage)
	}
}

func TestOpenAIUsagePreservesEstimatedThinkingWhenUpstreamOmitsIt(t *testing.T) {
	result := &proxy.CallResult{
		TextParts: []model.TextPart{
			{Text: "private reasoning", Thought: true},
			{Text: "answer"},
		},
		UsageMetadata: map[string]interface{}{
			"promptTokenCount":     float64(8),
			"candidatesTokenCount": float64(2),
		},
	}
	usage, estimated := openAIUsage(model.ChatCompletionRequest{}, result)
	if !estimated {
		t.Fatal("missing thoughtsTokenCount should keep usage estimated when reasoning was returned")
	}
	if usage.CompletionTokens <= 2 || usage.CompletionTokensDetails == nil || usage.CompletionTokensDetails.ReasoningTokens <= 0 {
		t.Fatalf("usage = %+v, want upstream candidates plus estimated reasoning", usage)
	}
}

func TestOpenAIUsageDerivesMissingThoughtsFromExactTotal(t *testing.T) {
	result := &proxy.CallResult{UsageMetadata: map[string]interface{}{
		"promptTokenCount":     float64(8),
		"candidatesTokenCount": float64(2),
		"totalTokenCount":      float64(15),
	}}
	usage, estimated := openAIUsage(model.ChatCompletionRequest{}, result)
	if estimated {
		t.Fatal("a missing component derivable from exact upstream totals should remain exact")
	}
	if usage.CompletionTokens != 7 || usage.CompletionTokensDetails == nil || usage.CompletionTokensDetails.ReasoningTokens != 5 {
		t.Fatalf("usage = %+v, want candidates=2 plus derived reasoning=5", usage)
	}
}

func TestAnthropicUsagePreservesExactZeroOutput(t *testing.T) {
	result := &proxy.CallResult{UsageMetadata: map[string]interface{}{
		"promptTokenCount":     float64(0),
		"candidatesTokenCount": float64(0),
	}}
	usage, estimated := anthropicUsage(model.AnthropicMessageRequest{}, result)
	if estimated || usage.InputTokens != 0 || usage.OutputTokens != 0 {
		t.Fatalf("usage = %+v estimated=%v, want exact zero values", usage, estimated)
	}
}

func TestCompletionEstimatorUsesCanonicalPartsAndIgnoresSignature(t *testing.T) {
	result := &proxy.CallResult{Parts: []model.VertexPart{
		{ExecutableCode: map[string]interface{}{"language": "PYTHON", "code": strings.Repeat("print(1)\n", 20)}},
		{CodeExecutionResult: map[string]interface{}{"outcome": "OUTCOME_OK", "output": strings.Repeat("1\n", 20)}},
		{FileData: map[string]interface{}{"mimeType": "text/plain", "fileUri": "gs://bucket/output.txt"}},
	}}
	candidateTokens, reasoningTokens := estimateCompletionTokenBreakdown(result)
	if candidateTokens <= 0 || reasoningTokens != 0 {
		t.Fatalf("canonical output estimate = candidates %d reasoning %d", candidateTokens, reasoningTokens)
	}

	signatureOnly := &proxy.CallResult{TextParts: []model.TextPart{{Thought: true, ThoughtSignature: strings.Repeat("A", 4000)}}}
	if got := estimateOpenAICompletionTokens(signatureOnly); got != 0 {
		t.Fatalf("opaque thought signature counted as %d output tokens", got)
	}
}

func TestPromptEstimatorsIncludeStructuredOutputSchemas(t *testing.T) {
	chatBase := model.ChatCompletionRequest{Messages: []model.ChatMessage{{Role: "user", Content: "extract"}}}
	chatStructured := chatBase
	chatStructured.ResponseFormat = map[string]interface{}{
		"type": "json_schema",
		"json_schema": map[string]interface{}{"schema": map[string]interface{}{
			"type": "object", "properties": map[string]interface{}{"answer": map[string]interface{}{"type": "string"}},
		}},
	}
	if estimateOpenAIPromptTokens(chatStructured) <= estimateOpenAIPromptTokens(chatBase) {
		t.Fatal("OpenAI response_format did not increase the prompt estimate")
	}

	anthropicBase := model.AnthropicMessageRequest{Messages: []model.AnthropicInputMessage{{Role: "user", Content: "extract"}}}
	anthropicStructured := anthropicBase
	anthropicStructured.OutputConfig = map[string]interface{}{
		"format": map[string]interface{}{"type": "json_schema", "schema": map[string]interface{}{"type": "object"}},
	}
	if estimateAnthropicInputTokens(anthropicStructured) <= estimateAnthropicInputTokens(anthropicBase) {
		t.Fatal("Anthropic output_config did not increase the input estimate")
	}
}
