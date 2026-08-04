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
