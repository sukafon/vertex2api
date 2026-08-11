package proxy

import (
	"context"
	"strings"
)

const (
	CompatibilityLayerGeminiNative          = "Gemini Native"
	CompatibilityLayerOpenAIChatCompletions = "OpenAI Chat Completions"
	CompatibilityLayerOpenAIResponses       = "OpenAI Responses"
	CompatibilityLayerAnthropicMessages     = "Anthropic Messages"
	CompatibilityLayerOpenAIImages          = "OpenAI Images"
)

type compatibilityLayerContextKey struct{}

// WithCompatibilityLayer identifies the public compatibility protocol that
// originated an upstream request. It affects diagnostics only.
func WithCompatibilityLayer(ctx context.Context, layer string) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	layer = strings.TrimSpace(layer)
	if layer == "" {
		layer = CompatibilityLayerGeminiNative
	}
	return context.WithValue(ctx, compatibilityLayerContextKey{}, layer)
}

func compatibilityLayerFromContext(ctx context.Context) string {
	if ctx != nil {
		if layer, ok := ctx.Value(compatibilityLayerContextKey{}).(string); ok {
			if layer = strings.TrimSpace(layer); layer != "" {
				return layer
			}
		}
	}
	return CompatibilityLayerGeminiNative
}

func compatibilityLayerLogMessage(ctx context.Context, message string) string {
	message = strings.TrimSpace(message)
	if message == "" {
		return compatibilityLayerFromContext(ctx)
	}
	return compatibilityLayerFromContext(ctx) + " " + message
}
