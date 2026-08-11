package proxy

import (
	"context"
	"testing"
)

func TestCompatibilityLayerFromContext(t *testing.T) {
	if got := compatibilityLayerFromContext(context.Background()); got != CompatibilityLayerGeminiNative {
		t.Fatalf("default compatibility layer = %q", got)
	}
	ctx := WithCompatibilityLayer(context.Background(), CompatibilityLayerOpenAIResponses)
	if got := compatibilityLayerFromContext(ctx); got != CompatibilityLayerOpenAIResponses {
		t.Fatalf("compatibility layer = %q", got)
	}
	if got := compatibilityLayerLogMessage(ctx, "stream failed, retrying on current token..."); got != "OpenAI Responses stream failed, retrying on current token..." {
		t.Fatalf("compatibility log message = %q", got)
	}
}
