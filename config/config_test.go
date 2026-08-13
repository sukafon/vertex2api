package config

import (
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestParseEnvListTrimsAndDropsEmptyItems(t *testing.T) {
	got := parseEnvList(" https://p1.example/ ,https://p2.example/,, ")
	want := []string{"https://p1.example/", "https://p2.example/"}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("parseEnvList() = %#v, want %#v", got, want)
	}
}

func TestValidateFailsClosedWithoutAPIKey(t *testing.T) {
	cfg := validTestConfig()
	cfg.APIKeys = nil

	if err := cfg.Validate(); err == nil {
		t.Fatal("Validate() succeeded without API_KEY or explicit unauthenticated mode")
	}

	cfg.AllowUnauthenticated = true
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate() rejected explicit unauthenticated mode: %v", err)
	}
}

func TestValidateRequiresBrowserChainKeys(t *testing.T) {
	for _, mutate := range []func(*Config){
		func(cfg *Config) { cfg.GraphQLAPIKey = "" },
		func(cfg *Config) { cfg.RecaptchaKey = "" },
	} {
		cfg := validTestConfig()
		mutate(cfg)
		if err := cfg.Validate(); err == nil {
			t.Fatal("Validate() succeeded without a required browser-chain key")
		}
	}
}

func TestUpstreamLogValueHonorsRedactionAndTruncation(t *testing.T) {
	if got := UpstreamLogValue("line 1\nline 2", false, 0); got != "line 1 line 2" {
		t.Fatalf("unredacted log value = %q", got)
	}
	if got := UpstreamLogValue("sensitive", true, 0); got != "[REDACTED]" {
		t.Fatalf("redacted log value = %q", got)
	}
	if got := UpstreamLogValue("abcdef", false, 3); got != "abc..." {
		t.Fatalf("truncated log value = %q", got)
	}
}

func TestLoadDefaultsAutoFetchModelsAndUpstreamResponseRedaction(t *testing.T) {
	t.Setenv("AUTO_FETCH_MODELS", "")
	t.Setenv("REDACT_UPSTREAM_RESPONSES", "")
	t.Setenv("LOG_CODE3_REQUEST_BODIES", "")
	t.Setenv("RANDOM_FINGERPRINT", "")
	t.Setenv("TLS_CLIENT_PROFILE", "")
	t.Setenv("ALLOW_CUSTOM_MODEL_NAMES", "")
	t.Setenv("GEMINI_STRICT_ALT_SSE", "")
	t.Setenv("REJECT_CHAT_LIVENESS_PROBES", "")
	t.Setenv("RESPOND_CHAT_LIVENESS_PROBES", "")
	cfg := Load()

	if !cfg.AutoFetchModels {
		t.Fatal("AUTO_FETCH_MODELS default = false, want true")
	}
	if cfg.RedactUpstreamResponses {
		t.Fatal("REDACT_UPSTREAM_RESPONSES default = true, want false")
	}
	if cfg.LogCode3RequestBodies {
		t.Fatal("LOG_CODE3_REQUEST_BODIES default = true, want false")
	}
	if cfg.RandomFingerprint {
		t.Fatal("RANDOM_FINGERPRINT default = true, want false")
	}
	if cfg.TLSClientProfile != "chrome_146" {
		t.Fatalf("TLS_CLIENT_PROFILE default = %q, want chrome_146", cfg.TLSClientProfile)
	}
	if cfg.AllowCustomModelNames {
		t.Fatal("ALLOW_CUSTOM_MODEL_NAMES default = true, want false")
	}
	if cfg.GeminiStrictAltSSE {
		t.Fatal("GEMINI_STRICT_ALT_SSE default = true, want false")
	}
	if cfg.RejectChatLivenessProbe {
		t.Fatal("REJECT_CHAT_LIVENESS_PROBES default = true, want false")
	}
	if cfg.ReplyChatLivenessProbe {
		t.Fatal("RESPOND_CHAT_LIVENESS_PROBES default = true, want false")
	}
}

func TestLoadCanEnableUpstreamResponseRedaction(t *testing.T) {
	t.Setenv("REDACT_UPSTREAM_RESPONSES", "true")
	if cfg := Load(); !cfg.RedactUpstreamResponses {
		t.Fatal("REDACT_UPSTREAM_RESPONSES=true did not enable response redaction")
	}
}

func TestLoadCanEnableCode3RequestBodyLogging(t *testing.T) {
	t.Setenv("LOG_CODE3_REQUEST_BODIES", "true")
	if cfg := Load(); !cfg.LogCode3RequestBodies {
		t.Fatal("LOG_CODE3_REQUEST_BODIES = false, want true")
	}
}

func TestLoadCanEnableRandomFingerprint(t *testing.T) {
	t.Setenv("RANDOM_FINGERPRINT", "true")
	if cfg := Load(); !cfg.RandomFingerprint {
		t.Fatal("RANDOM_FINGERPRINT = false, want true")
	}
}

func TestLoadCanEnableChatLivenessProbeRejection(t *testing.T) {
	t.Setenv("REJECT_CHAT_LIVENESS_PROBES", "true")
	if cfg := Load(); !cfg.RejectChatLivenessProbe {
		t.Fatal("REJECT_CHAT_LIVENESS_PROBES = false, want true")
	}
}

func TestLoadCanEnableConstructedChatLivenessProbeResponses(t *testing.T) {
	t.Setenv("RESPOND_CHAT_LIVENESS_PROBES", "true")
	if cfg := Load(); !cfg.ReplyChatLivenessProbe {
		t.Fatal("RESPOND_CHAT_LIVENESS_PROBES = false, want true")
	}
}

func TestLoadGeneratesAPIKeyWhenMissing(t *testing.T) {
	t.Setenv("API_KEY", "")
	t.Setenv("ALLOW_UNAUTHENTICATED", "false")
	t.Setenv("API_KEY_FILE", filepath.Join(t.TempDir(), "api-key"))

	cfg := Load()
	if cfg.GeneratedAPIKey == "" {
		t.Fatal("GeneratedAPIKey is empty")
	}
	if len(cfg.APIKeys) != 1 || cfg.APIKeys[0] != cfg.GeneratedAPIKey {
		t.Fatalf("APIKeys = %#v, want the generated API key", cfg.APIKeys)
	}
	if !strings.HasPrefix(cfg.GeneratedAPIKey, generatedAPIKeyPrefix) {
		t.Fatalf("generated API key = %q, want %q prefix", cfg.GeneratedAPIKey, generatedAPIKeyPrefix)
	}
	if len(cfg.GeneratedAPIKey) != len(generatedAPIKeyPrefix)+64 {
		t.Fatalf("generated API key length = %d, want %d", len(cfg.GeneratedAPIKey), len(generatedAPIKeyPrefix)+64)
	}
	for _, char := range cfg.GeneratedAPIKey[len(generatedAPIKeyPrefix):] {
		if !strings.ContainsRune("0123456789abcdef", char) {
			t.Fatalf("generated API key contains non-hex character %q", char)
		}
	}
}

func TestLoadReusesPersistedAPIKey(t *testing.T) {
	t.Setenv("API_KEY", "")
	t.Setenv("ALLOW_UNAUTHENTICATED", "false")
	t.Setenv("API_KEY_FILE", filepath.Join(t.TempDir(), "api-key"))

	first := Load()
	if first.GeneratedAPIKey == "" {
		t.Fatal("first Load() did not generate an API key")
	}

	second := Load()
	if second.GeneratedAPIKey != "" {
		t.Fatal("second Load() generated a new API key instead of reusing the persisted key")
	}
	if len(second.APIKeys) != 1 || second.APIKeys[0] != first.GeneratedAPIKey {
		t.Fatalf("second APIKeys = %#v, want persisted key %q", second.APIKeys, first.GeneratedAPIKey)
	}
}

func TestLoadDoesNotGenerateAPIKeyForExplicitUnauthenticatedMode(t *testing.T) {
	t.Setenv("API_KEY", "")
	t.Setenv("ALLOW_UNAUTHENTICATED", "true")

	cfg := Load()
	if cfg.GeneratedAPIKey != "" {
		t.Fatalf("GeneratedAPIKey = %q, want empty", cfg.GeneratedAPIKey)
	}
	if len(cfg.APIKeys) != 0 {
		t.Fatalf("APIKeys = %#v, want empty", cfg.APIKeys)
	}
}

func TestLoadAllowsCustomModelNamesWhenConfigured(t *testing.T) {
	t.Setenv("ALLOW_CUSTOM_MODEL_NAMES", "true")
	if cfg := Load(); !cfg.AllowCustomModelNames {
		t.Fatal("ALLOW_CUSTOM_MODEL_NAMES = false, want true")
	}
}

func TestLoadCanRequireGeminiAltSSE(t *testing.T) {
	t.Setenv("GEMINI_STRICT_ALT_SSE", "true")
	if cfg := Load(); !cfg.GeminiStrictAltSSE {
		t.Fatal("GEMINI_STRICT_ALT_SSE = false, want true")
	}
}

func TestLoadHost(t *testing.T) {
	for _, want := range []string{"0.0.0.0", "127.0.0.1"} {
		t.Setenv("HOST", want)
		if got := Load().Host; got != want {
			t.Fatalf("HOST = %q, want %q", got, want)
		}
	}
}

func TestValidateRejectsPlaceholderAndMalformedValues(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Config)
	}{
		{name: "placeholder key", mutate: func(c *Config) { c.APIKeys = []string{"replace-with-random-secret"} }},
		{name: "invalid port", mutate: func(c *Config) { c.Port = "70000" }},
		{name: "invalid base URL", mutate: func(c *Config) { c.VertexBaseURL = "not-a-url" }},
		{name: "obsolete GraphQL host", mutate: func(c *Config) { c.VertexBaseURL = "https://content-aiplatform.googleapis.com" }},
		{name: "invalid proxy", mutate: func(c *Config) { c.Proxy = "://bad" }},
		{name: "invalid retry", mutate: func(c *Config) { c.MaxRetry = 0 }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := validTestConfig()
			tt.mutate(cfg)
			if err := cfg.Validate(); err == nil {
				t.Fatal("Validate() succeeded for invalid configuration")
			}
		})
	}
}

func TestValidateRejectsShortSecrets(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Config)
	}{
		{name: "api key", mutate: func(c *Config) { c.APIKeys = []string{"short"} }},
		{name: "stats key", mutate: func(c *Config) { c.StatsKey = "short" }},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cfg := validTestConfig()
			tt.mutate(cfg)
			if err := cfg.Validate(); err == nil {
				t.Fatal("Validate() accepted a secret shorter than 16 characters")
			}
		})
	}
}

func validTestConfig() *Config {
	return &Config{
		APIKeys:             []string{"test-secret-123456"},
		Host:                "0.0.0.0",
		Port:                "8080",
		VertexBaseURL:       "https://vertex.example",
		GraphQLAPIKey:       "test-graphql-key",
		RecaptchaBase:       "https://recaptcha.example",
		RecaptchaKey:        "test-recaptcha-key",
		MaxRetry:            3,
		MaxRefresh:          3,
		RetryDelayMs:        100,
		HTTPTimeoutSeconds:  30,
		WriteTimeoutSeconds: 60,
	}
}
