package config

import (
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

func TestLoadDefaultsAutoFetchModelsAndUpstreamLogRedaction(t *testing.T) {
	t.Setenv("AUTO_FETCH_MODELS", "")
	t.Setenv("REDACT_UPSTREAM_LOGS", "")
	t.Setenv("ALLOW_CUSTOM_MODEL_NAMES", "")
	cfg := Load()

	if !cfg.AutoFetchModels {
		t.Fatal("AUTO_FETCH_MODELS default = false, want true")
	}
	if cfg.RedactUpstreamLogs {
		t.Fatal("REDACT_UPSTREAM_LOGS default = true, want false")
	}
	if cfg.AllowCustomModelNames {
		t.Fatal("ALLOW_CUSTOM_MODEL_NAMES default = true, want false")
	}
}

func TestLoadGeneratesAPIKeyWhenMissing(t *testing.T) {
	t.Setenv("API_KEY", "")
	t.Setenv("ALLOW_UNAUTHENTICATED", "false")

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
