package config

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode/utf8"

	"github.com/joho/godotenv"
	"github.com/rs/zerolog/log"
)

type Config struct {
	APIKeys                 []string
	GeneratedAPIKey         string
	APIKeyGenerationError   error
	APIKeyFile              string
	AllowUnauthenticated    bool
	AllowCustomModelNames   bool
	StatsKey                string
	Host                    string
	Port                    string
	VertexBaseURL           string
	GraphQLAPIKey           string
	PrefixVertexBaseURLs    []string
	RecaptchaBase           string
	RecaptchaKey            string
	PrefixRecaptchaBaseURLs []string
	Proxy                   string
	MaxRetry                int
	MaxRefresh              int
	RetryDelayMs            int
	HTTPTimeoutSeconds      int
	WriteTimeoutSeconds     int
	AutoFetchModels         bool
	AutoFetchCron           string
	RedactUpstreamResponses bool
	LogCode3RequestBodies   bool
	RandomFingerprint       bool
	TLSClientProfile        string
	CORSAllowOrigin         string
}

const DefaultVertexBaseURL = "https://cloudconsole-pa.clients6.google.com"

const defaultAPIKeyFile = ".vertex2api-api-key"

func Load() *Config {
	if err := godotenv.Load(); err != nil {
		log.Debug().Msg("No .env file found, using environment variables")
	}
	cfg := &Config{
		APIKeyFile:              strings.TrimSpace(getEnv("API_KEY_FILE", defaultAPIKeyFile)),
		AllowUnauthenticated:    getEnvBool("ALLOW_UNAUTHENTICATED", false),
		AllowCustomModelNames:   getEnvBool("ALLOW_CUSTOM_MODEL_NAMES", false),
		StatsKey:                getEnv("STATS_KEY", ""),
		Host:                    getEnv("HOST", "0.0.0.0"),
		Port:                    getEnv("PORT", "8080"),
		VertexBaseURL:           getEnv("VERTEX_BASE_URL", DefaultVertexBaseURL),
		GraphQLAPIKey:           getEnv("GRAPHQL_API_KEY", ""),
		PrefixVertexBaseURLs:    parseEnvList(getEnv("PREFIX_VERTEX_BASE_URL", "")),
		RecaptchaBase:           getEnv("RECAPTCHA_BASE_URL", "https://www.recaptcha.net"),
		RecaptchaKey:            getEnv("RECAPTCHA_KEY", ""),
		PrefixRecaptchaBaseURLs: parseEnvList(getEnv("PREFIX_RECAPTCHA_BASE_URL", "")),
		Proxy:                   getEnv("PROXY", ""),
		MaxRetry:                getEnvInt("MAX_RETRY", 3),
		MaxRefresh:              getEnvInt("MAX_REFRESH", 3),
		RetryDelayMs:            getEnvInt("RETRY_DELAY_MS", 1000),
		HTTPTimeoutSeconds:      getEnvInt("HTTP_TIMEOUT_SECONDS", 120),
		WriteTimeoutSeconds:     getEnvInt("WRITE_TIMEOUT_SECONDS", 600),
		AutoFetchModels:         getEnvBool("AUTO_FETCH_MODELS", true),
		AutoFetchCron:           getEnv("AUTO_FETCH_CRON", "0 0,4 * * *"),
		RedactUpstreamResponses: getEnvBool("REDACT_UPSTREAM_RESPONSES", false),
		LogCode3RequestBodies:   getEnvBool("LOG_CODE3_REQUEST_BODIES", false),
		RandomFingerprint:       getEnvBool("RANDOM_FINGERPRINT", false),
		TLSClientProfile:        getEnv("TLS_CLIENT_PROFILE", "chrome_146"),
		CORSAllowOrigin:         strings.TrimSpace(getEnv("CORS_ALLOW_ORIGIN", "")),
	}

	// 解析逗号分隔的 API Key 数组。未提供 API_KEY 时，优先复用持久化密钥，
	// 不存在时才生成并保存一个新的随机密钥。
	raw := getEnv("API_KEY", "")
	if raw != "" {
		cfg.APIKeys = parseEnvList(raw)
	}
	if len(cfg.APIKeys) == 0 && !cfg.AllowUnauthenticated {
		persistedKey, err := loadPersistedAPIKey(cfg.APIKeyFile)
		if err != nil {
			cfg.APIKeyGenerationError = fmt.Errorf("load persisted API_KEY: %w", err)
		} else if persistedKey != "" {
			cfg.APIKeys = []string{persistedKey}
		} else {
			generatedKey, generateErr := generateAPIKey()
			if generateErr != nil {
				cfg.APIKeyGenerationError = generateErr
			} else {
				persistedKey, persistErr := persistAPIKey(cfg.APIKeyFile, generatedKey)
				if persistErr != nil {
					cfg.APIKeyGenerationError = fmt.Errorf("persist API_KEY: %w", persistErr)
				} else {
					cfg.APIKeys = []string{persistedKey}
					if persistedKey == generatedKey {
						cfg.GeneratedAPIKey = generatedKey
					}
				}
			}
		}
	}

	return cfg
}

// Validate rejects insecure or malformed startup configuration.
func (c *Config) Validate() error {
	if c == nil {
		return fmt.Errorf("configuration is nil")
	}
	if c.APIKeyGenerationError != nil {
		return fmt.Errorf("generate API_KEY: %w", c.APIKeyGenerationError)
	}
	if len(c.APIKeys) == 0 && !c.AllowUnauthenticated {
		return fmt.Errorf("API_KEY is required unless ALLOW_UNAUTHENTICATED=true")
	}
	for name, key := range map[string]string{
		"API_KEY":         strings.Join(c.APIKeys, ","),
		"STATS_KEY":       c.StatsKey,
		"GRAPHQL_API_KEY": c.GraphQLAPIKey,
		"RECAPTCHA_KEY":   c.RecaptchaKey,
	} {
		if isPlaceholderSecret(key) {
			return fmt.Errorf("placeholder %s must be replaced before startup", name)
		}
	}
	for _, key := range c.APIKeys {
		if secretLength(key) < 16 {
			return fmt.Errorf("API_KEY entries must be at least 16 characters")
		}
	}
	if c.StatsKey != "" && secretLength(c.StatsKey) < 16 {
		return fmt.Errorf("STATS_KEY must be at least 16 characters")
	}
	if strings.TrimSpace(c.GraphQLAPIKey) == "" {
		return fmt.Errorf("GRAPHQL_API_KEY is required")
	}
	if strings.TrimSpace(c.RecaptchaKey) == "" {
		return fmt.Errorf("RECAPTCHA_KEY is required")
	}
	port, err := strconv.Atoi(c.Port)
	if err != nil || port < 1 || port > 65535 {
		return fmt.Errorf("PORT must be an integer from 1 to 65535")
	}
	if err := validateHTTPURL("VERTEX_BASE_URL", c.VertexBaseURL); err != nil {
		return err
	}
	if strings.TrimRight(c.VertexBaseURL, "/") == "https://content-aiplatform.googleapis.com" {
		return fmt.Errorf("VERTEX_BASE_URL points to a host that does not serve the anonymous GraphQL endpoint; use %s", DefaultVertexBaseURL)
	}
	if err := validateHTTPURL("RECAPTCHA_BASE_URL", c.RecaptchaBase); err != nil {
		return err
	}
	for _, value := range c.PrefixVertexBaseURLs {
		if err := validateHTTPURL("PREFIX_VERTEX_BASE_URL", value); err != nil {
			return err
		}
	}
	for _, value := range c.PrefixRecaptchaBaseURLs {
		if err := validateHTTPURL("PREFIX_RECAPTCHA_BASE_URL", value); err != nil {
			return err
		}
	}
	if c.Proxy != "" {
		parsed, err := url.Parse(c.Proxy)
		if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https" && parsed.Scheme != "socks5") {
			return fmt.Errorf("PROXY must be a valid http, https, or socks5 URL")
		}
	}
	if c.CORSAllowOrigin != "" && c.CORSAllowOrigin != "*" {
		parsed, err := url.Parse(c.CORSAllowOrigin)
		if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") || parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" || (parsed.Path != "" && parsed.Path != "/") {
			return fmt.Errorf("CORS_ALLOW_ORIGIN must be *, or an http/https origin without a path")
		}
	}
	if c.MaxRetry <= 0 {
		return fmt.Errorf("MAX_RETRY must be greater than zero")
	}
	if c.MaxRefresh <= 0 {
		return fmt.Errorf("MAX_REFRESH must be greater than zero")
	}
	if c.RetryDelayMs < 0 {
		return fmt.Errorf("RETRY_DELAY_MS must not be negative")
	}
	if c.HTTPTimeoutSeconds <= 0 {
		return fmt.Errorf("HTTP_TIMEOUT_SECONDS must be greater than zero")
	}
	if c.WriteTimeoutSeconds <= 0 {
		return fmt.Errorf("WRITE_TIMEOUT_SECONDS must be greater than zero")
	}
	return nil
}

const generatedAPIKeyPrefix = "sk-"

func generateAPIKey() (string, error) {
	bytes := make([]byte, 32)
	if _, err := rand.Read(bytes); err != nil {
		return "", fmt.Errorf("read secure random bytes: %w", err)
	}
	return generatedAPIKeyPrefix + hex.EncodeToString(bytes), nil
}

func loadPersistedAPIKey(path string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return "", nil
	}

	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return "", nil
	}
	if err != nil {
		return "", err
	}

	key := strings.TrimSpace(string(data))
	if key == "" {
		return "", nil
	}
	if len(parseEnvList(key)) != 1 || secretLength(key) < 16 {
		return "", fmt.Errorf("%s contains an invalid API key", path)
	}
	return key, nil
}

func persistAPIKey(path, generatedKey string) (string, error) {
	path = strings.TrimSpace(path)
	if path == "" {
		return generatedKey, nil
	}

	dir := filepath.Dir(path)
	if dir != "." {
		if err := os.MkdirAll(dir, 0o700); err != nil {
			return "", err
		}
	}

	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if os.IsExist(err) {
		// Another process may have generated the key concurrently. Reuse the
		// key that won the race instead of authenticating with an unpersisted key.
		return loadPersistedAPIKey(path)
	}
	if err != nil {
		return "", err
	}

	if _, err := file.WriteString(generatedKey + "\n"); err != nil {
		_ = file.Close()
		return "", err
	}
	if err := file.Close(); err != nil {
		return "", err
	}
	return generatedKey, nil
}

// ValidateKey 检查给定 key 是否在白名单中
func (c *Config) ValidateKey(key string) bool {
	valid := 0
	for _, k := range c.APIKeys {
		valid |= constantTimeSecretMatch(k, key)
	}
	return valid == 1
}

func (c *Config) ValidateStatsKey(key string) bool {
	return c.StatsKey != "" && constantTimeSecretMatch(c.StatsKey, key) == 1
}

func constantTimeSecretMatch(expected, actual string) int {
	expectedHash := sha256.Sum256([]byte(expected))
	actualHash := sha256.Sum256([]byte(actual))
	return subtle.ConstantTimeCompare(expectedHash[:], actualHash[:])
}

func validateHTTPURL(name, value string) error {
	parsed, err := url.Parse(value)
	if err != nil || parsed.Host == "" || (parsed.Scheme != "http" && parsed.Scheme != "https") {
		return fmt.Errorf("%s must be a valid http or https URL", name)
	}
	return nil
}

func isPlaceholderSecret(value string) bool {
	value = strings.ToLower(strings.TrimSpace(value))
	return strings.Contains(value, "replace-with") || strings.Contains(value, "change-me") || strings.Contains(value, "changeme")
}

func secretLength(value string) int {
	return utf8.RuneCountInString(strings.TrimSpace(value))
}

// UpstreamLogValue keeps upstream error details out of logs when requested.
// The default is deliberately non-redacting for diagnostics; callers decide
// the maximum length before logging.
func UpstreamLogValue(value string, redact bool, maxLen int) string {
	if redact {
		return "[REDACTED]"
	}
	value = strings.TrimSpace(value)
	value = strings.ReplaceAll(value, "\r\n", " ")
	value = strings.ReplaceAll(value, "\n", " ")
	if maxLen > 0 {
		runes := []rune(value)
		if len(runes) > maxLen {
			return string(runes[:maxLen]) + "..."
		}
	}
	return value
}

// UpstreamLogError formats an upstream error using the configured redaction
// policy without exposing its wrapped error object to the logger.
func UpstreamLogError(err error, redact bool, maxLen int) string {
	if err == nil {
		return ""
	}
	return UpstreamLogValue(err.Error(), redact, maxLen)
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func parseEnvList(raw string) []string {
	var values []string
	for _, item := range strings.Split(raw, ",") {
		item = strings.TrimSpace(item)
		if item != "" {
			values = append(values, item)
		}
	}
	return values
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func getEnvBool(key string, fallback bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
		vLower := strings.ToLower(strings.TrimSpace(v))
		return vLower == "true" || vLower == "1" || vLower == "yes"
	}
	return fallback
}
