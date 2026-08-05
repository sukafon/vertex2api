package recaptcha

import (
	"context"
	"fmt"
	"math/rand"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"sync"
	"time"

	"vertex2api/client"
	"vertex2api/config"

	"github.com/rs/zerolog/log"
)

const (
	cacheTTL = 90 * time.Second

	recaptchaCo = "aHR0cHM6Ly9jb25zb2xlLmNsb3VkLmdvb2dsZS5jb206NDQz"
	recaptchaV  = "jdMmXeCQEkPbnFDy9T04NbgJ"
	recaptchaHL = "zh-CN"
	recaptchaVH = "6581054572"
)

var rrespRegex = regexp.MustCompile(`rresp","(.*?)"`)
var tokenRegex = regexp.MustCompile(`id="recaptcha-token"\s+value="([^"]+)"`)

type cachedToken struct {
	token     string
	baseURL   string
	expiresAt time.Time
	inUse     bool
}

type TokenLease struct {
	cache   *TokenCache
	token   string
	baseURL string
	once    sync.Once
}

func (tl *TokenLease) Token() string {
	if tl == nil {
		return ""
	}
	return tl.token
}

func (tl *TokenLease) BaseURL() string {
	if tl == nil {
		return ""
	}
	return tl.baseURL
}

func (tl *TokenLease) Release() {
	if tl == nil || tl.cache == nil {
		return
	}
	tl.once.Do(func() {
		tl.cache.release(tl.token)
	})
}

// Retire removes a token that exhausted the upstream retry budget. A retired
// token is never returned to the cache for a later request.
func (tl *TokenLease) Retire() {
	if tl == nil || tl.cache == nil {
		return
	}
	tl.once.Do(func() {
		tl.cache.retire(tl.token)
	})
}

// TokenCache 管理 recaptcha token 的获取与缓存
type TokenCache struct {
	mu                 sync.Mutex
	tokens             map[string]*cachedToken
	httpClient         *client.HTTPClient
	baseURL            string
	prefixBaseURLs     []string
	recaptchaKey       string
	redactUpstreamLogs bool
	randomFingerprint  bool
}

func NewTokenCache(httpClient *client.HTTPClient, cfg *config.Config) *TokenCache {
	return &TokenCache{
		httpClient:         httpClient,
		baseURL:            cfg.RecaptchaBase,
		prefixBaseURLs:     cfg.PrefixRecaptchaBaseURLs,
		recaptchaKey:       cfg.RecaptchaKey,
		redactUpstreamLogs: cfg.RedactUpstreamLogs,
		randomFingerprint:  cfg.RandomFingerprint,
		tokens:             make(map[string]*cachedToken),
	}
}

func (tc *TokenCache) selectedRecaptchaBaseURL() string {
	prefixes := tc.prefixBaseURLs
	if len(prefixes) == 0 {
		return tc.baseURL
	}

	return selectRecaptchaBaseURL(tc.baseURL, prefixes, rand.Intn(10) < 4, rand.Intn(len(prefixes)))
}

func selectRecaptchaBaseURL(baseURL string, prefixes []string, direct bool, prefixIndex int) string {
	if direct || len(prefixes) == 0 {
		return baseURL
	}
	return prefixes[prefixIndex%len(prefixes)] + baseURL
}

func (tc *TokenCache) GetTokenContext(ctx context.Context) (*TokenLease, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if lease := tc.acquireCached(time.Now()); lease != nil {
		return lease, nil
	}

	const maxTokenCollisions = 3
	for i := 0; i < maxTokenCollisions; i++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}

		token, baseURL, err := tc.fetchWithRetry(ctx, 3)
		if err != nil {
			return nil, err
		}
		if lease := tc.storeFetchedToken(token, baseURL, time.Now()); lease != nil {
			return lease, nil
		}
		log.Warn().Msg("Fetched recaptcha token is already in use, fetching another token...")
	}
	return nil, fmt.Errorf("failed to get an unused recaptcha token")
}

func (tc *TokenCache) acquireCached(now time.Time) *TokenLease {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.pruneExpiredLocked(now)
	for token, item := range tc.tokens {
		if item.inUse {
			continue
		}
		item.inUse = true
		return &TokenLease{cache: tc, token: token, baseURL: item.baseURL}
	}
	return nil
}

func (tc *TokenCache) storeFetchedToken(token, baseURL string, now time.Time) *TokenLease {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.pruneExpiredLocked(now)
	if item, ok := tc.tokens[token]; ok && item.inUse {
		return nil
	}

	tc.tokens[token] = &cachedToken{
		token:     token,
		baseURL:   baseURL,
		expiresAt: now.Add(cacheTTL),
		inUse:     true,
	}
	return &TokenLease{cache: tc, token: token, baseURL: baseURL}
}

func (tc *TokenCache) release(token string) {
	tc.mu.Lock()
	defer tc.mu.Unlock()

	item, ok := tc.tokens[token]
	if !ok {
		return
	}
	if time.Now().After(item.expiresAt) {
		delete(tc.tokens, token)
		return
	}
	item.inUse = false
}

func (tc *TokenCache) retire(token string) {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	delete(tc.tokens, token)
}

func (tc *TokenCache) pruneExpiredLocked(now time.Time) {
	for token, item := range tc.tokens {
		if now.After(item.expiresAt) {
			delete(tc.tokens, token)
		}
	}
}

func (tc *TokenCache) fetchWithRetry(ctx context.Context, maxRetry int) (string, string, error) {
	var lastErr error
	for i := 0; i < maxRetry; i++ {
		if err := ctx.Err(); err != nil {
			return "", "", err
		}

		token, baseURL, err := tc.fetchToken(ctx)
		if err == nil {
			log.Info().Msg("recaptcha token acquired successfully")
			return token, baseURL, nil
		}
		lastErr = err
		log.Warn().Str("err", config.UpstreamLogError(err, tc.redactUpstreamLogs, 120)).Int("attempt", i+1).Msg("Failed to get recaptcha token, retrying...")
	}
	return "", "", fmt.Errorf("failed to get recaptcha token after %d attempts: %w", maxRetry, lastErr)
}

func (tc *TokenCache) fetchToken(ctx context.Context) (string, string, error) {
	cb := randomString(10)
	baseURL := tc.selectedRecaptchaBaseURL()
	recaptchaKey := tc.recaptchaKey
	// Keep one browser profile for the anchor/reload pair when enabled. A new
	// fetch (and therefore a new retry) selects another profile, matching the
	// reference implementation's per-attempt emulation behavior.
	var fingerprint client.Fingerprint
	if tc.randomFingerprint {
		fingerprint = client.NewRandomFingerprint()
	}

	anchorURL := fmt.Sprintf(
		"%s/recaptcha/enterprise/anchor?ar=1&k=%s&co=%s&hl=%s&v=%s&size=invisible&anchor-ms=20000&execute-ms=15000&cb=%s",
		baseURL, recaptchaKey, recaptchaCo, recaptchaHL, recaptchaV, cb,
	)
	reloadURL := fmt.Sprintf(
		"%s/recaptcha/enterprise/reload?k=%s",
		baseURL, recaptchaKey,
	)

	// Step 1: GET anchor page → 提取 recaptcha-token
	anchorReq, err := http.NewRequestWithContext(ctx, "GET", anchorURL, nil)
	if err != nil {
		return "", "", fmt.Errorf("build anchor request: %w", err)
	}
	if tc.randomFingerprint {
		fingerprint.ApplyNavigationHeaders(anchorReq)
	}

	anchorBody, statusCode, err := tc.httpClient.Do(anchorReq)
	if err != nil {
		return "", "", fmt.Errorf("anchor request failed: %w", err)
	}
	if statusCode != 200 {
		return "", "", fmt.Errorf("anchor request status %d", statusCode)
	}

	matches := tokenRegex.FindSubmatch(anchorBody)
	if matches == nil {
		return "", "", fmt.Errorf("recaptcha-token element not found in anchor HTML")
	}
	baseToken := string(matches[1])

	// Step 2: POST reload → 换取最终 token
	if err := ctx.Err(); err != nil {
		return "", "", err
	}

	form := url.Values{
		"v":      {recaptchaV},
		"reason": {"q"},
		"k":      {recaptchaKey},
		"c":      {baseToken},
		"co":     {recaptchaCo},
		"hl":     {recaptchaHL},
		"size":   {"invisible"},
		"vh":     {recaptchaVH},
		"chr":    {""},
		"bg":     {""},
	}

	reloadReq, err := http.NewRequestWithContext(ctx, "POST", reloadURL, strings.NewReader(form.Encode()))
	if err != nil {
		return "", "", fmt.Errorf("build reload request: %w", err)
	}
	if tc.randomFingerprint {
		fingerprint.ApplyXHRHeaders(
			reloadReq,
			"application/x-www-form-urlencoded;charset=UTF-8",
			"*/*",
			strings.TrimRight(baseURL, "/"),
			anchorURL,
			"same-origin",
		)
	} else {
		reloadReq.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}

	reloadBody, statusCode, err := tc.httpClient.Do(reloadReq)
	if err != nil {
		return "", "", fmt.Errorf("reload request failed: %w", err)
	}
	if statusCode != 200 {
		return "", "", fmt.Errorf("reload request status %d", statusCode)
	}

	rrespMatch := rrespRegex.FindSubmatch(reloadBody)
	if rrespMatch == nil {
		return "", "", fmt.Errorf("rresp not found in reload response")
	}

	return string(rrespMatch[1]), baseURL, nil
}

// randomString 生成指定长度的随机字符串
func randomString(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, n)
	for i := range b {
		b[i] = letters[rand.Intn(len(letters))]
	}
	return string(b)
}
