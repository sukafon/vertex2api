package middleware

import (
	"fmt"
	"net/http"
	"strings"
	"time"

	"vertex2api/config"

	"github.com/bytedance/sonic"
	"github.com/rs/zerolog/log"
)

// Auth returns API key middleware.
// It supports Authorization: Bearer, x-api-key, x-goog-api-key, and ?key=.
func Auth(cfg *config.Config) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if (len(cfg.APIKeys) == 0 && cfg.AllowUnauthenticated) || r.URL.Path == "/v1/stats" || r.URL.Path == "/health" {
				next.ServeHTTP(w, r)
				return
			}

			key := extractKey(r)
			if key == "" {
				writeAuthError(w, r, http.StatusUnauthorized, "Missing API key. Provide via Authorization header, x-api-key header, x-goog-api-key header, or key query parameter.", "missing_api_key")
				return
			}

			if !cfg.ValidateKey(key) {
				writeAuthError(w, r, http.StatusUnauthorized, "Invalid API key.", "invalid_api_key")
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

func extractKey(r *http.Request) string {
	if auth := r.Header.Get("Authorization"); auth != "" {
		scheme, value, found := strings.Cut(strings.TrimSpace(auth), " ")
		if found && strings.EqualFold(scheme, "Bearer") {
			return strings.TrimSpace(value)
		}
	}

	if key := r.Header.Get("x-goog-api-key"); key != "" {
		return key
	}

	if key := r.Header.Get("x-api-key"); key != "" {
		return key
	}

	return r.URL.Query().Get("key")
}

func writeAuthError(w http.ResponseWriter, r *http.Request, status int, message, code string) {
	if isAnthropicRequest(r) {
		errType := "authentication_error"
		if status == http.StatusForbidden {
			errType = "permission_error"
		}
		data, _ := sonic.Marshal(map[string]interface{}{
			"type":       "error",
			"request_id": fmt.Sprintf("req_%d", time.Now().UnixNano()),
			"error": map[string]interface{}{
				"type":    errType,
				"message": message,
			},
		})
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(status)
		_, _ = w.Write(data)
		return
	}

	if isGeminiRequest(r) {
		statusStr := "UNAUTHENTICATED"
		if status == http.StatusForbidden {
			statusStr = "PERMISSION_DENIED"
		}
		data, _ := sonic.Marshal(map[string]interface{}{
			"error": map[string]interface{}{
				"code":    status,
				"message": message,
				"status":  statusStr,
			},
		})
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		w.WriteHeader(status)
		_, _ = w.Write(data)
		return
	}

	resp := map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "invalid_request_error",
			"code":    code,
		},
	}
	data, err := sonic.Marshal(resp)
	if err != nil {
		log.Error().Err(err).Msg("Auth error response marshal failed")
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	_, _ = w.Write(data)
}

func isAnthropicRequest(r *http.Request) bool {
	if r == nil {
		return false
	}
	if strings.HasPrefix(r.URL.Path, "/v1/messages") {
		return true
	}
	if r.Header.Get("anthropic-version") != "" || r.Header.Get("anthropic-beta") != "" {
		return true
	}
	ua := strings.ToLower(r.Header.Get("User-Agent"))
	return strings.Contains(ua, "claude") || strings.Contains(ua, "anthropic")
}

func isGeminiRequest(r *http.Request) bool {
	if r == nil {
		return false
	}
	path := r.URL.Path
	return strings.HasPrefix(path, "/v1beta") || strings.HasPrefix(path, "/v1beta1") ||
		(strings.HasPrefix(path, "/v1/models/") && strings.Contains(path, ":"))
}
