package handler

import (
	"net/http"
	"strings"

	"vertex2api/config"
	"vertex2api/stats"
)

func Stats(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if cfg.StatsKey == "" {
			WriteJSON(w, http.StatusForbidden, map[string]string{"error": "Stats endpoint disabled (no STATS_KEY configured)"})
			return
		}

		key := ""
		if scheme, value, found := strings.Cut(strings.TrimSpace(r.Header.Get("Authorization")), " "); found && strings.EqualFold(scheme, "Bearer") {
			key = strings.TrimSpace(value)
		}
		if key == "" {
			key = r.Header.Get("x-goog-api-key")
			if key == "" {
				key = r.Header.Get("x-api-key")
			}
		}
		if key == "" {
			key = r.URL.Query().Get("key")
		}

		if !cfg.ValidateStatsKey(key) {
			WriteJSON(w, http.StatusUnauthorized, map[string]string{"error": "Invalid STATS_KEY"})
			return
		}

		WriteJSON(w, http.StatusOK, map[string]interface{}{
			"success": true,
			"stats":   stats.GetStats(),
		})
	}
}
