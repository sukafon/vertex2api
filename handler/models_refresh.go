package handler

import (
	"context"
	"net/http"

	"vertex2api/config"

	"github.com/rs/zerolog/log"
)

// ModelCatalogRefreshFunc refreshes the in-memory model catalog and returns
// the number of upstream models applied to it.
type ModelCatalogRefreshFunc func(context.Context) (int, error)

// RefreshModels handles POST /v1/models/refresh using the dedicated STATS_KEY.
func RefreshModels(cfg *config.Config, refresh ModelCatalogRefreshFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !requireStatsKey(w, r, cfg, "Model refresh endpoint disabled (no STATS_KEY configured)") {
			return
		}
		if refresh == nil {
			log.Error().Msg("Manual model catalog refresh is unavailable")
			WriteJSON(w, http.StatusInternalServerError, map[string]interface{}{
				"success": false,
				"error":   "Model catalog refresh is unavailable",
			})
			return
		}

		count, err := refresh(r.Context())
		if err != nil {
			log.Error().Str("err", config.UpstreamLogError(err, false, 512)).Msg("Manual model catalog refresh failed; keeping existing catalog")
			WriteJSON(w, http.StatusBadGateway, map[string]interface{}{
				"success": false,
				"error":   "Failed to refresh model catalog from upstream",
			})
			return
		}
		if count == 0 {
			log.Warn().Msg("Manual model catalog refresh returned no models; keeping existing catalog")
			WriteJSON(w, http.StatusBadGateway, map[string]interface{}{
				"success": false,
				"error":   "Upstream model catalog was empty; existing catalog was kept",
			})
			return
		}

		WriteJSON(w, http.StatusOK, map[string]interface{}{
			"success":     true,
			"updated":     true,
			"model_count": count,
		})
	}
}
