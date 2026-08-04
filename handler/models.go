package handler

import (
	"net/http"

	"vertex2api/model"
)

// ModelsList handles GET /v1/models for both OpenAI-compatible and
// Anthropic-compatible clients.
func ModelsList() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if isAnthropicRequest(r) {
			WriteJSON(w, http.StatusOK, model.AnthropicModelList())
			return
		}
		WriteJSON(w, http.StatusOK, model.OpenAIModelList())
	})
}

func RetrieveModel() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		modelID := r.PathValue("modelID")
		if isAnthropicRequest(r) {
			anthropicResp := model.AnthropicModelList()
			for _, item := range anthropicResp.Data {
				if item.ID == modelID {
					WriteJSON(w, http.StatusOK, item)
					return
				}
			}
			writeAnthropicError(w, http.StatusNotFound, "not_found_error", "model not found")
			return
		}

		openAIResp := model.OpenAIModelList()
		for _, item := range openAIResp.Data {
			if item.ID == modelID {
				WriteJSON(w, http.StatusOK, item)
				return
			}
		}
		WriteJSON(w, http.StatusNotFound, model.ErrorResponse{
			Error: &model.APIError{Message: "model not found", Type: "invalid_request_error", Code: "model_not_found"},
		})
	})
}

// GeminiListModels handles GET /v1beta/models.
func GeminiListModels() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		WriteJSON(w, http.StatusOK, model.GeminiModelList())
	})
}
