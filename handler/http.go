package handler

import (
	"context"
	"errors"
	"io"
	"net/http"
	"strings"

	"vertex2api/model"
	"vertex2api/proxy"

	"github.com/bytedance/sonic"
	"github.com/rs/zerolog/log"
)

const publicServerErrorMessage = "Upstream service error"

func publicServerErrorResponse(err error) model.ErrorResponse {
	return model.ErrorResponse{
		Error: &model.APIError{Message: publicServerErrorMessageFor(err), Type: "server_error"},
	}
}

func publicUpstreamErrorMessage(vp *proxy.VertexProxy, err error) string {
	if vp != nil && vp.RedactUpstreamResponses() {
		return publicServerErrorMessage
	}
	return publicServerErrorMessageFor(err)
}

func publicUpstreamErrorResponse(vp *proxy.VertexProxy, err error) model.ErrorResponse {
	return model.ErrorResponse{
		Error: &model.APIError{Message: publicUpstreamErrorMessage(vp, err), Type: "server_error"},
	}
}

func upstreamLogError(vp *proxy.VertexProxy, err error) string {
	if vp == nil {
		if err == nil {
			return ""
		}
		return err.Error()
	}
	return vp.UpstreamLogError(err)
}

func publicServerErrorMessageFor(err error) string {
	if err == nil {
		return publicServerErrorMessage
	}
	return err.Error()
}

// WriteJSON writes a JSON response using the same encoder as the rest of the service.
func WriteJSON(w http.ResponseWriter, status int, value interface{}) {
	data, err := sonic.Marshal(value)
	if err != nil {
		log.Error().Err(err).Msg("JSON response marshal failed")
		http.Error(w, http.StatusText(http.StatusInternalServerError), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(status)
	if _, err := w.Write(data); err != nil {
		log.Debug().Err(err).Msg("JSON response write failed")
	}
}

func readJSONRequest(w http.ResponseWriter, r *http.Request, dst interface{}) ([]byte, bool) {
	body, err := io.ReadAll(r.Body)
	if err != nil {
		var maxBytesErr *http.MaxBytesError
		if errors.As(err, &maxBytesErr) {
			WriteProtocolError(w, r, http.StatusRequestEntityTooLarge, "Request body too large", "invalid_request_error")
			return nil, false
		}
		WriteProtocolError(w, r, http.StatusBadRequest, "Failed to read request body: "+err.Error(), "invalid_request_error")
		return nil, false
	}

	if err := sonic.Unmarshal(body, dst); err != nil {
		WriteProtocolError(w, r, http.StatusBadRequest, "Invalid request body: "+err.Error(), "invalid_request_error")
		return nil, false
	}
	return body, true
}

func requestContextCanceled(ctx context.Context, err error) bool {
	return ctx.Err() != nil || errors.Is(err, context.Canceled)
}

func setSSEHeaders(w http.ResponseWriter) {
	header := w.Header()
	header.Set("Content-Type", "text/event-stream")
	header.Set("Cache-Control", "no-cache")
	header.Set("Connection", "keep-alive")
	header.Set("X-Accel-Buffering", "no")
}

func flushResponse(w http.ResponseWriter) error {
	return http.NewResponseController(w).Flush()
}

func WriteProtocolError(w http.ResponseWriter, r *http.Request, status int, message string, errType string) {
	if isAnthropicRequest(r) {
		anthropicType := anthropicErrorType(status, errType)
		writeAnthropicError(w, status, anthropicType, message)
		return
	}
	if isGeminiRequest(r) {
		WriteJSON(w, status, map[string]interface{}{
			"error": map[string]interface{}{
				"code":    status,
				"message": message,
				"status":  geminiErrorStatus(status),
			},
		})
		return
	}
	WriteJSON(w, status, model.ErrorResponse{
		Error: &model.APIError{Message: message, Type: errType},
	})
}

func writeUpstreamProtocolError(w http.ResponseWriter, r *http.Request, vp *proxy.VertexProxy, err error) {
	status := proxy.HTTPStatusForError(err)
	WriteProtocolError(w, r, status, publicUpstreamErrorMessage(vp, err), openAIErrorType(status))
}

func openAIErrorType(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	default:
		return "server_error"
	}
}

func anthropicErrorType(status int, fallback string) string {
	switch status {
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusServiceUnavailable:
		return "overloaded_error"
	default:
		if fallback != "" && fallback != "server_error" {
			return fallback
		}
		return "api_error"
	}
}

func geminiErrorStatus(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge:
		return "INVALID_ARGUMENT"
	case http.StatusUnauthorized:
		return "UNAUTHENTICATED"
	case http.StatusForbidden:
		return "PERMISSION_DENIED"
	case http.StatusNotFound:
		return "NOT_FOUND"
	case http.StatusTooManyRequests:
		return "RESOURCE_EXHAUSTED"
	case http.StatusServiceUnavailable:
		return "UNAVAILABLE"
	default:
		return "INTERNAL"
	}
}

func isGeminiRequest(r *http.Request) bool {
	if r == nil {
		return false
	}
	path := r.URL.Path
	return strings.HasPrefix(path, "/v1beta") || strings.HasPrefix(path, "/v1beta1") ||
		(strings.HasPrefix(path, "/v1/models/") && strings.Contains(path, ":"))
}
