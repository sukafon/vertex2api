package handler

import (
	"encoding/base64"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"vertex2api/model"
	"vertex2api/proxy"

	"github.com/rs/zerolog/log"
)

const defaultOpenAIImageModel = "gemini-3-pro-image"

// ImageGenerations handles POST /v1/images/generations.
func ImageGenerations(vp *proxy.VertexProxy, allowCustomModelNames bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(proxy.WithCompatibilityLayer(r.Context(), proxy.CompatibilityLayerOpenAIImages))
		contentType := r.Header.Get("Content-Type")

		var prompt, modelName string
		var imageB64 string
		var imageMime string
		var n int
		var nProvided bool
		var responseFormat string
		var size, quality string

		if isMultipart(contentType) {
			if err := r.ParseMultipartForm(32 << 20); err != nil {
				WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{
					Error: &model.APIError{Message: "Invalid multipart form: " + err.Error(), Type: "invalid_request_error"},
				})
				return
			}

			prompt = r.FormValue("prompt")
			modelName = r.FormValue("model")
			responseFormat = r.FormValue("response_format")
			size = r.FormValue("size")
			quality = r.FormValue("quality")
			if nStr := r.FormValue("n"); nStr != "" {
				nProvided = true
				parsed, err := strconv.Atoi(nStr)
				if err != nil {
					writeOpenAIImageValidationError(w, "n must be an integer")
					return
				}
				n = parsed
			}

			file, fileHeader, err := r.FormFile("image")
			if err == nil && file != nil {
				defer file.Close()
				data, readErr := io.ReadAll(file)
				if readErr != nil {
					WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{
						Error: &model.APIError{Message: "Failed to read image file", Type: "invalid_request_error"},
					})
					return
				}
				imageB64 = base64.StdEncoding.EncodeToString(data)
				imageMime = detectMimeType(fileHeader.Header.Get("Content-Type"), fileHeader.Filename)
			}
		} else {
			var req model.ImageGenerationRequest
			_, ok := readJSONRequest(w, r, &req)
			if !ok {
				return
			}

			prompt = req.Prompt
			modelName = req.Model
			if req.N != nil {
				n = *req.N
				nProvided = true
			}
			responseFormat = req.ResponseFormat
			size = req.Size
			quality = req.Quality
			if req.Image != "" {
				imageMime, imageB64 = parseDataURL(req.Image)
			}
		}

		if prompt == "" {
			WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{
				Error: &model.APIError{Message: "prompt is required", Type: "invalid_request_error"},
			})
			return
		}
		if strings.TrimSpace(modelName) == "" {
			modelName = defaultOpenAIImageModel
		}
		if message := validateModelName(modelName, allowCustomModelNames); message != "" {
			writeOpenAIImageValidationError(w, message)
			return
		}
		if !nProvided {
			n = 1
		}
		if n < 1 || n > 10 {
			writeOpenAIImageValidationError(w, "n must be between 1 and 10")
			return
		}
		if responseFormat != "" && responseFormat != "b64_json" {
			writeOpenAIImageValidationError(w, "only response_format=b64_json is supported")
			return
		}

		generateImages(w, r, vp, modelName, prompt, imageB64, imageMime, n, size, quality)
	})
}

// ImageEdits handles POST /v1/images/edits.
func ImageEdits(vp *proxy.VertexProxy, allowCustomModelNames bool) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r = r.WithContext(proxy.WithCompatibilityLayer(r.Context(), proxy.CompatibilityLayerOpenAIImages))
		if err := r.ParseMultipartForm(32 << 20); err != nil {
			WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{
				Error: &model.APIError{Message: "Invalid multipart form: " + err.Error(), Type: "invalid_request_error"},
			})
			return
		}

		prompt := r.FormValue("prompt")
		modelName := r.FormValue("model")
		var n int
		var nProvided bool
		size := r.FormValue("size")
		quality := r.FormValue("quality")
		if nStr := r.FormValue("n"); nStr != "" {
			nProvided = true
			parsed, err := strconv.Atoi(nStr)
			if err != nil {
				writeOpenAIImageValidationError(w, "n must be an integer")
				return
			}
			n = parsed
		}

		if prompt == "" {
			WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{
				Error: &model.APIError{Message: "prompt is required", Type: "invalid_request_error"},
			})
			return
		}
		if strings.TrimSpace(modelName) == "" {
			modelName = defaultOpenAIImageModel
		}
		if message := validateModelName(modelName, allowCustomModelNames); message != "" {
			writeOpenAIImageValidationError(w, message)
			return
		}
		if !nProvided {
			n = 1
		}
		if n < 1 || n > 10 {
			writeOpenAIImageValidationError(w, "n must be between 1 and 10")
			return
		}

		file, fileHeader, err := r.FormFile("image")
		if err != nil || file == nil {
			WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{
				Error: &model.APIError{Message: "image file is required for /v1/images/edits", Type: "invalid_request_error"},
			})
			return
		}
		defer file.Close()

		data, err := io.ReadAll(file)
		if err != nil {
			WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{
				Error: &model.APIError{Message: "Failed to read image file", Type: "invalid_request_error"},
			})
			return
		}

		imageB64 := base64.StdEncoding.EncodeToString(data)
		imageMime := detectMimeType(fileHeader.Header.Get("Content-Type"), fileHeader.Filename)

		generateImages(w, r, vp, modelName, prompt, imageB64, imageMime, n, size, quality)
	})
}

func generateImages(
	w http.ResponseWriter,
	r *http.Request,
	vp *proxy.VertexProxy,
	modelName, prompt, imageB64, imageMime string,
	n int,
	size, quality string,
) {
	parts := []map[string]interface{}{{"text": prompt}}

	if imageB64 != "" {
		if imageMime == "" {
			imageMime = "image/png"
		}
		parts = append(parts, map[string]interface{}{
			"inlineData": map[string]interface{}{
				"mimeType": imageMime,
				"data":     imageB64,
			},
		})
	}

	contents := []map[string]interface{}{
		{"parts": parts, "role": "user"},
	}
	genConfig := map[string]interface{}{
		"responseModalities": []string{"TEXT", "IMAGE"},
	}
	if aspectRatio := openAIImageAspectRatio(size); aspectRatio != "" {
		genConfig["imageConfig"] = map[string]interface{}{"aspectRatio": aspectRatio}
	} else if strings.TrimSpace(size) != "" && !strings.EqualFold(strings.TrimSpace(size), "auto") {
		w.Header().Set("X-Vertex2API-Warning", "OpenAI image size has no exact Vertex aspect-ratio equivalent; the model default was used")
	}
	if quality != "" && quality != "auto" {
		w.Header().Add("X-Vertex2API-Warning", "OpenAI image quality has no exact Vertex equivalent; the model default was used")
	}

	var allImages []model.ImageData
	for i := 0; i < n; i++ {
		result, err := vp.CallWithTokenWithOptionsContext(r.Context(), modelName, contents, genConfig, nil, nil, nil)
		if err != nil {
			if requestContextCanceled(r.Context(), err) {
				log.Debug().Err(err).Str("model", modelName).Int("n", i+1).Msg("Image generation request canceled")
				return
			}
			log.Error().Str("err", vp.UpstreamLogError(err)).Str("model", modelName).Int("n", i+1).Msg("OpenAI Images generation failed")
			WriteJSON(w, http.StatusInternalServerError, publicUpstreamErrorResponse(vp, err))
			return
		}
		for _, img := range result.ImageParts {
			allImages = append(allImages, model.ImageData{B64JSON: img.Data})
		}
		if len(allImages) >= n {
			break
		}
	}

	if len(allImages) > n {
		allImages = allImages[:n]
	}
	if len(allImages) == 0 {
		WriteJSON(w, http.StatusBadGateway, model.ErrorResponse{Error: &model.APIError{
			Message: "Upstream returned no generated images",
			Type:    "server_error",
		}})
		return
	}

	WriteJSON(w, http.StatusOK, model.ImageGenerationResponse{
		Created: time.Now().Unix(),
		Data:    allImages,
	})
}

func openAIImageAspectRatio(size string) string {
	size = strings.ToLower(strings.TrimSpace(size))
	if size == "" || size == "auto" {
		return ""
	}
	parts := strings.SplitN(size, "x", 2)
	if len(parts) != 2 {
		return ""
	}
	width, widthErr := strconv.Atoi(parts[0])
	height, heightErr := strconv.Atoi(parts[1])
	if widthErr != nil || heightErr != nil || width <= 0 || height <= 0 {
		return ""
	}
	divisor := greatestCommonDivisor(width, height)
	ratio := strconv.Itoa(width/divisor) + ":" + strconv.Itoa(height/divisor)
	switch ratio {
	case "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9":
		return ratio
	default:
		return ""
	}
}

func greatestCommonDivisor(a, b int) int {
	for b != 0 {
		a, b = b, a%b
	}
	return a
}

func writeOpenAIImageValidationError(w http.ResponseWriter, message string) {
	WriteJSON(w, http.StatusBadRequest, model.ErrorResponse{Error: &model.APIError{
		Message: message,
		Type:    "invalid_request_error",
	}})
}

func isMultipart(contentType string) bool {
	return strings.HasPrefix(contentType, "multipart/form-data")
}

func detectMimeType(contentType, filename string) string {
	if contentType != "" && contentType != "application/octet-stream" {
		return contentType
	}

	switch {
	case hasExt(filename, ".png"):
		return "image/png"
	case hasExt(filename, ".jpg"), hasExt(filename, ".jpeg"):
		return "image/jpeg"
	case hasExt(filename, ".gif"):
		return "image/gif"
	case hasExt(filename, ".webp"):
		return "image/webp"
	default:
		return "image/png"
	}
}

func hasExt(filename, ext string) bool {
	return strings.HasSuffix(strings.ToLower(filename), ext)
}
