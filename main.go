package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime/debug"
	"strings"
	"syscall"
	"time"

	"vertex2api/client"
	"vertex2api/config"
	"vertex2api/handler"
	"vertex2api/middleware"
	"vertex2api/model"
	"vertex2api/proxy"
	"vertex2api/recaptcha"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const maxRequestBodyBytes = 32 * 1024 * 1024

var (
	version   = "1.0.5"
	commit    = "unknown"
	buildDate = "unknown"
)

func main() {
	zerolog.TimeFieldFormat = time.RFC3339
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr, TimeFormat: "15:04:05"})
	if err := run(); err != nil {
		log.Fatal().Err(err).Msg("vertex2api stopped")
	}
}

func run() error {
	cfg := config.Load()
	if err := cfg.Validate(); err != nil {
		return fmt.Errorf("invalid configuration: %w", err)
	}
	if cfg.GeneratedAPIKey != "" {
		log.Info().Str("api_key", cfg.GeneratedAPIKey).Msg("API_KEY was not provided; generated a random API key")
	}
	log.Info().
		Str("host", cfg.Host).
		Str("port", cfg.Port).
		Str("version", version).
		Str("commit", commit).
		Int("api_keys", len(cfg.APIKeys)).
		Int("max_retry", cfg.MaxRetry).
		Int("max_refresh", cfg.MaxRefresh).
		Str("tls_client_profile", cfg.TLSClientProfile).
		Int("http_timeout_seconds", cfg.HTTPTimeoutSeconds).
		Int("write_timeout_seconds", cfg.WriteTimeoutSeconds).
		Bool("auto_fetch_models", cfg.AutoFetchModels).
		Bool("allow_custom_model_names", cfg.AllowCustomModelNames).
		Bool("redact_upstream_responses", cfg.RedactUpstreamResponses).
		Msg("Configuration loaded")

	httpClient, err := client.New(cfg)
	if err != nil {
		return fmt.Errorf("create HTTP client: %w", err)
	}
	defer httpClient.CloseIdleConnections()
	tokenCache := recaptcha.NewTokenCache(httpClient, cfg)
	vertexProxy := proxy.NewVertexProxy(httpClient, tokenCache, cfg)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	model.StartAutoFetcher(ctx, httpClient, cfg)

	app := newApplication(cfg, vertexProxy)
	addr := net.JoinHostPort(cfg.Host, cfg.Port)
	server := &http.Server{
		Addr:              addr,
		Handler:           app,
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      time.Duration(cfg.WriteTimeoutSeconds) * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    1 << 20,
	}

	serverErr := make(chan error, 1)
	go func() {
		serverErr <- server.ListenAndServe()
	}()

	log.Info().Str("addr", addr).Msg("Starting vertex2api server")
	select {
	case err := <-serverErr:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return fmt.Errorf("serve HTTP: %w", err)
	case <-ctx.Done():
		log.Info().Msg("Shutdown signal received")
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			return fmt.Errorf("graceful shutdown: %w", err)
		}
		return nil
	}
}

func newApplication(cfg *config.Config, vertexProxy *proxy.VertexProxy) http.Handler {
	mux := http.NewServeMux()
	responsesAPI := handler.NewResponsesAPI(vertexProxy, cfg.AllowCustomModelNames, strings.Join(cfg.APIKeys, "\x00"))
	mux.Handle("POST /v1/messages", handler.AnthropicMessages(vertexProxy, cfg.AllowCustomModelNames))
	mux.Handle("POST /v1/messages/count_tokens", handler.AnthropicCountTokens(cfg.AllowCustomModelNames))
	mux.Handle("POST /v1/chat/completions", handler.ChatCompletions(vertexProxy, cfg.AllowCustomModelNames))
	mux.Handle("POST /v1/responses", responsesAPI.Responses())
	mux.Handle("POST /v1/responses/compact", responsesAPI.Compact())
	mux.Handle("POST /v1/images/generations", handler.ImageGenerations(vertexProxy, cfg.AllowCustomModelNames))
	mux.Handle("POST /v1/images/edits", handler.ImageEdits(vertexProxy, cfg.AllowCustomModelNames))
	mux.Handle("GET /v1/models", handler.ModelsList())
	mux.Handle("GET /v1/models/{modelID}", handler.RetrieveModel())
	mux.Handle("GET /v1/stats", handler.Stats(cfg))

	geminiHandler := handler.GeminiGenerate(vertexProxy, cfg.AllowCustomModelNames)
	mux.Handle("POST /v1beta/models/{modelAction}", geminiHandler)
	mux.Handle("POST /v1beta1/models/{modelAction}", geminiHandler)
	mux.Handle("POST /v1/models/{modelAction}", geminiHandler)
	mux.Handle("GET /v1beta/models", handler.GeminiListModels())

	mux.HandleFunc("GET /health", func(w http.ResponseWriter, r *http.Request) {
		handler.WriteJSON(w, http.StatusOK, map[string]string{"status": "ok", "version": version})
	})
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		handler.WriteJSON(w, http.StatusOK, map[string]interface{}{
			"service":    "vertex2api",
			"version":    version,
			"commit":     commit,
			"build_date": buildDate,
			"endpoints": []string{
				"POST /v1/messages",
				"POST /v1/messages/count_tokens",
				"POST /v1/chat/completions",
				"POST /v1/responses",
				"POST /v1/responses/compact",
				"POST /v1/images/generations",
				"POST /v1/images/edits",
				"GET  /v1/models",
				"POST /v1beta1/models/{model}:generateContent",
				"POST /v1beta1/models/{model}:streamGenerateContent",
				"POST /v1beta1/models/{model}:countTokens",
			},
		})
	})

	var app http.Handler = mux
	app = bodyLimitMiddleware(maxRequestBodyBytes, app)
	app = middleware.Auth(cfg)(app)
	app = middleware.RejectPathTraversal(app)
	app = corsMiddleware(cfg.CORSAllowOrigin, app)
	app = recoverMiddleware(app)
	return app
}

func recoverMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if v := recover(); v != nil {
				log.Error().
					Interface("panic", v).
					Bytes("stack", debug.Stack()).
					Msg("HTTP handler panic")
				handler.WriteProtocolError(w, r, http.StatusInternalServerError, "Internal server error", "server_error")
			}
		}()
		next.ServeHTTP(w, r)
	})
}

func corsMiddleware(allowOrigin string, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header := w.Header()
		origin := r.Header.Get("Origin")
		if allowOrigin == "*" || (allowOrigin != "" && origin == allowOrigin) {
			header.Set("Access-Control-Allow-Origin", allowOrigin)
			if allowOrigin != "*" {
				header.Add("Vary", "Origin")
			}
		}
		header.Set("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, HEAD, OPTIONS")
		if requested := r.Header.Get("Access-Control-Request-Headers"); requested != "" {
			header.Set("Access-Control-Allow-Headers", requested)
		} else {
			header.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, Accept, x-goog-api-key")
		}

		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		next.ServeHTTP(w, r)
	})
}

func bodyLimitMiddleware(limit int64, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, limit)
		next.ServeHTTP(w, r)
	})
}
