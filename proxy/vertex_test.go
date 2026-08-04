package proxy

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"vertex2api/client"
	"vertex2api/config"
	"vertex2api/recaptcha"

	"github.com/bytedance/sonic"
)

func TestSelectVertexBaseURL(t *testing.T) {
	baseURL := "https://vertex.example"
	prefixes := []string{"https://prefix1.example/", "https://prefix2.example/"}

	tests := []struct {
		name        string
		direct      bool
		prefixes    []string
		prefixIndex int
		want        string
	}{
		{
			name:     "direct",
			direct:   true,
			prefixes: prefixes,
			want:     baseURL,
		},
		{
			name:        "prefixed",
			prefixes:    prefixes,
			prefixIndex: 1,
			want:        "https://prefix2.example/https://vertex.example",
		},
		{
			name:   "no prefixes falls back to direct",
			direct: false,
			want:   baseURL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := selectVertexBaseURL(baseURL, tt.prefixes, tt.direct, tt.prefixIndex); got != tt.want {
				t.Fatalf("selectVertexBaseURL() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestBuildVertexAPIURL(t *testing.T) {
	baseURL := "https://prefix.example/https://vertex.example"
	apiKey := "test-vertex-key"

	got := buildVertexAPIURL(baseURL, apiKey)
	want := "https://prefix.example/https://vertex.example" + vertexAPIPath + "?key=" + apiKey + "&prettyPrint=false"
	if got != want {
		t.Fatalf("buildVertexAPIURL() = %q, want %q", got, want)
	}
}

func TestUpstreamLogErrorHonorsConfig(t *testing.T) {
	redacted := &VertexProxy{cfg: &config.Config{RedactUpstreamLogs: true}}
	if got := redacted.upstreamLogError(errors.New("upstream secret")); got != "[REDACTED]" {
		t.Fatalf("redacted upstream error = %q", got)
	}

	unredacted := &VertexProxy{cfg: &config.Config{RedactUpstreamLogs: false}}
	if got := unredacted.upstreamLogError(errors.New("upstream detail")); got != "upstream detail" {
		t.Fatalf("unredacted upstream error = %q", got)
	}
}

func TestCallDoesNotRetryVertexNotFoundError(t *testing.T) {
	requests := 0
	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		_, _ = w.Write([]byte(`[{"results":[{"errors":[{"message":"Publisher Model gemini-2.0-flash was not found or your project does not have access to it.","extensions":{"status":{"code":5}}}]}]}]`))
	}))
	defer vertexServer.Close()

	httpClient := &client.HTTPClient{HTTP: vertexServer.Client()}
	cfg := &config.Config{
		VertexBaseURL: vertexServer.URL,
		MaxRetry:      4,
		RetryDelayMs:  0,
	}
	vp := NewVertexProxy(httpClient, nil, cfg)

	bodyJSON, err := BuildVertexBody("gemini-test", []map[string]interface{}{
		{"role": "user", "parts": []map[string]interface{}{{"text": "hello"}}},
	}, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	result, err := vp.CallContext(context.Background(), bodyJSON)
	if err == nil {
		t.Fatal("Call returned nil error, want vertex not found error")
	}
	if result != nil {
		t.Fatalf("Call result = %#v, want nil", result)
	}
	if strings.Contains(err.Error(), "after retries") {
		t.Fatalf("Call error = %q, should not be wrapped as a retry exhaustion", err)
	}
	var vertexErr *vertexAPIError
	if !errors.As(err, &vertexErr) {
		t.Fatalf("Call error = %v, want vertexAPIError", err)
	}
	if vertexErr.Code != 5 {
		t.Fatalf("vertex error code = %d, want 5", vertexErr.Code)
	}
	if requests != 1 {
		t.Fatalf("vertex requests = %d, want 1", requests)
	}
}

func TestStreamDoesNotRetryVertexNotFoundError(t *testing.T) {
	requests := 0
	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		_, _ = w.Write([]byte(`[{"results":[{"errors":[{"message":"Publisher Model gemini-2.0-flash was not found or your project does not have access to it.","extensions":{"status":{"code":5}}}]}]}]`))
	}))
	defer vertexServer.Close()

	httpClient := &client.HTTPClient{HTTP: vertexServer.Client()}
	cfg := &config.Config{
		VertexBaseURL: vertexServer.URL,
		MaxRetry:      4,
		RetryDelayMs:  0,
	}
	vp := NewVertexProxy(httpClient, nil, cfg)

	bodyJSON, err := BuildVertexBody("gemini-test", []map[string]interface{}{
		{"role": "user", "parts": []map[string]interface{}{{"text": "hello"}}},
	}, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	err = vp.StreamContext(context.Background(), bodyJSON, func(_ *CallResult) error {
		t.Fatal("stream callback should not be called")
		return nil
	})
	if err == nil {
		t.Fatal("Stream returned nil error, want vertex not found error")
	}
	if strings.Contains(err.Error(), "after retries") {
		t.Fatalf("Stream error = %q, should not be wrapped as a retry exhaustion", err)
	}
	var vertexErr *vertexAPIError
	if !errors.As(err, &vertexErr) {
		t.Fatalf("Stream error = %v, want vertexAPIError", err)
	}
	if vertexErr.Code != 5 {
		t.Fatalf("vertex error code = %d, want 5", vertexErr.Code)
	}
	if requests != 1 {
		t.Fatalf("vertex requests = %d, want 1", requests)
	}
}

func TestCallRetriesWithoutRejectedThoughtSignature(t *testing.T) {
	requests := 0
	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		body, _ := io.ReadAll(r.Body)
		part := firstPartFromBody(t, body)

		if requests == 1 {
			if _, ok := part["thoughtSignature"]; !ok {
				t.Fatal("first request should include thoughtSignature")
			}
			_, _ = w.Write([]byte(`[{"results":[{"errors":[{"message":"Unable to submit request because Thought signature is not valid..","extensions":{"status":{"code":3}}}]}]}]`))
			return
		}

		if got := part["thoughtSignature"]; got != thoughtSignatureBypassValue() {
			t.Fatalf("retried request thoughtSignature = %v, want bypass", got)
		}
		_, _ = w.Write([]byte(`[{"results":[{"data":{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"ok"}]}}]}}]}]`))
	}))
	defer vertexServer.Close()

	httpClient := &client.HTTPClient{HTTP: vertexServer.Client()}
	cfg := &config.Config{
		VertexBaseURL: vertexServer.URL,
		MaxRetry:      1,
		RetryDelayMs:  0,
	}
	vp := NewVertexProxy(httpClient, nil, cfg)

	bodyJSON, err := BuildVertexBody("gemini-test", []map[string]interface{}{
		{
			"role": "model",
			"parts": []map[string]interface{}{
				{
					"functionCall": map[string]interface{}{
						"name": "lookup",
						"args": map[string]interface{}{"query": "weather"},
					},
					"thoughtSignature": base64.StdEncoding.EncodeToString([]byte("stale signature")),
				},
			},
		},
	}, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	result, err := vp.CallContext(context.Background(), bodyJSON)
	if err != nil {
		t.Fatalf("Call returned error: %v", err)
	}
	if got := result.TextParts[0].Text; got != "ok" {
		t.Fatalf("response text = %q, want ok", got)
	}
	if requests != 2 {
		t.Fatalf("vertex requests = %d, want 2", requests)
	}
}

func TestCallRetriesOnThoughtSignatureFieldError(t *testing.T) {
	requests := 0
	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		body, _ := io.ReadAll(r.Body)
		part := firstPartFromBody(t, body)

		if requests == 1 {
			if _, ok := part["thoughtSignature"]; !ok {
				t.Fatal("first request should include thoughtSignature")
			}
			_, _ = w.Write([]byte(`[{"results":[{"errors":[{"message":"Invalid value at 'contents[0].parts[0].thought_signature' (TYPE_BYTES), thought_signature failed validation.","extensions":{"status":{"code":3}}}]}]}]`))
			return
		}

		if got := part["thoughtSignature"]; got != thoughtSignatureBypassValue() {
			t.Fatalf("retried request thoughtSignature = %v, want bypass", got)
		}
		_, _ = w.Write([]byte(`[{"results":[{"data":{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"ok"}]}}]}}]}]`))
	}))
	defer vertexServer.Close()

	httpClient := &client.HTTPClient{HTTP: vertexServer.Client()}
	cfg := &config.Config{
		VertexBaseURL: vertexServer.URL,
		MaxRetry:      1,
		RetryDelayMs:  0,
	}
	vp := NewVertexProxy(httpClient, nil, cfg)

	bodyJSON, err := BuildVertexBody("gemini-test", []map[string]interface{}{
		{
			"role": "model",
			"parts": []map[string]interface{}{
				{
					"functionCall": map[string]interface{}{
						"name": "lookup",
						"args": map[string]interface{}{"query": "weather"},
					},
					"thoughtSignature": base64.StdEncoding.EncodeToString([]byte("stale signature")),
				},
			},
		},
	}, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	result, err := vp.CallContext(context.Background(), bodyJSON)
	if err != nil {
		t.Fatalf("Call returned error: %v", err)
	}
	if got := result.TextParts[0].Text; got != "ok" {
		t.Fatalf("response text = %q, want ok", got)
	}
	if requests != 2 {
		t.Fatalf("vertex requests = %d, want 2", requests)
	}
}

func TestIsThoughtSignatureInvalidErrorIgnoresGenericSignatureError(t *testing.T) {
	if !isThoughtSignatureInvalidError(errors.New("Invalid value at 'contents[0].parts[0].thought_signature' (TYPE_BYTES)")) {
		t.Fatal("thought_signature field error should be treated as thought signature related")
	}
	if isThoughtSignatureInvalidError(errors.New("request signature mismatch")) {
		t.Fatal("generic signature error should not be treated as thought signature related")
	}
}

func TestClassifyRecaptchaRetryError(t *testing.T) {
	tests := []struct {
		name string
		msg  string
		want recaptchaRetryReason
	}{
		{
			name: "verification failed",
			msg:  "Failed to verify action",
			want: recaptchaVerifyFailed,
		},
		{
			name: "expired or invalid token",
			msg:  "Recaptcha token is invalid, please refresh the page or log in, and try again.",
			want: recaptchaTokenInvalid,
		},
		{
			name: "model turn validation",
			msg:  "Requests ending with a model turn are not supported.",
			want: recaptchaRetryNone,
		},
		{
			name: "thought signature validation",
			msg:  "Thought signature is not valid.",
			want: recaptchaRetryNone,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := classifyRecaptchaRetryError(&vertexAPIError{Code: 3, Message: tt.msg}); got != tt.want {
				t.Fatalf("classifyRecaptchaRetryError() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCallDoesNotRetryNonRecaptchaCode3(t *testing.T) {
	requests := 0
	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		_, _ = w.Write([]byte(`[{
			"results":[{"errors":[{"message":"Requests ending with a model turn are not supported.","extensions":{"status":{"code":3}}}]}]
		}]`))
	}))
	defer vertexServer.Close()

	vp := NewVertexProxy(&client.HTTPClient{HTTP: vertexServer.Client()}, nil, &config.Config{
		VertexBaseURL: vertexServer.URL,
		MaxRetry:      3,
		MaxRefresh:    3,
		RetryDelayMs:  0,
	})
	bodyJSON, err := BuildVertexBody("gemini-3.6-flash", []map[string]interface{}{
		{"role": "user", "parts": []map[string]interface{}{{"text": "hello"}}},
	}, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	_, err = vp.CallContext(context.Background(), bodyJSON)
	if err == nil {
		t.Fatal("CallContext unexpectedly succeeded")
	}
	if requests != 1 {
		t.Fatalf("vertex requests = %d, want 1", requests)
	}
}

func TestStreamRetriesWithoutRejectedThoughtSignature(t *testing.T) {
	requests := 0
	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		body, _ := io.ReadAll(r.Body)
		part := firstPartFromBody(t, body)

		if requests == 1 {
			if _, ok := part["thoughtSignature"]; !ok {
				t.Fatal("first request should include thoughtSignature")
			}
			_, _ = w.Write([]byte(`[{"results":[{"errors":[{"message":"Unable to submit request because Thought signature is not valid..","extensions":{"status":{"code":3}}}]}]}]`))
			return
		}

		if got := part["thoughtSignature"]; got != thoughtSignatureBypassValue() {
			t.Fatalf("retried request thoughtSignature = %v, want bypass", got)
		}
		_, _ = w.Write([]byte(`[{"results":[{"data":{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"ok"}]}}]}}]}]`))
	}))
	defer vertexServer.Close()

	httpClient := &client.HTTPClient{HTTP: vertexServer.Client()}
	cfg := &config.Config{
		VertexBaseURL: vertexServer.URL,
		MaxRetry:      1,
		RetryDelayMs:  0,
	}
	vp := NewVertexProxy(httpClient, nil, cfg)

	bodyJSON, err := BuildVertexBody("gemini-test", []map[string]interface{}{
		{
			"role": "model",
			"parts": []map[string]interface{}{
				{
					"functionCall": map[string]interface{}{
						"name": "lookup",
						"args": map[string]interface{}{"query": "weather"},
					},
					"thoughtSignature": base64.StdEncoding.EncodeToString([]byte("stale signature")),
				},
			},
		},
	}, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	var text strings.Builder
	err = vp.StreamContext(context.Background(), bodyJSON, func(result *CallResult) error {
		for _, part := range result.TextParts {
			text.WriteString(part.Text)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("Stream returned error: %v", err)
	}
	if got := text.String(); got != "ok" {
		t.Fatalf("streamed text = %q, want ok", got)
	}
	if requests != 2 {
		t.Fatalf("vertex requests = %d, want 2", requests)
	}
}
func TestCallWithTokenRefreshesOnResourceExhausted(t *testing.T) {
	var mu sync.Mutex
	reloads := 0
	tokenRequests := map[string]int{}

	recaptchaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/anchor"):
			_, _ = w.Write([]byte(`<input id="recaptcha-token" value="anchor-token">`))
		case strings.Contains(r.URL.Path, "/reload"):
			mu.Lock()
			reloads++
			token := fmt.Sprintf("token-%d", reloads)
			mu.Unlock()
			_, _ = w.Write([]byte(fmt.Sprintf(`["rresp","%s"]`, token)))
		default:
			http.NotFound(w, r)
		}
	}))
	defer recaptchaServer.Close()

	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var raw map[string]interface{}
		if err := sonic.Unmarshal(body, &raw); err != nil {
			t.Fatalf("unmarshal vertex request: %v", err)
		}
		variables := raw["variables"].(map[string]interface{})
		token := variables["recaptchaToken"].(string)

		mu.Lock()
		tokenRequests[token]++
		mu.Unlock()

		if token == "token-1" {
			_, _ = w.Write([]byte(`[{"results":[{"errors":[{"message":"Resource has been exhausted (e.g. check quota).","extensions":{"status":{"code":8}}}]}]}]`))
			return
		}
		_, _ = w.Write([]byte(`[{"results":[{"data":{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"ok"}]}}]}}]}]`))
	}))
	defer vertexServer.Close()

	httpClient := &client.HTTPClient{HTTP: vertexServer.Client()}
	cfg := &config.Config{
		VertexBaseURL: vertexServer.URL,
		RecaptchaBase: recaptchaServer.URL,
		MaxRetry:      6,
		MaxRefresh:    3,
		RetryDelayMs:  0,
	}
	vp := NewVertexProxy(httpClient, recaptcha.NewTokenCache(httpClient, cfg), cfg)

	contents := []map[string]interface{}{
		{"role": "user", "parts": []map[string]interface{}{{"text": "hello"}}},
	}
	result, err := vp.CallWithTokenWithOptionsContext(context.Background(), "gemini-test", contents, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("CallWithToken returned error: %v", err)
	}
	if got := result.TextParts[0].Text; got != "ok" {
		t.Fatalf("response text = %q, want ok", got)
	}

	mu.Lock()
	defer mu.Unlock()
	if reloads != 2 {
		t.Fatalf("reloads = %d, want 2", reloads)
	}
	if got := tokenRequests["token-1"]; got != 7 {
		t.Fatalf("token-1 requests = %d, want 7", got)
	}
	if got := tokenRequests["token-2"]; got != 1 {
		t.Fatalf("token-2 requests = %d, want 1", got)
	}
}

func TestStreamWithTokenRefreshesImmediatelyOnInvalidRecaptchaToken(t *testing.T) {
	var mu sync.Mutex
	reloads := 0
	tokenRequests := map[string]int{}

	recaptchaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/anchor"):
			_, _ = w.Write([]byte(`<input id="recaptcha-token" value="anchor-token">`))
		case strings.Contains(r.URL.Path, "/reload"):
			mu.Lock()
			reloads++
			token := fmt.Sprintf("token-%d", reloads)
			mu.Unlock()
			_, _ = w.Write([]byte(fmt.Sprintf(`["rresp","%s"]`, token)))
		default:
			http.NotFound(w, r)
		}
	}))
	defer recaptchaServer.Close()

	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var raw map[string]interface{}
		if err := sonic.Unmarshal(body, &raw); err != nil {
			t.Fatalf("unmarshal vertex request: %v", err)
		}
		variables := raw["variables"].(map[string]interface{})
		token := variables["recaptchaToken"].(string)

		mu.Lock()
		tokenRequests[token]++
		mu.Unlock()

		if token == "token-1" {
			_, _ = w.Write([]byte(`[{"results":[{"errors":[{"message":"Recaptcha token is invalid, please refresh the page or log in, and try again.","extensions":{"status":{"code":3}}}]}]}]`))
			return
		}
		_, _ = w.Write([]byte(`[{"results":[{"data":{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"ok"}]}}]}}]}]`))
	}))
	defer vertexServer.Close()

	httpClient := &client.HTTPClient{HTTP: vertexServer.Client()}
	cfg := &config.Config{
		VertexBaseURL: vertexServer.URL,
		RecaptchaBase: recaptchaServer.URL,
		MaxRetry:      3,
		RetryDelayMs:  0,
	}
	vp := NewVertexProxy(httpClient, recaptcha.NewTokenCache(httpClient, cfg), cfg)

	contents := []map[string]interface{}{
		{"role": "user", "parts": []map[string]interface{}{{"text": "hello"}}},
	}
	bodyJSON, tokenLease, err := vp.BuildBodyWithTokenWithOptionsContext(context.Background(), "gemini-test", contents, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("BuildBodyWithToken returned error: %v", err)
	}

	var text strings.Builder
	err = vp.StreamWithTokenContext(context.Background(), bodyJSON, tokenLease, func(result *CallResult) error {
		for _, part := range result.TextParts {
			text.WriteString(part.Text)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("StreamWithToken returned error: %v", err)
	}
	if got := text.String(); got != "ok" {
		t.Fatalf("streamed text = %q, want ok", got)
	}

	mu.Lock()
	defer mu.Unlock()
	if reloads != 2 {
		t.Fatalf("reloads = %d, want 2", reloads)
	}
	if got := tokenRequests["token-1"]; got != 1 {
		t.Fatalf("token-1 requests = %d, want 1", got)
	}
	if got := tokenRequests["token-2"]; got != 1 {
		t.Fatalf("token-2 requests = %d, want 1", got)
	}
}

func TestCallWithTokenRefreshesImmediatelyOnExpiredRecaptchaToken(t *testing.T) {
	var mu sync.Mutex
	reloads := 0
	tokenRequests := map[string]int{}

	recaptchaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case strings.Contains(r.URL.Path, "/anchor"):
			_, _ = w.Write([]byte(`<input id="recaptcha-token" value="anchor-token">`))
		case strings.Contains(r.URL.Path, "/reload"):
			mu.Lock()
			reloads++
			token := fmt.Sprintf("token-%d", reloads)
			mu.Unlock()
			_, _ = w.Write([]byte(fmt.Sprintf(`["rresp","%s"]`, token)))
		default:
			http.NotFound(w, r)
		}
	}))
	defer recaptchaServer.Close()

	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var raw map[string]interface{}
		if err := sonic.Unmarshal(body, &raw); err != nil {
			t.Fatalf("unmarshal vertex request: %v", err)
		}
		variables := raw["variables"].(map[string]interface{})
		token := variables["recaptchaToken"].(string)

		mu.Lock()
		tokenRequests[token]++
		mu.Unlock()

		if token == "token-1" {
			_, _ = w.Write([]byte(`[{"results":[{"errors":[{"message":"Recaptcha token is invalid, please refresh the page or log in, and try again.","extensions":{"status":{"code":3}}}]}]}]`))
			return
		}
		_, _ = w.Write([]byte(`[{"results":[{"data":{"candidates":[{"finishReason":"STOP","content":{"parts":[{"text":"ok"}]}}]}}]}]`))
	}))
	defer vertexServer.Close()

	httpClient := &client.HTTPClient{HTTP: vertexServer.Client()}
	cfg := &config.Config{
		VertexBaseURL: vertexServer.URL,
		RecaptchaBase: recaptchaServer.URL,
		MaxRetry:      3,
		MaxRefresh:    3,
		RetryDelayMs:  0,
	}
	vp := NewVertexProxy(httpClient, recaptcha.NewTokenCache(httpClient, cfg), cfg)

	contents := []map[string]interface{}{
		{"role": "user", "parts": []map[string]interface{}{{"text": "hello"}}},
	}
	result, err := vp.CallWithTokenWithOptionsContext(context.Background(), "gemini-test", contents, nil, nil, nil, nil)
	if err != nil {
		t.Fatalf("CallWithToken returned error: %v", err)
	}
	if got := result.TextParts[0].Text; got != "ok" {
		t.Fatalf("response text = %q, want ok", got)
	}

	mu.Lock()
	defer mu.Unlock()
	if reloads != 2 {
		t.Fatalf("reloads = %d, want 2", reloads)
	}
	if got := tokenRequests["token-1"]; got != 1 {
		t.Fatalf("token-1 requests = %d, want 1", got)
	}
	if got := tokenRequests["token-2"]; got != 1 {
		t.Fatalf("token-2 requests = %d, want 1", got)
	}
}

func TestStreamContextCancelsUpstreamRequest(t *testing.T) {
	upstreamCanceled := make(chan struct{})

	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"results":[{"data":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]}}]}}]}` + "\n\n"))
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}

		<-r.Context().Done()
		close(upstreamCanceled)
	}))
	defer vertexServer.Close()

	httpClient := &client.HTTPClient{HTTP: vertexServer.Client()}
	cfg := &config.Config{
		VertexBaseURL: vertexServer.URL,
		MaxRetry:      1,
	}
	vp := NewVertexProxy(httpClient, nil, cfg)

	contents := []map[string]interface{}{
		{"role": "user", "parts": []map[string]interface{}{{"text": "hello"}}},
	}
	bodyJSON, err := BuildVertexBody("gemini-test", contents, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var text strings.Builder
	err = vp.StreamContext(ctx, bodyJSON, func(result *CallResult) error {
		for _, part := range result.TextParts {
			text.WriteString(part.Text)
		}
		cancel()
		return context.Canceled
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("StreamContext error = %v, want context.Canceled", err)
	}
	if got := text.String(); got != "hello" {
		t.Fatalf("streamed text = %q, want hello", got)
	}

	select {
	case <-upstreamCanceled:
	case <-time.After(2 * time.Second):
		t.Fatal("upstream request was not canceled")
	}
}

func TestStreamCallbackErrorClosesUpstreamRequest(t *testing.T) {
	upstreamCanceled := make(chan struct{})

	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"results":[{"data":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello"}]}}]}}]}` + "\n\n"))
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}

		<-r.Context().Done()
		close(upstreamCanceled)
	}))
	defer vertexServer.Close()

	httpClient := &client.HTTPClient{HTTP: vertexServer.Client()}
	cfg := &config.Config{
		VertexBaseURL: vertexServer.URL,
		MaxRetry:      1,
	}
	vp := NewVertexProxy(httpClient, nil, cfg)

	contents := []map[string]interface{}{
		{"role": "user", "parts": []map[string]interface{}{{"text": "hello"}}},
	}
	bodyJSON, err := BuildVertexBody("gemini-test", contents, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	writeErr := errors.New("client stream write failed: connection closed")
	err = vp.StreamContext(context.Background(), bodyJSON, func(_ *CallResult) error {
		return writeErr
	})
	if !errors.Is(err, writeErr) {
		t.Fatalf("Stream error = %v, want writeErr", err)
	}

	select {
	case <-upstreamCanceled:
	case <-time.After(2 * time.Second):
		t.Fatal("upstream request was not closed after callback error")
	}
}

func TestBuildVertexBodyAlwaysUsesStreamingOperation(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role": "user",
			"parts": []map[string]interface{}{
				{"text": "Hello"},
			},
		},
	}

	body, err := BuildVertexBody("gemini-test", contents, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	var raw map[string]interface{}
	if err := sonic.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if got := raw["operationName"]; got != streamGenerateContentOperationName {
		t.Fatalf("operationName = %v, want %s", got, streamGenerateContentOperationName)
	}
	if got := raw["querySignature"]; got != streamGenerateContentQuerySignature {
		t.Fatalf("querySignature = %v, want %s", got, streamGenerateContentQuerySignature)
	}
}

func TestBuildVertexBodyDefaultsMissingContentRoleToUser(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"parts": []map[string]interface{}{
				{"text": "Hello"},
			},
		},
	}

	body, err := BuildVertexBody("gemini-test", contents, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	variables := variablesFromBody(t, body)
	normalizedContents := variables["contents"].([]interface{})
	content := normalizedContents[0].(map[string]interface{})
	if got := content["role"]; got != "user" {
		t.Fatalf("content role = %v, want user", got)
	}
}

func TestBuildVertexBodyPassesToolsAndToolConfigTopLevel(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role":  "user",
			"parts": []map[string]interface{}{{"text": "Hello"}},
		},
	}
	tools := []interface{}{
		map[string]interface{}{
			"functionDeclarations": []interface{}{
				map[string]interface{}{
					"name":        "lookup",
					"description": "look up data",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"query": map[string]interface{}{
								"type":        "string",
								"description": "search query",
							},
						},
						"required": []interface{}{"query"},
					},
				},
			},
		},
	}
	toolConfig := map[string]interface{}{
		"functionCallingConfig": map[string]interface{}{"mode": "ANY"},
	}
	genConfig := map[string]interface{}{"temperature": 0.7}

	body, err := BuildVertexBodyWithOptions("gemini-test", contents, genConfig, nil, nil, "token", &VertexRequestOptions{
		Tools:      tools,
		ToolConfig: toolConfig,
	})
	if err != nil {
		t.Fatalf("BuildVertexBodyWithOptions returned error: %v", err)
	}

	variables := variablesFromBody(t, body)
	if _, ok := variables["generationConfig"].(map[string]interface{})["tools"]; ok {
		t.Fatal("generationConfig should not contain tools")
	}
	if _, ok := variables["tools"]; !ok {
		t.Fatal("variables should contain top-level tools")
	}
	if _, ok := variables["toolConfig"]; !ok {
		t.Fatal("variables should contain top-level toolConfig")
	}

	normalizedTools := variables["tools"].([]interface{})
	functionDeclarations := normalizedTools[0].(map[string]interface{})["functionDeclarations"].([]interface{})
	declaration := functionDeclarations[0].(map[string]interface{})
	parameters := declaration["parametersJsonSchema"].(map[string]interface{})
	if got := parameters["type"]; got != "object" {
		t.Fatalf("parameters type = %v, want object", got)
	}
	properties := parameters["properties"].(map[string]interface{})
	propertyValue := properties["query"].(map[string]interface{})
	if got := propertyValue["type"]; got != "string" {
		t.Fatalf("property value type = %v, want string", got)
	}
}

func TestBuildVertexBodyNormalizesArraySchemaItemsToObject(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role":  "user",
			"parts": []map[string]interface{}{{"text": "Hello"}},
		},
	}
	tools := []interface{}{
		map[string]interface{}{
			"functionDeclarations": []interface{}{
				map[string]interface{}{
					"name": "send_messages",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"messages": map[string]interface{}{
								"type": "array",
								"items": []interface{}{
									map[string]interface{}{
										"type": "object",
										"properties": map[string]interface{}{
											"content": map[string]interface{}{"type": "string"},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	body, err := BuildVertexBodyWithOptions("gemini-test", contents, nil, nil, nil, "token", &VertexRequestOptions{
		Tools: tools,
	})
	if err != nil {
		t.Fatalf("BuildVertexBodyWithOptions returned error: %v", err)
	}

	variables := variablesFromBody(t, body)
	normalizedTools := variables["tools"].([]interface{})
	functionDeclarations := normalizedTools[0].(map[string]interface{})["functionDeclarations"].([]interface{})
	declaration := functionDeclarations[0].(map[string]interface{})
	parameters := declaration["parametersJsonSchema"].(map[string]interface{})
	properties := parameters["properties"].(map[string]interface{})
	messagesSchema := properties["messages"].(map[string]interface{})
	items := messagesSchema["items"].(map[string]interface{})
	if got := items["type"]; got != "object" {
		t.Fatalf("messages.items type = %v, want object", got)
	}
	itemProperties := items["properties"].(map[string]interface{})
	if _, ok := itemProperties["content"]; !ok {
		t.Fatalf("messages.items properties = %v, want content", itemProperties)
	}
}

func TestBuildVertexBodyNormalizesParametersJSONSchemaItemsToObject(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role":  "user",
			"parts": []map[string]interface{}{{"text": "Hello"}},
		},
	}
	tools := []interface{}{
		map[string]interface{}{
			"functionDeclarations": []interface{}{
				map[string]interface{}{
					"name": "send_messages",
					"parametersJsonSchema": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"messages": map[string]interface{}{
								"type": "array",
								"items": []interface{}{
									map[string]interface{}{
										"type": "object",
										"properties": map[string]interface{}{
											"content": map[string]interface{}{"type": "string"},
										},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	body, err := BuildVertexBodyWithOptions("gemini-test", contents, nil, nil, nil, "token", &VertexRequestOptions{
		Tools: tools,
	})
	if err != nil {
		t.Fatalf("BuildVertexBodyWithOptions returned error: %v", err)
	}

	variables := variablesFromBody(t, body)
	normalizedTools := variables["tools"].([]interface{})
	functionDeclarations := normalizedTools[0].(map[string]interface{})["functionDeclarations"].([]interface{})
	declaration := functionDeclarations[0].(map[string]interface{})
	parameters := declaration["parametersJsonSchema"].(map[string]interface{})
	properties := parameters["properties"].(map[string]interface{})
	messagesSchema := properties["messages"].(map[string]interface{})
	items := messagesSchema["items"].(map[string]interface{})
	if got := items["type"]; got != "object" {
		t.Fatalf("messages.items type = %v, want object", got)
	}
	itemProperties := items["properties"].(map[string]interface{})
	if _, ok := itemProperties["content"]; !ok {
		t.Fatal("messages.items.properties should contain content")
	}
}

func TestBuildVertexBodyOmitsToolsWithoutToolType(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role":  "user",
			"parts": []map[string]interface{}{{"text": "Hello"}},
		},
	}

	body, err := BuildVertexBodyWithOptions("gemini-test", contents, nil, nil, nil, "token", &VertexRequestOptions{
		Tools: []interface{}{
			map[string]interface{}{},
			map[string]interface{}{"functionDeclarations": []interface{}{}},
			map[string]interface{}{"type": "function"},
		},
	})
	if err != nil {
		t.Fatalf("BuildVertexBodyWithOptions returned error: %v", err)
	}

	variables := variablesFromBody(t, body)
	if _, ok := variables["tools"]; ok {
		t.Fatal("variables should omit tools when every tool lacks a valid tool_type")
	}
}

func TestStreamVertexResponseEmitsArrayChunks(t *testing.T) {
	body := `[{"results":[{"data":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello "}]} }]}},{"data":{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"world"}]}}]}}]}]`

	var chunks []CallResult
	status, err := streamVertexResponse(strings.NewReader(body), func(result *CallResult) error {
		chunks = append(chunks, *result)
		return nil
	})
	if err != nil {
		t.Fatalf("streamVertexResponse returned error: %v", err)
	}
	if status != 0 {
		t.Fatalf("status = %d, want 0", status)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks length = %d, want 2", len(chunks))
	}
	if got := chunks[0].TextParts[0].Text; got != "hello " {
		t.Fatalf("first chunk text = %q, want hello", got)
	}
	if got := chunks[1].TextParts[0].Text; got != "world" {
		t.Fatalf("second chunk text = %q, want world", got)
	}
	if got := chunks[1].FinishReason; got != "STOP" {
		t.Fatalf("second chunk finishReason = %q, want STOP", got)
	}
}

func TestStreamVertexResponseDelaysFinishOnlyChunkUntilEnd(t *testing.T) {
	body := `[{"results":[{"data":{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":null}}]}},{"data":{"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_current_time","args":{"timezone":"Asia/Shanghai"}}}]}}]}}]}]`

	var chunks []CallResult
	status, err := streamVertexResponse(strings.NewReader(body), func(result *CallResult) error {
		chunks = append(chunks, *result)
		return nil
	})
	if err != nil {
		t.Fatalf("streamVertexResponse returned error: %v", err)
	}
	if status != 0 {
		t.Fatalf("status = %d, want 0", status)
	}
	if len(chunks) != 2 {
		t.Fatalf("chunks length = %d, want 2", len(chunks))
	}
	if got := len(chunks[0].FunctionCalls); got != 1 {
		t.Fatalf("first chunk function calls = %d, want 1", got)
	}
	if got := chunks[0].FunctionCalls[0].Name; got != "get_current_time" {
		t.Fatalf("first chunk function call name = %q, want get_current_time", got)
	}
	if got := chunks[0].FinishReason; got != "" {
		t.Fatalf("first chunk finishReason = %q, want empty", got)
	}
	if got := chunks[1].FinishReason; got != "STOP" {
		t.Fatalf("second chunk finishReason = %q, want STOP", got)
	}
	if chunks[1].HasContent() {
		t.Fatal("second chunk should only contain finishReason")
	}
}

func TestStreamVertexResponseTreatsUnspecifiedFinishReasonAsEmpty(t *testing.T) {
	body := `[{"results":[{"data":{"candidates":[{"finishReason":"FINISH_REASON_UNSPECIFIED","content":{"role":"model","parts":[{"text":"hello"}]}}]}}]}]`

	var chunks []CallResult
	status, err := streamVertexResponse(strings.NewReader(body), func(result *CallResult) error {
		chunks = append(chunks, *result)
		return nil
	})
	if err != nil {
		t.Fatalf("streamVertexResponse returned error: %v", err)
	}
	if status != 0 {
		t.Fatalf("status = %d, want 0", status)
	}
	if len(chunks) != 1 {
		t.Fatalf("chunks length = %d, want 1", len(chunks))
	}
	if got := chunks[0].FinishReason; got != "" {
		t.Fatalf("finishReason = %q, want empty", got)
	}
}

func TestStreamVertexResponseEmitsSSEChunks(t *testing.T) {
	body := "data: {\"results\":[{\"data\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hello\"}]}}]}}]}\n\n" +
		"data: {\"results\":[{\"data\":{\"candidates\":[{\"finishReason\":\"STOP\",\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\" world\"}]}}]}}]}\n\n"

	var text strings.Builder
	status, err := streamVertexResponse(strings.NewReader(body), func(result *CallResult) error {
		for _, part := range result.TextParts {
			text.WriteString(part.Text)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("streamVertexResponse returned error: %v", err)
	}
	if status != 0 {
		t.Fatalf("status = %d, want 0", status)
	}
	if got := text.String(); got != "hello world" {
		t.Fatalf("streamed text = %q, want hello world", got)
	}
}

func TestParseVertexResponseCoalescesStreamedTextForNonStream(t *testing.T) {
	body := []byte(`[{"results":[{"data":{"candidates":[{"content":{"role":"model","parts":[{"text":"hello "}]} }]}},{"data":{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"world"}]}}]}}]}]`)

	result, status, err := parseVertexResponse(body)
	if err != nil {
		t.Fatalf("parseVertexResponse returned error: %v", err)
	}
	if status != 0 {
		t.Fatalf("status = %d, want 0", status)
	}
	if len(result.TextParts) != 1 {
		t.Fatalf("TextParts length = %d, want 1", len(result.TextParts))
	}
	if got := result.TextParts[0].Text; got != "hello world" {
		t.Fatalf("text = %q, want hello world", got)
	}
}

func TestParseVertexResponseExpandsNestedGraphQLPayloadAndMetadata(t *testing.T) {
	body := []byte(`[{"results":[{"data":{"ui":{"streamGenerateContentAnonymous":[{"responseId":"response-1","candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"hello "}]}}]},{"modelVersion":"gemini-test-001","usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":2,"totalTokenCount":7},"candidates":[{"index":0,"finishReason":"STOP","content":{"role":"model","parts":[{"text":"world"}]}}]}]}}}]}]`)

	result, status, err := parseVertexResponse(body)
	if err != nil {
		t.Fatalf("parseVertexResponse returned error: %v", err)
	}
	if status != 0 {
		t.Fatalf("status = %d, want 0", status)
	}
	if got := result.ResponseID; got != "response-1" {
		t.Fatalf("ResponseID = %q, want response-1", got)
	}
	if got := result.ModelVersion; got != "gemini-test-001" {
		t.Fatalf("ModelVersion = %q, want gemini-test-001", got)
	}
	if got := result.UsageMetadata["totalTokenCount"]; got != float64(7) {
		t.Fatalf("totalTokenCount = %v, want 7", got)
	}
	if len(result.Candidates) != 1 {
		t.Fatalf("Candidates length = %d, want 1", len(result.Candidates))
	}
	if got := result.Candidates[0].TextParts[0].Text; got != "hello world" {
		t.Fatalf("candidate text = %q, want hello world", got)
	}
	if got := result.FinishReason; got != "STOP" {
		t.Fatalf("FinishReason = %q, want STOP", got)
	}
}

func TestParseVertexResponseKeepsCandidatesSeparate(t *testing.T) {
	body := []byte(`[{"results":[{"data":{"candidates":[{"index":0,"finishReason":"STOP","content":{"parts":[{"text":"first"}]}},{"index":1,"finishReason":"MAX_TOKENS","content":{"parts":[{"text":"second"}]}}]}}]}]`)

	result, _, err := parseVertexResponse(body)
	if err != nil {
		t.Fatalf("parseVertexResponse returned error: %v", err)
	}
	if len(result.Candidates) != 2 {
		t.Fatalf("Candidates length = %d, want 2", len(result.Candidates))
	}
	if got := result.Candidates[0].TextParts[0].Text; got != "first" {
		t.Fatalf("candidate 0 text = %q, want first", got)
	}
	if got := result.Candidates[1].TextParts[0].Text; got != "second" {
		t.Fatalf("candidate 1 text = %q, want second", got)
	}
	if got := result.TextParts[0].Text; got != "first" {
		t.Fatalf("primary text = %q, want first", got)
	}
}

func TestBuildVertexBodyNormalizesInlineData(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role": "user",
			"parts": []map[string]interface{}{
				{
					"inlineData": map[string]interface{}{
						"mimeType": "image/jpeg",
						"data":     "data:image/png;base64,aG Vs\nbG8=",
					},
				},
			},
		},
	}

	body, err := BuildVertexBody("gemini-test", contents, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	inlineData := inlineDataFromBody(t, body)
	if got := inlineData["mimeType"]; got != "image/png" {
		t.Fatalf("mimeType = %v, want image/png", got)
	}
	if got := inlineData["data"]; got != "aGVsbG8=" {
		t.Fatalf("data = %v, want normalized base64", got)
	}
}

func TestBuildVertexBodyRejectsRemoteImageURLAsInlineData(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role": "user",
			"parts": []map[string]interface{}{
				{
					"inlineData": map[string]interface{}{
						"mimeType": "image/png",
						"data":     "https://example.com/image.png",
					},
				},
			},
		},
	}

	_, err := BuildVertexBody("gemini-test", contents, nil, nil, nil, "token")
	if err == nil {
		t.Fatal("BuildVertexBody returned nil error")
	}
	if !strings.Contains(err.Error(), "remote URL") {
		t.Fatalf("error = %v, want remote URL message", err)
	}
}

func TestBuildVertexBodyNormalizesThoughtSignature(t *testing.T) {
	rawSignature := base64.RawURLEncoding.EncodeToString([]byte("real signature bytes"))
	contents := []map[string]interface{}{
		{
			"role": "model",
			"parts": []interface{}{
				map[string]interface{}{
					"text":             "hello",
					"thoughtSignature": rawSignature,
				},
			},
		},
	}

	body, err := BuildVertexBody("gemini-test", contents, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	part := firstPartFromBody(t, body)
	want := base64.StdEncoding.EncodeToString([]byte("real signature bytes"))
	if got := part["thoughtSignature"]; got != want {
		t.Fatalf("thoughtSignature = %v, want %s", got, want)
	}
}

func TestBuildVertexBodyDropsInvalidThoughtSignature(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role": "model",
			"parts": []interface{}{
				map[string]interface{}{
					"text":             "hello",
					"thoughtSignature": "not base64!",
				},
			},
		},
	}

	body, err := BuildVertexBody("gemini-test", contents, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	part := firstPartFromBody(t, body)
	if got, ok := part["thoughtSignature"]; ok {
		t.Fatalf("thoughtSignature should be omitted, got %v", got)
	}
}

func TestBuildVertexBodyNormalizesBypassThoughtSignature(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role": "model",
			"parts": []interface{}{
				map[string]interface{}{
					"functionCall": map[string]interface{}{
						"name": "lookup",
						"args": map[string]interface{}{"query": "weather"},
					},
					"thoughtSignature": thoughtSignatureBypassValue(),
				},
			},
		},
	}

	body, err := BuildVertexBody("gemini-3-pro", contents, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	part := firstPartFromBody(t, body)
	if got := part["thoughtSignature"]; got != thoughtSignatureBypassValue() {
		t.Fatalf("thoughtSignature = %v, want bypass", got)
	}
}

func TestBuildVertexBodyAddsBypassThoughtSignatureToFunctionCall(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role": "model",
			"parts": []map[string]interface{}{
				{
					"functionCall": map[string]interface{}{
						"name": "lookup",
						"args": map[string]interface{}{"query": "weather"},
					},
				},
			},
		},
	}

	body, err := BuildVertexBody("gemini-3-pro", contents, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	part := firstPartFromBody(t, body)
	if got := part["thoughtSignature"]; got != thoughtSignatureBypassValue() {
		t.Fatalf("thoughtSignature = %v, want bypass", got)
	}
}

func TestBuildVertexBodyPreservesFunctionCallThoughtSignature(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role": "model",
			"parts": []map[string]interface{}{
				{
					"functionCall": map[string]interface{}{
						"name": "lookup",
						"args": map[string]interface{}{"query": "weather"},
					},
					"thoughtSignature": base64.StdEncoding.EncodeToString([]byte("real signature")),
				},
			},
		},
		{
			"role":  "user",
			"parts": []map[string]interface{}{{"text": "continue"}},
		},
	}

	body, err := BuildVertexBody("gemini-3.6-flash", contents, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	part := firstPartFromBody(t, body)
	want := base64.StdEncoding.EncodeToString([]byte("real signature"))
	if got := part["thoughtSignature"]; got != want {
		t.Fatalf("thoughtSignature = %v, want %s", got, want)
	}
}

func TestParseVertexResponsePreservesThoughtFlag(t *testing.T) {
	body := []byte(`[{"results":[{"data":{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"text":"thinking","thought":true},{"text":"answer"}]}}]}}]}]`)

	result, status, err := parseVertexResponse(body)
	if err != nil {
		t.Fatalf("parseVertexResponse returned error: %v", err)
	}
	if status != 0 {
		t.Fatalf("status = %d, want 0", status)
	}
	if len(result.TextParts) != 2 {
		t.Fatalf("TextParts length = %d, want 2", len(result.TextParts))
	}
	if !result.TextParts[0].Thought {
		t.Fatal("first text part should be marked as thought")
	}
	if result.TextParts[1].Thought {
		t.Fatal("second text part should not be marked as thought")
	}
}

func TestParseVertexResponsePreservesFunctionCall(t *testing.T) {
	body := []byte(`[{"results":[{"data":{"candidates":[{"finishReason":"STOP","content":{"role":"model","parts":[{"functionCall":{"name":"lookup","args":{"query":"weather"}},"thoughtSignature":"sig"}]}}]}}]}]`)

	result, status, err := parseVertexResponse(body)
	if err != nil {
		t.Fatalf("parseVertexResponse returned error: %v", err)
	}
	if status != 0 {
		t.Fatalf("status = %d, want 0", status)
	}
	if len(result.FunctionCalls) != 1 {
		t.Fatalf("FunctionCalls length = %d, want 1", len(result.FunctionCalls))
	}
	if len(result.TextParts) != 0 {
		t.Fatalf("TextParts length = %d, want 0", len(result.TextParts))
	}
	if got := result.FunctionCalls[0].Name; got != "lookup" {
		t.Fatalf("function name = %q, want lookup", got)
	}
	if got := result.FunctionCalls[0].Args["query"]; got != "weather" {
		t.Fatalf("function args query = %v, want weather", got)
	}
	if got := result.FunctionCalls[0].ThoughtSignature; got != "sig" {
		t.Fatalf("function thought signature = %q, want sig", got)
	}
}

func TestBuildVertexBodySanitizesThinkingConfig(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role": "user",
			"parts": []map[string]interface{}{
				{"text": "Hello"},
			},
		},
	}

	tests := []struct {
		name       string
		modelName  string
		wantLevel  bool
		wantBudget bool
	}{
		{
			name:      "gemini 3 keeps thinkingLevel and removes thinkingBudget",
			modelName: "gemini-3-pro",
			wantLevel: true,
		},
		{
			name:       "gemini 2.5 keeps thinkingBudget and removes thinkingLevel",
			modelName:  "gemini-2.5-pro",
			wantBudget: true,
		},
		{
			name:      "other models remove both thinking fields",
			modelName: "gemini-2.0-flash",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			genConfig := map[string]interface{}{
				"temperature": 0.8,
				"thinkingConfig": map[string]interface{}{
					"thinkingLevel":   "high",
					"thinkingBudget":  1024,
					"includeThoughts": true,
				},
			}

			body, err := BuildVertexBody(tt.modelName, contents, genConfig, nil, nil, "token")
			if err != nil {
				t.Fatalf("BuildVertexBody returned error: %v", err)
			}

			generationConfig := generationConfigFromBody(t, body)
			thinkingConfig := generationConfig["thinkingConfig"].(map[string]interface{})

			if _, ok := thinkingConfig["thinkingLevel"]; ok != tt.wantLevel {
				t.Fatalf("thinkingLevel presence = %v, want %v", ok, tt.wantLevel)
			}
			if _, ok := thinkingConfig["thinkingBudget"]; ok != tt.wantBudget {
				t.Fatalf("thinkingBudget presence = %v, want %v", ok, tt.wantBudget)
			}
			if _, ok := thinkingConfig["includeThoughts"]; !ok {
				t.Fatal("thinkingConfig should keep unrelated fields")
			}
			if tt.wantLevel {
				if got, ok := thinkingConfig["thinkingLevel"].(string); !ok || got != "HIGH" {
					t.Fatalf("thinkingLevel = %v, want HIGH", got)
				}
			}
		})
	}
}

func TestBuildVertexBodyUppercasesThinkingLevel(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role": "user",
			"parts": []map[string]interface{}{
				{"text": "Hello"},
			},
		},
	}
	genConfig := map[string]interface{}{
		"temperature": 0.8,
		"thinkingConfig": map[string]interface{}{
			"thinkingLevel": "low",
		},
	}

	body, err := BuildVertexBody("gemini-3-pro", contents, genConfig, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	generationConfig := generationConfigFromBody(t, body)
	thinkingConfig := generationConfig["thinkingConfig"].(map[string]interface{})
	if got, ok := thinkingConfig["thinkingLevel"].(string); !ok || got != "LOW" {
		t.Fatalf("thinkingLevel = %v, want LOW", got)
	}
}

func TestBuildVertexBodyKeepsGemini3ThinkingBudgetWithoutThinkingLevel(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role": "user",
			"parts": []map[string]interface{}{
				{"text": "Hello"},
			},
		},
	}
	genConfig := map[string]interface{}{
		"temperature": 0.8,
		"thinkingConfig": map[string]interface{}{
			"thinkingBudget": 1024,
		},
	}

	body, err := BuildVertexBody("gemini-3-pro", contents, genConfig, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	generationConfig := generationConfigFromBody(t, body)
	thinkingConfig := generationConfig["thinkingConfig"].(map[string]interface{})
	if _, ok := thinkingConfig["thinkingBudget"]; !ok {
		t.Fatal("thinkingConfig should keep thinkingBudget for gemini-3 when thinkingLevel is absent")
	}
}

func TestBuildVertexBodySanitizesUnsupportedImageResponseModality(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role":  "user",
			"parts": []map[string]interface{}{{"text": "Hello"}},
		},
	}
	genConfig := map[string]interface{}{
		"responseModalities": []interface{}{"TEXT", "IMAGE"},
	}

	body, err := BuildVertexBody("gemini-2.5-pro", contents, genConfig, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	generationConfig := generationConfigFromBody(t, body)
	modalities := generationConfig["responseModalities"].([]interface{})
	if len(modalities) != 1 || modalities[0] != "TEXT" {
		t.Fatalf("responseModalities = %v, want [TEXT]", modalities)
	}
}

func TestBuildVertexBodyKeepsSupportedImageResponseModality(t *testing.T) {
	contents := []map[string]interface{}{
		{
			"role":  "user",
			"parts": []map[string]interface{}{{"text": "Hello"}},
		},
	}
	genConfig := map[string]interface{}{
		"responseModalities": []interface{}{"TEXT", "IMAGE"},
	}

	body, err := BuildVertexBody("gemini-3-pro-image", contents, genConfig, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody returned error: %v", err)
	}

	generationConfig := generationConfigFromBody(t, body)
	modalities := generationConfig["responseModalities"].([]interface{})
	if len(modalities) != 2 || modalities[0] != "TEXT" || modalities[1] != "IMAGE" {
		t.Fatalf("responseModalities = %v, want [TEXT IMAGE]", modalities)
	}
}

func inlineDataFromBody(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()

	part := firstPartFromBody(t, body)
	return part["inlineData"].(map[string]interface{})
}

func firstPartFromBody(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()

	variables := variablesFromBody(t, body)
	contents := variables["contents"].([]interface{})
	content := contents[0].(map[string]interface{})
	parts := content["parts"].([]interface{})
	return parts[0].(map[string]interface{})
}

func generationConfigFromBody(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()

	variables := variablesFromBody(t, body)
	return variables["generationConfig"].(map[string]interface{})
}

func variablesFromBody(t *testing.T, body []byte) map[string]interface{} {
	t.Helper()

	var raw map[string]interface{}
	if err := sonic.Unmarshal(body, &raw); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}

	return raw["variables"].(map[string]interface{})
}

func TestHTTPStatusForError(t *testing.T) {
	if got := HTTPStatusForError(&vertexAPIError{Code: 8, Message: "quota"}); got != http.StatusTooManyRequests {
		t.Fatalf("status = %d, want 429", got)
	}
	if got := HTTPStatusForError(&upstreamHTTPError{Status: http.StatusBadGateway, Body: "bad gateway"}); got != http.StatusBadGateway {
		t.Fatalf("status = %d, want 502", got)
	}
}

func TestTrimTrailingEmptyModelTurns(t *testing.T) {
	// Empty model turn at the end should be trimmed
	emptyModelEnd := []map[string]interface{}{
		{"role": "user", "parts": []interface{}{map[string]interface{}{"text": "hi"}}},
		{"role": "model", "parts": []interface{}{map[string]interface{}{"text": ""}}},
	}
	res1, err := normalizeContents("gemini-2.5-flash", emptyModelEnd)
	if err != nil {
		t.Fatalf("normalizeContents error: %v", err)
	}
	if len(res1) != 1 || res1[0]["role"] != "user" {
		t.Fatalf("empty model turn was not trimmed, got len=%d, lastRole=%v", len(res1), res1[len(res1)-1]["role"])
	}

	// Other Gemini models preserve a non-empty model prefill as before.
	textModelEnd := []map[string]interface{}{
		{"role": "user", "parts": []interface{}{map[string]interface{}{"text": "hi"}}},
		{"role": "model", "parts": []interface{}{map[string]interface{}{"text": "I am model"}}},
	}
	res2, err := normalizeContents("gemini-2.5-flash", textModelEnd)
	if err != nil {
		t.Fatalf("normalizeContents error: %v", err)
	}
	if len(res2) != 2 || res2[len(res2)-1]["role"] != "model" {
		t.Fatalf("model prefill was modified, got len=%d, lastRole=%v", len(res2), res2[len(res2)-1]["role"])
	}
	res35, err := normalizeContents("gemini-3.5-flash", textModelEnd)
	if err != nil {
		t.Fatalf("normalizeContents(gemini-3.5-flash) error: %v", err)
	}
	if len(res35) != 2 || res35[len(res35)-1]["role"] != "model" {
		t.Fatalf("gemini-3.5 model prefill was modified, got len=%d, lastRole=%v", len(res35), res35[len(res35)-1]["role"])
	}

	for _, modelName := range []string{"gemini-3.6-flash"} {
		res, err := normalizeContents(modelName, textModelEnd)
		if err != nil {
			t.Fatalf("normalizeContents(%s) error: %v", modelName, err)
		}
		if len(res) != 1 || res[0]["role"] != "user" {
			t.Fatalf("%s model prefill was not removed, got len=%d, lastRole=%v", modelName, len(res), res[len(res)-1]["role"])
		}
	}

	multipleModelEnd := []map[string]interface{}{
		{"role": "user", "parts": []interface{}{map[string]interface{}{"text": "hi"}}},
		{"role": "model", "parts": []interface{}{map[string]interface{}{"text": "first"}}},
		{"role": "model", "parts": []interface{}{map[string]interface{}{"text": "second"}}},
	}
	res4, err := normalizeContents("gemini-3.6-flash", multipleModelEnd)
	if err != nil {
		t.Fatalf("normalizeContents with multiple model turns error: %v", err)
	}
	if len(res4) != 1 || res4[0]["role"] != "user" {
		t.Fatalf("multiple trailing model turns were not removed, got len=%d", len(res4))
	}

	trailingEmptyUser := []map[string]interface{}{
		{"role": "user", "parts": []interface{}{map[string]interface{}{"text": "hi"}}},
		{"role": "model", "parts": []interface{}{map[string]interface{}{"text": "answer"}}},
		{"role": "user", "parts": []interface{}{map[string]interface{}{"text": "  "}}},
	}
	res5, err := normalizeContents("gemini-3.6-flash", trailingEmptyUser)
	if err != nil {
		t.Fatalf("normalizeContents with trailing empty user error: %v", err)
	}
	if len(res5) != 1 || res5[0]["role"] != "user" {
		t.Fatalf("trailing empty user did not expose and remove model prefill, got len=%d", len(res5))
	}

	longText := strings.Repeat("中", 1100)
	if got := config.UpstreamLogValue(longText, false, 1024); len([]rune(got)) != 1027 {
		t.Fatalf("truncated model turn text rune length = %d, want 1027 including ellipsis", len([]rune(got)))
	}
}

func TestExtractModelName(t *testing.T) {
	body, err := BuildVertexBody("gemini-2.5-flash", []map[string]interface{}{
		{"role": "user", "parts": []interface{}{map[string]interface{}{"text": "hello"}}},
	}, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody error: %v", err)
	}

	model := extractModelName(body)
	if model != "gemini-2.5-flash" {
		t.Errorf("extractModelName(body) = %q, want %q", model, "gemini-2.5-flash")
	}

	if emptyModel := extractModelName(nil); emptyModel != "" {
		t.Errorf("extractModelName(nil) = %q, want empty string", emptyModel)
	}
}
