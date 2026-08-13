package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync"
	"testing"

	"vertex2api/client"
	"vertex2api/config"
)

func TestCode3RequestLogDeduplicatesByErrorAcrossBodiesAndRestarts(t *testing.T) {
	path := filepath.Join(t.TempDir(), code3RequestLogFilename)
	requestLog := newCode3RequestLog(path)
	errorA := &vertexAPIError{Code: 3, Message: "invalid request shape"}
	errorB := &vertexAPIError{Code: 3, Message: "model turn is not supported"}

	captured, err := requestLog.capture(errorA, []byte(`{"request":"first"}`))
	if err != nil || !captured {
		t.Fatalf("first capture = (%v, %v), want (true, nil)", captured, err)
	}
	captured, err = requestLog.capture(errorA, []byte(`{"request":"different body"}`))
	if err != nil || captured {
		t.Fatalf("duplicate capture = (%v, %v), want (false, nil)", captured, err)
	}
	captured, err = requestLog.capture(errorB, []byte(`{"request":"second error"}`))
	if err != nil || !captured {
		t.Fatalf("different error capture = (%v, %v), want (true, nil)", captured, err)
	}

	// A new logger represents a restarted process. The file remains the source
	// of truth, so the same error must still be suppressed.
	restartedLog := newCode3RequestLog(path)
	captured, err = restartedLog.capture(errorA, []byte(`{"request":"after restart"}`))
	if err != nil || captured {
		t.Fatalf("post-restart duplicate capture = (%v, %v), want (false, nil)", captured, err)
	}

	entries := readCode3RequestLogEntries(t, path)
	if len(entries) != 2 {
		t.Fatalf("entry count = %d, want 2", len(entries))
	}
	if got := requestField(t, entries[0]); got != "first" {
		t.Fatalf("first stored request = %q, want first", got)
	}
	if got := requestField(t, entries[1]); got != "second error" {
		t.Fatalf("second stored request = %q, want second error", got)
	}
}

func TestCode3RequestLogRedactsRecaptchaTokenBeforeWriting(t *testing.T) {
	path := filepath.Join(t.TempDir(), code3RequestLogFilename)
	requestLog := newCode3RequestLog(path)
	body := []byte(`{"variables":{"recaptchaToken":"secret-token","nested":[{"RecaptchaToken":"nested-secret"}]},"request":"kept"}`)

	captured, err := requestLog.capture(&vertexAPIError{Code: 3, Message: "redact me"}, body)
	if err != nil || !captured {
		t.Fatalf("capture = (%v, %v), want (true, nil)", captured, err)
	}
	if !bytes.Contains(body, []byte("secret-token")) {
		t.Fatal("capture mutated the upstream request body")
	}

	entries := readCode3RequestLogEntries(t, path)
	var stored map[string]interface{}
	if err := json.Unmarshal(entries[0].RequestBody, &stored); err != nil {
		t.Fatalf("decode stored request: %v", err)
	}
	variables := stored["variables"].(map[string]interface{})
	if variables["recaptchaToken"] != redactedCode3RequestValue {
		t.Fatalf("recaptchaToken = %v, want redacted", variables["recaptchaToken"])
	}
	nested := variables["nested"].([]interface{})[0].(map[string]interface{})
	if nested["RecaptchaToken"] != redactedCode3RequestValue {
		t.Fatalf("nested RecaptchaToken = %v, want redacted", nested["RecaptchaToken"])
	}
	if stored["request"] != "kept" {
		t.Fatalf("non-secret field changed: %#v", stored)
	}
}

func TestRedactCode3RequestBodyOmitsUnparseableInput(t *testing.T) {
	body, err := redactCode3RequestBody([]byte(`{"recaptchaToken":"secret"`))
	if err != nil {
		t.Fatalf("redactCode3RequestBody: %v", err)
	}
	if bytes.Contains(body, []byte("secret")) || string(body) != `"[UNPARSEABLE REQUEST BODY OMITTED]"` {
		t.Fatalf("unparseable body was not safely omitted: %s", body)
	}
}

func TestCode3RequestLogPathUsesAPIKeyDirectory(t *testing.T) {
	if got := code3RequestLogPath(".vertex2api-api-key"); got != code3RequestLogFilename {
		t.Fatalf("default log path = %q, want %q", got, code3RequestLogFilename)
	}
	want := filepath.Join("state", code3RequestLogFilename)
	if got := code3RequestLogPath(filepath.Join("state", "api-key")); got != want {
		t.Fatalf("state log path = %q, want %q", got, want)
	}
}

func TestCode3RequestLogDeletionResetsDeduplication(t *testing.T) {
	path := filepath.Join(t.TempDir(), code3RequestLogFilename)
	requestLog := newCode3RequestLog(path)
	code3Err := &vertexAPIError{Code: 3, Message: "same error"}

	if captured, err := requestLog.capture(code3Err, []byte(`{"request":"before deletion"}`)); err != nil || !captured {
		t.Fatalf("initial capture = (%v, %v), want (true, nil)", captured, err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatalf("remove log: %v", err)
	}
	if captured, err := requestLog.capture(code3Err, []byte(`{"request":"after deletion"}`)); err != nil || !captured {
		t.Fatalf("capture after deletion = (%v, %v), want (true, nil)", captured, err)
	}

	entries := readCode3RequestLogEntries(t, path)
	if len(entries) != 1 {
		t.Fatalf("entry count after deletion = %d, want 1", len(entries))
	}
	if got := requestField(t, entries[0]); got != "after deletion" {
		t.Fatalf("stored request after deletion = %q, want after deletion", got)
	}
}

func TestCode3RequestLogConcurrentDuplicatesWriteOnce(t *testing.T) {
	path := filepath.Join(t.TempDir(), code3RequestLogFilename)
	requestLog := newCode3RequestLog(path)
	code3Err := &vertexAPIError{Code: 3, Message: "concurrent error"}

	const workers = 32
	var wg sync.WaitGroup
	results := make(chan bool, workers)
	errorsFound := make(chan error, workers)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			captured, err := requestLog.capture(code3Err, []byte(`{"request":"concurrent"}`))
			results <- captured
			errorsFound <- err
		}()
	}
	wg.Wait()
	close(results)
	close(errorsFound)

	for err := range errorsFound {
		if err != nil {
			t.Fatalf("concurrent capture returned error: %v", err)
		}
	}
	writes := 0
	for captured := range results {
		if captured {
			writes++
		}
	}
	if writes != 1 {
		t.Fatalf("successful writes = %d, want 1", writes)
	}
	if entries := readCode3RequestLogEntries(t, path); len(entries) != 1 {
		t.Fatalf("entry count = %d, want 1", len(entries))
	}
}

func TestCode3RequestLogIgnoresOtherErrors(t *testing.T) {
	path := filepath.Join(t.TempDir(), code3RequestLogFilename)
	requestLog := newCode3RequestLog(path)

	for _, err := range []error{
		&vertexAPIError{Code: 5, Message: "not found"},
		&vertexAPIError{Code: 3, Message: "Failed to verify action"},
		errors.New("ordinary error"),
	} {
		if captured, captureErr := requestLog.capture(err, []byte(`{"request":"ignored"}`)); captureErr != nil || captured {
			t.Fatalf("capture(%v) = (%v, %v), want (false, nil)", err, captured, captureErr)
		}
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("log file exists for ignored errors: %v", err)
	}
}

func TestDoStreamCapturesCode3RequestWhenEnabled(t *testing.T) {
	vertexServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[{"results":[{"errors":[{"message":"captured through doStream","extensions":{"status":{"code":3}}}]}]}]`))
	}))
	defer vertexServer.Close()

	requestLogPath := filepath.Join(t.TempDir(), code3RequestLogFilename)
	vp := NewVertexProxy(
		&client.HTTPClient{HTTP: vertexServer.Client()},
		nil,
		&config.Config{
			VertexBaseURL:         vertexServer.URL,
			MaxRetry:              1,
			RetryDelayMs:          0,
			LogCode3RequestBodies: true,
		},
	)
	vp.code3RequestLog = newCode3RequestLog(requestLogPath)

	body, err := BuildVertexBody("gemini-test", []map[string]interface{}{
		{"role": "user", "parts": []map[string]interface{}{{"text": "body to capture"}}},
	}, nil, nil, nil, "token")
	if err != nil {
		t.Fatalf("BuildVertexBody: %v", err)
	}
	if _, err := vp.CallContext(context.Background(), body); err == nil {
		t.Fatal("CallContext returned nil error, want Code 3")
	}

	entries := readCode3RequestLogEntries(t, requestLogPath)
	if len(entries) != 1 {
		t.Fatalf("entry count = %d, want 1", len(entries))
	}
	if entries[0].Error != "captured through doStream" {
		t.Fatalf("stored error = %q", entries[0].Error)
	}
	var stored map[string]interface{}
	if err := json.Unmarshal(entries[0].RequestBody, &stored); err != nil {
		t.Fatalf("decode stored request: %v", err)
	}
	variables := stored["variables"].(map[string]interface{})
	if variables["recaptchaToken"] != redactedCode3RequestValue {
		t.Fatalf("stored recaptchaToken = %v, want redacted", variables["recaptchaToken"])
	}
	contents := variables["contents"].([]interface{})
	parts := contents[0].(map[string]interface{})["parts"].([]interface{})
	if parts[0].(map[string]interface{})["text"] != "body to capture" {
		t.Fatalf("stored diagnostic body lost content: %#v", stored)
	}
}

func readCode3RequestLogEntries(t *testing.T, path string) []code3RequestLogEntry {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("open Code 3 request log: %v", err)
	}
	defer file.Close()

	var entries []code3RequestLogEntry
	decoder := json.NewDecoder(file)
	for decoder.More() {
		var entry code3RequestLogEntry
		if err := decoder.Decode(&entry); err != nil {
			t.Fatalf("decode Code 3 request log: %v", err)
		}
		entries = append(entries, entry)
	}
	return entries
}

func requestField(t *testing.T, entry code3RequestLogEntry) string {
	t.Helper()
	var body map[string]string
	if err := json.Unmarshal(entry.RequestBody, &body); err != nil {
		t.Fatalf("decode stored request body: %v", err)
	}
	return body["request"]
}
