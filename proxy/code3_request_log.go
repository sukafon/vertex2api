package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
)

const code3RequestLogFilename = "code3_request_bodies.log"

type code3RequestLog struct {
	path string
	mu   sync.Mutex
}

type code3RequestLogEntry struct {
	CapturedAt  string          `json:"captured_at"`
	Code        int             `json:"code"`
	Error       string          `json:"error"`
	RequestBody json.RawMessage `json:"request_body"`
}

func newCode3RequestLog(path string) *code3RequestLog {
	return &code3RequestLog{path: path}
}

func code3RequestLogPath(apiKeyFile string) string {
	dir := filepath.Dir(strings.TrimSpace(apiKeyFile))
	if dir == "" || dir == "." {
		return code3RequestLogFilename
	}
	return filepath.Join(dir, code3RequestLogFilename)
}

// capture writes only the first occurrence of an exact Code 3 error message.
// The log file is the persistent deduplication state: deleting it immediately
// allows the same error to be captured again, without restarting the process.
func (l *code3RequestLog) capture(err error, requestBody []byte) (bool, error) {
	if l == nil {
		return false, nil
	}

	var vertexErr *vertexAPIError
	if !errors.As(err, &vertexErr) || vertexErr.Code != 3 {
		return false, nil
	}
	if classifyRecaptchaRetryError(vertexErr) == recaptchaVerifyFailed {
		// "Failed to verify action" is a transient reCAPTCHA failure handled by
		// the retry path, not a malformed request that needs manual diagnosis.
		return false, nil
	}
	errorMessage := normalizeCode3ErrorMessage(vertexErr.Message)

	l.mu.Lock()
	defer l.mu.Unlock()

	seen, readErr := code3ErrorAlreadyLogged(l.path, errorMessage)
	if readErr != nil {
		return false, fmt.Errorf("read Code 3 request log: %w", readErr)
	}
	if seen {
		return false, nil
	}

	body := json.RawMessage(bytes.Clone(requestBody))
	if !json.Valid(body) {
		encodedBody, marshalErr := json.Marshal(string(requestBody))
		if marshalErr != nil {
			return false, fmt.Errorf("encode non-JSON request body: %w", marshalErr)
		}
		body = encodedBody
	}

	entry := code3RequestLogEntry{
		CapturedAt:  time.Now().UTC().Format(time.RFC3339Nano),
		Code:        vertexErr.Code,
		Error:       errorMessage,
		RequestBody: body,
	}
	encodedEntry, marshalErr := json.Marshal(entry)
	if marshalErr != nil {
		return false, fmt.Errorf("encode Code 3 request log entry: %w", marshalErr)
	}
	encodedEntry = append(encodedEntry, '\n')

	file, openErr := os.OpenFile(l.path, os.O_WRONLY|os.O_CREATE|os.O_APPEND, 0o600)
	if openErr != nil {
		return false, fmt.Errorf("open Code 3 request log: %w", openErr)
	}
	if _, writeErr := file.Write(encodedEntry); writeErr != nil {
		_ = file.Close()
		return false, fmt.Errorf("write Code 3 request log: %w", writeErr)
	}
	if syncErr := file.Sync(); syncErr != nil {
		_ = file.Close()
		return false, fmt.Errorf("sync Code 3 request log: %w", syncErr)
	}
	if closeErr := file.Close(); closeErr != nil {
		return false, fmt.Errorf("close Code 3 request log: %w", closeErr)
	}
	return true, nil
}

func code3ErrorAlreadyLogged(path, errorMessage string) (bool, error) {
	file, err := os.Open(path)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	defer file.Close()

	reader := bufio.NewReader(file)
	for {
		line, readErr := reader.ReadBytes('\n')
		line = bytes.TrimSpace(line)
		if len(line) > 0 {
			var entry code3RequestLogEntry
			if unmarshalErr := json.Unmarshal(line, &entry); unmarshalErr == nil &&
				entry.Code == 3 && normalizeCode3ErrorMessage(entry.Error) == errorMessage {
				return true, nil
			}
		}

		switch {
		case readErr == nil:
			continue
		case errors.Is(readErr, io.EOF):
			return false, nil
		default:
			return false, readErr
		}
	}
}

func normalizeCode3ErrorMessage(message string) string {
	return strings.TrimSpace(strings.ReplaceAll(message, "\r\n", "\n"))
}

func (vp *VertexProxy) captureCode3Request(err error, requestBody []byte) {
	if vp == nil || vp.code3RequestLog == nil {
		return
	}

	captured, captureErr := vp.code3RequestLog.capture(err, requestBody)
	if captureErr != nil {
		log.Error().Err(captureErr).Str("path", vp.code3RequestLog.path).Msg("Failed to capture Code 3 upstream request body")
		return
	}
	if captured {
		log.Warn().Str("path", vp.code3RequestLog.path).Msg("Captured a new Code 3 upstream request body")
	}
}
