package client

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"vertex2api/config"
)

func TestNewRejectsMalformedProxy(t *testing.T) {
	_, err := New(&config.Config{Proxy: "://bad", HTTPTimeoutSeconds: 1})
	if err == nil {
		t.Fatal("New() succeeded with malformed proxy URL")
	}
}

func TestDoRejectsOversizedBufferedResponse(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "16777217")
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	c, err := New(&config.Config{HTTPTimeoutSeconds: 5})
	if err != nil {
		t.Fatalf("New(): %v", err)
	}
	defer c.CloseIdleConnections()
	req, err := http.NewRequest(http.MethodGet, server.URL, nil)
	if err != nil {
		t.Fatalf("NewRequest(): %v", err)
	}
	if _, status, err := c.Do(req); err == nil || status != http.StatusOK {
		t.Fatalf("Do() status=%d err=%v, want status 200 and size error", status, err)
	}
}
