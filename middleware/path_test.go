package middleware

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestRejectPathTraversalRejectsEncodedAndDoubleEncodedDots(t *testing.T) {
	paths := []string{
		"/v1/models/../../health",
		"/%2e%2e/health",
		"/%252e%252e/health",
	}

	for _, path := range paths {
		t.Run(path, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/health", nil)
			req.URL.Path = path
			req.URL.RawPath = path
			rec := httptest.NewRecorder()

			called := false
			RejectPathTraversal(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
				called = true
			})).ServeHTTP(rec, req)

			if called {
				t.Fatal("next handler was called for a traversal path")
			}
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d, want 400", rec.Code)
			}
		})
	}
}
