package middleware

import (
	"net/http"
	"net/url"
	"strings"
)

// RejectPathTraversal rejects dot-dot path traversal before authentication or
// ServeMux path cleaning can reinterpret the request as another route.
func RejectPathTraversal(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hasPathTraversal(r.URL.Path) || hasPathTraversal(r.URL.EscapedPath()) {
			http.Error(w, "invalid request path", http.StatusBadRequest)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func hasPathTraversal(path string) bool {
	for i := 0; i < 3; i++ {
		if strings.Contains(path, "..") {
			return true
		}

		decoded, err := url.PathUnescape(path)
		if err != nil {
			return true
		}
		if decoded == path {
			return false
		}
		path = decoded
	}
	return strings.Contains(path, "..")
}
