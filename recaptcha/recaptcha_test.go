package recaptcha

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"vertex2api/client"
	"vertex2api/config"
)

func TestSelectRecaptchaBaseURL(t *testing.T) {
	baseURL := "https://recaptcha.example"
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
			want:        "https://prefix2.example/https://recaptcha.example",
		},
		{
			name:   "no prefixes falls back to direct",
			direct: false,
			want:   baseURL,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := selectRecaptchaBaseURL(baseURL, tt.prefixes, tt.direct, tt.prefixIndex); got != tt.want {
				t.Fatalf("selectRecaptchaBaseURL() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestTokenCacheLeasesOnlyUnusedTokens(t *testing.T) {
	var mu sync.Mutex
	reloads := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.URL.Query().Get("k"); got != "test-recaptcha-key" {
			t.Errorf("recaptcha key = %q, want test-recaptcha-key", got)
		}
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
	defer server.Close()

	tc := NewTokenCache(&client.HTTPClient{HTTP: server.Client()}, &config.Config{RecaptchaBase: server.URL, RecaptchaKey: "test-recaptcha-key"})

	first, err := tc.GetTokenContext(context.Background())
	if err != nil {
		t.Fatalf("first GetToken returned error: %v", err)
	}
	second, err := tc.GetTokenContext(context.Background())
	if err != nil {
		t.Fatalf("second GetToken returned error: %v", err)
	}

	if first.Token() == second.Token() {
		t.Fatalf("concurrent leases got the same token %q", first.Token())
	}
	if got := reloadCount(&mu, &reloads); got != 2 {
		t.Fatalf("reloads = %d, want 2 while first token is still leased", got)
	}

	first.Release()
	first.Release()

	third, err := tc.GetTokenContext(context.Background())
	if err != nil {
		t.Fatalf("third GetToken returned error: %v", err)
	}
	if got, want := third.Token(), first.Token(); got != want {
		t.Fatalf("released token was not reused: got %q, want %q", got, want)
	}
	if got := reloadCount(&mu, &reloads); got != 2 {
		t.Fatalf("reloads = %d, want 2 after reusing released token", got)
	}

	second.Release()
	third.Retire()
	fourth, err := tc.GetTokenContext(context.Background())
	if err != nil {
		t.Fatalf("fourth GetToken returned error: %v", err)
	}
	if fourth.Token() == first.Token() {
		t.Fatalf("retired token was reused: %q", fourth.Token())
	}
	fourth.Release()
}

func reloadCount(mu *sync.Mutex, reloads *int) int {
	mu.Lock()
	defer mu.Unlock()
	return *reloads
}
