package client

import (
	"net/http"
	"strings"
	"testing"
)

func TestRandomFingerprintProfilesAreConsistent(t *testing.T) {
	if got, want := len(fingerprintProfiles), 34; got != want {
		t.Fatalf("fingerprint profile count = %d, want %d", got, want)
	}

	for _, profile := range fingerprintProfiles {
		if profile.UserAgent == "" || profile.SecCHUA == "" || profile.FullVersion == "" {
			t.Fatal("fingerprint profile has an empty required field")
		}
		if !strings.Contains(profile.SecCHUAFullVersionList, profile.FullVersion) {
			t.Fatalf("full version %q is missing from Sec-CH-UA-Full-Version-List %q", profile.FullVersion, profile.SecCHUAFullVersionList)
		}
		if !strings.Contains(profile.UserAgent, "Chrome/"+profile.FullVersion) {
			t.Fatalf("User-Agent %q does not contain Chrome/%s", profile.UserAgent, profile.FullVersion)
		}
	}
}

func TestFingerprintApplyXHRHeaders(t *testing.T) {
	profile := fingerprintProfiles[0]
	req, err := http.NewRequest(http.MethodPost, "https://example.test", nil)
	if err != nil {
		t.Fatalf("NewRequest(): %v", err)
	}

	profile.ApplyXHRHeaders(req, "application/json", "*/*", "https://origin.test", "https://origin.test/", "cross-site")

	for name, want := range map[string]string{
		"User-Agent":             profile.UserAgent,
		"Sec-CH-UA":              profile.SecCHUA,
		"Sec-CH-UA-Full-Version": `"` + profile.FullVersion + `"`,
		"Content-Type":           "application/json",
		"Accept":                 "*/*",
		"Origin":                 "https://origin.test",
		"Referer":                "https://origin.test/",
		"Sec-Fetch-Site":         "cross-site",
		"Sec-Fetch-Mode":         "cors",
		"Sec-Fetch-Dest":         "empty",
	} {
		if got := req.Header.Get(name); got != want {
			t.Fatalf("%s = %q, want %q", name, got, want)
		}
	}
}
