package client

import (
	"fmt"
	"math/rand"
	"net/http"
)

// Fingerprint describes the browser-facing part of an upstream request
// fingerprint. It intentionally keeps all browser version fields consistent
// so the User-Agent and Sec-CH-UA headers describe the same client.
//
// The standard library HTTP transport does not expose a way to reproduce a
// browser's TLS ClientHello. These headers are therefore a request-layer
// emulation, not a complete TLS fingerprint.
type Fingerprint struct {
	UserAgent              string
	SecCHUA                string
	SecCHUAFullVersionList string
	Platform               string
	PlatformVersion        string
	Architecture           string
	Bitness                string
	FullVersion            string
}

// NewRandomFingerprint selects a Chromium-family browser profile. The pool
// mirrors the browser families and version range used by the reference
// implementation. Callers should keep the returned profile for the lifetime
// of one browser-like flow, then select a new one for the next retry.
func NewRandomFingerprint() Fingerprint {
	return fingerprintProfiles[rand.Intn(len(fingerprintProfiles))]
}

// Apply sets the common client-hint headers shared by the upstream requests.
func (f Fingerprint) Apply(req *http.Request) {
	if req == nil {
		return
	}
	req.Header.Set("User-Agent", f.UserAgent)
	req.Header.Set("Sec-CH-UA", f.SecCHUA)
	req.Header.Set("Sec-CH-UA-Mobile", "?0")
	req.Header.Set("Sec-CH-UA-Platform", fmt.Sprintf("%q", f.Platform))
	req.Header.Set("Sec-CH-UA-Arch", fmt.Sprintf("%q", f.Architecture))
	req.Header.Set("Sec-CH-UA-Bitness", fmt.Sprintf("%q", f.Bitness))
	req.Header.Set("Sec-CH-UA-Full-Version", fmt.Sprintf("%q", f.fullVersion()))
	req.Header.Set("Sec-CH-UA-Full-Version-List", f.SecCHUAFullVersionList)
	req.Header.Set("Sec-CH-UA-Platform-Version", fmt.Sprintf("%q", f.PlatformVersion))
	req.Header.Set("Sec-CH-UA-Model", `""`)
	req.Header.Set("Sec-CH-UA-WoW64", "?0")
	req.Header.Set("Sec-CH-UA-Form-Factors", `"Desktop"`)
	req.Header.Set("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8,en-GB;q=0.7,en-US;q=0.6")
}

// ApplyNavigationHeaders applies the headers used for the reCAPTCHA anchor
// navigation request.
func (f Fingerprint) ApplyNavigationHeaders(req *http.Request) {
	f.Apply(req)
	if req == nil {
		return
	}
	req.Header.Set("Upgrade-Insecure-Requests", "1")
	req.Header.Set("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8")
	req.Header.Set("Sec-Fetch-Site", "cross-site")
	req.Header.Set("Sec-Fetch-Mode", "navigate")
	req.Header.Set("Sec-Fetch-Dest", "iframe")
}

// ApplyXHRHeaders applies the headers used for browser XHR/fetch requests.
func (f Fingerprint) ApplyXHRHeaders(req *http.Request, contentType, accept, origin, referer, site string) {
	f.Apply(req)
	if req == nil {
		return
	}
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if accept != "" {
		req.Header.Set("Accept", accept)
	}
	if origin != "" {
		req.Header.Set("Origin", origin)
	}
	if referer != "" {
		req.Header.Set("Referer", referer)
	}
	if site != "" {
		req.Header.Set("Sec-Fetch-Site", site)
	}
	req.Header.Set("Sec-Fetch-Mode", "cors")
	req.Header.Set("Sec-Fetch-Dest", "empty")
	req.Header.Set("Priority", "u=1, i")
}

func (f Fingerprint) fullVersion() string {
	// SecCHUAFullVersionList is the source of truth for the profile. The
	// individual full-version hint is derived from its Chromium version.
	return f.FullVersion
}

var fingerprintProfiles = buildFingerprintProfiles()

func buildFingerprintProfiles() []Fingerprint {
	profiles := make([]Fingerprint, 0, 34)
	for major := 131; major <= 147; major++ {
		// Keep a stable, realistic-looking build number for each major version;
		// the important property is that all fields agree on the major version.
		build := 6000 + (major-131)*97
		profiles = append(profiles,
			newChromiumFingerprint("Chrome", "Google Chrome", major, build),
			newChromiumFingerprint("Edge", "Microsoft Edge", major, build+31),
		)
	}
	return profiles
}

func newChromiumFingerprint(uaName, brand string, major, build int) Fingerprint {
	fullVersion := fmt.Sprintf("%d.0.%d.0", major, build)
	uaVersion := fullVersion
	if uaName == "Edge" {
		uaVersion = fullVersion + " Edg/" + fullVersion
	}

	return Fingerprint{
		UserAgent:              fmt.Sprintf("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/%s Safari/537.36", uaVersion),
		SecCHUA:                fmt.Sprintf(`"Not;A=Brand";v="8", "Chromium";v="%d", "%s";v="%d"`, major, brand, major),
		SecCHUAFullVersionList: fmt.Sprintf(`"Not;A=Brand";v="8.0.0.0", "Chromium";v="%s", "%s";v="%s"`, fullVersion, brand, fullVersion),
		Platform:               "Windows",
		PlatformVersion:        "19.0.0",
		Architecture:           "x86",
		Bitness:                "64",
		FullVersion:            fullVersion,
	}
}
