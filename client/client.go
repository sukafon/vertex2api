package client

import (
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"vertex2api/config"

	fhttp "github.com/bogdanfinn/fhttp"
	tls_client "github.com/bogdanfinn/tls-client"
	"github.com/bogdanfinn/tls-client/profiles"
)

// HTTPClient keeps the standard-library request interface used by the
// application while using tls-client for upstream requests created by New.
// The HTTP field remains as a compatibility fallback for tests and callers
// that construct HTTPClient literals directly.
type HTTPClient struct {
	HTTP        *http.Client
	tlsClient   tls_client.HttpClient
	fingerprint Fingerprint
}

const maxBufferedResponseBytes = 16 * 1024 * 1024

func New(cfg *config.Config) (*HTTPClient, error) {
	if cfg == nil {
		return nil, fmt.Errorf("config is nil")
	}

	profileName := strings.ToLower(strings.TrimSpace(cfg.TLSClientProfile))
	if profileName == "" {
		profileName = "chrome_146"
	}
	profile, ok := profiles.MappedTLSClients[profileName]
	if !ok {
		return nil, fmt.Errorf("unsupported TLS_CLIENT_PROFILE %q", cfg.TLSClientProfile)
	}

	timeoutSeconds := cfg.HTTPTimeoutSeconds
	if timeoutSeconds <= 0 {
		timeoutSeconds = 30
	}
	transportOptions := &tls_client.TransportOptions{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 20,
		MaxConnsPerHost:     50,
	}
	idleConnTimeout := 90 * time.Second
	transportOptions.IdleConnTimeout = &idleConnTimeout

	options := []tls_client.HttpClientOption{
		tls_client.WithTimeoutSeconds(timeoutSeconds),
		tls_client.WithClientProfile(profile),
		tls_client.WithTransportOptions(transportOptions),
		tls_client.WithDisableHttp3(),
		tls_client.WithCookieJar(tls_client.NewCookieJar()),
	}
	if cfg.Proxy != "" {
		proxyURL, err := url.Parse(cfg.Proxy)
		if err != nil {
			return nil, fmt.Errorf("parse proxy URL: %w", err)
		}
		if proxyURL.Host == "" {
			return nil, fmt.Errorf("proxy URL must include a host")
		}
		options = append(options, tls_client.WithProxyUrl(cfg.Proxy))
	}

	upstreamClient, err := tls_client.NewHttpClient(tls_client.NewNoopLogger(), options...)
	if err != nil {
		return nil, fmt.Errorf("create tls-client: %w", err)
	}

	return &HTTPClient{
		tlsClient:   upstreamClient,
		fingerprint: fingerprintForTLSProfile(profileName),
	}, nil
}

func (c *HTTPClient) CloseIdleConnections() {
	if c != nil && c.tlsClient != nil {
		c.tlsClient.CloseIdleConnections()
	}
	if c != nil && c.HTTP != nil {
		c.HTTP.CloseIdleConnections()
	}
}

// UsesTLSClient reports whether this client was created by New and uses the
// browser-emulating transport rather than the compatibility fallback.
func (c *HTTPClient) UsesTLSClient() bool {
	return c != nil && c.tlsClient != nil
}

// Do 执行请求并返回响应 body 字节切片
func (c *HTTPClient) Do(req *http.Request) ([]byte, int, error) {
	resp, err := c.DoRaw(req)
	if err != nil {
		return nil, 0, err
	}
	defer resp.Body.Close()
	if resp.ContentLength > maxBufferedResponseBytes {
		return nil, resp.StatusCode, fmt.Errorf("response body exceeds %d bytes", maxBufferedResponseBytes)
	}

	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBufferedResponseBytes+1))
	if err != nil {
		return nil, resp.StatusCode, err
	}
	if len(body) > maxBufferedResponseBytes {
		return nil, resp.StatusCode, fmt.Errorf("response body exceeds %d bytes", maxBufferedResponseBytes)
	}

	return body, resp.StatusCode, nil
}

// DoRaw executes the request and returns the raw response so callers can
// consume the body incrementally. The caller must close resp.Body.
func (c *HTTPClient) DoRaw(req *http.Request) (*http.Response, error) {
	if c == nil {
		return nil, fmt.Errorf("HTTP client is nil")
	}
	if req == nil {
		return nil, fmt.Errorf("HTTP request is nil")
	}
	if c.tlsClient != nil {
		return c.doTLSClient(req)
	}
	if c.HTTP == nil {
		return nil, fmt.Errorf("HTTP transport is unavailable")
	}
	return c.HTTP.Do(req)
}

func (c *HTTPClient) doTLSClient(req *http.Request) (*http.Response, error) {
	prepared := req.Clone(req.Context())
	prepared.Body = req.Body
	// tls-client does not apply default headers when a request already has
	// headers. Apply the stable browser profile here so the HTTP headers and
	// TLS ClientHello describe the same browser family and version.
	c.fingerprint.Apply(prepared)

	tlsReq, err := fhttp.NewRequestWithContext(prepared.Context(), prepared.Method, prepared.URL.String(), prepared.Body)
	if err != nil {
		return nil, fmt.Errorf("build tls-client request: %w", err)
	}
	tlsReq.Header = cloneTLSHeaders(prepared.Header)
	tlsReq.Host = prepared.Host
	tlsReq.ContentLength = prepared.ContentLength
	tlsReq.Close = prepared.Close

	tlsResp, err := c.tlsClient.Do(tlsReq)
	if err != nil {
		return nil, err
	}
	return convertTLSResponse(tlsResp, req), nil
}

func cloneTLSHeaders(src http.Header) fhttp.Header {
	dst := make(fhttp.Header, len(src))
	for key, values := range src {
		dst[key] = append([]string(nil), values...)
	}
	return dst
}

func convertTLSResponse(src *fhttp.Response, req *http.Request) *http.Response {
	if src == nil {
		return nil
	}
	return &http.Response{
		Status:           src.Status,
		StatusCode:       src.StatusCode,
		Proto:            src.Proto,
		ProtoMajor:       src.ProtoMajor,
		ProtoMinor:       src.ProtoMinor,
		Header:           cloneStandardHeaders(src.Header),
		Body:             src.Body,
		ContentLength:    src.ContentLength,
		TransferEncoding: append([]string(nil), src.TransferEncoding...),
		Close:            src.Close,
		Uncompressed:     src.Uncompressed,
		Request:          req,
	}
}

func cloneStandardHeaders(src fhttp.Header) http.Header {
	dst := make(http.Header, len(src))
	for key, values := range src {
		dst[key] = append([]string(nil), values...)
	}
	return dst
}

func fingerprintForTLSProfile(profileName string) Fingerprint {
	major := 146
	if strings.HasPrefix(profileName, "chrome_") {
		value := strings.TrimPrefix(profileName, "chrome_")
		value = strings.SplitN(value, "_", 2)[0]
		if parsed, err := strconv.Atoi(value); err == nil && parsed >= 100 {
			major = parsed
		}
	}
	build := 6000 + (major-131)*97
	return newChromiumFingerprint("Chrome", "Google Chrome", major, build)
}
