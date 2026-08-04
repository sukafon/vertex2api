package client

import (
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"time"

	"vertex2api/config"
)

// HTTPClient 封装 net/http 客户端，支持连接池和 HTTP/2
type HTTPClient struct {
	HTTP *http.Client
}

const maxBufferedResponseBytes = 16 * 1024 * 1024

func New(cfg *config.Config) (*HTTPClient, error) {
	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 20,
		MaxConnsPerHost:     50,
		IdleConnTimeout:     90 * time.Second,
		TLSHandshakeTimeout: 10 * time.Second,
		ForceAttemptHTTP2:   true,
		TLSClientConfig: &tls.Config{
			MinVersion: tls.VersionTLS12,
		},
	}

	// 配置 HTTP 代理
	if cfg.Proxy != "" {
		proxyURL, err := url.Parse(cfg.Proxy)
		if err != nil {
			return nil, fmt.Errorf("parse proxy URL: %w", err)
		}
		if proxyURL.Host == "" {
			return nil, fmt.Errorf("proxy URL must include a host")
		}
		transport.Proxy = http.ProxyURL(proxyURL)
	}

	return &HTTPClient{
		HTTP: &http.Client{
			Transport: transport,
			Timeout:   time.Duration(cfg.HTTPTimeoutSeconds) * time.Second,
		},
	}, nil
}

func (c *HTTPClient) CloseIdleConnections() {
	if c != nil && c.HTTP != nil {
		c.HTTP.CloseIdleConnections()
	}
}

// Do 执行请求并返回响应 body 字节切片
func (c *HTTPClient) Do(req *http.Request) ([]byte, int, error) {
	resp, err := c.HTTP.Do(req)
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
	return c.HTTP.Do(req)
}
