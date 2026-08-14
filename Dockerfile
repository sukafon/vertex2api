# syntax=docker/dockerfile:1

ARG GO_VERSION=1.26.6
FROM golang:${GO_VERSION}-alpine AS build

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .

ARG VERSION=1.0.6
ARG COMMIT=unknown
ARG BUILD_DATE=unknown
ARG GOAMD64=v3
RUN CGO_ENABLED=0 GOOS=linux GOAMD64=${GOAMD64} go build -trimpath -pgo=auto \
  -ldflags="-s -w -X main.version=${VERSION} -X main.commit=${COMMIT} -X main.buildDate=${BUILD_DATE}" \
  -o /out/vertex2api .

FROM alpine:3.24.1

RUN apk add --no-cache ca-certificates tzdata \
  && addgroup -S vertex2api \
  && adduser -S -G vertex2api vertex2api

# These are public browser-chain identifiers, not the service access secret.
# They can be overridden at build time if the upstream values are rotated.
ARG GRAPHQL_API_KEY=AIzaSyCI-zsRP85UVOi0DjtiCwWBwQ1djDy741g
ARG RECAPTCHA_KEY=6LdCjtspAAAAAMcV4TGdWLJqRTEk1TfpdLqEnKdj
ENV GRAPHQL_API_KEY=${GRAPHQL_API_KEY} \
    RECAPTCHA_KEY=${RECAPTCHA_KEY} \
    API_KEY_FILE=/data/api-key

WORKDIR /app
RUN mkdir -p /data && chown vertex2api:vertex2api /data
COPY --from=build --chown=vertex2api:vertex2api /out/vertex2api /app/vertex2api

USER vertex2api
VOLUME ["/data"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/health >/dev/null || exit 1

ENTRYPOINT ["/app/vertex2api"]
