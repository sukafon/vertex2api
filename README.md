# vertex2api

`vertex2api` 将 Google Cloud Console 使用的匿名 Vertex GraphQL 链路转换为 OpenAI、Gemini 和 Anthropic 风格的 HTTP API。项目重点是协议字段、流式事件、错误信封、工具 Schema 与 usage 的协议转换。

## 已实现接口

| 接口 | 路径 |
| --- | --- |
| OpenAI | `POST /v1/chat/completions`、`POST /v1/responses`、`POST /v1/responses/compact`、`GET /v1/models`、`GET /v1/models/{model}`、`POST /v1/images/generations`、`POST /v1/images/edits` |
| Gemini | `POST /v1beta/models/{model}:generateContent`、`:streamGenerateContent`、`:countTokens`；同时接受 `v1beta1` 和 `v1` 生成路径；`GET /v1beta/models` |
| Anthropic | `POST /v1/messages`、`POST /v1/messages/count_tokens` |

兼容性须知：

- 将 OpenAI/Anthropic 工具 Schema 归一化为 Gemini 可接受的子集，不修改调用者的原始对象。
- 保留 Gemini 候选边界、结束原因、工具调用、思考签名和原生候选元数据。
- Chat Completions 的显式 `reasoning_effort` 会映射到 Gemini 3 `thinkingLevel` 或 Gemini 2.5 `thinkingBudget`；`xhigh` 会静默降级为 Gemini 可表达的 `high`，省略该字段时保留模型默认思考配置，不依据 token 上限自动降级。
- 对 `gemini-3.6-*` 移除末尾 model 预填充，适配该版本不接受 model turn 结尾的请求约束；这不是把内容改写成 `systemInstruction`，其他模型保留原行为。
- Responses 支持无状态字符串/输入项数组、文本与 data URL 图片、函数工具、custom、local_shell、shell、apply_patch、结构化输出和 SSE 生命周期事件；不实现 `previous_response_id`、后台任务或 OpenAI 托管工具。
- `/v1/responses/compact` 与 `context_management.compact_threshold` 会调用当前 Vertex 模型生成状态检查点，再输出 AES-GCM 认证加密的 opaque compaction item。该 item 只保证由相同 API_KEY 列表配置的 vertex2api 实例解码，不能与 OpenAI 官方 compaction item 互换；修改密钥内容或顺序会使旧 item 失效。无密钥运行时使用进程内随机密钥，重启后旧 item 失效。

每个接口只实现仓库中列出的字段和行为，未列出的字段不会被视为已生效。

## 快速开始

要求 Go `1.26.5` 或更高的兼容版本。

```bash
cp .env.example .env
# 可将 API_KEY 替换为固定密钥；删除该变量则首次启动时生成并持久化
go run .
```

默认监听所有网卡（`HOST=0.0.0.0`），通常通过 `http://127.0.0.1:8080` 访问。`GET /health` 不需要鉴权；其他 API 默认需要密钥。

### OpenAI 示例

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Authorization: Bearer your-random-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-3.6-flash",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": false
  }'
```

Responses（包括 Codex 使用的无状态工具循环）也可直接调用：

```bash
curl http://127.0.0.1:8080/v1/responses \
  -H 'Authorization: Bearer your-random-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-3.6-flash",
    "input": "检查当前目录中的 Go 测试",
    "store": false,
    "stream": false
  }'
```

显式压缩一个无状态上下文窗口：

```bash
curl http://127.0.0.1:8080/v1/responses/compact \
  -H 'Authorization: Bearer your-random-secret' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-3.6-flash",
    "input": [{"type":"message","role":"user","content":"继续之前的编码任务"}]
  }'
```

把响应的 `output` 数组原样作为下一次 `/v1/responses` 的 `input` 前缀。服务端自动压缩则在普通 Responses 请求中添加 `"context_management":[{"type":"compaction","compact_threshold":200000}]`。

### Gemini 示例

```bash
curl 'http://127.0.0.1:8080/v1beta/models/gemini-3.6-flash:generateContent' \
  -H 'x-goog-api-key: your-random-secret' \
  -H 'Content-Type: application/json' \
  -d '{"contents":[{"role":"user","parts":[{"text":"你好"}]}]}'
```

### Anthropic 示例

```bash
curl http://127.0.0.1:8080/v1/messages \
  -H 'x-api-key: your-random-secret' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "gemini-3.6-flash",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "你好"}]
  }'
```

## 配置

配置可来自环境变量或工作目录中的 `.env`。仓库不会把 `.env` 打进二进制或容器镜像。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `API_KEY` | 未提供时自动生成并持久化 | API 密钥，多个值用逗号分隔；省略时首次生成 `sk-` 开头的随机密钥并保存到 `API_KEY_FILE` |
| `API_KEY_FILE` | `.vertex2api-api-key` | 自动生成的 API 密钥持久化文件路径；手动设置 `API_KEY` 时不使用 |
| `TZ` | `Asia/Shanghai` | 日志和自动拉取任务使用的时区；可填写其他 IANA 时区 |
| `ALLOW_UNAUTHENTICATED` | `false` | 显式允许无鉴权运行，仅建议本地开发使用 |
| `ALLOW_CUSTOM_MODEL_NAMES` | `false` | 是否允许调用不在当前模型目录中的模型名称；开启后仍拒绝路径分隔符和 `..` 序列 |
| `STATS_KEY` | 无 | `/v1/stats` 独立密钥；留空时该接口不可用 |
| `HOST` | `0.0.0.0` | 服务监听地址，例如 `0.0.0.0` 或 `127.0.0.1` |
| `PORT` | `8080` | 监听端口 |
| `VERTEX_BASE_URL` | `https://cloudconsole-pa.clients6.google.com` | 匿名 Vertex GraphQL 上游基址 |
| `GRAPHQL_API_KEY` | 无 | GraphQL 浏览器链路的公开 API 标识，必填 |
| `PREFIX_VERTEX_BASE_URL` | 无 | 可选前缀地址，多个值用逗号分隔 |
| `RECAPTCHA_BASE_URL` | `https://www.recaptcha.net` | reCAPTCHA 上游基址 |
| `RECAPTCHA_KEY` | 无 | reCAPTCHA 浏览器链路的公开站点密钥，必填 |
| `PREFIX_RECAPTCHA_BASE_URL` | 无 | 可选前缀地址，多个值用逗号分隔 |
| `PROXY` | 无 | `http`、`https` 或 `socks5` 代理 URL |
| `MAX_RETRY` | `3` | 每个 token 的最大上游重试次数 |
| `MAX_REFRESH` | `3` | reCAPTCHA token 最大刷新次数 |
| `RETRY_DELAY_MS` | `1000` | 重试间隔，毫秒 |
| `HTTP_TIMEOUT_SECONDS` | `120` | 上游 HTTP 超时 |
| `WRITE_TIMEOUT_SECONDS` | `600` | 下游响应写超时，需覆盖长流式请求 |
| `AUTO_FETCH_MODELS` | `true` | 启动并定时拉取上游模型目录 |
| `AUTO_FETCH_CRON` | `0 0,4 * * *` | 标准五段 Cron 表达式 |
| `REDACT_UPSTREAM_LOGS` | `false` | 是否将上游错误/响应详情替换为 `[REDACTED]` |
| `RANDOM_FINGERPRINT` | `false` | 是否附加浏览器请求层 Header；TLS ClientHello 由 `TLS_CLIENT_PROFILE` 控制 |
| `TLS_CLIENT_PROFILE` | `chrome_146` | `tls-client` 浏览器 TLS/HTTP2 profile |
| `CORS_ALLOW_ORIGIN` | 无 | 浏览器跨域 Origin；默认不授权跨域，谨慎使用 `*` |

上游 reCAPTCHA 和 Vertex 请求默认使用 `tls-client` 的 Chrome profile，模拟 TLS ClientHello、HTTP/2 设置和请求层浏览器特征；`RANDOM_FINGERPRINT=true` 时额外附加请求层 Header。建议保持同一个 `TLS_CLIENT_PROFILE` 稳定使用，不要在每次重试时切换 profile。

密钥可通过 `Authorization: Bearer`、`x-api-key`、`x-goog-api-key` 或 `?key=` 传递；手动设置的 `API_KEY` 和 `STATS_KEY` 至少需要 16 个字符。未设置 `API_KEY` 时，程序首次启动会生成密钥并写入 `API_KEY_FILE`，后续重启会复用该密钥；生产环境建议手动设置固定密钥，并通过反向代理启用 TLS、限流和访问日志脱敏。

## 模型目录

项目内置一份小型回退目录，因此发布包不需要提交动态生成的 `model.json`。启用 `AUTO_FETCH_MODELS=true` 后，程序会在启动时和 Cron 触发时从相同的 Vertex GraphQL 链路刷新内存目录；拉取失败时保留最后一份可用目录，不写入磁盘。

## Docker

镜像采用多阶段构建，运行时使用非 root 用户，也不会把本地 `.env` 复制进镜像。

### 本地构建镜像

如果不使用 Docker Hub，也可以直接从源码在本地构建镜像。以下命令适用于 WSL/Linux Bash，在项目根目录执行：

```bash
docker build --pull \
  --build-arg VERSION=local \
  --build-arg COMMIT="$(git rev-parse --short HEAD 2>/dev/null || printf 'local')" \
  --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  -t vertex2api:local \
  .
```

最后的 `.` 表示当前项目目录，不能省略。构建时如果出现 `SecretsUsedInArgOrEnv` 提示，这是 Docker 对镜像内置公开浏览器链路标识的警告，不会阻止构建。

本地构建完成后运行：

```bash
docker run -d \
  --name vertex2api \
  --restart unless-stopped \
  -v vertex2api-data:/data \
  -p 8080:8080 \
  -e TZ=Asia/Shanghai \
  vertex2api:local
```

### 使用 Docker Hub 镜像

```bash
docker pull sukafon6/vertex2api:latest
docker run -d \
  --name vertex2api \
  --restart unless-stopped \
  -v vertex2api-data:/data \
  -p 8080:8080 \
  -e TZ=Asia/Shanghai \
  sukafon6/vertex2api:latest
```

容器启动后，如果没有提供 `API_KEY`，程序会生成一个 `sk-` 开头的随机密钥并保存到 `/data/api-key`，同时打印到首次启动日志；命名卷 `vertex2api-data` 会让该密钥在容器重建后仍然保留。如需固定密钥，可额外添加 `-e API_KEY=YOUR_API_KEY_AT_LEAST_16_CHARS`。通过 `http://127.0.0.1:8080` 访问；健康检查地址为 `http://127.0.0.1:8080/health`。如需使用其他宿主机端口，只需修改 `-p` 左侧端口，例如 `-p 28888:8080`。如需让其他机器访问，应在宿主机防火墙和反向代理层配置 TLS、限流及访问控制。

常用管理命令：

```bash
docker logs -f --tail=1000 vertex2api
docker start vertex2api
docker restart vertex2api
docker stop vertex2api
docker rm -f vertex2api
docker rmi -f vertex2api
```

也可以把上述 `-e` 参数放入用户自己的 `.env` 文件，再使用 `--env-file .env`，避免把访问密钥直接留在 shell 历史中：

```bash
docker run -d \
  --name vertex2api \
  --restart unless-stopped \
  --env-file .env \
  -e TZ=Asia/Shanghai \
  -v vertex2api-data:/data \
  -p 8080:8080 \
  sukafon6/vertex2api:latest
```

## 开发与发布检查

```bash
go mod tidy
go test ./...
go vet ./...
go run honnef.co/go/tools/cmd/staticcheck@v0.7.0 ./...
go run golang.org/x/vuln/cmd/govulncheck@v1.6.0 ./...
CGO_ENABLED=0 go build -trimpath -o vertex2api .
```

CI 会执行同一组测试、静态检查、漏洞可达性扫描和无 CGO 构建。贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)，安全问题见 [SECURITY.md](SECURITY.md)。

## 许可证

项目基于 [GNU Affero General Public License v3.0](LICENSE)（`AGPL-3.0-only`）发布。
