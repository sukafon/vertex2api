# vertex2api

`vertex2api` 将 Google Cloud Console 使用的匿名 Vertex GraphQL 链路转换为 OpenAI、Gemini 和 Anthropic 风格的 HTTP API。项目重点是协议字段、流式事件、错误信封、工具 Schema 与 usage 的协议转换。

## 已实现接口

| 接口 | 路径 |
| --- | --- |
| OpenAI | `POST /v1/chat/completions`、`POST /v1/responses`、`POST /v1/responses/compact`、`GET /v1/models`、`GET /v1/models/{model}`、`POST /v1/images/generations`、`POST /v1/images/edits` |
| Gemini | `POST /v1beta/models/{model}:generateContent`、`:streamGenerateContent`、`:countTokens`；同时接受 `v1beta1` 和 `v1` 生成路径；`GET /v1beta/models` |
| Anthropic | `POST /v1/messages`、`POST /v1/messages/count_tokens` |

兼容性须知：

- 内部使用有序 Vertex Part 表示响应，不先压平成文本；原生 Gemini 会保留候选边界、Part 顺序、角色、图片/文件、函数调用与结果、代码执行、思考签名、结束原因、grounding、安全评级和候选元数据。
- 将 OpenAI/Anthropic 工具 Schema 归一化为 Vertex 可接受的 JSON Schema，不修改调用者的原始对象；协议没有等价字段时采用下表中的显式降级，不伪造另一个供应商的 opaque 数据。
- Chat Completions 和 Responses 的显式 `reasoning_effort` 会映射到 Gemini 3 `thinkingLevel` 或 Gemini 2.5 `thinkingBudget`；`xhigh`/`max` 会降级为 Gemini 可表达的 `high`，其他 Gemini 3 等级会根据上游模型目录发布的 `thinking_level` 枚举约束到最近可用值。省略该字段时保留模型默认思考配置，不依据 token 上限自动降级；Anthropic Messages 省略 thinking/effort 时则使用目录中该模型支持的最低思考等级，目录未发布能力时不猜测。Responses 的 `namespace` 工具会展开为 `namespace__tool` 形式的 Vertex 函数声明；响应时恢复为独立的 `namespace` 与子工具 `name` 供 Codex 分派，下一轮输入再无损还原为 Vertex 扁平名。
- 对 `gemini-3.5-flash-lite*` 和 `gemini-3.6+` 移除末尾 model 预填充，适配这些模型不接受 model turn 结尾的请求约束；这不是把内容改写成 `systemInstruction`，其他较早模型保留原行为。
- Responses 支持无状态字符串/输入项数组、文本、data URL/远程图片、内联/远程文件、音频、函数工具、custom、local_shell、shell、apply_patch、Vertex 等价的 `web_search`、`code_interpreter`、`url_context`、结构化输出、refusal、URL citation、web-search call 和完整 SSE 生命周期；不实现 `previous_response_id`、OpenAI Files、后台任务或 `file_search`。
- Anthropic Messages 支持思考及签名流、工具流、图片、PDF/文本/URL document、`output_config.format` 结构化输出和 `output_config.effort`。Vertex grounding 无法生成 Anthropic web-search citation 所要求的 `encrypted_index`/`encrypted_content`，因此不会伪造 Anthropic citation block。
- 输入历史包含不完整的并行工具记录时，会按 tool call ID、名称依次配对，只保留真实存在的 function call/response 对并按调用顺序发送给 Vertex；不会为未执行的工具伪造结果。该兼容处理不对不完整历史的来源作归因。
- 图片接口把 OpenAI `size` 中可精确表达的宽高比映射为 Vertex `imageConfig.aspectRatio`；`quality` 或不可表达的尺寸使用模型默认值，并通过 `X-Vertex2API-Warning` 明示降级。省略模型时使用 `gemini-3-pro-image`。
- `/v1/responses/compact` 与 `context_management.compact_threshold` 会调用当前 Vertex 模型生成状态检查点，再输出 AES-GCM 认证加密的 opaque compaction item。该 item 只保证由相同 API_KEY 列表配置的 vertex2api 实例解码，不能与 OpenAI 官方 compaction item 互换；修改密钥内容或顺序会使旧 item 失效。无密钥运行时使用进程内随机密钥，重启后旧 item 失效。

### 转换保真度

| 能力 | Gemini 原生 | OpenAI Chat | OpenAI Responses | Anthropic Messages |
| --- | --- | --- | --- | --- |
| 文本、思考、工具调用顺序 | 原样 | 文本/思考分字段，工具调用使用标准字段 | Vertex thought 映射为独立、可流式展示的 Responses `assistant` commentary message；最终文本与工具调用使用独立 item | 使用 thinking/text/tool_use block 与实时 delta |
| 思考签名 | 原样 | 工具调用 ID 携带 opaque 签名以便回传 | 工具调用 ID 携带 opaque 签名以便回传 | thinking signature 原样；工具调用 ID 携带 opaque 签名 |
| 图片、文件、音频输入 | 原样 | 映射为 `inlineData`/`fileData` | 映射为 `inlineData`/`fileData` | image/document 映射为 `inlineData`/`fileData` |
| 图片、文件、代码执行输出 | 原样 | 无标准等价项时转为可见 Markdown/data URL | 无标准等价项时转为可见 `output_text` | 转为协议合法的可见 text block/Markdown data URL |
| Google Search grounding | 原样 | URL citation annotations | `web_search_call` 与 URL citation annotations | 不伪造缺少 Anthropic opaque 索引的 citation |
| safety/prompt metadata | 按 Gemini Developer API 规范化 `promptFeedback`/finish reason | `content_filter` | refusal item | `end_turn`（不伪装成 Anthropic 模型的 `refusal`） |
| usage/cache/thinking token | 原样 metadata | 映射 usage details；缺失时标记估算 | 映射 input/output details；缺失时标记估算 | 映射 cache-read/thinking usage；缺失时标记估算 |

“原样”指该匿名 Vertex GraphQL 上游实际返回或接受的字段；并不代表另一个协议中不存在的服务端资源（例如 OpenAI Files、container ID、持久化 response 或 Anthropic 加密搜索索引）可以被本地伪造。无法等价表达但仍可展示的输出会显式转为可见内容；会改变调用语义的请求字段则返回参数错误或在响应头中给出降级警告。

## 快速开始

要求 Go `1.26.6` 或更高的兼容版本。

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

Chat Completions 流式响应保持同一 `id` 和 `created`，每个 `delta` 都显式返回 `role: "assistant"`、`content`、`reasoning_content` 和 `tool_calls`；后两项无内容时为 `null`，有内容时返回实际思考文本或工具调用数组，并以 `data: [DONE]` 结束。中间 chunk 始终省略 `usage`；仅当请求设置 `stream_options.include_usage=true` 时，`[DONE]` 前包含结束原因的最后一个消息 chunk 才返回完整 usage。`include_obfuscation` 默认开启，可显式设为 `false`。上游提供完整且自洽的 Vertex usage 时直接映射；缺失、过期或自相矛盾时会补全并通过 `X-Usage-Estimated: true` 标记。

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

Gemini SSE 流式请求应显式携带 `alt=sse`：

```bash
curl --no-buffer 'http://127.0.0.1:8080/v1beta/models/gemini-3.6-flash:streamGenerateContent?alt=sse' \
  -H 'x-goog-api-key: your-random-secret' \
  -H 'Accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{"contents":[{"role":"user","parts":[{"text":"你好"}]}]}'
```

启用 `GEMINI_STRICT_ALT_SSE=true` 后，缺少 `alt=sse` 的 `streamGenerateContent` 会按非流模式等待完成，再一次性返回单个 `application/json` 响应对象；关闭时继续兼容不带该参数的旧 SSE 客户端。规范的非流调用仍应使用 `generateContent`。

Gemini SSE 的每个普通消息块都会携带当前累计的 `usageMetadata`、`modelVersion` 和 `responseId`。每个块优先采用与当前累计输出时点匹配的最近上游 usage；上游仅提供部分字段时由项目内置算法补齐，上游未提供或其后又产生新输出时则按该块对应的请求和累计输出重新估算，后续总量不会倒灌到先前块。末尾仅含 `thoughtSignature` 和空文本的传输片段会并入最后一个实际消息；如果整条流只有签名，则签名会附着到结束 candidate，避免产生空普通块。结束 candidate 携带候选索引和 `finishReason`，并继续携带相同响应级元数据。同一流在首个下游块实际发出时锁定此前观察到的首个上游真实 `modelVersion` 和 `responseId`；匿名上游未提供时，分别回退为请求模型名和一次性生成的会话响应 ID，整个流保持不变。

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
| `GEMINI_STRICT_ALT_SSE` | `false` | 是否严格要求 Gemini `streamGenerateContent` 使用 `alt=sse` 才返回 SSE；开启后缺少或使用其他 `alt` 值会等待完成并返回单个完整 JSON 响应对象 |
| `REJECT_CHAT_LIVENESS_PROBES` | `false` | 是否拒绝仅包含单条 `"hi"` 用户消息的 OpenAI Chat 请求；开启后应使用 `GET /health` 验活 |
| `RESPOND_CHAT_LIVENESS_PROBES` | `false` | 是否对上述验活请求在本地构造协议合法的正常响应并跳过上游；与拒绝开关同时开启时本项优先 |
| `STATS_KEY` | 无 | `/v1/stats` 和 `POST /v1/models/refresh` 的独立密钥；留空时这两个接口不可用 |
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
| `REDACT_UPSTREAM_RESPONSES` | `false` | 是否隐藏返回给下游调用方的上游错误详情；服务端日志始终保留真实错误 |
| `LOG_CODE3_REQUEST_BODIES` | `false` | 是否把首次出现的每种 Vertex Code 3 错误及完整上游请求体写入 `API_KEY_FILE` 所在目录的 `code3_request_bodies.log`；相同错误即使请求体不同也跳过，删除日志后才重新记录 |
| `RANDOM_FINGERPRINT` | `false` | 是否附加浏览器请求层 Header；TLS ClientHello 由 `TLS_CLIENT_PROFILE` 控制 |
| `TLS_CLIENT_PROFILE` | `chrome_146` | `tls-client` 浏览器 TLS/HTTP2 profile |
| `CORS_ALLOW_ORIGIN` | 无 | 浏览器跨域 Origin；默认不授权跨域，谨慎使用 `*` |

上游 reCAPTCHA 和 Vertex 请求默认使用 `tls-client` 的 Chrome profile，模拟 TLS ClientHello、HTTP/2 设置和请求层浏览器特征；`RANDOM_FINGERPRINT=true` 时额外附加请求层 Header。建议保持同一个 `TLS_CLIENT_PROFILE` 稳定使用，不要在每次重试时切换 profile。

`LOG_CODE3_REQUEST_BODIES=true` 仅用于逐项诊断需要人工处理的 Vertex Code 3 错误，不会跳过请求、重试或错误返回；可自动重试的 `Failed to verify action` 不写入该文件。日志采用 JSON Lines 格式，错误消息是去重键：一种错误只保存第一次出现时的请求体，之后即使请求体不同也不再写入；运行中删除 `code3_request_bodies.log` 即可清空去重状态并允许重新捕获。本地默认路径是 `./code3_request_bodies.log`，官方镜像中随默认 `API_KEY_FILE=/data/api-key` 写入持久化卷 `/data/code3_request_bodies.log`。文件包含完整上游请求体，可能含对话内容和临时 reCAPTCHA token，请限制文件访问并在排查后删除。

如果容器名为 `vertex2api`，且启动日志显示 Code 3 文件写在容器工作目录 `/app`，可用以下命令将其导出到宿主机：

```bash
docker cp vertex2api:/app/code3_request_bodies.log ~/vertex2api/code3_request_bodies.log
```

`docker cp` 会复制文件，但不会可靠地代替你准备目标父目录；请先确保宿主机的 `~/vertex2api` 已存在。使用官方镜像默认 `API_KEY_FILE=/data/api-key` 时，容器内源路径应改为 `/data/code3_request_bodies.log`。

密钥可通过 `Authorization: Bearer`、`x-api-key`、`x-goog-api-key` 或 `?key=` 传递；手动设置的 `API_KEY` 和 `STATS_KEY` 至少需要 16 个字符。未设置 `API_KEY` 时，程序首次启动会生成密钥并写入 `API_KEY_FILE`，后续重启会复用该密钥；生产环境建议手动设置固定密钥，并通过反向代理启用 TLS、限流和访问日志脱敏。

## 模型目录

项目内置一份小型回退目录，因此发布包不需要提交动态生成的 `model.json`。启用 `AUTO_FETCH_MODELS=true` 后，程序会在启动时和 Cron 触发时从相同的 Vertex GraphQL 链路刷新内存目录；拉取失败时保留最后一份可用目录，不写入磁盘。

也可以随时使用 `STATS_KEY` 手动刷新；该操作不受 `AUTO_FETCH_MODELS` 开关影响：

```bash
curl -X POST 'http://127.0.0.1:8080/v1/models/refresh' \
  -H 'Authorization: Bearer your-stats-key'
```

刷新成功返回更新后的 `model_count`；上游失败或返回空目录时保留当前内存目录。

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
