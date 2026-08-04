# 贡献指南

感谢参与 `vertex2api`。改动应优先保持三套公开协议的可观察行为规范，并将 Vertex GraphQL 的特殊处理限制在代理层。

## 开发流程

1. 使用 `go.mod` 指定的 Go 版本。
2. 从 `.env.example` 创建本地 `.env`，不要提交密钥或真实请求数据。
3. 为协议字段、GraphQL 解包、首包错误、usage 或 Schema 行为补充最小测试。
4. 提交前运行：

```bash
go mod tidy
go fmt ./...
go test ./...
go vet ./...
go run honnef.co/go/tools/cmd/staticcheck@v0.7.0 ./...
go run golang.org/x/vuln/cmd/govulncheck@v1.6.0 ./...
```

## 设计约束

- 不加入测试模型、公开调试入口、请求体打印或内置凭据。
- 不注入隐藏提示词，不通过删改用户内容实现“抗截断”。
- 上游缺少 usage 时可以估算，但必须通过 `X-Usage-Estimated: true` 明确标识。
- 流式首包前的错误应返回协议对应的正常 HTTP 错误；响应已提交后不得伪造新的 HTTP 状态。
- 工具 Schema 归一化必须操作副本，不能修改调用者请求对象。
- 新的配置默认值应安全；绕过鉴权、开放 CORS 等行为必须显式启用。

Pull Request 请说明外部可观察变化、测试覆盖和已知限制。不要附带与改动无关的机械重排。
