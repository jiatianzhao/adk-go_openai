# 快速开始指南

## 核心流程说明

这个示例展示了如何使用 adk-go 的 openai 模型对接 KimiK2，构建一个完整的 Agent 工作流。

### 完整链路

```
1. 配置模型 (openai.NewModel)
   ↓
2. 创建工具 (functiontool.New)
   ↓
3. 创建 Agent (llmagent.New)
   ↓
4. 创建 Session 和 Runner
   ↓
5. 调用 Agent (runner.Run)
   ↓
6. Agent 自动调用工具
   ↓
7. 输出最终结论
```

## 关键代码片段

### 1. 创建模型

```go
model, err := openai.NewModel(modelName, &openai.Config{
    BaseURL: baseURL,  // KimiK2 API 地址
    APIKey:  apiKey,   // API Key
})
```

### 2. 创建工具

```go
readFileTool, err := functiontool.New(
    functiontool.Config{
        Name:        "read_trace_file",
        Description: "读取 trace 数据文件",
    },
    readFileTool, // 工具函数
)
```

### 3. 创建 Agent

```go
agent, err := llmagent.New(llmagent.Config{
    Name:        "root_cause_analyzer",
    Model:       model,
    Description: "根因分析助手",
    Instruction: "你是专业的根因分析专家...",
    Tools: []tool.Tool{
        readFileTool,
    },
})
```

### 4. 执行 Agent

```go
runner, err := runner.New(runner.Config{
    AppName:         appName,
    Agent:           agent,
    SessionService:  sessionService,
    ArtifactService: artifactService,
})

// 调用 Agent
for event, err := range runner.Run(ctx, userID, sessionID, userMsg, agent.RunConfig{
    StreamingMode: agent.StreamingModeNone,
}) {
    // 处理响应事件
}
```

## 工具函数签名

工具函数必须遵循以下签名：

```go
func toolFunction(ctx tool.Context, input InputType) (OutputType, error)
```

其中：
- `InputType` 和 `OutputType` 必须是结构体类型
- 字段使用 JSON tag 定义
- 函数会被自动转换为工具供 Agent 调用

## 环境变量配置

```bash
# 必需
export KIMIK2_API_KEY="your_api_key"

# 可选（有默认值）
export KIMIK2_BASE_URL="https://api.moonshot.cn/v1"
export KIMIK2_MODEL="moonshot-v1-8k"
export DATA_DIR="./data"
```

## 运行示例

```bash
# 方式1: 使用脚本
./run.sh

# 方式2: 直接运行
export KIMIK2_API_KEY="your_key"
go run main.go
```

## 常见问题

### Q: Agent 如何知道何时调用工具？

A: Agent 会根据用户输入和工具描述自动决定是否需要调用工具。你只需要在 `Instruction` 中说明何时使用工具即可。

### Q: 如何添加更多工具？

A: 创建更多工具函数，然后在 `llmagent.Config.Tools` 中注册即可。

### Q: 如何自定义 Agent 行为？

A: 修改 `llmagent.Config.Instruction` 字段，详细描述 Agent 的角色和行为。

### Q: 支持流式输出吗？

A: 支持。将 `StreamingMode` 设置为 `agent.StreamingModeSSE` 即可。

## 下一步

- 查看 `main.go` 了解完整实现
- 阅读 `README.md` 获取详细文档
- 根据你的需求修改工具和 Agent 配置

