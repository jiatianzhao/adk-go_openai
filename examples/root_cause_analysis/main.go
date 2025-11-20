// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// 根因分析示例：演示如何使用 adk-go 的 openai 模型对接 KimiK2，
// 构建一个带工具的 agent 来分析 trace 数据并输出根因分析结论。
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"time"

	agentA "google.golang.org/adk/agent"
	"google.golang.org/adk/agent/llmagent"
	"google.golang.org/adk/artifact"
	"google.golang.org/adk/examples/root_cause_analysis/tools"
	"google.golang.org/adk/model/openai"
	runnerA "google.golang.org/adk/runner"
	sessionA "google.golang.org/adk/session"
	"google.golang.org/adk/tool"
	"google.golang.org/adk/tool/functiontool"
	"google.golang.org/genai"

	_ "github.com/go-sql-driver/mysql" // MySQL驱动
)

func main() {
	ctx := context.Background()

	// ========== 步骤 1: 配置 KimiK2 API ==========
	// 从环境变量读取配置，如果没有设置则使用默认值（测试阶段）
	baseURL := os.Getenv("KIMIK2_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.moonshot.cn/v1"
	}

	apiKey := os.Getenv("KIMIK2_API_KEY")
	if apiKey == "" {
		apiKey = "sk-4nGV86STuhZhzE55008lpNSwA4qx7JW1w0PsKSWjBhWOm7pN"
	}

	modelName := os.Getenv("KIMIK2_MODEL")
	if modelName == "" {
		modelName = "kimi-k2-0905-preview"
	}

	log.Printf("配置信息:")
	log.Printf("  BaseURL: %s", baseURL)
	log.Printf("  Model: %s", modelName)

	// ========== 步骤 2: 创建 OpenAI 兼容的模型适配器 ==========
	model, err := openai.NewModel(modelName, &openai.Config{
		BaseURL: baseURL,
		APIKey:  apiKey,

		MaxRetries:       1,                 // 最大重试次数
		MaxHistoryLength: 50,                // 最大历史消息数
		DebugLogging:     true,              // 是否开启调试日志
		Timeout:          120 * time.Second, // 请求超时时间
		SessionTTL:       1 * time.Hour,     // Session TTL: 1小时
	})
	if err != nil {
		log.Fatalf("创建模型失败: %v", err)
	}

	log.Println("✓ 模型创建成功")

	// ========== 步骤 3: 创建读取文件的工具 ==========
	readFileToolInstance, err := functiontool.New(
		functiontool.Config{
			Name:        "read_trace_file",
			Description: "读取存储在数据目录中的 trace 数据文件。用于获取 trace 的上下游链路信息和异常数据。文件路径应该是相对于数据目录的相对路径。",
		},
		tools.ReadFileTool,
	)
	if err != nil {
		log.Fatalf("创建文件读取工具失败: %v", err)
	}

	log.Println("✓ 文件读取工具创建成功")

	// ========== 步骤 3.1: 创建数据库查询工具 ==========
	queryDBToolInstance, err := functiontool.New(
		functiontool.Config{
			Name:        "query_trace_database",
			Description: "查询 trace 数据库获取服务调用记录、错误日志、性能指标等信息。支持 SQL SELECT 查询，可以查询 trace_records、error_logs、service_metrics 等表。只允许执行 SELECT 查询，不允许执行 INSERT、UPDATE、DELETE 等操作。",
		},
		tools.QueryDBTool,
	)
	if err != nil {
		log.Fatalf("创建数据库查询工具失败: %v", err)
	}
	log.Println("✓ 数据库查询工具创建成功")

	// ========== 步骤 3.2: 创建文件写入工具 ==========
	writeFileToolInstance, err := functiontool.New(
		functiontool.Config{
			Name:        "write_analysis_report",
			Description: "将分析结果、报告等写入文件。支持文本、JSON、CSV 等格式。文件会保存在数据目录中，文件路径应该是相对于数据目录的相对路径。",
		},
		tools.WriteFileTool,
	)
	if err != nil {
		log.Fatalf("创建文件写入工具失败: %v", err)
	}
	log.Println("✓ 文件写入工具创建成功")

	// ========== 步骤 4: 创建 LLM Agent ==========
	agent, err := llmagent.New(llmagent.Config{
		Name:        "root_cause_analyzer",
		Model:       model,
		Description: "一个专业的根因分析助手，能够分析 trace 数据，识别异常，并输出根因分析结论",
		Instruction: `你是一个专业的根因分析专家。你的任务是：

1. 分析用户提供的 trace 数据（包括上下游链路和异常信息）
2. 识别关键异常点和潜在问题
3. 使用工具获取更多信息：
   - 使用 query_trace_database 工具查询数据库获取 trace 记录、错误日志、服务指标等详细信息
   - 使用 read_trace_file 工具读取相关的 trace 数据文件（如果用户提供了文件路径）
4. 综合分析 trace 的调用链路、时间序列、错误信息、数据库记录等
5. 生成完整的根因分析报告，包括：
   - 问题定位：哪个服务/模块出现了问题
   - 根本原因：导致问题的根本原因
   - 影响范围：问题影响了哪些下游服务
   - 建议措施：如何修复和预防类似问题
6. 使用 write_analysis_report 工具将分析报告保存到文件中（建议保存为 JSON 格式）

工作流程建议：
- 先查询数据库获取相关的 trace 记录和错误日志
- 如果用户提供了文件路径，读取相关文件获取更多上下文
- 综合分析所有信息
- 生成详细的分析报告
- 将报告保存到文件（例如：reports/root_cause_analysis_<trace_id>.json）

请使用专业但易懂的语言，确保分析结论准确且有价值。`,
		Tools: []tool.Tool{
			readFileToolInstance,
			queryDBToolInstance,
			writeFileToolInstance,
		},
	})
	if err != nil {
		log.Fatalf("创建 Agent 失败: %v", err)
	}

	log.Println("✓ Agent 创建成功")

	// ========== 步骤 5: 创建 Session 和 Runner ==========
	userID := "root_cause_user"
	appName := "root_cause_analysis"

	sessionService := sessionA.InMemoryService()
	resp, err := sessionService.Create(ctx, &sessionA.CreateRequest{
		AppName: appName,
		UserID:  userID,
	})
	if err != nil {
		log.Fatalf("创建 Session 失败: %v", err)
	}

	session := resp.Session
	log.Printf("✓ Session 创建成功: %s", session.ID())

	artifactService := artifact.InMemoryService()

	runner, err := runnerA.New(runnerA.Config{
		AppName:         appName,
		Agent:           agent,
		SessionService:  sessionService,
		ArtifactService: artifactService,
	})
	if err != nil {
		log.Fatalf("创建 Runner 失败: %v", err)
	}

	log.Println("✓ Runner 创建成功")

	// ========== 步骤 6: 构建用户请求（模拟 trace 数据分析请求）==========
	// 这里模拟一个根因分析的场景：用户提供 trace 信息，要求分析问题
	userPrompt := `请帮我分析以下 trace 数据：

Trace ID: trace-12345
时间范围: 2024-01-15 10:00:00 - 10:05:00

调用链路：
1. 用户服务 (UserService) -> API网关 (APIGateway) -> 订单服务 (OrderService)
2. 订单服务调用支付服务 (PaymentService) 时出现超时
3. 支付服务调用银行接口 (BankAPI) 返回 500 错误

异常信息：
- PaymentService: 连接超时 (30秒)
- BankAPI: HTTP 500 Internal Server Error
- 错误消息: "Database connection pool exhausted"

请执行以下步骤：
1. 查询数据库获取 trace-12345 相关的所有记录和错误日志
2. 读取 data/payment_service_logs.txt 文件获取更多详细信息
3. 综合分析所有信息，找出根本原因
4. 生成完整的根因分析报告并保存到文件 reports/root_cause_analysis_trace_12345.json`

	log.Println("\n========== 开始执行 Agent ==========")
	log.Printf("用户输入: %s\n", userPrompt)

	// ========== 步骤 7: 调用 Agent 并处理响应 ==========
	userMsg := genai.NewContentFromText(userPrompt, genai.RoleUser)

	fmt.Println("\n========== Agent 响应 ==========")
	var finalResponse string
	eventCount := 0

	// 使用非流式模式执行（也可以使用 StreamingModeSSE 进行流式输出）
	for event, err := range runner.Run(ctx, userID, session.ID(), userMsg, agentA.RunConfig{
		StreamingMode: agentA.StreamingModeNone,
	}) {
		if err != nil {
			log.Printf("执行错误: %v", err)
			continue
		}

		eventCount++

		// 处理 LLM 响应事件
		if event.LLMResponse.Content != nil {
			for _, part := range event.LLMResponse.Content.Parts {
				if part.Text != "" {
					finalResponse += part.Text
					fmt.Print(part.Text)
				}
			}
		}

	}

	fmt.Println("\n\n========== 执行完成 ==========")
	log.Printf("处理了 %d 个事件", eventCount)
	log.Printf("最终响应长度: %d 字符", len(finalResponse))

}
