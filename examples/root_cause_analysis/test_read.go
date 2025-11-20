package main

import (
	"context"
	"fmt"
	"log"
	"os"

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
)

func main() {
	ctx := context.Background()

	baseURL := os.Getenv("KIMIK2_BASE_URL")
	if baseURL == "" {
		baseURL = "https://api.moonshot.cn/v1"
	}

	apiKey := os.Getenv("KIMIK2_API_KEY")
	if apiKey == "" {
		log.Fatal("请设置 KIMIK2_API_KEY 环境变量")
	}

	modelName := os.Getenv("KIMIK2_MODEL")
	if modelName == "" {
		modelName = "kimi-k2-0905-preview"
	}

	model, err := openai.NewModel(modelName, &openai.Config{
		BaseURL: baseURL,
		APIKey:  apiKey,
	})
	if err != nil {
		log.Fatalf("创建模型失败: %v", err)
	}

	readFileTool, err := functiontool.New(
		functiontool.Config{
			Name:        "read_trace_file",
			Description: "读取存储在数据目录中的 trace 数据文件。文件路径应该是相对于数据目录的相对路径。",
		},
		tools.ReadFileTool,
	)
	if err != nil {
		log.Fatalf("创建工具失败: %v", err)
	}

	agent, err := llmagent.New(llmagent.Config{
		Name:        "read_file_test",
		Model:       model,
		Description: "测试文件读取工具的 Agent",
		Instruction: "你是一个文件读取助手。当用户要求读取文件时，使用 read_trace_file 工具读取文件并返回文件内容。",
		Tools: []tool.Tool{
			readFileTool,
		},
	})
	if err != nil {
		log.Fatalf("创建 Agent 失败: %v", err)
	}

	sessionService := sessionA.InMemoryService()
	resp, err := sessionService.Create(ctx, &sessionA.CreateRequest{
		AppName: "read_test",
		UserID:  "test_user",
	})
	if err != nil {
		log.Fatalf("创建 Session 失败: %v", err)
	}

	runner, err := runnerA.New(runnerA.Config{
		AppName:         "read_test",
		Agent:           agent,
		SessionService:  sessionService,
		ArtifactService: artifact.InMemoryService(),
	})
	if err != nil {
		log.Fatalf("创建 Runner 失败: %v", err)
	}

	userMsg := genai.NewContentFromText("请读取 data/payment_service_logs.txt 文件的内容", genai.RoleUser)

	fmt.Println("========== Agent 响应 ==========")
	for event, err := range runner.Run(ctx, "test_user", resp.Session.ID(), userMsg, agentA.RunConfig{
		StreamingMode: agentA.StreamingModeNone,
	}) {
		if err != nil {
			log.Printf("执行错误: %v", err)
			continue
		}

		if event.LLMResponse.Content != nil {
			for _, part := range event.LLMResponse.Content.Parts {
				if part.Text != "" {
					fmt.Print(part.Text)
				}
			}
		}

	}

	fmt.Println("\n========== 完成 ==========")
}
