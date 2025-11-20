package tools

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"google.golang.org/adk/tool"
)

// ReadFileInput 定义读取文件工具的输入参数
type ReadFileInput struct {
	FilePath string `json:"file_path"` // 要读取的文件路径（相对于数据目录）
}

// ReadFileOutput 定义读取文件工具的输出结果
type ReadFileOutput struct {
	Content string `json:"content"`         // 文件内容
	Success bool   `json:"success"`         // 是否成功读取
	Error   string `json:"error,omitempty"` // 错误信息（如果有）
}

// ReadFileTool 是一个读取本地文件的工具函数
// 用于读取存储在数据目录中的 trace 数据文件
func ReadFileTool(ctx tool.Context, input ReadFileInput) (ReadFileOutput, error) {
	// 获取数据目录路径（可以通过环境变量配置，默认使用当前目录下的 data 目录）
	dataDir := os.Getenv("DATA_DIR")
	if dataDir == "" {
		dataDir = "data"
	}

	// 构建完整文件路径
	fullPath := filepath.Join(dataDir, input.FilePath)

	// 安全检查：确保文件在数据目录内，防止路径遍历攻击
	absDataDir, err := filepath.Abs(dataDir)
	if err != nil {
		return ReadFileOutput{
			Success: false,
			Error:   fmt.Sprintf("无法解析数据目录路径: %v", err),
		}, nil
	}

	absFilePath, err := filepath.Abs(fullPath)
	if err != nil {
		return ReadFileOutput{
			Success: false,
			Error:   fmt.Sprintf("无法解析文件路径: %v", err),
		}, nil
	}

	// 检查文件是否在数据目录内
	relPath, err := filepath.Rel(absDataDir, absFilePath)
	if err != nil || relPath == ".." || len(relPath) >= 2 && relPath[:2] == ".." {
		return ReadFileOutput{
			Success: false,
			Error:   "不允许访问数据目录之外的文件",
		}, nil
	}

	// 读取文件内容
	content, err := os.ReadFile(absFilePath)
	if err != nil {
		return ReadFileOutput{
			Success: false,
			Error:   fmt.Sprintf("读取文件失败: %v", err),
		}, nil
	}

	fmt.Println("--> 模型调用了read工具" + time.Now().Format("2006-01-02 15:04:05"))
	return ReadFileOutput{
		Content: string(content),
		Success: true,
	}, nil
}
