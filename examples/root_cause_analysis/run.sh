#!/bin/bash

# 根因分析 Agent 示例运行脚本

# 设置默认值（如果环境变量未设置）
export KIMIK2_BASE_URL="${KIMIK2_BASE_URL:-https://api.moonshot.cn/v1}"
export KIMIK2_MODEL="${KIMIK2_MODEL:-moonshot-v1-8k}"
export DATA_DIR="${DATA_DIR:-./data}"

# 检查 API Key
if [ -z "$KIMIK2_API_KEY" ]; then
    echo "警告: 未设置 KIMIK2_API_KEY 环境变量"
    echo "请设置: export KIMIK2_API_KEY='your_api_key_here'"
    echo ""
    echo "继续运行（将使用占位符）..."
fi

# 确保数据目录存在
if [ ! -d "$DATA_DIR" ]; then
    echo "创建数据目录: $DATA_DIR"
    mkdir -p "$DATA_DIR"
fi

# 运行程序
echo "========== 启动根因分析 Agent =========="
echo "BaseURL: $KIMIK2_BASE_URL"
echo "Model: $KIMIK2_MODEL"
echo "DataDir: $DATA_DIR"
echo ""

cd "$(dirname "$0")"
go run main.go

