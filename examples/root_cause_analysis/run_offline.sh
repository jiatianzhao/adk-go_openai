#!/bin/bash
# 离线运行脚本 - 使用 vendor 模式

export KIMIK2_API_KEY="${KIMIK2_API_KEY:-}"
if [ -z "$KIMIK2_API_KEY" ]; then
    echo "请设置 KIMIK2_API_KEY 环境变量"
    exit 1
fi

cd "$(dirname "$0")"
go run -mod=vendor test_read.go

