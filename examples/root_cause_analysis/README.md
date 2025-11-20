# 根因分析 Agent 示例

这个示例演示了如何使用 adk-go 的 openai 模型对接 KimiK2，构建一个带工具的 agent 来进行根因分析。

## 功能说明

1. **对接 KimiK2**: 使用 OpenAI 兼容的 API 接口对接 KimiK2 模型
2. **文件读取工具**: 封装了一个读取本地文件的工具，用于读取 trace 数据
3. **数据库查询工具**: 封装了一个查询数据库的工具，用于查询 trace 记录、错误日志、服务指标等
4. **文件写入工具**: 封装了一个写文件的工具，用于保存分析报告
5. **Agent 构建**: 创建一个专业的根因分析 agent
6. **工具调用**: Agent 可以自动调用工具查询数据库、读取文件、写入报告
7. **根因分析**: 基于 trace 数据、数据库记录和异常信息输出根因分析结论

## 使用步骤

### 1. 配置环境变量

```bash
# KimiK2 API 配置（测试阶段使用官方 API）
export KIMIK2_BASE_URL="https://api.moonshot.cn/v1"
export KIMIK2_API_KEY="your_api_key_here"
export KIMIK2_MODEL="moonshot-v1-8k"

# 数据目录配置（可选，默认为 ./data）
export DATA_DIR="./data"

# 数据库配置（可选，默认连接本地2881端口的MySQL）
export DB_HOST="localhost"
export DB_PORT="2881"
export DB_USER="root"
export DB_PASSWORD=""
export DB_NAME="trace_db"
```

### 2. 初始化数据库

数据库会在程序运行时自动初始化，你也可以手动初始化：

```bash
# 连接MySQL数据库（假设使用MySQL客户端）
mysql -h localhost -P 2881 -u root < init_trace_db.sql
```

数据库包含以下表：
- `trace_records`: trace 调用记录
- `error_logs`: 错误日志
- `service_metrics`: 服务指标

### 3. 准备数据文件

将你的 trace 数据文件放在 `data/` 目录下。示例中已经包含了一个示例文件：
- `data/payment_service_logs.txt` - 支付服务日志示例

### 4. 运行示例

```bash
cd examples/root_cause_analysis
go run main.go
```

## 代码结构说明

### main.go

主要包含以下部分：

1. **readFileTool**: 文件读取工具函数
   - 输入：文件路径（相对于数据目录）
   - 输出：文件内容
   - 包含安全检查，防止路径遍历攻击

2. **queryDBTool**: 数据库查询工具函数
   - 输入：SQL SELECT 查询语句
   - 输出：查询结果（JSON格式）
   - 只允许执行 SELECT 查询，确保数据安全
   - 支持查询 trace_records、error_logs、service_metrics 等表

3. **writeFileTool**: 文件写入工具函数
   - 输入：文件路径、内容、格式（text/json/csv）
   - 输出：写入结果和文件大小
   - 包含安全检查，只允许写入数据目录内
   - 支持文本、JSON、CSV 等格式

4. **模型配置**: 
   - 使用 `openai.NewModel` 创建 OpenAI 兼容的模型适配器
   - 支持自定义 BaseURL 和 APIKey
   - 可配置超时、重试等参数

3. **Agent 创建**:
   - 使用 `llmagent.New` 创建 LLM Agent
   - 配置专业的根因分析指令
   - 注册文件读取工具

4. **Runner 执行**:
   - 创建 Session 和 Runner
   - 调用 `runner.Run` 执行 Agent
   - 处理响应事件并输出结果

## 完整链路流程

```
用户输入 (trace 数据描述)
    ↓
Agent 接收请求
    ↓
Agent 决定调用工具
    ├─→ query_trace_database (查询数据库获取 trace 记录和错误日志)
    ├─→ read_trace_file (读取相关文件获取更多上下文)
    └─→ write_analysis_report (保存分析报告到文件)
    ↓
工具执行并返回结果
    ↓
Agent 综合分析所有信息
    ↓
Agent 生成根因分析报告
    ↓
Agent 将报告保存到文件
    ↓
输出最终结论
```

## 工具使用示例

### 数据库查询示例

Agent 可以执行以下类型的 SQL 查询：

```sql
-- 查询特定 trace 的所有记录
SELECT * FROM trace_records WHERE trace_id = 'trace-12345';

-- 查询错误日志
SELECT * FROM error_logs WHERE trace_id = 'trace-12345';

-- 查询服务指标
SELECT * FROM service_metrics WHERE service_name = 'PaymentService';

-- 关联查询
SELECT tr.*, el.error_message 
FROM trace_records tr 
LEFT JOIN error_logs el ON tr.span_id = el.span_id 
WHERE tr.trace_id = 'trace-12345';
```

### 文件写入示例

Agent 可以将分析结果保存为不同格式：

- **JSON 格式**: `write_analysis_report` 工具会自动将结构化数据转换为 JSON
- **文本格式**: 直接写入文本内容
- **CSV 格式**: 保存表格数据

## 自定义扩展

### 添加更多工具

示例中已经包含了三个工具：
- `read_trace_file`: 读取文件
- `query_trace_database`: 查询数据库
- `write_analysis_report`: 写入文件

你可以添加更多工具来增强 Agent 的能力，例如：

- 日志搜索工具
- 指标查询工具
- 依赖关系分析工具
- 图表生成工具

示例：

```go
// 添加新的工具
newTool, err := functiontool.New(
    functiontool.Config{
        Name:        "your_tool_name",
        Description: "工具描述",
    },
    yourToolFunction,
)

// 在 Agent 中注册
Tools: []tool.Tool{
    readFileToolInstance,
    queryDBToolInstance,
    writeFileToolInstance,
    newTool,
}
```

### 修改 Agent 指令

你可以根据实际需求修改 Agent 的 `Instruction`，使其更适合你的根因分析场景。

## 注意事项

1. **API Key**: 请确保设置正确的 KIMIK2_API_KEY，否则请求会失败
2. **数据库连接**: 确保数据库服务运行在 2881 端口，并且已经初始化了表结构
3. **数据目录**: 确保数据目录存在且包含必要的文件
4. **文件路径**: 工具只允许访问数据目录内的文件，确保安全
5. **SQL 安全**: 数据库查询工具只允许执行 SELECT 查询，不允许执行 INSERT、UPDATE、DELETE 等操作
6. **模型选择**: 根据你的 KimiK2 部署选择合适的模型名称

## 生产环境部署

在生产环境中：

1. 将 `KIMIK2_BASE_URL` 设置为你自己部署的 KimiK2 服务地址
2. 使用环境变量或密钥管理服务来管理 API Key
3. 根据需要调整超时时间和重试策略
4. 添加日志和监控
5. 考虑添加错误处理和重试机制

