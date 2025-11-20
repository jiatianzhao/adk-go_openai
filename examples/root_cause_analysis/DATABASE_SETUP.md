# 数据库设置说明

## 快速开始

### 1. 初始化数据库

使用提供的 SQL 文件初始化数据库：

```bash
# 方式1: 使用 MySQL 客户端
mysql -h localhost -P 2881 -u root < init_trace_db.sql

# 方式2: 如果设置了密码
mysql -h localhost -P 2881 -u root -p < init_trace_db.sql

# 方式3: 使用 MySQL 命令行工具
mysql -h localhost -P 2881 -u root
mysql> source init_trace_db.sql
mysql> exit
```

### 2. 验证数据库

连接数据库并检查表是否创建成功：

```bash
mysql -h localhost -P 2881 -u root

mysql> USE trace_db;
mysql> SHOW TABLES;
# 应该看到：
# - trace_records
# - error_logs
# - service_metrics

mysql> SELECT COUNT(*) FROM trace_records;
# 应该返回 5（示例数据）

mysql> SELECT COUNT(*) FROM error_logs;
# 应该返回 3（示例数据）

mysql> SELECT COUNT(*) FROM service_metrics;
# 应该返回 11（示例数据）
```

## 数据库表结构

### trace_records 表

存储 trace 调用记录：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键 |
| trace_id | VARCHAR(64) | Trace ID |
| span_id | VARCHAR(64) | Span ID |
| parent_span_id | VARCHAR(64) | 父 Span ID |
| service_name | VARCHAR(128) | 服务名称 |
| operation_name | VARCHAR(256) | 操作名称 |
| start_time | TIMESTAMP | 开始时间 |
| duration_ms | INT | 持续时间（毫秒） |
| status_code | VARCHAR(16) | 状态码：OK, ERROR, TIMEOUT |
| tags | JSON | 标签信息 |

### error_logs 表

存储错误日志：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键 |
| trace_id | VARCHAR(64) | Trace ID |
| span_id | VARCHAR(64) | Span ID |
| service_name | VARCHAR(128) | 服务名称 |
| error_type | VARCHAR(64) | 错误类型 |
| error_message | TEXT | 错误消息 |
| error_stack | TEXT | 错误堆栈 |
| occurred_at | TIMESTAMP | 发生时间 |

### service_metrics 表

存储服务指标：

| 字段 | 类型 | 说明 |
|------|------|------|
| id | BIGINT | 主键 |
| service_name | VARCHAR(128) | 服务名称 |
| metric_name | VARCHAR(128) | 指标名称 |
| metric_value | DECIMAL(20,4) | 指标值 |
| metric_unit | VARCHAR(32) | 单位 |
| recorded_at | TIMESTAMP | 记录时间 |

## 环境变量配置

数据库连接通过环境变量配置：

```bash
export DB_HOST="localhost"      # 数据库主机（默认：localhost）
export DB_PORT="2881"           # 数据库端口（默认：2881）
export DB_USER="root"            # 数据库用户（默认：root）
export DB_PASSWORD=""            # 数据库密码（默认：空）
export DB_NAME="trace_db"        # 数据库名称（默认：trace_db）
```

## 示例查询

### 查询特定 trace 的所有记录

```sql
SELECT * FROM trace_records WHERE trace_id = 'trace-12345';
```

### 查询错误日志

```sql
SELECT * FROM error_logs WHERE trace_id = 'trace-12345';
```

### 查询服务指标

```sql
SELECT * FROM service_metrics WHERE service_name = 'PaymentService';
```

### 关联查询 trace 记录和错误日志

```sql
SELECT 
    tr.trace_id,
    tr.span_id,
    tr.service_name,
    tr.operation_name,
    tr.status_code,
    tr.duration_ms,
    el.error_type,
    el.error_message
FROM trace_records tr
LEFT JOIN error_logs el ON tr.span_id = el.span_id
WHERE tr.trace_id = 'trace-12345'
ORDER BY tr.start_time;
```

### 查询连接池使用情况

```sql
SELECT 
    service_name,
    metric_name,
    metric_value,
    recorded_at
FROM service_metrics
WHERE metric_name LIKE '%connection%'
ORDER BY recorded_at;
```

## 故障排查

### 连接失败

1. 检查数据库服务是否运行：
   ```bash
   # 检查端口是否监听
   netstat -an | grep 2881
   # 或
   lsof -i :2881
   ```

2. 检查数据库用户权限：
   ```sql
   SHOW GRANTS FOR 'root'@'localhost';
   ```

3. 检查防火墙设置

### 表不存在

确保已经执行了 `init_trace_db.sql` 文件。

### 权限错误

确保数据库用户有 SELECT 权限（查询工具只需要 SELECT 权限）。

## 注意事项

1. **只读访问**: 数据库查询工具只允许执行 SELECT 查询，确保数据安全
2. **连接超时**: 查询超时时间设置为 10 秒
3. **JSON 字段**: tags 字段使用 JSON 类型，查询时会自动解析
4. **时区**: 时间字段使用服务器的本地时区

