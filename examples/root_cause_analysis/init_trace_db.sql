-- Trace 数据库初始化脚本
-- 用于根因分析示例的数据库表结构和示例数据

-- 创建 trace_records 表：存储 trace 调用记录
CREATE TABLE IF NOT EXISTS trace_records (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trace_id VARCHAR(64) NOT NULL COMMENT 'Trace ID',
    span_id VARCHAR(64) NOT NULL COMMENT 'Span ID',
    parent_span_id VARCHAR(64) COMMENT '父 Span ID',
    service_name VARCHAR(128) NOT NULL COMMENT '服务名称',
    operation_name VARCHAR(256) NOT NULL COMMENT '操作名称',
    start_time TIMESTAMP NOT NULL COMMENT '开始时间',
    duration_ms INT NOT NULL COMMENT '持续时间（毫秒）',
    status_code VARCHAR(16) DEFAULT 'OK' COMMENT '状态码：OK, ERROR, TIMEOUT',
    tags JSON COMMENT '标签信息（JSON格式）',
    INDEX idx_trace_id (trace_id),
    INDEX idx_service_name (service_name),
    INDEX idx_start_time (start_time)
) COMMENT='Trace调用记录表';

-- 创建 error_logs 表：存储错误日志
CREATE TABLE IF NOT EXISTS error_logs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    trace_id VARCHAR(64) NOT NULL COMMENT 'Trace ID',
    span_id VARCHAR(64) COMMENT 'Span ID',
    service_name VARCHAR(128) NOT NULL COMMENT '服务名称',
    error_type VARCHAR(64) NOT NULL COMMENT '错误类型',
    error_message TEXT NOT NULL COMMENT '错误消息',
    error_stack TEXT COMMENT '错误堆栈',
    occurred_at TIMESTAMP NOT NULL COMMENT '发生时间',
    INDEX idx_trace_id (trace_id),
    INDEX idx_service_name (service_name),
    INDEX idx_occurred_at (occurred_at)
) COMMENT='错误日志表';

-- 创建 service_metrics 表：存储服务指标
CREATE TABLE IF NOT EXISTS service_metrics (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    service_name VARCHAR(128) NOT NULL COMMENT '服务名称',
    metric_name VARCHAR(128) NOT NULL COMMENT '指标名称',
    metric_value DECIMAL(20, 4) NOT NULL COMMENT '指标值',
    metric_unit VARCHAR(32) COMMENT '单位',
    recorded_at TIMESTAMP NOT NULL COMMENT '记录时间',
    INDEX idx_service_name (service_name),
    INDEX idx_recorded_at (recorded_at)
) COMMENT='服务指标表';

-- 插入示例 trace 记录数据
INSERT INTO trace_records (trace_id, span_id, parent_span_id, service_name, operation_name, start_time, duration_ms, status_code, tags) VALUES
('trace-12345', 'span-001', NULL, 'UserService', 'handleUserRequest', '2024-01-15 10:00:00', 50, 'OK', '{"user_id": "user123", "request_type": "order"}'),
('trace-12345', 'span-002', 'span-001', 'APIGateway', 'routeRequest', '2024-01-15 10:00:01', 30, 'OK', '{"route": "/api/order", "method": "POST"}'),
('trace-12345', 'span-003', 'span-002', 'OrderService', 'createOrder', '2024-01-15 10:00:02', 1200, 'OK', '{"order_id": "order-12345", "amount": 199.99}'),
('trace-12345', 'span-004', 'span-003', 'PaymentService', 'processPayment', '2024-01-15 10:00:03', 30000, 'TIMEOUT', '{"payment_method": "credit_card", "amount": 199.99}'),
('trace-12345', 'span-005', 'span-004', 'BankAPI', 'chargeCard', '2024-01-15 10:00:33', 500, 'ERROR', '{"bank": "test_bank", "error_code": "DB_CONNECTION_ERROR"}');

-- 插入示例错误日志数据
INSERT INTO error_logs (trace_id, span_id, service_name, error_type, error_message, error_stack, occurred_at) VALUES
('trace-12345', 'span-004', 'PaymentService', 'TIMEOUT', '连接超时 (30秒)', 'PaymentService.processPayment: timeout after 30s\n  at PaymentService.connectToBank()', '2024-01-15 10:00:33'),
('trace-12345', 'span-005', 'BankAPI', 'DATABASE_ERROR', 'Database connection pool exhausted', 'BankAPI.chargeCard: failed to get database connection\n  at BankAPI.getConnection()\n  at BankAPI.chargeCard()', '2024-01-15 10:00:33'),
('trace-12345', 'span-005', 'BankAPI', 'HTTP_ERROR', 'HTTP 500 Internal Server Error', NULL, '2024-01-15 10:00:33');

-- 插入示例服务指标数据
INSERT INTO service_metrics (service_name, metric_name, metric_value, metric_unit, recorded_at) VALUES
('PaymentService', 'connection_pool_usage', 85.0, 'percent', '2024-01-15 10:00:20'),
('PaymentService', 'connection_pool_usage', 92.0, 'percent', '2024-01-15 10:00:25'),
('PaymentService', 'connection_pool_usage', 100.0, 'percent', '2024-01-15 10:00:30'),
('PaymentService', 'active_connections', 20, 'count', '2024-01-15 10:00:30'),
('PaymentService', 'max_connections', 20, 'count', '2024-01-15 10:00:30'),
('PaymentService', 'waiting_connections', 5, 'count', '2024-01-15 10:00:30'),
('BankAPI', 'database_connection_pool_size', 20, 'count', '2024-01-15 10:00:30'),
('BankAPI', 'database_connection_pool_active', 20, 'count', '2024-01-15 10:00:30'),
('BankAPI', 'database_connection_pool_idle', 0, 'count', '2024-01-15 10:00:30'),
('OrderService', 'cpu_usage', 45.0, 'percent', '2024-01-15 10:00:30'),
('OrderService', 'memory_usage', 78.0, 'percent', '2024-01-15 10:00:30');

