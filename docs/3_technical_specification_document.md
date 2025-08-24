# 3. 技术规格文档 (Technical Specification Document - TSD)

## 3.1 系统架构

### 整体架构
HydrAI-SWE采用模块化微服务架构，支持本地部署和云平台扩展：

- **本地部署**: 单机多进程架构，适合开发和中小规模应用
- **云平台扩展**: AWS 或 Google Cloud Platform (GCP) 支持
- **容器化**: Docker容器化部署，支持Kubernetes编排

### 核心组件
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources │    │  Processing     │    │   API & UI      │
│                │    │  Pipeline       │    │                 │
├─────────────────┤    ├─────────────────┤    ├─────────────────┤
│ • ECCC GRIB2   │───▶│ • ETL Engine    │───▶│ • FastAPI       │
│ • NASA MODIS   │    │ • High-Res      │    │ • Web UI        │
│ • Sentinel-2   │    │   Integration   │    │ • Authentication│
│ • HYDAT SQLite │    │ • QC & Gap      │    │ • Rate Limiting │
│ • DEM SRTM     │    │   Filling       │    └─────────────────┘
└─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  ML Models      │
                       │                 │
                       ├─────────────────┤
                       │ • NeuralHydrology│
                       │ • LSTM/TCN/GRU │
                       │ • Cross-Validation│
                       │ • Model Versioning│
                       └─────────────────┘
```

### 数据存储架构
- **原始数据**: 本地文件系统 + 云存储 (S3/GCS)
- **处理数据**: PostgreSQL with PostGIS extension
- **模型文件**: 本地存储 + 版本控制
- **日志**: 本地文件 + Elasticsearch (可选)

## 3.2 技术栈

### 核心编程语言
- **Python**: 3.9+ (主要开发语言)
- **SQL**: PostgreSQL 14+ with PostGIS
- **Shell**: Bash scripting for automation

### 数据处理库
- **地理空间**: GeoPandas, Rasterio, GDAL, Shapely
- **数值计算**: NumPy, Pandas, Xarray
- **图像处理**: Pillow, OpenCV
- **数据验证**: Pydantic, Cerberus

### 机器学习框架
- **深度学习**: PyTorch 2.0+, NeuralHydrology
- **传统ML**: Scikit-learn, XGBoost
- **模型管理**: MLflow, DVC
- **超参数优化**: Optuna, Hyperopt

### Web框架和API
- **后端**: FastAPI 0.100+
- **前端**: HTML5 + JavaScript + Chart.js
- **认证**: JWT, API Key
- **文档**: OpenAPI/Swagger

### 基础设施
- **容器化**: Docker, Docker Compose
- **配置管理**: Pydantic Settings, Environment Variables
- **日志**: Python logging, Structured logging
- **监控**: Prometheus + Grafana (可选)

## 3.3 数据库设计

### 核心表结构

#### 水文站点表 (hydrometric_stations)
```sql
CREATE TABLE hydrometric_stations (
    station_id VARCHAR(20) PRIMARY KEY,  -- e.g., '05OC001'
    station_name VARCHAR(100) NOT NULL,  -- e.g., 'Red River at Emerson'
    latitude DECIMAL(10,8) NOT NULL,
    longitude DECIMAL(11,8) NOT NULL,
    drainage_area_km2 DECIMAL(10,2),
    elevation_m DECIMAL(8,2),
    region VARCHAR(50) NOT NULL,         -- e.g., 'red_river_basin'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### 径流预测表 (runoff_forecasts)
```sql
CREATE TABLE runoff_forecasts (
    id SERIAL PRIMARY KEY,
    station_id VARCHAR(20) NOT NULL REFERENCES hydrometric_stations(station_id),
    forecast_date DATE NOT NULL,         -- 预测生成日期
    target_date DATE NOT NULL,           -- 预测目标日期
    flow_median_m3s REAL,                -- 预测中位数流量 (m³/s)
    flow_p05_m3s REAL,                   -- 5%分位数流量
    flow_p95_m3s REAL,                   -- 95%分位数流量
    confidence_level VARCHAR(20),         -- 置信度级别
    model_version VARCHAR(20),            -- 模型版本
    region VARCHAR(50) NOT NULL,          -- 地理区域
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(station_id, target_date, model_version)
);
```

#### SWE数据表 (swe_data)
```sql
CREATE TABLE swe_data (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    region VARCHAR(50) NOT NULL,
    resolution_m INTEGER NOT NULL,        -- 分辨率(米)
    data_source VARCHAR(50) NOT NULL,     -- 数据源
    file_path TEXT,                       -- 文件路径
    metadata JSONB,                       -- 元数据
    quality_score DECIMAL(3,2),           -- 质量评分
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### 模型训练记录表 (model_training_logs)
```sql
CREATE TABLE model_training_logs (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(20) NOT NULL,
    training_start TIMESTAMP WITH TIME ZONE NOT NULL,
    training_end TIMESTAMP WITH TIME ZONE,
    region VARCHAR(50) NOT NULL,
    hyperparameters JSONB,
    metrics JSONB,                        -- 训练指标
    status VARCHAR(20) DEFAULT 'running', -- running, completed, failed
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### 空间索引
```sql
-- 为地理查询创建空间索引
CREATE INDEX idx_stations_location ON hydrometric_stations USING GIST (
    ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)
);

-- 为时间查询创建索引
CREATE INDEX idx_forecasts_date ON runoff_forecasts (target_date);
CREATE INDEX idx_swe_date ON swe_data (date);
```

## 3.4 API设计规范

### RESTful API端点

#### 1. 径流预测 API
```http
GET /api/v1/runoff-forecast
```

**查询参数:**
- `station_id` (必需): 水文站点ID
- `start_date` (必需): 开始日期 (YYYY-MM-DD)
- `end_date` (必需): 结束日期 (YYYY-MM-DD)
- `mode` (可选): 预测模式 (nowcast/scenario)
- `scenario_year` (可选): 情景年份
- `region` (可选): 地理区域

**响应示例 (200 OK):**
```json
{
  "station_id": "05OC001",
  "station_name": "Red River at Emerson",
  "region": "red_river_basin",
  "forecasts": [
    {
      "target_date": "2025-04-01",
      "flow_median_m3s": 1250.5,
      "flow_p05_m3s": 980.0,
      "flow_p95_m3s": 1520.0,
      "confidence_level": "high",
      "model_version": "lstm_v2.1"
    }
  ],
  "metadata": {
    "generated_at": "2025-03-25T10:30:00Z",
    "data_sources": ["ECCC", "NASA_MODIS", "HYDAT"],
    "last_updated": "2025-03-25T06:00:00Z"
  }
}
```

#### 2. SWE数据 API
```http
GET /api/v1/swe
```

**查询参数:**
- `date` (必需): 日期 (YYYY-MM-DD)
- `region` (必需): 地理区域
- `resolution` (可选): 分辨率 (10m, 100m, 500m, 1000m)
- `format` (可选): 输出格式 (json, geotiff, netcdf)

#### 3. 风险评估 API
```http
GET /api/v1/risk-assessment
```

**查询参数:**
- `date` (必需): 评估日期
- `region` (必需): 地理区域
- `threshold` (可选): 风险阈值

**响应示例:**
```json
{
  "date": "2025-03-25",
  "region": "red_river_basin",
  "risk_level": "medium",
  "risk_score": 0.65,
  "factors": {
    "swe_anomaly": 0.3,
    "temperature_trend": 0.4,
    "precipitation_forecast": 0.2
  },
  "recommendations": [
    "Monitor temperature trends",
    "Prepare for moderate flood risk"
  ]
}
```

### 认证和授权
```http
# API密钥认证
X-API-Key: your_api_key_here

# JWT令牌认证 (未来版本)
Authorization: Bearer your_jwt_token_here
```

### 速率限制
- **免费用户**: 100 requests/hour
- **认证用户**: 1000 requests/hour
- **企业用户**: 10000 requests/hour

## 3.5 数据流程

### ETL流程
```
1. 数据提取 (Extract)
   ├── ECCC: 每日06:00 UTC自动下载
   ├── NASA: 每日12:00 UTC检查更新
   ├── Sentinel-2: 按需下载 (5-7天重访周期)
   └── HYDAT: 每月更新

2. 数据转换 (Transform)
   ├── 坐标系统统一 (WGS84 → UTM Zone 14N)
   ├── 分辨率标准化 (10m → 100m → 500m → 1000m)
   ├── 质量控制 (异常值检测、缺失值填充)
   └── 特征工程 (NDSI、冻融指数、地形特征)

3. 数据加载 (Load)
   ├── 关系数据 → PostgreSQL
   ├── 栅格数据 → 本地存储 + 云存储
   └── 元数据 → 数据库索引
```

### 模型训练流程
```
1. 数据准备
   ├── 时间序列对齐
   ├── 特征标准化
   └── 训练/验证/测试分割

2. 模型训练
   ├── 超参数优化 (Optuna)
   ├── 交叉验证 (时间前向)
   └── 模型选择 (AIC/BIC)

3. 模型评估
   ├── 性能指标 (NSE, RMSE, MAE)
   ├── 不确定性量化
   └── 模型解释性分析

4. 模型部署
   ├── 版本控制
   ├── A/B测试
   └── 性能监控
```

## 3.6 性能指标

### 系统性能目标
- **API响应时间**: <200ms (95th percentile)
- **数据处理延迟**: <24小时
- **模型训练时间**: <12小时 (GPU)
- **系统可用性**: 99.9%

### 模型性能目标
- **SWE估算精度**: RMSE <20% (相比地面观测)
- **径流预测精度**: NSE >0.85
- **洪水预警**: 提前7天，准确率>80%

### 资源使用
- **内存**: 8GB-32GB (根据区域大小)
- **存储**: 100GB-1TB (根据数据保留策略)
- **CPU**: 4-16核心 (根据处理需求)
- **GPU**: 推荐用于模型训练

## 3.7 安全考虑

### 数据安全
- **加密传输**: HTTPS/TLS 1.3
- **数据加密**: 敏感数据AES-256加密
- **访问控制**: 基于角色的权限管理
- **审计日志**: 完整的操作记录

### 网络安全
- **防火墙**: 端口限制和IP白名单
- **DDoS防护**: 速率限制和异常检测
- **入侵检测**: 实时安全监控
- **定期更新**: 安全补丁和依赖更新

## 3.8 部署和运维

### 部署选项
1. **本地部署**: 单机Docker容器
2. **云部署**: AWS/GCP容器服务
3. **混合部署**: 本地处理 + 云存储

### 监控和告警
- **系统监控**: CPU、内存、磁盘、网络
- **应用监控**: API响应时间、错误率
- **业务监控**: 数据质量、模型性能
- **告警通知**: 邮件、Slack、短信

### 备份和恢复
- **数据备份**: 每日增量 + 每周全量
- **配置备份**: 版本控制和配置管理
- **灾难恢复**: 多区域备份和快速恢复
- **测试恢复**: 定期恢复测试验证
