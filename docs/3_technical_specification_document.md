# 3. 技术规格文档 (Technical Specification Document - TSD)

## 3.1 系统架构

*   **云平台:** AWS 或 Google Cloud Platform (GCP)
*   **数据存储:**
    *   **原始和处理后数据:** Amazon S3 或 Google Cloud Storage
    *   **关系型数据 (模型输出):** PostgreSQL with PostGIS extension
    *   **日志:** Elasticsearch 或 CloudWatch Logs
*   **计算:**
    *   **数据处理:** 使用Docker容器化的Python脚本，通过AWS Batch或Kubernetes进行编排。
    *   **模型训练:** 使用带有GPU的虚拟机实例 (例如 AWS EC2 P3 或 GCP A2 系列)。
    *   **API网关:** Amazon API Gateway 或 Google Cloud Endpoints，用于API管理、认证和速率限制。

## 3.2 技术栈

*   **编程语言:** Python 3.9+
*   **核心库:**
    *   **数据处理:** Pandas, GeoPandas, Rasterio, NumPy
    *   **机器学习:** Scikit-learn, TensorFlow 2.x 或 PyTorch
    *   **水文模型:** 使用LSTM, ANN, 或 SVM的自定义模型
    *   **API框架:** FastAPI 或 Flask
    *   **数据库:** PostgreSQL 14+ with PostGIS
    *   **基础设施即代码 (IaC):** Terraform 或 AWS CloudFormation

## 3.3 数据库模式 (示例)

```sql
CREATE TABLE runoff_forecasts (
    id SERIAL PRIMARY KEY,
    station_id VARCHAR(20) NOT NULL, -- e.g., '05OC001'
    forecast_date DATE NOT NULL,     -- The date the forecast was generated
    target_date DATE NOT NULL,       -- The date the forecast is for
    flow_median_m3s REAL,            -- Predicted median flow in m^3/s
    flow_p05_m3s REAL,               -- 5th percentile flow
    flow_p95_m3s REAL,               -- 95th percentile flow
    model_version VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## 3.4 API 端点示例

*   **获取径流预测:**
    *   **请求:** `GET /api/v1/runoff-forecast?station_id=05OC001&start_date=2025-04-01&end_date=2025-04-07`
    *   **认证:** `Header: X-API-Key: <user_api_key>`
    *   **响应 (200 OK):**
        ```json
        {
          "station_id": "05OC001",
          "station_name": "Red River at Emerson",
          "forecasts": [
            {
              "target_date": "2025-04-01",
              "flow_median_m3s": 1250.5,
              "flow_p05_m3s": 980.0,
              "flow_p95_m3s": 1520.0
            },
            {
              "target_date": "2025-04-02",
              "flow_median_m3s": 1310.0,
              "flow_p05_m3s": 1050.0,
              "flow_p95_m3s": 1600.0
            }
          ]
        }
        ```
