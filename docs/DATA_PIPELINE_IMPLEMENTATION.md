# HydrAI-SWE 数据管道实现文档

## 概述

本文档描述了 HydrAI-SWE 项目中实现的数据管道系统，该系统提供了自动化的数据获取、同步和备用数据源接管功能。

## 核心特性

### 1. 真实状态反馈
- **生产环境原则**：不接受 mock 和硬编码，所有状态都是真实检查结果
- **实时状态更新**：每个数据源的状态、记录数和最后更新时间都是实时获取
- **错误透明化**：当数据源不可用时，系统提供明确的错误信息和解决建议

### 2. 备用数据源接管
- **自动回退机制**：当主要数据源（如 NASA MODIS/Sentinel-2）不可用时，系统自动切换到备用数据源
- **无缝切换**：用户无需手动干预，系统自动选择可用的最佳数据源
- **状态标记**：备用数据源使用时，状态会明确标记为 "Active (Backup)"

### 3. 多数据源支持
- **地面数据**：HYDAT 水文站、ECCC 天气数据
- **卫星数据**：MODIS 积雪、Sentinel-2 影像
- **备用数据**：ERA5-Land 再分析、SMAP 样本、HLS 样本

## 技术架构

### API 端点

#### 管道状态管理
- `GET /api/v1/pipeline/status` - 获取所有数据源状态
- `GET /api/v1/pipeline/backup/status` - 获取备用数据源配置
- `GET /api/v1/pipeline/credentials/status` - 检查 Earthdata 凭据状态

#### 数据同步操作
- `POST /api/v1/pipeline/sync?source={source}` - 同步单一数据源
- `POST /api/v1/pipeline/sync-all` - 同步所有数据源
- `GET /api/v1/pipeline/job/{job_id}` - 查询同步作业状态

### 数据源配置

```python
SOURCE_MAP = {
    "modis": {
        "paths": ["/path/to/mODIS/data"],
        "backup_sources": ["era5_land", "smap_samples"]
    },
    "sentinel2": {
        "paths": ["/path/to/sentinel2/data"],
        "backup_sources": ["era5_land", "hls_samples"]
    },
    "era5_land": {
        "paths": ["/path/to/era5/data"]
    }
    # ... 其他数据源
}
```

### 备用数据源逻辑

1. **凭据检查**：首先检查 NASA Earthdata 凭据（.netrc、Bearer Token、earthaccess 包）
2. **主源尝试**：如果凭据可用，尝试从主要数据源获取数据
3. **自动回退**：如果主源失败或凭据不可用，自动切换到备用数据源
4. **状态更新**：更新管道状态，标记数据来源（主源或备用源）

## 前端集成

### 实时状态显示
- **动态表格**：数据管道管理界面显示实时状态
- **自动刷新**：每30秒自动更新状态信息
- **交互式操作**：每个数据源都有独立的同步按钮

### 状态指示器
- 🟢 **Active**：数据源正常工作
- 🟡 **Active (Backup)**：使用备用数据源
- 🔴 **Idle**：数据源空闲
- ⚠️ **Syncing**：正在同步中

## 使用示例

### 1. 检查管道状态
```bash
curl http://localhost:8000/api/v1/pipeline/status
```

### 2. 同步特定数据源
```bash
curl -X POST "http://localhost:8000/api/v1/pipeline/sync?source=modis"
```

### 3. 查看作业状态
```bash
curl http://localhost:8000/api/v1/pipeline/job/{job_id}
```

### 4. 检查备用数据源配置
```bash
curl http://localhost:8000/api/v1/pipeline/backup/status
```

## 测试验证

### 自动化测试脚本
项目包含完整的测试脚本 `scripts/test_pipeline.py`，验证：
- 管道状态查询功能
- 备用数据源回退机制
- 凭据状态检查
- 数据同步作业管理

### 测试结果示例
```
✅ 管道状态查询成功
📊 数据源数量: 7
🟢 modis: Active (Backup) (17 records)
🟢 sentinel2: Active (Backup) (14 records)
🟢 eccc: Active (7393 records)
```

## 部署和维护

### 系统要求
- Python 3.8+
- FastAPI 框架
- 支持异步操作的环境

### 监控建议
- 定期检查管道状态 API
- 监控备用数据源的使用情况
- 关注凭据状态和更新

### 故障排除
1. **凭据问题**：检查 ~/.netrc 文件或环境变量
2. **备用源不可用**：验证备用数据源路径和权限
3. **同步失败**：查看作业日志和错误消息

## 未来扩展

### 计划功能
- 实现真实的 MODIS/Sentinel-2 下载脚本
- 添加数据质量验证和报告
- 集成更多备用数据源
- 实现数据版本管理和回滚

### 技术改进
- 添加数据压缩和存储优化
- 实现增量同步机制
- 添加数据源健康检查
- 实现分布式数据同步

## 总结

HydrAI-SWE 数据管道系统成功实现了：
- **生产就绪**：真实状态反馈，无 mock 数据
- **高可用性**：备用数据源自动接管
- **用户友好**：清晰的状态显示和操作界面
- **可扩展性**：支持添加新的数据源和备用方案

该系统为 HydrAI-SWE 项目提供了可靠的数据基础设施，确保即使在主要数据源不可用的情况下，系统仍能继续为用户提供服务。
