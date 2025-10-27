# 📊 HydrAI-SWE Data Strategy

## 概述

本文档详细说明了 HydrAI-SWE 项目的数据管理策略，确保数据质量和完整性。

## 📅 数据时间范围策略

### 1. 2010-2020年：真实历史数据 ✅

- **数据来源**：Manitoba 每日 SWE 测量数据
- **数据条数**：4,018 条
- **数据性质**：100% 真实历史数据
- **数据质量**：高，来源可靠

### 2. 2021-2024年：基于真实规律生成的模拟数据 ✅

- **数据来源**：基于 2010-2020 年真实数据规律生成
- **数据条数**：1,461 条（约每年 365 条）
- **数据性质**：模拟数据，但遵循真实数据的规律和趋势
- **生成方法**：
  - 分析 2010-2020 年各年平均 SWE 值
  - 计算年际变化趋势
  - 基于历史模式和季节性规律生成
  - 使用平滑曲线而非随机数据

### 3. 2025年：实时同步的真实数据 ✅

- **数据来源**：
  - OpenMeteo 历史气象数据
  - Manitoba 洪水预警系统
- **数据条数**：持续增长（每日同步）
- **数据性质**：100% 真实数据
- **同步频率**：每日凌晨 2:00 自动同步

## 🔄 数据同步机制

### 每日同步服务

```bash
# 启动每日同步服务
python3 daily_sync_service.py
```

**同步内容**：
1. 从 OpenMeteo 获取最新气象数据（降水、降雪、温度）
2. 从 Manitoba 洪水预警系统获取洪水预警数据
3. 基于气象数据计算 SWE 值
4. 自动更新到 SQLite 数据库

**SWE 计算逻辑**：
- 降雪量 × 0.3 → SWE
- 低温降水（最高温 < 2°C）→ SWE 增加
- 高温（平均温 > 0°C）→ SWE 融化

## 📁 关键文件

### 数据管理脚本

1. **`implement_correct_data_strategy.py`**
   - 实现完整的数据策略
   - 清理旧数据
   - 生成 2021-2024 年模拟数据
   - 同步 2025 年真实数据

2. **`daily_sync_service.py`**
   - 每日自动同步服务
   - 从多个数据源获取最新数据
   - 自动更新数据库

3. **`simple_swe_api.py`**
   - 轻量级 FastAPI 服务器
   - SQLite 数据库查询接口
   - 支持时间窗口筛选
   - 支持分页查询

### 数据清理脚本

- **`clean_and_sync_real_data.py`**：清理模拟数据，重新同步真实数据
- **`generate_realistic_swe_data.py`**：生成符合真实数据规律的模拟数据
- **`fix_data_gap.py`**：修复 2020-2025 年间的数据空白

## 🗄️ 数据库结构

### SQLite 数据库：`swe_data.db`

```sql
CREATE TABLE swe_data (
    timestamp TEXT PRIMARY KEY,
    swe_mm REAL,
    data_source TEXT
);
```

### 数据源标识

- **`historical`**：2010-2020 年真实历史数据
- **`simulated_2021`**：2021 年模拟数据
- **`simulated_2022`**：2022 年模拟数据
- **`simulated_2023`**：2023 年模拟数据
- **`simulated_2024`**：2024 年模拟数据
- **`openmeteo_2025`**：来自 OpenMeteo 的 2025 年真实数据
- **`manitoba_flood_2025`**：来自 Manitoba 洪水预警系统的 2025 年真实数据

## 📊 当前数据统计

```
总记录数: 5,540
时间范围: 2010-01-01 到 2025-06-17

数据源分布:
- historical: 4,018 条 (2010-2020)
- simulated_2021: 365 条
- simulated_2022: 365 条
- simulated_2023: 365 条
- simulated_2024: 366 条
- manitoba_flood_2025: 35 条
- openmeteo_2025: 26 条
```

## 🚀 使用指南

### 1. 初始化数据库

```bash
cd /home/sean/hydrai_swe
source venv/bin/activate
python3 implement_correct_data_strategy.py
```

### 2. 启动简单 API 服务器

```bash
python3 simple_swe_api.py
```

API 端点：
- `GET /api/swe/historical` - 查询历史数据
- `GET /api/swe/realtime` - 查询最新数据
- `GET /health` - 健康检查

### 3. 启动每日同步服务

```bash
python3 daily_sync_service.py
```

### 4. 启动完整前端

```bash
# 在一个终端启动简单 API
python3 simple_swe_api.py

# 在另一个终端启动前端服务器（如使用 Python http.server）
cd templates/ui
python3 -m http.server 3000
```

访问 `http://localhost:3000/enhanced_dashboard.html`

## ✅ 数据质量保证

1. **真实数据优先**：2010-2020 和 2025 年使用真实数据
2. **模式匹配**：模拟数据遵循真实数据的规律和趋势
3. **平滑曲线**：模拟数据生成平滑、一致的曲线，避免不合理的随机性
4. **自动同步**：2025 年数据每日自动更新，确保最新
5. **数据标注**：每条数据都有明确的数据源标识

## 📝 注意事项

1. **不要手动修改数据源标识**：系统依赖数据源标识来区分真实数据和模拟数据
2. **定期检查同步服务**：确保每日同步服务正常运行
3. **监控数据完整性**：定期检查数据库中的数据是否完整
4. **备份重要数据**：定期备份 `swe_data.db` 文件

---

**最后更新**: 2025-01-20  
**维护者**: HydrAI-SWE Team