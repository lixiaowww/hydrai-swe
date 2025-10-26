# 数据策略文档

## 📊 数据管理策略

本文档详细说明了 HydrAI-SWE 项目的数据管理策略，包括数据分类、来源和处理方法。

## 🎯 数据分类

### 1. 历史真实数据 (2010-2020)
- **来源**: Manitoba Daily SWE 测量数据
- **数量**: 4,018 条记录
- **特征**: 
  - 每日 SWE 测量值
  - 平滑连续的曲线
  - 高数据质量
- **用途**: 训练模型和生成模拟数据的参考模式

### 2. 模拟数据 (2021-2024)
- **生成方法**: 基于 2010-2020 年真实数据的历史规律
- **数量**: 1,461 条记录（每天一条）
- **特征**:
  - 遵循历史数据的季节模式
  - 保持年际变化趋势（+1.277mm/年）
  - 平滑且连续，与真实数据一致
- **数据源标记**: `simulated_2021`, `simulated_2022`, `simulated_2023`, `simulated_2024`

### 3. 实时真实数据 (2025)
- **来源**: 
  - OpenMeteo 历史气象数据
  - Manitoba 洪水预警系统
- **数量**: 持续增长（每天同步）
- **特征**:
  - 从降雪、降水和温度数据推断 SWE
  - 包含洪水预警期间的高 SWE 值
- **数据源标记**: `openmeteo_2025`, `manitoba_flood_2025`
- **同步时间**: 每天凌晨 2:00

## 🔄 数据同步机制

### 每日自动同步
- **启动服务**: `daily_sync_service.py`
- **同步时间**: 每天凌晨 2:00
- **同步内容**: 
  1. OpenMeteo 真实气象数据
  2. Manitoba 洪水预警系统数据

### 手动同步
运行以下命令手动同步 2025 年数据：
```bash
python3 implement_correct_data_strategy.py
```

## 🛠️ 核心文件

### 数据管理脚本
- **`implement_correct_data_strategy.py`**: 实现完整数据策略
- **`daily_sync_service.py`**: 每日自动同步服务
- **`clean_and_sync_real_data.py`**: 清理和重新同步真实数据
- **`generate_realistic_swe_data.py`**: 生成 2021-2024 年模拟数据

### 数据库
- **`swe_data.db`**: SQLite 数据库
  - 表结构: `timestamp`, `swe_mm`, `data_source`
  - 索引: 时间戳（主键）

## 📊 数据统计

### 最终数据状态（示例）
```
总记录数: 5,540
时间范围: 2010-01-01 到 2025-06-17

数据源分布:
- historical: 4,018条 (2010-2020年真实数据)
- simulated_2021-2024: 1,461条 (模拟数据)
- openmeteo_2025: 26条 (真实数据)
- manitoba_flood_2025: 35条 (真实数据)

年均SWE:
- 2020: 32.18mm (真实)
- 2021: 27.58mm (模拟)
- 2022: 29.53mm (模拟)
- 2023: 30.97mm (模拟)
- 2024: 28.19mm (模拟)
- 2025: 42.45mm (真实)
```

## 🎨 数据特点

### 真实数据特征 (2010-2020)
- **平滑连续性**: 数据曲线基本一致，变化平滑
- **季节性**: 冬季高 SWE，夏季低 SWE
- **年际变化**: 平均年际趋势 +1.277mm/年

### 模拟数据特征 (2021-2024)
- **遵循历史模式**: 基于 2010-2020 年真实数据的季节模式
- **趋势延续**: 保持 +1.277mm/年的年际趋势
- **平滑连续**: 与真实数据保持一致的数据平滑性
- **真实性**: 看起来"真实"，但实际上是从历史模式推断的

### 实时数据特征 (2025)
- **每日更新**: 从最新可用数据中推断 SWE
- **多源融合**: 结合气象数据和洪水预警信息
- **真实准确**: 基于真实观测数据，非模拟

## 🚀 快速开始

### 初始化数据库
```bash
python3 simple_swe_api.py  # 创建数据库并加载初始数据
```

### 实施完整策略
```bash
python3 implement_correct_data_strategy.py
```

### 启动每日同步
```bash
python3 daily_sync_service.py
```

### 查看数据状态
```bash
sqlite3 swe_data.db "SELECT COUNT(*) as total, data_source, COUNT(*) as count FROM swe_data GROUP BY data_source"
```

## 📝 数据质量标准

### 真实数据
- ✅ 来自官方数据源
- ✅ 包含数据源标记
- ✅ 每日完整记录
- ✅ 通过质量检查

### 模拟数据
- ✅ 基于真实历史模式
- ✅ 遵循季节性和年际趋势
- ✅ 平滑且连续
- ⚠️ 标记为 `simulated_*`

### 实时数据
- ✅ 来自权威数据源
- ✅ 每日自动同步
- ✅ 包含数据源标记
- ✅ 通过验证检查

## 🔍 数据验证

运行以下命令验证数据完整性：
```bash
python3 -c "
import sqlite3
conn = sqlite3.connect('swe_data.db')
cursor = conn.cursor()
cursor.execute('SELECT data_source, COUNT(*) FROM swe_data GROUP BY data_source')
print('数据源分布:')
for row in cursor.fetchall():
    print(f'  {row[0]}: {row[1]}条')
"
```

## 📄 相关文档

- **`README_DATA_SOURCES.md`**: 数据源详细说明
- **`REAL_DATA_REQUIREMENTS.md`**: 真实数据需求
- **`DATA_ACQUISITION.md`**: 数据获取指南

## 📞 支持和问题

如有问题或需要更多信息，请参考：
- 项目文档: `README.md`
- 数据源文档: `README_DATA_SOURCES.md`
- 数据需求: `REAL_DATA_REQUIREMENTS.md`
