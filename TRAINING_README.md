# HydrAI-SWE Training Guide

## 概述

本指南说明如何使用真实数据运行HydrAI-SWE项目的完整训练流程。

## 前置要求

1. **Python环境**: Python 3.9+
2. **依赖包**: 已安装所有requirements.txt中的包
3. **NASA Earthdata账户**: 用于获取MODIS积雪数据
4. **ECCC GRIB2数据**: 加拿大环境与气候变化部的天气数据
5. **HYDAT数据库**: 加拿大水文数据库

## 配置认证信息

### 1. NASA Earthdata认证

编辑 `config/credentials.env` 文件：

```bash
# NASA Earthdata Credentials
NASA_EARTHDATA_USERNAME=your_username
NASA_EARTHDATA_PASSWORD=your_password
```

### 2. HYDAT数据库路径

编辑 `src/data/etl.py` 文件中的 `HYDAT_DB_PATH` 变量：

```python
HYDAT_DB_PATH = "/path/to/your/Hydat.sqlite3"
```

## 运行训练流程

### 方法1: 完整流程（推荐）

运行完整的训练管道：

```bash
# 使用默认区域（红河流域）
python run_full_training.py

# 指定特定区域
python run_full_training.py --region red_river_basin      # 红河流域
python run_full_training.py --region winnipeg_metro      # 温尼伯都市区
python run_full_training.py --region winnipeg_city       # 温尼伯市区
python run_full_training.py --region manitoba_province   # 曼尼托巴省
```

这将自动执行：
1. ETL数据获取和处理
2. 数据准备（NeuralHydrology格式）
3. 模型训练

### 方法2: 分步执行

#### 步骤1: ETL数据获取

```bash
python src/data/etl.py
```

#### 步骤2: 数据准备

```bash
python src/neuralhydrology/prepare_data.py
```

#### 步骤3: 模型训练

```bash
python src/models/train.py
```

## 数据流程

### 1. 数据获取 (ETL)

- **NASA MODIS积雪数据**: 通过Earthdata API获取
- **ECCC天气数据**: 从GRIB2文件处理
- **HYDAT径流数据**: 从SQLite数据库读取

### 2. 数据处理

- 地理坐标对齐
- 时间序列合并
- 缺失值处理
- 格式转换

### 3. 模型训练

- 使用NeuralHydrology框架
- LSTM模型架构
- 自动超参数优化
- 训练结果保存到`runs/`目录

## 输出文件

### 处理后的数据

- `data/processed/nasa_modis_snow_processed.nc`: 积雪数据
- `data/processed/eccc_weather_processed.nc`: 天气数据
- `data/processed/hydat_streamflow_processed.csv`: 径流数据

### 训练结果

- `runs/hydrai_swe_experiment_*/`: 训练结果目录
- 包含训练好的模型、配置文件和评估结果

## 故障排除

### 常见问题

1. **认证失败**
   - 检查credentials.env文件
   - 验证用户名和密码

2. **数据文件缺失**
   - 确保HYDAT数据库文件存在
   - 检查ECCC GRIB2数据文件

3. **路径错误**
   - 确保在项目根目录运行脚本
   - 检查相对路径配置

### 调试模式

可以单独测试各个组件：

```bash
# 测试NASA认证
python test_nasa_auth.py

# 测试ETL流程
python src/data/etl.py

# 测试数据准备
python src/neuralhydrology/prepare_data.py
```

## 性能优化

### 数据范围

- 时间范围: 2024-03-01 到 2024-03-07
- 地理范围: 曼尼托巴省 (-102.0, 49.0, -95.0, 53.0)
- 空间分辨率: 1km x 1km

### 训练参数

- 模型: LSTM
- 隐藏层大小: 64
- 训练轮数: 30
- 批次大小: 16
- 学习率: 0.001

## 下一步

训练完成后，可以使用训练好的模型进行预测：

```bash
python src/models/predict.py
```

确保更新预测脚本中的模型路径。
