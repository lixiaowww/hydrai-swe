# 地理区域选择指南

## 概述

HydrAI-SWE项目支持多个地理精度级别的数据获取和模型训练。您可以根据具体需求选择合适的地理范围。

## 可用区域

### 1. 红河流域 (Red River Basin) - 推荐 ⭐

**基本信息**
- **名称**: Red River Basin
- **面积**: ~116,000 km²
- **分辨率**: 500m x 500m
- **地理范围**: (-97.5, 49.0, -96.5, 50.5)

**适用场景**
- SWE建模和预测
- 洪水预警系统
- 水资源管理
- 水电运营优化

**包含的水文站点**
- 05OC001: Red River at Emerson
- 05OC011: Red River at Winnipeg  
- 05OC012: Red River at Lockport

**数据产品**
- NASA MODIS积雪数据
- ECCC天气数据 (GRIB2)
- HYDAT水文数据库
- Sentinel-2高分辨率数据

**优势**
- 覆盖完整的红河流域
- 包含关键水文站点
- 数据密度适中
- 计算资源需求合理

### 2. 温尼伯都市区 (Winnipeg Metropolitan Area)

**基本信息**
- **名称**: Winnipeg Metropolitan Area
- **面积**: ~5,300 km²
- **分辨率**: 250m x 250m
- **地理范围**: (-97.3, 49.7, -96.9, 50.1)

**适用场景**
- 都市洪水风险评估
- 城市基础设施规划
- 局部天气影响分析
- 高精度SWE估算

**包含的行政区**
- Winnipeg (温尼伯)
- Selkirk (塞尔柯克)
- St. Andrews (圣安德鲁斯)
- St. Clements (圣克莱门茨)

**数据产品**
- NASA MODIS积雪数据
- ECCC天气数据 (GRIB2)
- HYDAT水文数据库
- Sentinel-2高分辨率数据
- LiDAR地形数据

**优势**
- 高空间分辨率
- 适合城市应用
- 数据质量高
- 计算效率好

### 3. 温尼伯市区 (Winnipeg City Core)

**基本信息**
- **名称**: Winnipeg City Core
- **面积**: ~465 km²
- **分辨率**: 100m x 100m
- **地理范围**: (-97.2, 49.8, -97.0, 50.0)

**适用场景**
- 城市微气候分析
- 精确洪水建模
- 基础设施风险评估
- 实时监控系统

**包含的地标**
- Red River (红河)
- Assiniboine River (阿西尼博因河)
- Assiniboine Park (阿西尼博因公园)
- The Forks (福克斯)

**数据产品**
- NASA MODIS积雪数据
- ECCC天气数据 (GRIB2)
- HYDAT水文数据库
- Sentinel-2高分辨率数据
- LiDAR地形数据
- 高分辨率数字高程模型

**优势**
- 最高空间分辨率
- 适合精细建模
- 城市特征完整
- 实时应用友好

### 4. 曼尼托巴省 (Manitoba Province)

**基本信息**
- **名称**: Manitoba Province
- **面积**: ~650,000 km²
- **分辨率**: 1000m x 1000m
- **地理范围**: (-102.0, 49.0, -95.0, 53.0)

**适用场景**
- 省级水资源规划
- 大尺度气候研究
- 政策制定支持
- 区域趋势分析

**数据产品**
- NASA MODIS积雪数据
- ECCC天气数据 (GRIB2)
- HYDAT水文数据库

**优势**
- 覆盖全省范围
- 适合宏观分析
- 数据获取稳定
- 计算资源需求低

## 如何选择区域

### 选择标准

1. **应用精度要求**
   - 高精度应用 → 选择温尼伯市区或都市区
   - 中等精度应用 → 选择红河流域
   - 宏观分析 → 选择曼尼托巴省

2. **计算资源**
   - 有限资源 → 选择较小区域
   - 充足资源 → 可选择较大区域

3. **数据需求**
   - 实时应用 → 选择较小区域
   - 历史分析 → 可选择较大区域

4. **时间要求**
   - 快速结果 → 选择较小区域
   - 深度分析 → 可选择较大区域

### 推荐选择

- **首次使用**: 红河流域 (平衡精度和效率)
- **城市应用**: 温尼伯都市区
- **精细建模**: 温尼伯市区
- **省级规划**: 曼尼托巴省

## 使用方法

### 命令行参数

```bash
# 使用默认区域（红河流域）
python3 run_full_training.py

# 指定特定区域
python3 run_full_training.py --region red_river_basin
python3 run_full_training.py --region winnipeg_metro
python3 run_full_training.py --region winnipeg_city
python3 run_full_training.py --region manitoba_province
```

### 测试认证

```bash
# 测试特定区域的NASA认证
python3 test_nasa_auth.py --region red_river_basin
python3 test_nasa_auth.py --region winnipeg_metro
```

### ETL流程

```bash
# 运行ETL（区域选择在脚本中交互式进行）
python3 src/data/etl.py
```

## 性能对比

| 区域 | 面积 (km²) | 分辨率 | 数据量 | 处理时间 | 内存需求 |
|------|-------------|--------|--------|----------|----------|
| 温尼伯市区 | 465 | 100m | 低 | 快 | 低 |
| 温尼伯都市区 | 5,300 | 250m | 中 | 中 | 中 |
| 红河流域 | 116,000 | 500m | 中高 | 中 | 中 |
| 曼尼托巴省 | 650,000 | 1000m | 高 | 慢 | 高 |

## 注意事项

1. **数据可用性**: 较小区域可能有更丰富的高分辨率数据
2. **计算效率**: 区域越小，计算效率越高
3. **模型精度**: 较小区域通常能提供更精确的局部预测
4. **扩展性**: 较小区域的结果可以扩展到相似地理环境

## 下一步

选择合适的地理区域后，请参考：
- `DATA_ACQUISITION.md`: 数据获取说明
- `TRAINING_README.md`: 训练流程指南
- `README.md`: 项目总体说明
