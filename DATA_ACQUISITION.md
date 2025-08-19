# 数据获取说明

## 概述

HydrAI-SWE项目需要以下数据源来运行完整的训练流程：

## 1. NASA MODIS积雪数据

### 获取方式
- **自动获取**: 通过ETL脚本自动下载（需要NASA Earthdata账户）
- **账户配置**: 已在 `config/credentials.env` 中配置

### 数据产品
- **产品名称**: MOD10A1 (MODIS/Terra Snow Cover Daily L3 Global 500m SIN Grid V061)
- **时间范围**: 2024-03-01 到 2024-03-07
- **地理范围**: 可配置，支持以下区域：
  - **红河流域** (推荐): (-97.5, 49.0, -96.5, 50.5) ~116,000 km²
  - **温尼伯都市区**: (-97.3, 49.7, -96.9, 50.1) ~5,300 km²
  - **温尼伯市区**: (-97.2, 49.8, -97.0, 50.0) ~465 km²
  - **曼尼托巴省**: (-102.0, 49.0, -95.0, 53.0) ~650,000 km²
- **分辨率**: 根据区域自动调整（100m-1000m）

### 注意事项
- 需要NASA Earthdata账户
- 数据下载可能需要较长时间
- 确保网络连接稳定

## 2. ECCC天气数据 (GRIB2)

### 获取方式
- **手动下载**: 从加拿大环境与气候变化部网站下载
- **存储位置**: `data/raw/eccc_grib/`

### 数据要求
- **格式**: GRIB2
- **变量**: 
  - t2m (2米气温)
  - tp (总降水量)
- **时间范围**: 2024-03-01 到 2024-03-07
- **地理范围**: 曼尼托巴省

### 下载链接
- **ECCC数据门户**: https://dd.weather.gc.ca/
- **GRIB2数据**: https://dd.weather.gc.ca/model_gem_global/15km/grib2/lat_lon/

## 3. HYDAT水文数据库

### 获取方式
- **手动下载**: 从加拿大水文数据库网站下载
- **存储位置**: `data/raw/Hydat.sqlite3`

### 数据要求
- **格式**: SQLite数据库
- **内容**: 历史径流数据
- **站点**: 05OC001 (Red River at Emerson)

### 下载链接
- **HYDAT数据库**: https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/
- **选择**: HYDAT SQLite3 格式

## 4. 数据目录结构

```
data/
├── raw/
│   ├── nasa_modis_snow/          # NASA MODIS积雪数据
│   ├── eccc_grib/                # ECCC天气数据
│   └── Hydat.sqlite3            # HYDAT水文数据库
└── processed/                    # 处理后的数据
    ├── nasa_modis_snow_processed.nc
    ├── eccc_weather_processed.nc
    └── hydat_streamflow_processed.csv
```

## 5. 数据获取步骤

### 步骤1: 创建目录结构
```bash
mkdir -p data/raw/nasa_modis_snow data/raw/eccc_grib data/processed
```

### 步骤2: 下载ECCC GRIB2数据
1. 访问 https://dd.weather.gc.ca/model_gem_global/15km/grib2/lat_lon/
2. 下载2024年3月1-7日的GRIB2文件
3. 将文件保存到 `data/raw/eccc_grib/` 目录

### 步骤3: 下载HYDAT数据库
1. 访问 https://collaboration.cmc.ec.gc.ca/cmc/hydrometrics/www/
2. 下载HYDAT SQLite3格式数据库
3. 重命名为 `Hydat.sqlite3` 并保存到 `data/raw/` 目录

### 步骤4: 运行ETL流程
```bash
source venv/bin/activate

# 使用默认区域（红河流域）
python3 src/data/etl.py

# 区域选择在脚本中交互式进行
```

## 6. 故障排除

### 常见问题

1. **NASA数据下载失败**
   - 检查网络连接
   - 验证账户凭据
   - 确认数据产品可用性

2. **GRIB2文件无法读取**
   - 检查文件完整性
   - 确认文件格式正确
   - 验证变量名称

3. **HYDAT数据库错误**
   - 检查文件路径
   - 确认数据库格式
   - 验证站点ID存在

### 调试建议

- 使用 `test_nasa_auth.py` 测试NASA认证
- 检查数据文件大小和格式
- 查看ETL脚本的错误输出

## 7. 数据质量要求

- **完整性**: 确保时间序列连续
- **准确性**: 数据值在合理范围内
- **一致性**: 不同数据源的时间和空间对齐
- **时效性**: 数据更新及时

## 8. 下一步

完成数据获取后，可以运行完整的训练流程：

```bash
python3 run_full_training.py
```

这将自动执行ETL、数据准备和模型训练。
