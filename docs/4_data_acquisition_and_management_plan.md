# 4. 数据采集与管理计划 (Data Acquisition and Management Plan)

## 4.1 数据源

| 数据类型 | 来源 | 访问方式 | 格式 | 主要变量 | 更新频率 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 数值天气预报 (NWP) | 加拿大环境与气候变化部 (ECCC) | GeoPortal REST API / WMS | GeoJSON / GRIB2 | 气温、降水、风速等 | 每日 |
| 遥感数据 | NASA (MODIS), ESA (Sentinel-2, Sentinel-3) | Earthdata Search API, Copernicus Open Access Hub | GeoTIFF | SWE、地表温度、植被指数 | 每日/每周 |
| 地面实况数据 | 加拿大水文数据库 (HYDAT) | SQLite数据库下载 | SQLite | 历史径流数据 | 每年 |
| 积雪遥测 (SNOTEL) | 美国自然资源保护局 (NRCS) | NRCS API | CSV | SWE、雪深 | 每日 |
| 水文网络 | 加拿大水文网络 (GeoPortal) | Feature Service | Shapefile/GeoJSON | 流域边界、河流网络 | 静态 |

## 4.2 数据处理流程 (ETL)

*   **提取 (Extract):**
    *   开发Python脚本，定期从各个数据源的API或服务中下载最新数据。
    *   对于HYDAT等静态数据，进行一次性下载和处理。
*   **转换 (Transform):**
    *   将所有地理空间数据重新投影到统一的坐标参考系统（例如，NAD83 / UTM Zone 14N）。
    *   将所有栅格数据重采样到统一的1km网格。
    *   对数据进行清理（例如，处理缺失值、异常值）、插值和特征工程。
    *   将不同来源的数据进行融合，创建一个统一的、可用于模型训练的数据集。
*   **加载 (Load):**
    *   将处理后的关系型数据加载到PostgreSQL数据库中。
    *   将处理后的栅格数据（例如，GeoTIFF）存储在云存储（S3/GCS）中。

## 4.3 数据版本控制和沿袭

*   **版本控制:** 使用DVC (Data Version Control) 来跟踪数据集和模型的版本，确保实验的可复现性。
*   **沿袭:** 实施自动化的沿袭跟踪，记录从原始数据到最终产品的每个步骤，包括使用了哪个版本的代码和数据。
*   **元数据:** 为每个数据集维护详细的元数据，包括来源、处理步骤、质量指标和更新日期。
