# HydrAI-SWE 持续更新数据源文档

**文档版本**: 1.0  
**创建日期**: 2025-09-27  
**最后更新**: 2025-09-27  
**维护者**: HydrAI-SWE开发团队

## 🎯 **概述**

本文档记录了HydrAI-SWE项目中所有持续更新的官方数据源及其访问地址。这些数据源为系统提供实时、准确的水文、气象、水质和洪水预警数据。

## 📊 **数据源总览**

### **数据源状态**
- ✅ **可用数据源**: 6个
- 🔄 **实时更新**: 4个
- 📈 **定期更新**: 2个
- 📅 **历史数据**: 6个

### **数据完整性**
- **100% 官方权威数据源**
- **无模拟或虚构数据**
- **符合监管标准**
- **数据质量可追溯**

## 🚀 **实时更新数据源**

### **1. Manitoba洪水预警系统**
- **数据源**: Government of Manitoba - Hydrologic Forecast Centre (HFC)
- **更新频率**: 实时更新
- **数据类型**: Overland Flood Alerts (陆地洪水预警)
- **访问地址**: 
  - **主页面**: https://open.canada.ca/data/en/dataset/8ec189d2-d9c5-ed0b-5773-895200e3e815
  - **CSV下载**: https://geoportal.gov.mb.ca/api/download/v1/items/11b3d4f938924f789f32c620d13cd4f8/csv?layers=0
  - **GeoJSON下载**: https://geoportal.gov.mb.ca/api/download/v1/items/11b3d4f938924f789f32c620d13cd4f8/geojson?layers=0
  - **ArcGIS REST API**: https://services.arcgis.com/mMUesHYPkXjaFGfS/arcgis/rest/services/Overland_Flood_Alerts/FeatureServer/0
- **许可证**: OpenMB Information and Data Use License
- **联系邮箱**: manitobamaps@gov.mb.ca
- **数据字段**: alert_type, issue_time, area_name, coordinates, severity, description, recommendations
- **预警类型**: Overland Flood Warning, Overland Flood Watch

### **2. RDPS降水预报系统**
- **数据源**: Environment and Climate Change Canada - RDPS
- **更新频率**: 每日4次 (00:00, 06:00, 12:00, 18:00 UTC)
- **数据类型**: 84小时降水预报 (子流域聚合)
- **访问地址**:
  - **主页面**: https://open.canada.ca/data/en/dataset/96f7f50d-f3d7-a775-67e7-fa1d0cd295c2
  - **CSV下载**: https://geoportal.gov.mb.ca/api/download/v1/items/0e74476ac32744679ff1d6075dd7142e/csv?layers=0
  - **GeoJSON下载**: https://geoportal.gov.mb.ca/api/download/v1/items/0e74476ac32744679ff1d6075dd7142e/geojson?layers=0
  - **ArcGIS REST API**: https://services.arcgis.com/mMUesHYPkXjaFGfS/arcgis/rest/services/RDPS_SubBasins_Precipitation_Distribution_84_hrs/FeatureServer/0
- **许可证**: OpenMB Information and Data Use License
- **联系邮箱**: manitobamaps@gov.mb.ca
- **数据字段**: basin_id, basin_name, precipitation_mm, forecast_period, coordinates, risk_level
- **技术规格**: 10公里分辨率, 84小时预报时长

### **3. Winnipeg河流水位监测**
- **数据源**: City of Winnipeg - Water and Waste Department
- **更新频率**: 实时更新
- **数据类型**: 河流水位监测数据
- **访问地址**:
  - **主页面**: https://legacy.winnipeg.ca/waterandwaste/flood/riverLevels.stm
  - **数据提供方**: Water Survey of Canada/Environment Canada
- **参考水位**: James Avenue Datum
- **监测站点**: James Avenue, Red River, Assiniboine River沿线
- **数据字段**: station_id, water_level, datum, timestamp, status
- **水位标准**: 正常夏季水位 6.5 feet, 步道最低水位 8.5 feet

### **4. OpenMeteo气象数据**
- **数据源**: OpenMeteo Canada
- **更新频率**: 定期更新
- **数据类型**: 气象观测数据
- **访问地址**: 
  - **数据文件**: data/real/openmeteo/openmeteo_canada_20250915_184349.csv
- **数据字段**: time, city, temperature_2m_max, temperature_2m_min, precipitation_sum, soil_moisture_0_to_7cm
- **覆盖城市**: 加拿大主要城市
- **时间范围**: 2024年

## 📅 **定期更新数据源**

### **5. Winnipeg水质监测数据**
- **数据源**: City of Winnipeg - Water and Waste Department
- **更新频率**: 年度报告
- **数据类型**: 饮用水质量检测
- **访问地址**:
  - **主页面**: https://legacy.winnipeg.ca/waterandwaste/water/testResults/default.stm
  - **2024年数据**: https://legacy.winnipeg.ca/waterandwaste/water/testResults/winnipeg.stm
  - **历史数据**: 2001-2024年各年度链接
- **合规标准**: 加拿大饮用水质量指南 (Guidelines for Canadian Drinking Water Quality)
- **监测点位**: 
  - Shoal Lake (水源地)
  - Water Treatment Plant raw (水处理厂进水)
  - Water Treatment Plant treated (水处理厂出水)
  - Winnipeg distribution system (配水系统)
- **监测参数**: turbidity, total_coliforms, e_coli, chlorine_residual, ph, total_dissolved_solids, hardness, iron, manganese
- **数据格式**: PDF报告 + 在线数据

### **6. SWE历史数据**
- **数据源**: Manitoba Hydro / NSIDC
- **更新频率**: 静态数据 (2010-2020)
- **数据类型**: 雪水当量历史数据
- **访问地址**:
  - **数据文件**: data/processed/validation/manitoba_daily_swe_*.csv
- **数据字段**: timestamp, swe_mm, valid_points, total_points
- **时间范围**: 2010-2020年
- **空间覆盖**: Manitoba省

## 🔧 **数据获取技术实现**

### **自动化数据获取**
```python
# 数据获取服务架构
class ManitobaDataCollector:
    def __init__(self):
        self.flood_alerts_collector = ManitobaFloodAlertsCollector()
        self.precipitation_collector = ManitobaPrecipitationForecastCollector()
        self.river_levels_collector = WinnipegRiverLevelsCollector()
        self.water_quality_collector = WinnipegWaterQualityCollector()
        self.openmeteo_collector = OpenMeteoDataCollector()
        self.swe_collector = SWEDataCollector()
```

### **数据更新机制**
- **实时数据**: 每15分钟检查更新
- **预报数据**: 每日4次自动获取
- **历史数据**: 每周检查新数据
- **水质数据**: 年度更新检查

### **数据质量保证**
- **数据验证**: 自动验证数据完整性和格式
- **错误处理**: 自动重试和错误报告
- **数据备份**: 定期备份历史数据
- **监控告警**: 数据获取失败自动告警

## 📊 **数据存储结构**

```
data/real/
├── manitoba_flood_alerts/
│   ├── current_alerts.csv
│   ├── current_alerts.geojson
│   └── alert_history/
├── manitoba_precipitation_forecast/
│   ├── current_forecast.csv
│   ├── current_forecast.geojson
│   └── forecast_history/
├── winnipeg_river_levels/
│   ├── current_levels.csv
│   └── historical_levels/
├── winnipeg_water_quality/
│   ├── current_data/
│   ├── historical_data/
│   └── processed_data/
├── openmeteo/
│   └── openmeteo_canada_*.csv
└── swe_data/
    └── manitoba_daily_swe_*.csv
```

## 🚨 **数据获取状态监控**

### **监控指标**
- **数据可用性**: > 99%
- **更新延迟**: < 5分钟
- **数据完整性**: 100%
- **错误率**: < 1%

### **告警机制**
- **数据获取失败**: 立即告警
- **数据延迟**: 超过15分钟告警
- **数据异常**: 自动检测并告警
- **服务中断**: 自动故障转移

## 📞 **联系信息**

### **数据提供方联系方式**
- **Manitoba省政府**: manitobamaps@gov.mb.ca
- **Environment and Climate Change Canada**: 通过Manitoba省政府联系
- **City of Winnipeg**: 通过311联系
- **OpenMeteo**: 通过官方API文档联系

### **技术支持**
- **API文档**: 完整的API文档和示例
- **数据字典**: 详细的数据字段说明
- **故障排除**: 技术支持服务
- **更新日志**: 数据源变更记录

## 🔄 **数据源更新日志**

### **2025-09-27**
- ✅ 添加Manitoba洪水预警数据源
- ✅ 添加RDPS降水预报数据源
- ✅ 添加Winnipeg河流水位数据源
- ✅ 添加Winnipeg水质监测数据源
- ✅ 确认OpenMeteo气象数据源
- ✅ 确认SWE历史数据源

### **未来更新计划**
- 🔄 扩展历史数据时间范围
- 🔄 增加更多监测站点
- 🔄 优化数据获取性能
- 🔄 增强数据质量监控

## 📋 **使用说明**

### **数据访问权限**
- **公开数据**: 所有数据源均为公开访问
- **使用限制**: 遵循各数据源的许可证要求
- **引用要求**: 使用数据时请正确引用数据源

### **数据使用建议**
- **实时数据**: 用于实时监测和预警
- **历史数据**: 用于趋势分析和模型训练
- **预报数据**: 用于短期预测和规划
- **综合数据**: 用于多因子分析和决策支持

---

**注意**: 本文档会随着新数据源的发现和现有数据源的更新而持续维护。请定期检查更新。

**维护者**: HydrAI-SWE开发团队  
**最后更新**: 2025-09-27  
**文档版本**: 1.0
