# HydrAI-SWE 项目文档索引

**文档版本**: 2.1  
**最后更新**: 2025年8月20日  
**项目状态**: 农业模块集成完成，系统功能全面扩展  

## 📚 核心文档

### **项目概述文档**
- [README.md](README.md) - 项目总览和快速开始指南
- [PROJECT_STATUS_SUMMARY.md](PROJECT_STATUS_SUMMARY.md) - 项目状态总结 (最新更新)
- [PROJECT_DEVELOPMENT_PROGRESS_REPORT.md](PROJECT_DEVELOPMENT_PROGRESS_REPORT.md) - 项目开发进度报告 (最新更新)

### **技术规范文档**
- [API_SPECIFICATION.md](API_SPECIFICATION.md) - API接口规范
- [INTERFACE_ORGANIZATION.md](INTERFACE_ORGANIZATION.md) - 界面组织结构
- [ENHANCED_UI_EN.md](ENHANCED_UI_EN.md) - 增强英文界面说明

## 🌱 农业模块文档 (新增)

### **农业功能文档**
- [AGRICULTURE_DEVELOPMENT_ROADMAP.md](AGRICULTURE_DEVELOPMENT_ROADMAP.md) - 农业模块开发路线图
- [AGRICULTURE_MODULE_RESEARCH_REPORT.md](AGRICULTURE_MODULE_RESEARCH_REPORT.md) - 农业模块调研报告
- [docs/6_agriculture_functional_specification.md](docs/6_agriculture_functional_specification.md) - 农业功能规范文档

### **农业模块代码**
- [src/models/agriculture/soil_moisture_predictor.py](src/models/agriculture/soil_moisture_predictor.py) - 土壤水分预测器
- [src/api/routers/agriculture.py](src/api/routers/agriculture.py) - 农业模块API路由
- [test_agriculture_integration.py](test_agriculture_integration.py) - 农业模块集成测试

## 🏗️ 系统架构文档

### **核心系统**
- [docs/1_project_requirements_document.md](docs/1_project_requirements_document.md) - 项目需求文档
- [docs/2_functional_specification_document.md](docs/2_functional_specification_document.md) - 功能规范文档
- [docs/3_technical_specification_document.md](docs/3_technical_specification_document.md) - 技术规格文档

### **数据处理**
- [docs/4_data_acquisition_and_management_plan.md](docs/4_data_acquisition_and_management_plan.md) - 数据采集和管理计划
- [GEOGRAPHIC_REGIONS.md](GEOGRAPHIC_REGIONS.md) - 地理区域指南
- [DATA_ACQUISITION.md](DATA_ACQUISITION.md) - 数据获取说明

## 🧠 机器学习和训练文档

### **模型训练**
- [TRAINING_README.md](TRAINING_README.md) - 训练流程指南
- [train_swe_model.py](train_swe_model.py) - SWE模型训练脚本
- [run_full_training.py](run_full_training.py) - 完整训练管道
- [run_full_training_with_high_resolution.py](run_full_training_with_high_resolution.py) - 高分辨率训练管道

### **模型训练报告**
- [PROJECT_MODEL_TRAINING_REPORT.md](PROJECT_MODEL_TRAINING_REPORT.md) - 模型训练状态报告

## 🌊 应用功能文档

### **洪水预警系统**
- [src/models/flood_risk_assessment.py](src/models/flood_risk_assessment.py) - 洪水风险评估模型
- [src/api/routers/flood_warning.py](src/api/routers/flood_warning.py) - 洪水预警API
- [test_flood_warning_system.py](test_flood_warning_system.py) - 洪水预警系统测试

### **交叉验证系统**
- [src/models/flood_risk_cross_validation.py](src/models/flood_risk_cross_validation.py) - 交叉验证模型
- [src/api/routers/cross_validation.py](src/api/routers/cross_validation.py) - 交叉验证API
- [test_cross_validation_system.py](test_cross_validation_system.py) - 交叉验证系统测试

## 🔬 研究和分析文档

### **SWE分析系统**
- [src/models/swe_analysis_system.py](src/models/swe_analysis_system.py) - SWE综合分析系统
- [climate_change_analysis.py](climate_change_analysis.py) - 气候变化影响分析
- [swe_analysis_modules.json](swe_analysis_modules.json) - SWE分析模块搜索结果

### **高分辨率数据集成**
- [high_resolution_integration_status.md](high_resolution_integration_status.md) - 高分辨率数据集成状态
- [high_resolution_weight_analysis.py](high_resolution_weight_analysis.py) - 高分辨率权重分析
- [high_resolution_data_analysis.py](high_resolution_data_analysis.py) - 高分辨率数据分析

## 🧪 测试和验证文档

### **系统测试**
- [test_frontend_performance.py](test_frontend_performance.py) - 前端性能测试
- [test_date_fix.py](test_date_fix.py) - 日期逻辑测试
- [test_server.py](test_server.py) - 服务器测试

### **数据测试**
- [test_data_format.py](test_data_format.py) - 数据格式测试
- [debug_data_sources.py](debug_data_sources.py) - 数据源调试
- [data_analysis_report.py](data_analysis_report.py) - 数据分析报告

## 🚀 部署和运维文档

### **部署指南**
- [quick_start.sh](quick_start.sh) - 快速启动脚本
- [start_enhanced_ui.py](start_enhanced_ui.py) - 增强界面启动脚本
- [Dockerfile](Dockerfile) - Docker容器配置

### **运维文档**
- [AUTO_UPDATE_SCRIPT.md](AUTO_UPDATE_SCRIPT.md) - 自动更新脚本说明
- [WARP.md](WARP.md) - 项目包装说明

## 📊 成功案例和解决方案

### **成功案例**
- [SUCCESS_CASE_SOLUTIONS.md](SUCCESS_CASE_SOLUTIONS.md) - 成功案例解决方案
- [explain_relationship.py](explain_relationship.py) - 关系解释脚本
- [explain_year_range_significance.py](explain_year_range_significance.py) - 年份范围意义解释

### **演示脚本**
- [demo_smart_data_selection.py](demo_smart_data_selection.py) - 智能数据选择演示
- [demo_smart_date_selection.py](demo_smart_date_selection.py) - 智能日期选择演示

## 🔍 搜索和研究工具

### **研究工具**
- [search_research_cases.py](search_research_cases.py) - 研究案例搜索
- [search_swe_analysis_modules.py](search_swe_analysis_modules.py) - SWE分析模块搜索
- [research_cases_results.json](research_cases_results.json) - 研究案例搜索结果

## 📁 目录结构

```
hydrai_swe/
├── docs/                           # 技术文档
├── src/                            # 源代码
│   ├── api/                        # API路由
│   │   └── routers/               # 路由模块
│   │       ├── swe.py             # SWE预测API
│   │       ├── flood_warning.py   # 洪水预警API
│   │       ├── cross_validation.py # 交叉验证API
│   │       └── agriculture.py     # 农业模块API ← 新增
│   ├── models/                     # 模型定义
│   │   ├── agriculture/           # 农业模块 ← 新增
│   │   │   └── soil_moisture_predictor.py
│   │   ├── flood_risk_assessment.py
│   │   └── swe_analysis_system.py
│   └── neuralhydrology/           # NeuralHydrology集成
├── templates/                      # 前端模板
│   └── ui/                        # 用户界面
│       ├── enhanced_en.html       # 增强英文界面
│       ├── enhanced_fr.html       # 法语界面
│       └── enhanced_cr.html       # 克里语界面
├── agriculture_integration/        # 农业模块集成 ← 新增
│   ├── SoilWeatherPredictor/      # GitHub项目
│   └── crop_yield_prediction/     # GitHub项目
└── 测试和文档文件...
```

## 📅 文档更新历史

### **2025年8月20日 - 版本2.1**
- ✅ 农业模块集成完成
- ✅ 新增农业功能文档
- ✅ 更新项目状态和进度报告
- ✅ 完善技术架构文档

### **2025年8月17日 - 版本2.0**
- ✅ 洪水预警系统完成
- ✅ 交叉验证系统完成
- ✅ 高分辨率数据集成
- ✅ 多语言界面支持

### **2025年7月 - 版本1.0**
- ✅ 核心系统架构完成
- ✅ SWE预测系统完成
- ✅ 径流预测系统完成
- ✅ 基础用户界面完成

## 🔗 快速链接

### **核心功能**
- [🌐 用户界面](http://localhost:8000/ui) - 主要用户界面
- [📚 API文档](http://localhost:8000/docs) - 完整API文档
- [🔍 健康检查](http://localhost:8000/api/v1/agriculture/health) - 农业模块状态

### **开发资源**
- [🚀 快速开始](README.md#quick-start) - 项目启动指南
- [🧪 测试运行](test_agriculture_integration.py) - 农业模块测试
- [📊 项目状态](PROJECT_STATUS_SUMMARY.md) - 当前项目状态

---

**文档维护**: 每次重要更新后自动更新  
**最后更新**: 2025年8月20日  
**维护状态**: 活跃维护中
