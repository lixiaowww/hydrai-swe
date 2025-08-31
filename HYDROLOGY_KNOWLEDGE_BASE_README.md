# HydrAI-SWE 水文知识库系统

## 🌊 概述

HydrAI-SWE水文知识库系统是一个专业的、基于科学的水文知识库，专门为雪水当量(SWE)分析和水文预测提供深度解读。该系统集成了：

- **专业知识库**: 涵盖SWE、融雪过程、径流生成等核心概念
- **区域背景**: 曼尼托巴省和红河流域的特定水文特征
- **气候变化**: 全球和区域气候变化对水文系统的影响
- **技术解读**: 基于数据的专业水文解读模板

## 🎯 核心特性

### 1. 专业知识库
- **SWE基础知识**: 定义、测量方法、物理属性
- **融雪过程**: 能量平衡、融雪速率、影响因素
- **径流生成**: 机制、SWE贡献、时间关系
- **预测方法**: 统计、物理、机器学习方法

### 2. 区域特定知识
- **曼尼托巴省气候**: 三个气候带的详细特征
- **红河流域**: 地形、土壤、水文特征
- **历史事件**: 1997年和2009年洪水事件分析
- **区域挑战**: 平坦地形、粘土土壤、排水困难

### 3. 气候变化背景
- **全球趋势**: 温度、降水、积雪覆盖变化
- **区域影响**: 曼尼托巴省特定变化预测
- **适应策略**: 防洪、预警、管理建议

### 4. 专业解读系统
- **趋势分析**: 统计显著性、变化幅度、置信度
- **季节性模式**: 峰值时间、模式识别、水文意义
- **异常检测**: Z-score分析、异常程度、风险评估
- **管理建议**: 基于数据的专业建议

## 🏗️ 系统架构

```
src/
├── knowledge/
│   └── hydrology_knowledge_base.py    # 核心知识库
├── api/routers/
│   └── enhanced_interpretation.py     # 增强解读服务
└── test_knowledge_base.py             # 测试脚本
```

### 核心组件

#### 1. HydrologyKnowledgeBase
- 初始化专业知识库
- 提供区域背景信息
- 生成专业解读模板
- 支持多语言输出

#### 2. EnhancedInterpretationService
- 趋势显著性分析
- 季节性模式检测
- 异常分数计算
- 数据质量评估
- 专业解读生成

#### 3. API端点
- `/api/v1/interpretation/swe-comprehensive`: 综合SWE解读
- `/api/v1/interpretation/quick-assessment`: 快速评估
- `/api/v1/interpretation/knowledge-base/glossary`: 技术词汇表
- `/api/v1/interpretation/knowledge-base/climate-context`: 气候变化信息
- `/api/v1/interpretation/knowledge-base/regional-info`: 区域信息

## 🚀 使用方法

### 1. 基本使用

```python
from src.knowledge.hydrology_knowledge_base import HydrologyKnowledgeBase

# 创建知识库实例
kb = HydrologyKnowledgeBase()

# 获取SWE解读
interpretation = kb.get_swe_interpretation(
    trend_direction="increasing",
    trend_magnitude=25.5,
    seasonal_pattern="early_peak",
    anomaly_score=2.3,
    forecast_confidence=0.85
)

print(interpretation['trend_analysis']['description'])
print(interpretation['management_recommendations'])
```

### 2. API使用

```bash
# 获取技术词汇表
curl http://localhost:8000/api/v1/interpretation/knowledge-base/glossary

# 快速SWE评估
curl -X POST http://localhost:8000/api/v1/interpretation/quick-assessment \
  -H "Content-Type: application/json" \
  -d '{"values": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]}'

# 综合解读
curl -X POST http://localhost:8000/api/v1/interpretation/swe-comprehensive \
  -H "Content-Type: application/json" \
  -d '{
    "values": [50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
    "timestamps": ["2024-01-01", "2024-01-02", "2024-01-03"],
    "region": "manitoba"
  }'
```

### 3. 集成到现有系统

```python
from src.api.routers.enhanced_interpretation import EnhancedInterpretationService

# 创建服务实例
service = EnhancedInterpretationService()

# 分析数据
trend_analysis = service.analyze_trend_significance(values, timestamps)
seasonal_patterns = service.detect_seasonal_patterns(values, timestamps)
anomaly_score = service.calculate_anomaly_score(values)
data_quality = service.assess_data_quality(values, timestamps)

# 生成专业解读
interpretation = service.generate_professional_interpretation(
    trend_analysis, seasonal_patterns, anomaly_score, data_quality
)
```

## 📊 解读示例

### 趋势分析解读

**输入**: 增加趋势，25.5%变化，早期峰值，异常分数2.3

**输出**:
```json
{
  "trend_analysis": {
    "description": "Strong increasing trend in SWE indicates significant accumulation patterns that may reflect climate change impacts or natural variability cycles.",
    "magnitude": "25.5%",
    "strength": "strong",
    "implications": [
      "Potential for increased spring flood risk",
      "Enhanced water storage for summer months",
      "Possible climate change signal",
      "Need for updated flood protection standards"
    ]
  },
  "anomaly_assessment": {
    "description": "Extremely high SWE values represent significant departure from normal conditions.",
    "score": 2.3,
    "severity": "extreme"
  },
  "management_recommendations": [
    "Monitor flood protection infrastructure capacity",
    "Update flood risk assessments and emergency plans",
    "Consider adaptive management strategies for changing conditions"
  ]
}
```

## 🔬 技术特点

### 1. 科学性
- 基于水文学原理和科学文献
- 统计显著性检验
- 物理一致性检查
- 专家知识集成

### 2. 专业性
- 水文专业术语
- 区域特定知识
- 气候变化考虑
- 管理建议生成

### 3. 可扩展性
- 模块化设计
- 知识库可扩展
- 模板可定制
- 多语言支持

### 4. 实用性
- 直接API接口
- 快速评估服务
- 详细解读报告
- 管理建议

## 🧪 测试

运行测试脚本验证系统功能：

```bash
python test_knowledge_base.py
```

测试包括：
- 知识库基本功能
- 解读服务功能
- API端点功能

## 📚 知识来源

### 1. 科学文献
- 水文学教科书和期刊
- 气候变化研究报告
- 区域水文研究

### 2. 专家知识
- 水文专家咨询
- 区域经验总结
- 历史事件分析

### 3. 数据驱动
- 统计分析结果
- 模型验证经验
- 实际应用反馈

## 🔄 更新和维护

### 1. 知识库更新
- 定期更新科学发现
- 集成新的研究结果
- 更新区域信息

### 2. 解读模板优化
- 基于用户反馈
- 性能评估结果
- 新需求集成

### 3. 质量保证
- 专家审查
- 用户测试
- 持续改进

## 🌟 优势

### 1. 专业性
- 基于水文学原理
- 专家知识集成
- 科学严谨性

### 2. 实用性
- 直接应用价值
- 管理建议生成
- 风险评估支持

### 3. 可扩展性
- 模块化设计
- 知识库可扩展
- 多区域支持

### 4. 可靠性
- 统计验证
- 物理一致性
- 专家审查

## 🎯 应用场景

### 1. 水文预测
- SWE趋势分析
- 季节性模式识别
- 异常情况预警

### 2. 洪水管理
- 风险评估
- 预警系统
- 应急响应

### 3. 水资源管理
- 水量评估
- 供需平衡
- 规划决策

### 4. 气候变化适应
- 趋势分析
- 影响评估
- 适应策略

## 📞 支持

如有问题或建议，请联系：
- 项目维护者: Sean Li
- 邮箱: lixiaowww@gmail.com
- 项目地址: https://github.com/lixiaowww/hydrai-swe

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

---

**注意**: 本知识库系统基于科学原理和专家知识构建，但解读结果仅供参考。实际应用时请结合具体情况和专业判断。
