# Manitoba 实时水文数据 API 使用指南

## 🌊 概述
`http://localhost:8000/api/manitoba-flood/` 是整合了实时曼省水文数据的主要API接口，提供最新的洪水风险评估和水文监测数据。

## 📍 重要说明
- ✅ **活跃**: `http://localhost:8000/api/manitoba-flood/` - **这是您需要的最新实时数据**
- ❌ **已废弃**: `http://localhost:8000/ui/flood-warning` - 不再使用

## 🔧 主要端点

### 1. 实时风险评估
```bash
curl http://localhost:8000/api/manitoba-flood/real-time-risk
```
**返回**: 当前洪水风险等级、概率、多语言描述和建议

### 2. 洪水时间线
```bash
curl http://localhost:8000/api/manitoba-flood/flood-timeline
```
**返回**: 历史和预测的洪水风险时间序列数据

### 3. 风险评估报告
```bash
curl http://localhost:8000/api/manitoba-flood/risk-assessment
```
**返回**: 详细的风险评估分析

### 4. 健康检查
```bash
curl http://localhost:8000/api/manitoba-flood/health
```
**返回**: API状态和模型加载情况

## 📊 数据特色

### 实时数据整合
- **ECCC气象数据**: Environment and Climate Change Canada
- **HYDAT流量数据**: 加拿大水文数据库
- **机器学习模型**: 随机森林分类器进行风险预测
- **多语言支持**: 中文、英文、法文

### 数据示例
```json
{
  "status": "success",
  "timestamp": "2025-08-24T23:04:18.690197",
  "data_date": "2024-12-31T00:00:00",
  "current_risk": {
    "level": "LOW",
    "probability": 0.8,
    "description": {
      "zh": "洪水风险低，正常状态",
      "en": "Low flood risk, normal status", 
      "fr": "Risque d'inondation faible, état normal"
    }
  }
}
```

## 🚀 使用示例

### Python 示例
```python
import requests

# 获取实时风险
response = requests.get("http://localhost:8000/api/manitoba-flood/real-time-risk")
risk_data = response.json()

print(f"风险等级: {risk_data['current_risk']['level']}")
print(f"风险概率: {risk_data['current_risk']['probability']}")
```

### JavaScript 示例
```javascript
fetch('http://localhost:8000/api/manitoba-flood/real-time-risk')
  .then(response => response.json())
  .then(data => {
    console.log('风险等级:', data.current_risk.level);
    console.log('风险概率:', data.current_risk.probability);
  });
```

## 📈 系统架构

### 数据流
1. **实时数据收集** → ECCC气象站 + HYDAT水文站
2. **数据处理** → 特征工程 + 异常检测
3. **模型预测** → 随机森林分类器
4. **风险评级** → LOW/MODERATE/HIGH/CRITICAL
5. **API输出** → JSON格式实时数据

### 更新频率
- **实时风险**: 每15分钟更新
- **时间线数据**: 每小时更新
- **模型重训练**: 每周一次

## 🛠 技术规格

- **框架**: FastAPI + scikit-learn
- **数据库**: 实时流处理
- **响应时间**: < 500ms
- **数据覆盖**: 曼尼托巴省 + 红河流域
- **预测范围**: 1-30天

---

**系统状态**: 🟢 正常运行  
**最后更新**: 2025-08-25 04:05 UTC  
**联系**: lixiaowww@gmail.com
