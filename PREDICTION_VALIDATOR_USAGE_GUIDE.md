# HydrAI-SWE 预测验证器使用指南

**文档版本**: 1.0  
**创建日期**: 2025年8月22日  
**基于**: 预测验证器开发完成，测试通过  

## 🎯 概述

HydrAI-SWE预测验证器是一个完整的预测结果质量保证系统，确保生产环境中的预测结果可信、准确、符合物理约束。

### 核心功能
- **物理约束验证**: 检查预测值是否在合理范围内
- **统计异常检测**: 基于历史数据检测异常预测
- **多源一致性验证**: 验证不同数据源预测结果的一致性
- **实时质量监控**: 在线监控预测质量，实时告警
- **综合质量评估**: 多维度质量评分和建议

## 🏗️ 系统架构

### 组件结构
```
预测验证器系统
├── 预测质量验证器 (PredictionQualityValidator)
│   ├── 物理约束验证器 (PhysicalConstraintValidator)
│   ├── 统计异常检测器 (StatisticalAnomalyDetector)
│   └── 多源一致性验证器 (MultiSourceConsistencyValidator)
├── 实时监控验证器 (RealTimeValidator)
│   ├── 性能监控器 (PerformanceMonitor)
│   └── 数据漂移检测器 (DriftDetector)
└── API接口层 (prediction_validation.py)
    ├── 验证API端点
    ├── 实时监控API端点
    └── 批量验证API端点
```

### 支持的数据类型
- **土壤湿度** (soil_moisture): 0.0 - 1.0 m³/m³
- **积雪水当量** (snow_water_equivalent): 0.0 - 2000.0 mm
- **径流** (runoff): 0.0 - 10000.0 m³/s
- **温度** (temperature): -50.0 - 50.0 °C
- **降水量** (precipitation): 0.0 - 500.0 mm/day

## 🚀 快速开始

### 1. 基础使用

#### 创建验证器
```python
from src.models.validation.prediction_validator import PredictionQualityValidator

# 创建验证器实例
validator = PredictionQualityValidator()
```

#### 验证单个预测结果
```python
import pandas as pd
import numpy as np

# 准备预测数据
dates = pd.date_range('2024-01-01', periods=100, freq='D')
predictions = pd.DataFrame({
    'soil_moisture': np.random.uniform(0.1, 0.8, 100)
}, index=dates)

# 执行验证
result = validator.validate_prediction_quality(
    predictions=predictions,
    variable_type='soil_moisture',
    source_name='my_model'
)

# 查看结果
print(f"验证结果: {'有效' if result.is_valid else '无效'}")
print(f"置信度分数: {result.confidence_score:.2%}")
print(f"警告数量: {len(result.warnings)}")
print(f"错误数量: {len(result.errors)}")
```

### 2. 实时监控使用

#### 启动实时验证器
```python
from src.models.validation.real_time_validator import RealTimeValidator

# 创建实时验证器
real_time_validator = RealTimeValidator()

# 初始化参考分布（用于漂移检测）
reference_data = pd.DataFrame({
    'soil_moisture': np.random.uniform(0.1, 0.8, 1000)
})
real_time_validator.initialize_reference_distribution(reference_data)
```

#### 添加验证任务
```python
# 添加验证任务到队列
real_time_validator.add_validation_task(
    predictions=predictions,
    variable_type='soil_moisture',
    source_name='production_model',
    prediction_id='pred_001'
)

# 获取验证状态
status = real_time_validator.get_validation_status()
print(f"队列大小: {status['queue_size']}")
print(f"总验证数: {status['total_validations']}")

# 获取最近结果
recent_results = real_time_validator.get_recent_results(10)
for result in recent_results:
    print(f"预测ID: {result.prediction_id}, 质量分数: {result.quality_score:.2%}")
```

### 3. API接口使用

#### 启动API服务
```bash
# 确保API路由已集成到主应用中
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 验证预测结果
```bash
curl -X POST "http://localhost:8000/api/v1/prediction-validation/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {"timestamp": "2024-01-01T00:00:00", "soil_moisture": 0.5},
      {"timestamp": "2024-01-01T01:00:00", "soil_moisture": 0.6}
    ],
    "variable_type": "soil_moisture",
    "source_name": "api_test",
    "prediction_id": "api_pred_001",
    "include_historical_validation": true
  }'
```

#### 获取验证状态
```bash
# 健康检查
curl "http://localhost:8000/api/v1/prediction-validation/health"

# 获取验证指标
curl "http://localhost:8000/api/v1/prediction-validation/metrics"

# 获取实时验证状态
curl "http://localhost:8000/api/v1/prediction-validation/real-time/status"
```

## 📊 验证结果解读

### 验证结果结构
```python
@dataclass
class ValidationResult:
    is_valid: bool                    # 整体有效性
    confidence_score: float           # 置信度分数 (0.0-1.0)
    validation_details: Dict[str, Any] # 详细验证信息
    warnings: List[str]               # 警告信息
    errors: List[str]                 # 错误信息
    recommendations: List[str]        # 改进建议
    timestamp: datetime               # 验证时间
```

### 质量分数说明
- **0.9-1.0**: 优秀 - 预测质量很高，可以直接使用
- **0.7-0.9**: 良好 - 预测质量良好，建议轻微优化
- **0.5-0.7**: 中等 - 预测质量一般，建议优化
- **0.3-0.5**: 较差 - 预测质量较差，需要改进
- **0.0-0.3**: 很差 - 预测质量很差，不建议使用

### 常见警告和错误

#### 物理约束警告
- **超出范围值**: 预测值超出物理合理范围
- **异常跳跃**: 相邻时间点预测值变化过大
- **负值检测**: 检测到物理上不可能的负值

#### 统计异常警告
- **异常点检测**: 基于历史数据检测到统计异常
- **分布漂移**: 当前数据分布与历史参考分布差异过大

#### 一致性警告
- **多源差异**: 不同数据源预测结果差异过大
- **时间不匹配**: 不同数据源时间索引不匹配

## 🔧 高级配置

### 自定义物理约束
```python
from src.models.validation.prediction_validator import PhysicalConstraintValidator

# 创建自定义约束验证器
custom_validator = PhysicalConstraintValidator()

# 添加自定义约束
custom_validator.constraints['custom_variable'] = {
    'min': -100.0,
    'max': 100.0,
    'unit': 'custom_unit',
    'description': '自定义变量的物理约束'
}
```

### 调整异常检测参数
```python
from src.models.validation.prediction_validator import StatisticalAnomalyDetector

# 创建异常检测器，调整污染率阈值
anomaly_detector = StatisticalAnomalyDetector(contamination=0.05)  # 5%异常率

# 训练异常检测模型
anomaly_detector.fit(historical_data)
```

### 实时验证器配置
```python
# 自定义配置
config = {
    'performance_window_size': 200,      # 性能监控窗口大小
    'alert_threshold': 0.75,            # 告警阈值
    'reference_window_size': 2000,      # 参考分布窗口大小
    'drift_threshold': 0.15,            # 漂移检测阈值
    'validation_interval': 0.5,         # 验证间隔（秒）
    'save_interval': 30                 # 保存间隔（秒）
}

real_time_validator = RealTimeValidator(config)
```

## 📈 性能优化

### 批量验证
```python
# 使用批量验证API提高效率
batch_requests = [
    ValidationRequest(
        predictions=pred1,
        variable_type='soil_moisture',
        source_name='model1'
    ),
    ValidationRequest(
        predictions=pred2,
        variable_type='soil_moisture',
        source_name='model2'
    )
]

# 启动批量验证
response = await client.post(
    "/api/v1/prediction-validation/batch-validate",
    json=batch_requests
)
```

### 异步处理
```python
import asyncio

async def validate_multiple_predictions():
    tasks = []
    for i in range(10):
        task = asyncio.create_task(
            validate_single_prediction(f"pred_{i}")
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

### 缓存优化
```python
# 缓存历史数据，避免重复加载
@lru_cache(maxsize=128)
def get_historical_data(variable_type: str):
    # 加载历史数据的逻辑
    pass
```

## 🚨 故障排除

### 常见问题

#### 1. 验证器初始化失败
```python
# 错误：ModuleNotFoundError: No module named 'src.models.validation'
# 解决：确保Python路径正确
import sys
sys.path.append('/path/to/hydrai_swe')
```

#### 2. 数据格式错误
```python
# 错误：DataFrame索引类型不匹配
# 解决：确保时间索引格式一致
predictions.index = pd.to_datetime(predictions.index)
```

#### 3. 内存不足
```python
# 错误：内存不足导致验证失败
# 解决：分批处理大数据集
chunk_size = 1000
for i in range(0, len(large_dataset), chunk_size):
    chunk = large_dataset[i:i+chunk_size]
    result = validator.validate_prediction_quality(chunk, ...)
```

#### 4. 实时验证器停止响应
```python
# 错误：实时验证器无响应
# 解决：检查监控线程状态
status = real_time_validator.get_validation_status()
if not status['active_monitoring']:
    # 重新启动监控
    real_time_validator = RealTimeValidator()
```

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查验证器状态
print(f"预测验证器: {prediction_validator is not None}")
print(f"实时验证器: {real_time_validator is not None}")

# 检查数据格式
print(f"预测数据形状: {predictions.shape}")
print(f"数据类型: {predictions.dtypes}")
print(f"索引类型: {type(predictions.index)}")
```

## 📚 最佳实践

### 1. 数据预处理
- 确保时间索引格式一致
- 处理缺失值和异常值
- 标准化数值范围

### 2. 验证策略
- 根据业务需求调整质量阈值
- 定期更新参考分布
- 建立验证结果监控仪表板

### 3. 性能监控
- 监控验证处理时间
- 跟踪验证成功率
- 设置合理的告警阈值

### 4. 错误处理
- 实现优雅的错误处理
- 记录详细的错误日志
- 提供用户友好的错误信息

## 🔗 集成指南

### 集成到现有系统
```python
# 在现有预测流程中添加验证
class PredictionPipeline:
    def __init__(self):
        self.validator = PredictionQualityValidator()
    
    def predict_and_validate(self, input_data):
        # 执行预测
        predictions = self.model.predict(input_data)
        
        # 验证预测结果
        validation_result = self.validator.validate_prediction_quality(
            predictions, 'soil_moisture', 'pipeline_model'
        )
        
        # 根据验证结果决定是否返回预测
        if validation_result.is_valid:
            return predictions, validation_result
        else:
            raise ValueError(f"预测验证失败: {validation_result.errors}")
```

### 集成到监控系统
```python
# 集成到Prometheus监控
from prometheus_client import Gauge, Counter

# 定义指标
validation_success_gauge = Gauge('prediction_validation_success', 'Prediction validation success rate')
validation_duration_gauge = Gauge('prediction_validation_duration', 'Prediction validation duration')

# 在验证完成后更新指标
validation_success_gauge.set(result.confidence_score)
validation_duration_gauge.set(validation_time)
```

## 📞 技术支持

### 获取帮助
- **文档**: 查看本文档和相关代码注释
- **测试**: 运行 `python3 test_prediction_validator.py` 验证功能
- **日志**: 检查日志文件了解详细错误信息
- **社区**: 在项目GitHub页面提交Issue

### 报告问题
报告问题时请提供：
1. 错误信息和堆栈跟踪
2. 使用的数据格式和大小
3. 系统环境信息
4. 复现步骤

---

**总结**: HydrAI-SWE预测验证器提供了完整的预测结果质量保证解决方案，通过物理约束验证、统计异常检测、多源一致性验证和实时监控，确保生产环境中的预测结果可信可靠。
