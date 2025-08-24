#!/usr/bin/env python3
"""
HydrAI-SWE 预测验证器测试脚本
测试预测质量验证器和实时验证器的功能
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_prediction_validator():
    """测试预测质量验证器"""
    logger.info("🧪 开始测试预测质量验证器")
    
    try:
        # 导入验证器
        from src.models.validation.prediction_validator import PredictionQualityValidator
        
        # 创建验证器
        validator = PredictionQualityValidator()
        logger.info("✅ 预测质量验证器创建成功")
        
        # 生成测试数据
        np.random.seed(42)
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        
        # 测试1：正常的土壤湿度预测
        logger.info("\n" + "="*50)
        logger.info("测试1：正常的土壤湿度预测")
        
        normal_predictions = pd.DataFrame({
            'soil_moisture': np.random.uniform(0.1, 0.8, 100)
        }, index=dates)
        
        normal_result = validator.validate_prediction_quality(
            normal_predictions, 'soil_moisture', 'normal_model'
        )
        
        logger.info(f"正常预测验证结果:")
        logger.info(f"  有效性: {normal_result.is_valid}")
        logger.info(f"  置信度分数: {normal_result.confidence_score:.2%}")
        logger.info(f"  警告数量: {len(normal_result.warnings)}")
        logger.info(f"  错误数量: {len(normal_result.errors)}")
        
        # 测试2：异常的土壤湿度预测
        logger.info("\n" + "="*50)
        logger.info("测试2：异常的土壤湿度预测")
        
        abnormal_predictions = pd.DataFrame({
            'soil_moisture': np.random.uniform(-0.1, 1.2, 100)  # 包含负值和超过1的值
        }, index=dates)
        
        abnormal_result = validator.validate_prediction_quality(
            abnormal_predictions, 'soil_moisture', 'abnormal_model'
        )
        
        logger.info(f"异常预测验证结果:")
        logger.info(f"  有效性: {abnormal_result.is_valid}")
        logger.info(f"  置信度分数: {abnormal_result.confidence_score:.2%}")
        logger.info(f"  警告数量: {len(abnormal_result.warnings)}")
        logger.info(f"  错误数量: {len(abnormal_result.errors)}")
        
        # 测试3：多数据源一致性验证
        logger.info("\n" + "="*50)
        logger.info("测试3：多数据源一致性验证")
        
        source1_predictions = pd.DataFrame({
            'soil_moisture': np.random.uniform(0.2, 0.7, 50)
        }, index=dates[:50])
        
        source2_predictions = pd.DataFrame({
            'soil_moisture': np.random.uniform(0.2, 0.7, 50) + np.random.normal(0, 0.1, 50)
        }, index=dates[:50])
        
        multi_source_predictions = {
            'source1': source1_predictions,
            'source2': source2_predictions
        }
        
        multi_source_result = validator.validate_prediction_quality(
            multi_source_predictions, 'soil_moisture', 'multi_source'
        )
        
        logger.info(f"多源一致性验证结果:")
        logger.info(f"  有效性: {multi_source_result.is_valid}")
        logger.info(f"  置信度分数: {multi_source_result.confidence_score:.2%}")
        logger.info(f"  警告数量: {len(multi_source_result.warnings)}")
        logger.info(f"  错误数量: {len(multi_source_result.errors)}")
        
        # 生成验证报告
        logger.info("\n" + "="*50)
        logger.info("生成验证报告")
        
        normal_report = validator.generate_validation_report(normal_result)
        abnormal_report = validator.generate_validation_report(abnormal_result)
        multi_source_report = validator.generate_validation_report(multi_source_result)
        
        # 保存报告
        os.makedirs("test_validation_reports", exist_ok=True)
        
        with open("test_validation_reports/normal_validation_report.md", "w", encoding="utf-8") as f:
            f.write(normal_report)
        
        with open("test_validation_reports/abnormal_validation_report.md", "w", encoding="utf-8") as f:
            f.write(abnormal_report)
        
        with open("test_validation_reports/multi_source_validation_report.md", "w", encoding="utf-8") as f:
            f.write(multi_source_report)
        
        logger.info("✅ 预测质量验证器测试完成，报告已保存")
        return True
        
    except Exception as e:
        logger.error(f"❌ 预测质量验证器测试失败: {e}")
        return False

def test_real_time_validator():
    """测试实时验证器"""
    logger.info("\n🧪 开始测试实时验证器")
    
    try:
        # 导入验证器
        from src.models.validation.real_time_validator import RealTimeValidator
        
        # 创建验证器
        validator = RealTimeValidator()
        logger.info("✅ 实时验证器创建成功")
        
        # 生成参考数据
        np.random.seed(42)
        reference_dates = pd.date_range('2023-01-01', periods=1000, freq='H')
        reference_data = pd.DataFrame({
            'soil_moisture': np.random.uniform(0.1, 0.8, 1000)
        }, index=reference_dates)
        
        # 初始化参考分布
        validator.initialize_reference_distribution(reference_data)
        logger.info("✅ 参考分布初始化完成")
        
        # 测试实时验证
        logger.info("\n" + "="*50)
        logger.info("测试实时验证功能")
        
        for i in range(5):
            # 生成预测数据
            pred_dates = pd.date_range(f'2024-01-{i+1:02d}', periods=24, freq='H')
            predictions = pd.DataFrame({
                'soil_moisture': np.random.uniform(0.1, 0.8, 24)
            }, index=pred_dates)
            
            # 添加验证任务
            validator.add_validation_task(
                predictions, 'soil_moisture', 'test_model', f"test_pred_{i}"
            )
            
            logger.info(f"  已添加验证任务 {i+1}/5")
        
        # 等待任务处理
        import time
        logger.info("等待任务处理完成...")
        time.sleep(10)
        
        # 获取状态和结果
        status = validator.get_validation_status()
        recent_results = validator.get_recent_results(5)
        
        logger.info(f"实时验证状态:")
        logger.info(f"  队列大小: {status['queue_size']}")
        logger.info(f"  总验证数: {status['total_validations']}")
        logger.info(f"  监控状态: {status['active_monitoring']}")
        
        logger.info(f"最近验证结果:")
        for i, result in enumerate(recent_results):
            logger.info(f"  结果 {i+1}: 质量分数 {result.quality_score:.2%}, 有效: {result.is_valid}")
        
        # 停止监控
        validator.stop_monitoring()
        logger.info("✅ 实时验证器测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 实时验证器测试失败: {e}")
        return False

def test_api_integration():
    """测试API集成"""
    logger.info("\n🧪 开始测试API集成")
    
    try:
        # 导入API路由
        from src.api.routers.prediction_validation import router
        
        logger.info("✅ API路由导入成功")
        
        # 检查API端点
        routes = [route.path for route in router.routes]
        logger.info(f"可用的API端点:")
        for route in routes:
            logger.info(f"  {route}")
        
        # 测试数据模型
        from src.api.routers.prediction_validation import ValidationRequest, ValidationResponse
        
        # 创建测试请求
        test_predictions = [
            {'timestamp': '2024-01-01T00:00:00', 'soil_moisture': 0.5},
            {'timestamp': '2024-01-01T01:00:00', 'soil_moisture': 0.6},
            {'timestamp': '2024-01-01T02:00:00', 'soil_moisture': 0.4}
        ]
        
        test_request = ValidationRequest(
            predictions=test_predictions,
            variable_type='soil_moisture',
            source_name='test_api',
            prediction_id='test_001',
            include_historical_validation=True
        )
        
        logger.info(f"测试请求创建成功:")
        logger.info(f"  变量类型: {test_request.variable_type}")
        logger.info(f"  数据源: {test_request.source_name}")
        logger.info(f"  预测数量: {len(test_request.predictions)}")
        
        logger.info("✅ API集成测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ API集成测试失败: {e}")
        return False

def test_validation_workflow():
    """测试完整验证工作流"""
    logger.info("\n🧪 开始测试完整验证工作流")
    
    try:
        # 导入验证器
        from src.models.validation.prediction_validator import PredictionQualityValidator
        
        # 创建验证器
        validator = PredictionQualityValidator()
        
        # 模拟真实场景的验证工作流
        logger.info("模拟真实场景验证工作流...")
        
        # 1. 验证土壤湿度预测
        soil_moisture_data = pd.DataFrame({
            'soil_moisture': np.random.uniform(0.2, 0.7, 50)
        }, index=pd.date_range('2024-01-01', periods=50, freq='D'))
        
        soil_result = validator.validate_prediction_quality(
            soil_moisture_data, 'soil_moisture', 'agriculture_model'
        )
        
        # 2. 验证积雪水当量预测
        swe_data = pd.DataFrame({
            'snow_water_equivalent': np.random.uniform(0, 1500, 30)
        }, index=pd.date_range('2024-01-01', periods=30, freq='D'))
        
        swe_result = validator.validate_prediction_quality(
            swe_data, 'snow_water_equivalent', 'snow_model'
        )
        
        # 3. 验证径流预测
        runoff_data = pd.DataFrame({
            'runoff': np.random.uniform(0, 5000, 40)
        }, index=pd.date_range('2024-01-01', periods=40, freq='D'))
        
        runoff_result = validator.validate_prediction_quality(
            runoff_data, 'runoff', 'hydrology_model'
        )
        
        # 汇总结果
        logger.info("验证工作流结果汇总:")
        logger.info(f"  土壤湿度: 有效={soil_result.is_valid}, 分数={soil_result.confidence_score:.2%}")
        logger.info(f"  积雪水当量: 有效={swe_result.is_valid}, 分数={swe_result.confidence_score:.2%}")
        logger.info(f"  径流: 有效={runoff_result.is_valid}, 分数={runoff_result.confidence_score:.2%}")
        
        # 计算整体有效性
        overall_valid = all([soil_result.is_valid, swe_result.is_valid, runoff_result.is_valid])
        overall_score = (soil_result.confidence_score + swe_result.confidence_score + runoff_result.confidence_score) / 3
        
        logger.info(f"整体验证结果: 有效={overall_valid}, 平均分数={overall_score:.2%}")
        
        logger.info("✅ 完整验证工作流测试完成")
        return True
        
    except Exception as e:
        logger.error(f"❌ 完整验证工作流测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始HydrAI-SWE预测验证器全面测试")
    
    # 创建测试结果目录
    os.makedirs("test_results", exist_ok=True)
    
    # 记录测试开始时间
    start_time = datetime.now()
    
    # 执行测试
    test_results = {}
    
    # 测试1：预测质量验证器
    test_results['prediction_validator'] = test_prediction_validator()
    
    # 测试2：实时验证器
    test_results['real_time_validator'] = test_real_time_validator()
    
    # 测试3：API集成
    test_results['api_integration'] = test_api_integration()
    
    # 测试4：完整验证工作流
    test_results['validation_workflow'] = test_validation_workflow()
    
    # 生成测试报告
    end_time = datetime.now()
    duration = end_time - start_time
    
    test_summary = {
        'test_start_time': start_time.isoformat(),
        'test_end_time': end_time.isoformat(),
        'test_duration_seconds': duration.total_seconds(),
        'test_results': test_results,
        'overall_success': all(test_results.values()),
        'success_count': sum(test_results.values()),
        'total_tests': len(test_results)
    }
    
    # 保存测试报告
    with open("test_results/validation_test_summary.json", "w", encoding="utf-8") as f:
        json.dump(test_summary, f, indent=2, ensure_ascii=False, default=str)
    
    # 输出测试结果
    logger.info("\n" + "="*60)
    logger.info("🎯 测试结果汇总")
    logger.info("="*60)
    
    for test_name, result in test_results.items():
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n总体结果: {'✅ 全部通过' if test_summary['overall_success'] else '❌ 部分失败'}")
    logger.info(f"通过率: {test_summary['success_count']}/{test_summary['total_tests']}")
    logger.info(f"测试耗时: {duration.total_seconds():.1f} 秒")
    
    if test_summary['overall_success']:
        logger.info("\n🎉 所有测试通过！预测验证器功能正常")
    else:
        logger.info("\n⚠️ 部分测试失败，请检查相关功能")
    
    return test_summary['overall_success']

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("\n🛑 测试被用户中断")
        exit(1)
    except Exception as e:
        logger.error(f"❌ 测试执行失败: {e}")
        exit(1)
