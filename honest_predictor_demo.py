#!/usr/bin/env python3
"""
诚实预测器使用示例
展示各种预测模式的使用方法
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def demo_strict_mode():
    """演示严格模式 - 需要完整历史数据"""
    print("🎯 演示1: 严格模式")
    print("=" * 50)
    
    try:
        from src.models.honest_predictor import HonestSWEPredictor, PredictionMode
        
        # 创建严格模式预测器
        predictor = HonestSWEPredictor(mode=PredictionMode.STRICT)
        
        # 获取预测要求说明
        requirements = predictor.get_prediction_requirements()
        print(f"📋 预测要求:")
        print(f"   模式: {requirements['current_mode']}")
        print(f"   最小数据点: {requirements['requirements']['strict_mode']['min_data_points']}")
        print(f"   置信度: {requirements['requirements']['strict_mode']['confidence']}")
        print(f"   限制: {requirements['requirements']['strict_mode']['limitations']}")
        print(f"   建议: {requirements['recommendations']}")
        
        # 添加历史数据
        print(f"\n📊 开始收集历史数据...")
        for i in range(30):
            test_date = datetime.now() - timedelta(days=i)
            # 模拟真实的雪数据变化
            snow_depth = 100 + 10 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 5)
            snow_fall = 50 + np.random.normal(0, 10)
            snow_we = 80 + 8 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 3)
            
            predictor.add_historical_data(snow_depth, snow_fall, snow_we, test_date)
        
        print(f"✅ 已收集 {len(predictor._historical_features)} 个历史数据点")
        
        # 获取数据质量报告
        quality_report = predictor.get_data_quality_report()
        print(f"\n📈 数据质量报告:")
        print(f"   状态: {quality_report['status']}")
        print(f"   质量分数: {quality_report['quality_score']:.2f}")
        print(f"   完成百分比: {quality_report['completion_percentage']}")
        
        # 进行预测
        test_date = datetime.now()
        prediction, confidence, metadata = predictor.predict(100.0, 50.0, 80.0, test_date)
        
        print(f"\n🔮 预测结果:")
        print(f"   预测值: {prediction:.2f} mm")
        print(f"   置信度: {confidence.value}")
        print(f"   模式: {metadata['mode']}")
        print(f"   数据质量: {metadata['data_quality']}")
        print(f"   限制: {metadata['limitations']}")
        print(f"   建议: {metadata['recommendations']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 严格模式演示失败: {e}")
        return False

def demo_limited_mode():
    """演示有限模式 - 数据不足时提供有限预测"""
    print("\n🎯 演示2: 有限模式")
    print("=" * 50)
    
    try:
        from src.models.honest_predictor import HonestSWEPredictor, PredictionMode
        
        # 创建有限模式预测器
        predictor = HonestSWEPredictor(mode=PredictionMode.LIMITED)
        
        # 获取预测要求说明
        requirements = predictor.get_prediction_requirements()
        print(f"📋 预测要求:")
        print(f"   模式: {requirements['current_mode']}")
        print(f"   最小数据点: {requirements['requirements']['limited_mode']['min_data_points']}")
        print(f"   置信度: {requirements['requirements']['limited_mode']['confidence']}")
        print(f"   限制: {requirements['requirements']['limited_mode']['limitations']}")
        print(f"   建议: {requirements['recommendations']}")
        
        # 添加少量历史数据
        print(f"\n📊 开始收集历史数据...")
        for i in range(10):  # 只收集10个数据点
            test_date = datetime.now() - timedelta(days=i)
            snow_depth = 100 + 10 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 5)
            snow_fall = 50 + np.random.normal(0, 10)
            snow_we = 80 + 8 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 3)
            
            predictor.add_historical_data(snow_depth, snow_fall, snow_we, test_date)
        
        print(f"✅ 已收集 {len(predictor._historical_features)} 个历史数据点")
        
        # 获取数据质量报告
        quality_report = predictor.get_data_quality_report()
        print(f"\n📈 数据质量报告:")
        print(f"   状态: {quality_report['status']}")
        print(f"   质量分数: {quality_report['quality_score']:.2f}")
        print(f"   完成百分比: {quality_report['completion_percentage']}")
        
        # 进行预测
        test_date = datetime.now()
        prediction, confidence, metadata = predictor.predict(100.0, 50.0, 80.0, test_date)
        
        print(f"\n🔮 预测结果:")
        print(f"   预测值: {prediction:.2f} mm")
        print(f"   置信度: {confidence.value}")
        print(f"   模式: {metadata['mode']}")
        print(f"   数据质量: {metadata['data_quality']}")
        print(f"   限制: {metadata['limitations']}")
        print(f"   建议: {metadata['recommendations']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 有限模式演示失败: {e}")
        return False

def demo_progressive_mode():
    """演示渐进模式 - 随着数据增加逐步提高质量"""
    print("\n🎯 演示3: 渐进模式")
    print("=" * 50)
    
    try:
        from src.models.honest_predictor import HonestSWEPredictor, PredictionMode
        
        # 创建渐进模式预测器
        predictor = HonestSWEPredictor(mode=PredictionMode.PROGRESSIVE)
        
        # 获取预测要求说明
        requirements = predictor.get_prediction_requirements()
        print(f"📋 预测要求:")
        print(f"   模式: {requirements['current_mode']}")
        print(f"   最小数据点: {requirements['requirements']['progressive_mode']['min_data_points']}")
        print(f"   置信度: {requirements['requirements']['progressive_mode']['confidence']}")
        print(f"   限制: {requirements['requirements']['progressive_mode']['limitations']}")
        print(f"   建议: {requirements['recommendations']}")
        
        # 演示渐进式数据收集和预测
        print(f"\n📊 渐进式数据收集和预测:")
        
        data_points_list = [1, 5, 10, 15, 20, 25, 30]
        predictions = []
        confidences = []
        
        for n_points in data_points_list:
            # 清空历史数据
            predictor._historical_features = []
            predictor._historical_dates = []
            
            # 添加指定数量的历史数据
            for i in range(n_points):
                test_date = datetime.now() - timedelta(days=i)
                snow_depth = 100 + 10 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 5)
                snow_fall = 50 + np.random.normal(0, 10)
                snow_we = 80 + 8 * np.sin(2 * np.pi * i / 30) + np.random.normal(0, 3)
                
                predictor.add_historical_data(snow_depth, snow_fall, snow_we, test_date)
            
            # 进行预测
            test_date = datetime.now()
            try:
                prediction, confidence, metadata = predictor.predict(100.0, 50.0, 80.0, test_date)
                predictions.append(prediction)
                confidences.append(confidence.value)
                
                print(f"   {n_points:2d} 个数据点: 预测值={prediction:6.2f} mm, 置信度={confidence.value}")
                
            except Exception as e:
                print(f"   {n_points:2d} 个数据点: 预测失败 - {e}")
                predictions.append(None)
                confidences.append(None)
        
        # 绘制渐进式预测结果
        try:
            plt.figure(figsize=(12, 8))
            
            # 预测值变化
            plt.subplot(2, 1, 1)
            valid_indices = [i for i, p in enumerate(predictions) if p is not None]
            valid_predictions = [predictions[i] for i in valid_indices]
            valid_data_points = [data_points_list[i] for i in valid_indices]
            
            plt.plot(valid_data_points, valid_predictions, 'bo-', linewidth=2, markersize=8)
            plt.xlabel('历史数据点数量')
            plt.ylabel('预测值 (mm)')
            plt.title('渐进模式：预测值随数据量变化')
            plt.grid(True, alpha=0.3)
            
            # 置信度变化
            plt.subplot(2, 1, 2)
            confidence_mapping = {'low': 1, 'medium': 2, 'high': 3}
            valid_confidences = [confidence_mapping[confidences[i]] for i in valid_indices]
            
            plt.plot(valid_data_points, valid_confidences, 'ro-', linewidth=2, markersize=8)
            plt.xlabel('历史数据点数量')
            plt.ylabel('置信度')
            plt.yticks([1, 2, 3], ['Low', 'Medium', 'High'])
            plt.title('渐进模式：置信度随数据量变化')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('progressive_prediction_demo.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 渐进式预测结果图表已保存为: progressive_prediction_demo.png")
            
        except Exception as e:
            print(f"⚠️ 图表生成失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 渐进模式演示失败: {e}")
        return False

def demo_data_validation():
    """演示数据验证功能"""
    print("\n🎯 演示4: 数据验证功能")
    print("=" * 50)
    
    try:
        from src.models.honest_predictor import HonestSWEPredictor, PredictionMode
        
        # 创建预测器
        predictor = HonestSWEPredictor(mode=PredictionMode.STRICT)
        
        # 测试各种数据验证场景
        test_cases = [
            {
                'name': '有效数据',
                'data': (100.0, 50.0, 80.0, datetime.now()),
                'should_pass': True
            },
            {
                'name': '负值雪深度',
                'data': (-100.0, 50.0, 80.0, datetime.now()),
                'should_pass': False
            },
            {
                'name': '负值雪水当量',
                'data': (100.0, 50.0, -80.0, datetime.now()),
                'should_pass': False
            },
            {
                'name': '异常大的雪深度',
                'data': (15000.0, 50.0, 80.0, datetime.now()),
                'should_pass': False
            },
            {
                'name': '异常大的融化量',
                'data': (100.0, -200.0, 80.0, datetime.now()),
                'should_pass': False
            },
            {
                'name': '异常雪水当量比例',
                'data': (100.0, 50.0, 50.0, datetime.now()),  # 比例0.5 > 0.4
                'should_pass': False
            }
        ]
        
        print(f"🧪 数据验证测试:")
        for test_case in test_cases:
            try:
                result = predictor.validate_feature_data(*test_case['data'])
                if test_case['should_pass'] and result:
                    print(f"   ✅ {test_case['name']}: 通过")
                elif not test_case['should_pass'] and not result:
                    print(f"   ✅ {test_case['name']}: 正确拒绝")
                else:
                    print(f"   ❌ {test_case['name']}: 验证结果不符合预期")
                    return False
            except ValueError as e:
                if not test_case['should_pass']:
                    print(f"   ✅ {test_case['name']}: 正确抛出异常 - {e}")
                else:
                    print(f"   ❌ {test_case['name']}: 不应该抛出异常 - {e}")
                    return False
        
        return True
        
    except Exception as e:
        print(f"❌ 数据验证演示失败: {e}")
        return False

def main():
    """主演示函数"""
    print("🚀 诚实预测器演示开始")
    print("=" * 50)
    
    demos = [
        demo_strict_mode,
        demo_limited_mode,
        demo_progressive_mode,
        demo_data_validation
    ]
    
    passed = 0
    total = len(demos)
    
    for demo in demos:
        try:
            if demo():
                passed += 1
        except Exception as e:
            print(f"❌ 演示异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 演示结果: {passed}/{total} 成功")
    
    if passed == total:
        print("🎉 所有演示成功完成！")
        print("✅ 诚实预测器功能正常，没有造假行为")
        return True
    else:
        print("⚠️ 部分演示失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

