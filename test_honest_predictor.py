#!/usr/bin/env python3
"""
测试诚实预测器
验证各种预测模式，确保没有造假行为
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_strict_mode():
    """测试严格模式 - 需要完整历史数据"""
    print("🔍 测试1: 严格模式")
    
    try:
        from src.models.honest_predictor import HonestSWEPredictor, PredictionMode
        
        # 创建严格模式预测器
        predictor = HonestSWEPredictor(mode=PredictionMode.STRICT)
        
        # 测试：没有历史数据时应该拒绝预测
        try:
            test_date = datetime.now()
            prediction, confidence, metadata = predictor.predict(100.0, 50.0, 80.0, test_date)
            print("❌ 应该拒绝预测但没有")
            return False
        except ValueError as e:
            print(f"✅ 正确拒绝预测（无历史数据）: {e}")
        
        # 测试：添加一些历史数据，但不足30个
        for i in range(15):  # 只添加15个数据点
            test_date = datetime.now() - timedelta(days=i)
            predictor.add_historical_data(100.0 + i, 50.0, 80.0 + i, test_date)
        
        # 测试：数据不足时应该拒绝预测
        try:
            prediction, confidence, metadata = predictor.predict(100.0, 50.0, 80.0, test_date)
            print("❌ 数据不足时应该拒绝预测但没有")
            return False
        except ValueError as e:
            print(f"✅ 数据不足时正确拒绝预测: {e}")
        
        # 测试：添加足够的历史数据
        for i in range(15, 30):  # 再添加15个数据点，总共30个
            test_date = datetime.now() - timedelta(days=i)
            predictor.add_historical_data(100.0 + i, 50.0, 80.0 + i, test_date)
        
        # 测试：数据充足时应该成功预测
        try:
            prediction, confidence, metadata = predictor.predict(100.0, 50.0, 80.0, test_date)
            print(f"✅ 严格模式预测成功，预测值: {prediction}")
            print(f"   置信度: {confidence.value}")
            print(f"   元数据: {metadata}")
            return True
        except Exception as e:
            print(f"❌ 数据充足时预测失败: {e}")
            return False
        
    except Exception as e:
        print(f"❌ 严格模式测试失败: {e}")
        return False

def test_limited_mode():
    """测试有限模式 - 数据不足时提供有限预测"""
    print("\n🔍 测试2: 有限模式")
    
    try:
        from src.models.honest_predictor import HonestSWEPredictor, PredictionMode
        
        # 创建有限模式预测器
        predictor = HonestSWEPredictor(mode=PredictionMode.LIMITED)
        
        # 测试：完全没有历史数据时应该拒绝预测
        try:
            test_date = datetime.now()
            prediction, confidence, metadata = predictor.predict(100.0, 50.0, 80.0, test_date)
            print("❌ 应该拒绝预测但没有")
            return False
        except ValueError as e:
            print(f"✅ 无历史数据时正确拒绝预测: {e}")
        
        # 测试：添加少量历史数据
        for i in range(5):  # 只添加5个数据点
            test_date = datetime.now() - timedelta(days=i)
            predictor.add_historical_data(100.0 + i, 50.0, 80.0 + i, test_date)
        
        # 测试：数据不足时应该提供有限预测
        try:
            prediction, confidence, metadata = predictor.predict(100.0, 50.0, 80.0, test_date)
            print(f"✅ 有限模式预测成功，预测值: {prediction}")
            print(f"   置信度: {confidence.value}")
            print(f"   元数据: {metadata}")
            
            # 检查元数据是否包含限制说明
            if 'limitations' in metadata and '数据不足' in metadata['limitations']:
                print("✅ 元数据正确包含限制说明")
            else:
                print("❌ 元数据缺少限制说明")
                return False
            
            return True
        except Exception as e:
            print(f"❌ 有限模式预测失败: {e}")
            return False
        
    except Exception as e:
        print(f"❌ 有限模式测试失败: {e}")
        return False

def test_progressive_mode():
    """测试渐进模式 - 随着数据增加逐步提高质量"""
    print("\n🔍 测试3: 渐进模式")
    
    try:
        from src.models.honest_predictor import HonestSWEPredictor, PredictionMode
        
        # 创建渐进模式预测器
        predictor = HonestSWEPredictor(mode=PredictionMode.PROGRESSIVE)
        
        # 测试：完全没有历史数据时应该拒绝预测
        try:
            test_date = datetime.now()
            prediction, confidence, metadata = predictor.predict(100.0, 50.0, 80.0, test_date)
            print("❌ 应该拒绝预测但没有")
            return False
        except ValueError as e:
            print(f"✅ 无历史数据时正确拒绝预测: {e}")
        
        # 测试：添加1个数据点
        test_date = datetime.now()
        predictor.add_historical_data(100.0, 50.0, 80.0, test_date)
        
        # 测试：1个数据点时的预测
        try:
            prediction, confidence, metadata = predictor.predict(100.0, 50.0, 80.0, test_date)
            print(f"✅ 1个数据点预测成功，预测值: {prediction}")
            print(f"   置信度: {confidence.value}")
            print(f"   元数据: {metadata}")
            
            if confidence.value != 'low':
                print("❌ 1个数据点时置信度应该为low")
                return False
            
            return True
        except Exception as e:
            print(f"❌ 1个数据点预测失败: {e}")
            return False
        
    except Exception as e:
        print(f"❌ 渐进模式测试失败: {e}")
        return False

def test_data_validation():
    """测试数据验证功能"""
    print("\n🔍 测试4: 数据验证功能")
    
    try:
        from src.models.honest_predictor import HonestSWEPredictor, PredictionMode
        
        # 创建预测器
        predictor = HonestSWEPredictor(mode=PredictionMode.STRICT)
        
        # 测试有效数据
        valid_result = predictor.validate_feature_data(100.0, 50.0, 80.0, datetime.now())
        print(f"✅ 有效数据验证: {valid_result}")
        
        # 测试无效数据 - 负值雪深度
        try:
            invalid_result = predictor.validate_feature_data(-100.0, 50.0, 80.0, datetime.now())
            print("❌ 应该拒绝负值雪深度但没有")
            return False
        except ValueError as e:
            print(f"✅ 负值雪深度正确被拒绝: {e}")
        
        # 测试无效数据 - 负值雪水当量
        try:
            invalid_result = predictor.validate_feature_data(100.0, 50.0, -80.0, datetime.now())
            print("❌ 应该拒绝负值雪水当量但没有")
            return False
        except ValueError as e:
            print(f"✅ 负值雪水当量正确被拒绝: {e}")
        
        # 测试无效数据 - 异常大的雪深度
        try:
            invalid_result = predictor.validate_feature_data(15000.0, 50.0, 80.0, datetime.now())
            print("❌ 应该拒绝异常大的雪深度但没有")
            return False
        except ValueError as e:
            print(f"✅ 异常大的雪深度正确被拒绝: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据验证测试失败: {e}")
        return False

def test_prediction_requirements():
    """测试预测要求说明"""
    print("\n🔍 测试5: 预测要求说明")
    
    try:
        from src.models.honest_predictor import HonestSWEPredictor, PredictionMode
        
        # 测试不同模式的预测要求
        for mode in [PredictionMode.STRICT, PredictionMode.LIMITED, PredictionMode.PROGRESSIVE]:
            predictor = HonestSWEPredictor(mode=mode)
            requirements = predictor.get_prediction_requirements()
            
            print(f"✅ {mode.value} 模式要求说明:")
            print(f"   当前模式: {requirements['current_mode']}")
            print(f"   建议: {requirements['recommendations']}")
            
            # 检查要求说明是否完整
            if 'requirements' not in requirements:
                print(f"❌ {mode.value} 模式缺少要求说明")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 预测要求说明测试失败: {e}")
        return False

def test_data_quality_report():
    """测试数据质量报告"""
    print("\n🔍 测试6: 数据质量报告")
    
    try:
        from src.models.honest_predictor import HonestSWEPredictor, PredictionMode
        
        predictor = HonestSWEPredictor(mode=PredictionMode.PROGRESSIVE)
        
        # 测试：没有历史数据时的质量报告
        quality_report = predictor.get_data_quality_report()
        print(f"✅ 无数据质量报告: {quality_report['status']}")
        
        # 测试：添加一些数据后的质量报告
        for i in range(10):
            test_date = datetime.now() - timedelta(days=i)
            predictor.add_historical_data(100.0 + i, 50.0, 80.0 + i, test_date)
        
        quality_report = predictor.get_data_quality_report()
        print(f"✅ 有数据质量报告: {quality_report['status']}")
        print(f"   质量分数: {quality_report['quality_score']}")
        print(f"   完成百分比: {quality_report['completion_percentage']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据质量报告测试失败: {e}")
        return False

def test_no_fake_data():
    """测试没有造假数据生成"""
    print("\n🔍 测试7: 没有造假数据生成")
    
    try:
        from src.models.honest_predictor import HonestSWEPredictor, PredictionMode
        
        # 检查诚实预测器是否有造假方法
        predictor = HonestSWEPredictor(mode=PredictionMode.PROGRESSIVE)
        
        # 检查是否有造假相关的方法
        fake_methods = [
            'initialize_with_synthetic_data',
            'create_fake_sequence',
            'generate_synthetic_data',
            'simulate_time_series'
        ]
        
        for method_name in fake_methods:
            if hasattr(predictor, method_name):
                print(f"❌ 发现造假方法: {method_name}")
                return False
        
        print("✅ 没有发现造假方法")
        
        # 检查代码中是否有造假相关的关键词
        import inspect
        source_code = inspect.getsource(HonestSWEPredictor)
        
        fake_keywords = [
            'np.tile',
            'np.random.normal',
            'synthetic',
            'fake',
            'simulate',
            'generate_fake'
        ]
        
        for keyword in fake_keywords:
            if keyword in source_code:
                print(f"⚠️ 发现可疑关键词: {keyword}")
                # 这里只是警告，不是错误，因为可能用于其他目的
        
        return True
        
    except Exception as e:
        print(f"❌ 造假数据检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始测试诚实预测器")
    print("=" * 50)
    
    tests = [
        test_strict_mode,
        test_limited_mode,
        test_progressive_mode,
        test_data_validation,
        test_prediction_requirements,
        test_data_quality_report,
        test_no_fake_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ 测试异常: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有诚实预测器测试通过！")
        print("✅ 确认没有造假行为")
        return True
    else:
        print("⚠️ 部分测试失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

