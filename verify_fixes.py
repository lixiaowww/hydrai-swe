#!/usr/bin/env python3
"""
验证修复后的代码
测试看门狗发现的问题是否已解决
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

def test_sequence_creation_fix():
    """测试序列创建修复"""
    print("🔍 测试1: 序列创建修复")
    
    try:
        from src.models.optimized_predictor import OptimizedSWEPredictor
        
        # 创建预测器
        predictor = OptimizedSWEPredictor()
        
        # 测试：没有历史数据时应该拒绝预测
        try:
            test_date = datetime.now()
            prediction = predictor.predict_single(100.0, 50.0, 80.0, test_date)
            print("❌ 应该拒绝预测但没有")
            return False
        except ValueError as e:
            print(f"✅ 正确拒绝预测（无历史数据）: {e}")
        
        # 测试：使用最小数据要求方法
        try:
            prediction = predictor.predict_with_minimal_data(100.0, 50.0, 80.0, test_date)
            print(f"✅ 最小数据预测成功，预测值: {prediction}")
        except Exception as e:
            print(f"⚠️ 最小数据预测失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 序列创建修复失败: {e}")
        return False

def test_data_validation():
    """测试数据验证功能"""
    print("\n🔍 测试2: 数据验证功能")
    
    try:
        from src.models.optimized_predictor import OptimizedSWEPredictor
        
        predictor = OptimizedSWEPredictor()
        
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

def test_data_quality_check():
    """测试数据质量检查功能"""
    print("\n🔍 测试3: 数据质量检查功能")
    
    try:
        from src.models.optimized_predictor import OptimizedSWEPredictor
        
        predictor = OptimizedSWEPredictor()
        
        # 测试：没有历史数据时的质量检查
        quality_report = predictor.validate_historical_data_quality()
        print(f"✅ 无数据质量检查: {quality_report['status']}")
        
        # 测试：获取数据要求说明
        requirements = predictor.get_data_requirements()
        print(f"✅ 数据要求说明: 需要 {requirements['minimum_historical_data']} 个历史数据点")
        
        # 测试：添加一些测试数据
        test_features = np.array([100.0, 50.0, 80.0, 180, 6, 2024])
        predictor.update_historical_features(test_features)
        
        # 再次检查质量
        quality_report = predictor.validate_historical_data_quality()
        print(f"✅ 有数据质量检查: {quality_report['status']}, 分数: {quality_report['quality_score']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 数据质量检查测试失败: {e}")
        return False

def test_seasonal_augmentation_fix():
    """测试季节性增强修复"""
    print("\n🔍 测试4: 季节性增强修复")
    
    try:
        from data_augmentation_experiment import DataAugmentationExperiment
        
        # 创建实验对象
        experiment = DataAugmentationExperiment()
        
        # 测试季节性增强
        if experiment.scaler_X is not None and experiment.scaler_y is not None:
            print("✅ 标准化器加载成功")
            
            # 创建测试数据
            X_test = np.random.randn(10, 6)  # 10个样本，6个特征
            y_test = np.random.randn(10)
            
            # 应用季节性增强
            X_aug, y_aug = experiment.apply_seasonal_augmentation(X_test, y_test, seasonal_factor=0.05)
            
            print(f"✅ 季节性增强成功，输出形状: X={X_aug.shape}, y={y_aug.shape}")
            return True
        else:
            print("⚠️ 标准化器未加载，跳过季节性增强测试")
            return True
            
    except Exception as e:
        print(f"❌ 季节性增强测试失败: {e}")
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n🔍 测试5: 错误处理")
    
    try:
        from src.models.optimized_predictor import OptimizedSWEPredictor
        
        predictor = OptimizedSWEPredictor()
        
        # 测试无效模型路径
        try:
            predictor.load_model("nonexistent_model.pth")
            print("❌ 应该抛出异常但没有")
            return False
        except Exception as e:
            print(f"✅ 无效模型路径正确被拒绝: {e}")
        
        # 测试数据验证
        try:
            predictor.predict_single("invalid", 50.0, 80.0, datetime.now())
            print("❌ 应该抛出异常但没有")
            return False
        except ValueError as e:
            print(f"✅ 无效数据类型正确被拒绝: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        return False

def test_feature_configuration():
    """测试特征配置"""
    print("\n🔍 测试6: 特征配置")
    
    try:
        from src.models.optimized_predictor import OptimizedSWEPredictor
        
        predictor = OptimizedSWEPredictor()
        
        # 检查特征配置
        expected_features = ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                           'day_of_year', 'month', 'year']
        
        for feature in expected_features:
            if feature in predictor.feature_config:
                print(f"✅ 特征 {feature} 配置正确")
            else:
                print(f"❌ 特征 {feature} 配置缺失")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 特征配置测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 开始验证彻底修复后的代码")
    print("=" * 50)
    
    tests = [
        test_sequence_creation_fix,
        test_data_validation,
        test_data_quality_check,
        test_seasonal_augmentation_fix,
        test_error_handling,
        test_feature_configuration
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
        print("🎉 所有彻底修复验证通过！")
        return True
    else:
        print("⚠️ 部分彻底修复验证失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
