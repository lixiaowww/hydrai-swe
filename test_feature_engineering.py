#!/usr/bin/env python3
"""
测试特征工程，找出为什么风险等级总是LOW
"""
import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.advanced_flood_warning import advanced_flood_system

def test_feature_engineering():
    """测试特征工程"""
    
    print("🔍 测试特征工程...")
    
    # 加载修复后的测试数据
    data_file = "data/processed/flood_warning/flood_warning_fixed_features.csv"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return
    
    print(f"📁 加载数据: {data_file}")
    data = pd.read_csv(data_file)
    print(f"✅ 数据加载成功: {data.shape}")
    
    # 显示前几行数据
    print("\n📋 原始数据预览:")
    print(data.head())
    
    # 准备特征
    print("\n🔧 准备高级特征...")
    try:
        features_data = advanced_flood_system.prepare_advanced_features(data)
        print(f"✅ 特征准备成功: {features_data.shape}")
        
        # 显示特征统计
        print("\n📊 特征统计:")
        print(features_data.describe())
        
        # 检查是否有无穷值或NaN
        print("\n🔍 检查数据质量:")
        print(f"无穷值数量: {(features_data == np.inf).sum().sum()}")
        print(f"NaN数量: {features_data.isna().sum().sum()}")
        
        # 预测风险
        print("\n🎯 预测风险...")
        prediction_result = advanced_flood_system.predict_advanced_flood_risk(features_data)
        
        print(f"风险等级: {prediction_result['risk_level'][:10]}")
        print(f"风险概率: {prediction_result['risk_probability'][:10]}")
        
        # 分析为什么都是LOW
        risk_levels = prediction_result['risk_level']
        risk_probs = prediction_result['risk_probability']
        
        print(f"\n📊 风险分析:")
        print(f"总样本数: {len(risk_levels)}")
        print(f"高风险样本数: {sum(risk_levels)}")
        print(f"低风险样本数: {len(risk_levels) - sum(risk_levels)}")
        print(f"平均风险概率: {np.mean(risk_probs):.4f}")
        print(f"最大风险概率: {np.max(risk_probs):.4f}")
        print(f"最小风险概率: {np.min(risk_probs):.4f}")
        
        # 检查模型阈值
        print(f"\n🔍 模型阈值分析:")
        print(f"模型类型: {type(advanced_flood_system.model)}")
        
        # 如果是RandomForest，检查特征重要性
        if hasattr(advanced_flood_system.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': advanced_flood_system.feature_names,
                'importance': advanced_flood_system.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"\n📈 特征重要性:")
            print(feature_importance.head(10))
        
    except Exception as e:
        print(f"❌ 特征工程失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_engineering()
