#!/usr/bin/env python3
"""
测试洪水预测模块的数据流程
"""

import pandas as pd
import numpy as np
import os
import sys

# 添加src目录到路径
sys.path.append('src')

from api.routers.flood_warning import FloodWarningService

def test_data_flow():
    """测试数据流程"""
    print("🔍 测试洪水预测模块数据流程")
    print("=" * 50)
    
    # 1. 检查数据文件
    weather_path = "data/raw/eccc_recent/eccc_recent_combined.csv"
    flow_path = "data/processed/hydat_streamflow_processed.csv"
    
    print(f"📁 检查数据文件:")
    print(f"   天气数据: {weather_path} - {'✅ 存在' if os.path.exists(weather_path) else '❌ 不存在'}")
    print(f"   径流数据: {flow_path} - {'✅ 存在' if os.path.exists(flow_path) else '❌ 不存在'}")
    
    if not os.path.exists(weather_path) or not os.path.exists(flow_path):
        print("❌ 数据文件缺失，无法继续测试")
        return
    
    # 2. 加载数据
    print(f"\n📊 加载数据:")
    try:
        weather_data = pd.read_csv(weather_path)
        flow_data = pd.read_csv(flow_path)
        print(f"   天气数据: {weather_data.shape[0]} 行, {weather_data.shape[1]} 列")
        print(f"   径流数据: {flow_data.shape[0]} 行, {flow_data.shape[1]} 列")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 3. 检查日期范围
    print(f"\n📅 检查日期范围:")
    weather_data['Date/Time'] = pd.to_datetime(weather_data['Date/Time'])
    flow_data['date'] = pd.to_datetime(flow_data['date'])
    
    print(f"   天气数据: {weather_data['Date/Time'].min()} 到 {weather_data['Date/Time'].max()}")
    print(f"   径流数据: {flow_data['date'].min()} 到 {flow_data['date'].max()}")
    
    # 4. 检查数据合并
    print(f"\n🔗 测试数据合并:")
    try:
        merged_data = pd.merge(
            weather_data, 
            flow_data, 
            left_on='Date/Time', 
            right_on='date', 
            how='inner'
        )
        print(f"   合并后数据: {merged_data.shape[0]} 行, {merged_data.shape[1]} 列")
        print(f"   合并成功率: {merged_data.shape[0] / min(weather_data.shape[0], flow_data.shape[0]) * 100:.1f}%")
        
        if merged_data.empty:
            print("❌ 数据合并失败，没有匹配的日期")
            return
            
    except Exception as e:
        print(f"❌ 数据合并失败: {e}")
        return
    
    # 5. 测试特征准备
    print(f"\n⚙️ 测试特征准备:")
    try:
        flood_service = FloodWarningService()
        
        # 检查模型加载
        if flood_service.model is None:
            print("❌ 模型未加载")
            return
        
        print(f"✅ 模型加载成功: {type(flood_service.model).__name__}")
        print(f"   特征数量: {len(flood_service.feature_names)}")
        
        # 准备特征
        features_data = flood_service.prepare_features(weather_data, flow_data)
        print(f"✅ 特征准备成功: {features_data.shape[0]} 行, {features_data.shape[1]} 列")
        print(f"   特征列: {list(features_data.columns)}")
        
        # 检查特征数据质量
        print(f"\n📈 特征数据质量:")
        print(f"   NaN值数量: {features_data.isna().sum().sum()}")
        print(f"   无穷值数量: {np.isinf(features_data.values).sum()}")
        print(f"   零值数量: {(features_data == 0).sum().sum()}")
        
        # 6. 测试预测
        print(f"\n🎯 测试预测:")
        try:
            prediction_result = flood_service.predict_flood_risk(features_data)
            print(f"✅ 预测成功")
            print(f"   风险等级数量: {len(prediction_result['risk_level'])}")
            print(f"   风险概率数量: {len(prediction_result['risk_probability'])}")
            
            # 统计风险分布
            risk_levels = prediction_result['risk_level']
            high_risk_count = sum(1 for x in risk_levels if x == 1)
            total_count = len(risk_levels)
            
            print(f"   高风险样本: {high_risk_count}/{total_count} ({high_risk_count/total_count*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            
    except Exception as e:
        print(f"❌ 特征准备失败: {e}")
        return
    
    print(f"\n✅ 数据流程测试完成")

if __name__ == "__main__":
    test_data_flow()
