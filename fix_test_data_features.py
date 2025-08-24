#!/usr/bin/env python3
"""
修复测试数据特征名称，使其与模型期望的特征匹配
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def fix_test_data_features():
    """修复测试数据特征名称"""
    
    # 加载测试数据
    test_data_file = "data/processed/flood_warning/flood_warning_test_data.csv"
    if not os.path.exists(test_data_file):
        print(f"❌ 测试数据文件不存在: {test_data_file}")
        return False
    
    print("📁 加载测试数据...")
    data = pd.read_csv(test_data_file)
    print(f"✅ 数据加载成功: {data.shape}")
    
    # 修复特征名称，使其与模型期望的匹配
    print("🔧 修复特征名称...")
    
    # 重命名列以匹配模型期望
    column_mapping = {
        '05OC001': '05OC001_x',
        '05OC011': '05OC011_y', 
        '05OC012': '05OC012_y'
    }
    
    data = data.rename(columns=column_mapping)
    
    # 确保所有必需的特征都存在
    required_features = [
        'Month', '05OC001_x', '05OC001_y', '05OC011_y', '05OC012_y',
        'DayOfYear', 'WeekOfYear', 'day_of_year_sin', 'day_of_year_cos',
        'month_sin', 'month_cos', 'temp_anomaly', 'flow_change',
        'flow_volatility', 'flow_peak', 'flow_corr_2_3', 'flow_corr_2_4',
        'flow_corr_2_5', 'flow_corr_3_5', 'flow_corr_4_5'
    ]
    
    # 检查缺失的特征
    missing_features = [f for f in required_features if f not in data.columns]
    print(f"🔍 缺失特征: {missing_features}")
    
    # 添加缺失的特征
    for feature in missing_features:
        if feature == '05OC001_x' and '05OC001' in data.columns:
            data['05OC001_x'] = data['05OC001']
        elif feature == '05OC011_y' and '05OC011' in data.columns:
            data['05OC011_y'] = data['05OC011']
        elif feature == '05OC012_y' and '05OC012' in data.columns:
            data['05OC012_y'] = data['05OC012']
        else:
            # 用0填充缺失特征
            data[feature] = 0
            print(f"⚠️  用0填充缺失特征: {feature}")
    
    # 确保数据类型正确
    print("🔧 修复数据类型...")
    
    # 处理无穷值和NaN
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        # 替换无穷值
        data[col] = data[col].replace([np.inf, -np.inf], np.nan)
        # 用0填充NaN
        data[col] = data[col].fillna(0)
    
    # 确保所有特征都是数值型
    for feature in required_features:
        if feature in data.columns:
            data[feature] = pd.to_numeric(data[feature], errors='coerce').fillna(0)
    
    # 验证特征
    print("✅ 验证特征...")
    available_features = [f for f in required_features if f in data.columns]
    print(f"📊 可用特征数量: {len(available_features)}")
    print(f"📊 数据形状: {data.shape}")
    
    # 保存修复后的数据
    output_file = "data/processed/flood_warning/flood_warning_fixed_features.csv"
    data.to_csv(output_file, index=False)
    print(f"💾 修复后的数据已保存: {output_file}")
    
    # 显示前几行数据
    print("\n📋 修复后的数据预览:")
    print(data[required_features].head())
    
    return True

if __name__ == "__main__":
    print("🚀 开始修复测试数据特征...")
    fix_test_data_features()
