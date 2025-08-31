#!/usr/bin/env python3
"""
STL分解调试脚本
检查数据格式和分解结果
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_stl_decomposition():
    """测试STL分解"""
    print("🔍 开始STL分解调试...")
    
    try:
        from src.models.data_science_analyzer import DataScienceAnalyzer
        
        # 创建测试数据
        print("📊 创建测试数据...")
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        np.random.seed(42)
        
        # 创建有趋势和季节性的测试数据
        trend = np.linspace(10, 50, len(dates))  # 上升趋势
        seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)  # 年季节性
        noise = np.random.normal(0, 5, len(dates))  # 噪声
        
        test_data = trend + seasonal + noise
        
        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'snow_water_equivalent_mm': test_data
        })
        df.set_index('date', inplace=True)
        
        print(f"📈 测试数据统计:")
        print(f"   长度: {len(df)}")
        print(f"   均值: {df['snow_water_equivalent_mm'].mean():.2f}")
        print(f"   标准差: {df['snow_water_equivalent_mm'].std():.2f}")
        print(f"   范围: {df['snow_water_equivalent_mm'].min():.2f} - {df['snow_water_equivalent_mm'].max():.2f}")
        
        # 保存测试数据
        test_file = "test_stl_data.csv"
        df.to_csv(test_file)
        print(f"💾 测试数据已保存到: {test_file}")
        
        # 测试STL分解
        print("\n🔬 测试STL分解...")
        analyzer = DataScienceAnalyzer()
        analyzer.data = df
        
        # 直接调用STL分解
        series = df['snow_water_equivalent_mm']
        result = analyzer._stl_decomposition(series)
        
        print(f"\n📊 STL分解结果:")
        print(f"   趋势数据长度: {len(result['trend']['values'])}")
        print(f"   趋势数据范围: {min(result['trend']['values']):.2f} - {max(result['trend']['values']):.2f}")
        print(f"   季节性数据范围: {min(result['seasonal']['values']):.2f} - {max(result['seasonal']['values']):.2f}")
        print(f"   残差数据范围: {min(result['resid']['values']):.2f} - {max(result['resid']['values']):.2f}")
        print(f"   季节性强度: {result['seasonal_strength']:.3f}")
        print(f"   趋势强度: {result['trend_strength']:.3f}")
        
        # 检查数据格式
        print(f"\n🔍 数据格式检查:")
        print(f"   趋势数据类型: {type(result['trend']['values'])}")
        print(f"   趋势数据前5个值: {result['trend']['values'][:5]}")
        print(f"   趋势索引前5个值: {result['trend']['index'][:5]}")
        
        return result
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_stl_decomposition()
    if result:
        print("\n✅ STL分解测试完成")
    else:
        print("\n❌ STL分解测试失败")
