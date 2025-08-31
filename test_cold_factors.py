#!/usr/bin/env python3
"""
冷因素发现算法测试脚本
验证修复后的算法逻辑
"""

import pandas as pd
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_cold_factors_discovery():
    """测试冷因素发现算法"""
    print("🔍 开始冷因素发现算法测试...")
    
    try:
        from src.models.data_science_analyzer import DataScienceAnalyzer
        
        # 创建测试数据
        print("📊 创建测试数据...")
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        np.random.seed(42)
        
        # 创建有意义的测试数据
        df = pd.DataFrame({
            'date': dates,
            'snow_water_equivalent_mm': np.random.normal(30, 10, len(dates)),  # 目标变量
            'temperature': np.random.normal(0, 15, len(dates)),  # 常见特征
            'humidity': np.random.normal(60, 20, len(dates)),    # 常见特征
            'pressure': np.random.normal(1013, 50, len(dates)),  # 常见特征
            'wind_speed': np.random.exponential(5, len(dates)),  # 稀有特征
            'solar_radiation': np.random.gamma(2, 100, len(dates)),  # 稀有特征
            'year': dates.year,  # 时间特征
            'month': dates.month,  # 时间特征
            'day_of_year': dates.dayofyear,  # 时间特征
        })
        df.set_index('date', inplace=True)
        
        print(f"📈 测试数据统计:")
        print(f"   长度: {len(df)}")
        print(f"   特征数: {len(df.columns)}")
        print(f"   目标变量: {df['snow_water_equivalent_mm'].mean():.2f} ± {df['snow_water_equivalent_mm'].std():.2f}")
        
        # 测试冷因素发现
        print("\n🔬 测试冷因素发现算法...")
        analyzer = DataScienceAnalyzer()
        analyzer.data = df
        
        result = analyzer.discover_cold_factors('snow_water_equivalent_mm', top_k=5)
        
        print(f"\n📊 冷因素发现结果:")
        print(f"   目标变量: {result['target']}")
        print(f"   前5个候选因素:")
        
        for i, (factor, score) in enumerate(result['top_candidates']):
            impact = result['impact_scores'].get(factor, 0)
            coldness = result['coldness_scores'].get(factor, 0)
            predictive = result['predictive_scores'].get(factor, 0)
            
            print(f"   {i+1}. {factor}:")
            print(f"      综合得分: {score:.4f}")
            print(f"      影响力: {impact:.4f}")
            print(f"      冷门度: {coldness:.4f}")
            print(f"      预测价值: {predictive:.4f}")
        
        # 检查时间特征是否被正确惩罚
        print(f"\n🔍 时间特征惩罚检查:")
        time_features = ['year', 'month', 'day_of_year']
        for tf in time_features:
            if tf in result['impact_scores']:
                impact = result['impact_scores'][tf]
                coldness = result['coldness_scores'][tf]
                predictive = result['predictive_scores'][tf]
                print(f"   {tf}: 影响力={impact:.4f}, 冷门度={coldness:.4f}, 预测价值={predictive:.4f}")
        
        # 检查稀有特征是否获得高分
        print(f"\n🔍 稀有特征检查:")
        rare_features = ['wind_speed', 'solar_radiation']
        for rf in rare_features:
            if rf in result['impact_scores']:
                impact = result['impact_scores'][rf]
                coldness = result['coldness_scores'][rf]
                predictive = result['predictive_scores'][rf]
                print(f"   {rf}: 影响力={impact:.4f}, 冷门度={coldness:.4f}, 预测价值={predictive:.4f}")
        
        return result
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_cold_factors_discovery()
    if result:
        print("\n✅ 冷因素发现算法测试完成")
    else:
        print("\n❌ 冷因素发现算法测试失败")


