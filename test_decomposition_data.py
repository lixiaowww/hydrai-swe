#!/usr/bin/env python3
"""
测试分解分析数据格式
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.data_science_analyzer import DataScienceAnalyzer

def test_decomposition_data():
    """测试分解分析数据格式"""
    print("🔍 测试分解分析数据格式...")
    print("=" * 50)
    
    # 创建分析器
    analyzer = DataScienceAnalyzer()
    
    # 加载数据
    data_path = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
    analyzer.load_data(data_path)
    
    if analyzer.data is None:
        print("❌ 数据加载失败")
        return
    
    print(f"✅ 数据加载成功: {len(analyzer.data)} 条记录")
    
    # 执行分解分析
    try:
        results = analyzer.advanced_time_series_decomposition('snow_water_equivalent_mm')
        print("✅ 分解分析完成")
        
        # 检查数据结构
        if 'stl_decomposition' in results:
            stl = results['stl_decomposition']
            print("\n📊 STL分解数据结构:")
            print(f"  - trend: {type(stl.get('trend'))}")
            if stl.get('trend'):
                print(f"    - trend.index: {type(stl['trend'].get('index'))}")
                print(f"    - trend.values: {type(stl['trend'].get('values'))}")
                print(f"    - trend.index长度: {len(stl['trend'].get('index', []))}")
                print(f"    - trend.values长度: {len(stl['trend'].get('values', []))}")
                print(f"    - trend.index前5个: {stl['trend'].get('index', [])[:5]}")
                print(f"    - trend.values前5个: {stl['trend'].get('values', [])[:5]}")
            
            print(f"  - seasonal: {type(stl.get('seasonal'))}")
            print(f"  - resid: {type(stl.get('resid'))}")
            print(f"  - seasonal_strength: {stl.get('seasonal_strength')}")
            print(f"  - trend_strength: {stl.get('trend_strength')}")
        
        # 检查解释数据
        if 'interpretation' in results:
            interpretation = results['interpretation']
            print("\n📝 解释数据结构:")
            print(f"  - summary: {interpretation.get('summary')}")
            print(f"  - key_insights: {len(interpretation.get('key_insights', []))}")
            print(f"  - business_implications: {interpretation.get('business_implications')}")
            print(f"  - recommendations: {len(interpretation.get('recommendations', []))}")
        
        # 保存结果用于前端测试
        import json
        with open('decomposition_test_data.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("\n💾 结果已保存到: decomposition_test_data.json")
        
    except Exception as e:
        print(f"❌ 分解分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_decomposition_data()
