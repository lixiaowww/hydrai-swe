#!/usr/bin/env python3
"""
测试图表数据格式的脚本
"""

import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from models.data_science_analyzer import DataScienceAnalyzer

def test_chart_data_format():
    """测试图表数据格式"""
    print("🔍 测试图表数据格式...")
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
            
            # 检查trend数据
            if 'trend' in stl and stl['trend']:
                trend = stl['trend']
                print(f"  - trend: {type(trend)}")
                print(f"    - trend.index: {type(trend.get('index'))}")
                print(f"    - trend.values: {type(trend.get('values'))}")
                print(f"    - trend.index长度: {len(trend.get('index', []))}")
                print(f"    - trend.values长度: {len(trend.get('values', []))}")
                print(f"    - trend.index前5个: {trend.get('index', [])[:5]}")
                print(f"    - trend.values前5个: {trend.get('values', [])[:5]}")
                
                # 验证数据类型
                index_sample = trend.get('index', [])[:3]
                values_sample = trend.get('values', [])[:3]
                print(f"    - index类型检查: {[type(x) for x in index_sample]}")
                print(f"    - values类型检查: {[type(x) for x in values_sample]}")
                
                # 检查是否为数组
                print(f"    - index是数组: {isinstance(trend.get('index'), list)}")
                print(f"    - values是数组: {isinstance(trend.get('values'), list)}")
                
                # 检查数组长度
                if isinstance(trend.get('index'), list) and isinstance(trend.get('values'), list):
                    print(f"    - 数组长度匹配: {len(trend.get('index', [])) == len(trend.get('values', []))}")
            
            # 检查seasonal数据
            if 'seasonal' in stl and stl['seasonal']:
                seasonal = stl['seasonal']
                print(f"  - seasonal: {type(seasonal)}")
                print(f"    - seasonal.index长度: {len(seasonal.get('index', []))}")
                print(f"    - seasonal.values长度: {len(seasonal.get('values', []))}")
            
            # 检查resid数据
            if 'resid' in stl and stl['resid']:
                resid = stl['resid']
                print(f"  - resid: {type(resid)}")
                print(f"    - resid.index长度: {len(resid.get('index', []))}")
                print(f"    - resid.values长度: {len(resid.get('values', []))}")
            
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
        
        # 创建简化的测试数据
        test_data = {
            "stl_decomposition": {
                "trend": {
                    "index": ["1979-01-01", "1979-01-02", "1979-01-03"],
                    "values": [18.6, 18.7, 18.8]
                },
                "seasonal": {
                    "index": ["1979-01-01", "1979-01-02", "1979-01-03"],
                    "values": [0.1, 0.2, 0.3]
                },
                "resid": {
                    "index": ["1979-01-01", "1979-01-02", "1979-01-03"],
                    "values": [-0.1, -0.2, -0.3]
                }
            }
        }
        
        with open('simple_test_data.json', 'w') as f:
            json.dump(test_data, f, indent=2)
        print("💾 简化测试数据已保存到: simple_test_data.json")
        
    except Exception as e:
        print(f"❌ 分解分析失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chart_data_format()
