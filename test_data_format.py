#!/usr/bin/env python3
"""
测试数据格式脚本
检查NeuralHydrology数据准备是否正确
"""

import pandas as pd
import os

def test_data_format():
    """测试数据格式"""
    print("🔍 测试数据格式...")
    
    # 检查训练数据
    train_file = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
    if os.path.exists(train_file):
        print(f"\n✅ 训练数据文件存在: {train_file}")
        
        try:
            df = pd.read_csv(train_file)
            print(f"  记录数: {len(df)}")
            print(f"  列名: {df.columns.tolist()}")
            
            # 检查日期列
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                print(f"  日期范围: {df['date'].min()} 到 {df['date'].max()}")
                
                # 检查是否有重复日期
                duplicates = df['date'].duplicated().sum()
                print(f"  重复日期: {duplicates}")
                
                # 检查索引类型
                df_indexed = df.set_index('date')
                print(f"  索引类型: {type(df_indexed.index)}")
                print(f"  索引是否为MultiIndex: {isinstance(df_indexed.index, pd.MultiIndex)}")
                
                # 尝试推断频率
                try:
                    freq = pd.infer_freq(df_indexed.index)
                    print(f"  推断频率: {freq}")
                except Exception as e:
                    print(f"  频率推断错误: {e}")
                
            else:
                print("  ❌ 缺少date列")
                
        except Exception as e:
            print(f"  ❌ 读取训练数据失败: {e}")
    else:
        print(f"\n❌ 训练数据文件不存在: {train_file}")
    
    # 检查评估数据
    eval_file = "src/neuralhydrology/data/red_river_basin/timeseries_eval.csv"
    if os.path.exists(eval_file):
        print(f"\n✅ 评估数据文件存在: {eval_file}")
        
        try:
            df = pd.read_csv(eval_file)
            print(f"  记录数: {len(df)}")
            print(f"  列名: {df.columns.tolist()}")
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                print(f"  日期范围: {df['date'].min()} 到 {df['date'].max()}")
                
        except Exception as e:
            print(f"  ❌ 读取评估数据失败: {e}")
    else:
        print(f"\n❌ 评估数据文件不存在: {eval_file}")
    
    # 检查配置文件
    config_file = "src/neuralhydrology/config.yml"
    if os.path.exists(config_file):
        print(f"\n✅ 配置文件存在: {config_file}")
        
        try:
            with open(config_file, 'r') as f:
                content = f.read()
                print(f"  配置大小: {len(content)} 字符")
                print("  配置内容预览:")
                for line in content.split('\n')[:10]:
                    print(f"    {line}")
        except Exception as e:
            print(f"  ❌ 读取配置文件失败: {e}")
    else:
        print(f"\n❌ 配置文件不存在: {config_file}")

def test_neuralhydrology_import():
    """测试NeuralHydrology导入"""
    print("\n🔍 测试NeuralHydrology导入...")
    
    try:
        import neuralhydrology
        print(f"  ✅ NeuralHydrology版本: {neuralhydrology.__version__}")
        
        # 测试基本功能
        from neuralhydrology.datautils.utils import infer_frequency
        print("  ✅ 成功导入infer_frequency函数")
        
    except ImportError as e:
        print(f"  ❌ NeuralHydrology导入失败: {e}")
    except Exception as e:
        print(f"  ❌ 其他错误: {e}")

def test_data_loading():
    """测试数据加载"""
    print("\n🔍 测试数据加载...")
    
    try:
        from neuralhydrology.datasetzoo.basedataset import BaseDataset
        
        # 创建数据集实例
        dataset = BaseDataset(
            data_dir="src/neuralhydrology/data",
            basin="red_river_basin",
            variable_names=["snow_depth_mm", "streamflow_m3s"],
            target_variables=["streamflow_m3s"]
        )
        
        print("  ✅ 成功创建BaseDataset实例")
        
        # 尝试加载数据
        data = dataset.get_data()
        print(f"  ✅ 成功加载数据，形状: {data.shape}")
        
    except Exception as e:
        print(f"  ❌ 数据加载测试失败: {e}")
        print(f"    错误类型: {type(e).__name__}")

if __name__ == "__main__":
    print("🚀 HydrAI-SWE 数据格式测试")
    print("=" * 50)
    
    test_data_format()
    test_neuralhydrology_import()
    test_data_loading()
    
    print("\n" + "=" * 50)
    print("✅ 测试完成！")
