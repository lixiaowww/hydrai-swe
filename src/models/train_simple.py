#!/usr/bin/env python3
"""
简化的NeuralHydrology训练脚本
直接使用API进行训练，避免复杂的数据准备流程
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

def create_simple_training_data():
    """创建简单的训练数据"""
    print("Creating simple training data...")
    
    # 创建时间序列
    dates = pd.date_range('1979-01-01', '1998-12-31', freq='D')
    
    # 创建模拟数据
    np.random.seed(42)  # 确保可重复性
    
    # 积雪深度：季节性变化 + 随机噪声
    seasonal_snow = 100 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25) + 50
    snow_depth = np.maximum(0, seasonal_snow + np.random.normal(0, 20, len(dates)))
    
    # 降雪量：冬季较高
    snow_fall = np.where(dates.month.isin([12, 1, 2, 3]), 
                         np.random.exponential(10, len(dates)), 
                         np.random.exponential(2, len(dates)))
    
    # 雪水当量：积雪深度的30%
    snow_water_equivalent = snow_depth * 0.3
    
    # 径流：基于积雪融化的简化模型
    streamflow = 1000 + snow_depth * 0.1 + np.random.normal(0, 50, len(dates))
    streamflow = np.maximum(500, streamflow)  # 最小径流
    
    # 创建DataFrame
    data = pd.DataFrame({
        'date': dates,
        'snow_depth_mm': snow_depth,
        'snow_fall_mm': snow_fall,
        'snow_water_equivalent_mm': snow_water_equivalent,
        'day_of_year': dates.dayofyear,
        'month': dates.month,
        'year': dates.year,
        'streamflow_m3s': streamflow
    })
    
    return data

def prepare_neuralhydrology_data(data, output_dir):
    """准备NeuralHydrology格式的数据"""
    print("Preparing NeuralHydrology data...")
    
    # 创建输出目录
    basin_dir = os.path.join(output_dir, "red_river_basin")
    os.makedirs(basin_dir, exist_ok=True)
    
    # 保存时间序列数据
    timeseries_file = os.path.join(basin_dir, "timeseries.csv")
    data.to_csv(timeseries_file, index=False)
    print(f"Saved timeseries to: {timeseries_file}")
    
    # 创建basins文件
    basins_file = os.path.join(output_dir, "basins.txt")
    with open(basins_file, 'w') as f:
        f.write("red_river_basin\n")
    print(f"Created basins file: {basins_file}")
    
    return output_dir

def train_with_neuralhydrology():
    """使用NeuralHydrology训练模型"""
    print("Starting NeuralHydrology training...")
    
    # 创建数据
    data = create_simple_training_data()
    print(f"Created training data: {len(data)} records")
    
    # 准备NeuralHydrology数据
    data_dir = "src/neuralhydrology/data"
    prepare_neuralhydrology_data(data, data_dir)
    
    # 切换到NeuralHydrology目录
    original_dir = os.getcwd()
    neuralhydrology_dir = Path("src/neuralhydrology")
    
    try:
        os.chdir(neuralhydrology_dir)
        print(f"Changed to directory: {os.getcwd()}")
        
        # 导入NeuralHydrology
        from neuralhydrology import nh_run
        from neuralhydrology.utils.config import Config
        
        # 加载配置
        config_path = Path("config.yml")
        cfg = Config(config_path)
        
        print("Configuration loaded successfully")
        print(f"Model: {cfg.model}")
        print(f"Hidden size: {cfg.hidden_size}")
        print(f"Epochs: {cfg.epochs}")
        print(f"Batch size: {cfg.batch_size}")
        
        # 开始训练
        print("Starting training...")
        nh_run.start_training(cfg)
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 返回原目录
        os.chdir(original_dir)
        print(f"Returned to directory: {os.getcwd()}")

if __name__ == "__main__":
    train_with_neuralhydrology()
