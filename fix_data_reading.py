#!/usr/bin/env python3
"""
数据修复脚本
解决现有数据文件的读取问题，标准化数据格式
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def fix_eccc_data():
    """修复ECCC雪数据"""
    print("🔧 修复ECCC雪数据")
    
    input_file = "data/processed/eccc_manitoba_snow_processed.csv"
    output_file = "data/processed/eccc_manitoba_snow_fixed.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    try:
        # 读取原始数据
        df = pd.read_csv(input_file)
        print(f"✅ 读取原始数据: {len(df)} 条记录")
        
        # 检查列名
        print(f"   列名: {list(df.columns)}")
        
        # 重命名列
        column_mapping = {
            'Total Snow (cm)': 'snow_fall_mm',
            'Snow on Grnd (cm)': 'snow_depth_mm'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 转换单位：cm -> mm
        if 'snow_fall_mm' in df.columns:
            df['snow_fall_mm'] = df['snow_fall_mm'] * 10
        if 'snow_depth_mm' in df.columns:
            df['snow_depth_mm'] = df['snow_depth_mm'] * 10
        
        # 添加缺失的列
        df['snow_water_equivalent_mm'] = df['snow_depth_mm'] * 0.3  # 估算雪水当量
        
        # 处理日期
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df['day_of_year'] = df['date'].dt.dayofyear
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
        
        # 选择需要的列
        required_columns = ['date', 'snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                           'day_of_year', 'month', 'year']
        
        df_fixed = df[required_columns].copy()
        
        # 处理缺失值
        df_fixed = df_fixed.fillna(0)
        
        # 保存修复后的数据
        df_fixed.to_csv(output_file, index=False)
        print(f"✅ 修复后的数据已保存: {output_file}")
        print(f"   记录数: {len(df_fixed)}")
        print(f"   列: {list(df_fixed.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False

def fix_hydat_data():
    """修复HYDAT流量数据"""
    print("\n🔧 修复HYDAT流量数据")
    
    input_file = "data/processed/hydat_streamflow_processed.csv"
    output_file = "data/processed/hydat_streamflow_fixed.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    try:
        # 读取原始数据
        df = pd.read_csv(input_file)
        print(f"✅ 读取原始数据: {len(df)} 条记录")
        
        # 检查列名
        print(f"   列名: {list(df.columns)}")
        
        # 处理日期列
        date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_columns:
            date_col = date_columns[0]
            df['date'] = pd.to_datetime(df[date_col])
        else:
            # 如果没有日期列，创建示例日期
            df['date'] = pd.date_range('2020-01-01', periods=len(df), freq='D')
        
        # 添加雪相关列（HYDAT主要是流量数据，这里添加估算值）
        df['snow_depth_mm'] = 0  # 流量数据没有雪深度
        df['snow_fall_mm'] = 0   # 流量数据没有降雪量
        df['snow_water_equivalent_mm'] = 0  # 流量数据没有雪水当量
        
        # 添加时间特征
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # 选择需要的列
        required_columns = ['date', 'snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                           'day_of_year', 'month', 'year']
        
        df_fixed = df[required_columns].copy()
        
        # 保存修复后的数据
        df_fixed.to_csv(output_file, index=False)
        print(f"✅ 修复后的数据已保存: {output_file}")
        print(f"   记录数: {len(df_fixed)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False

def fix_comprehensive_data():
    """修复综合训练数据"""
    print("\n🔧 修复综合训练数据")
    
    input_file = "data/processed/comprehensive_training_dataset.csv"
    output_file = "data/processed/comprehensive_training_dataset_fixed.csv"
    
    if not os.path.exists(input_file):
        print(f"❌ 输入文件不存在: {input_file}")
        return False
    
    try:
        # 读取原始数据
        df = pd.read_csv(input_file, index_col=0)
        print(f"✅ 读取原始数据: {len(df)} 条记录")
        
        # 检查列名
        print(f"   列名: {list(df.columns)}")
        
        # 重置索引
        df = df.reset_index()
        
        # 处理日期列
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            # 如果没有日期列，使用索引
            df['date'] = pd.to_datetime(df.index)
        
        # 添加时间特征
        df['day_of_year'] = df['date'].dt.dayofyear
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        # 确保雪相关列存在
        required_snow_columns = ['snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm']
        for col in required_snow_columns:
            if col not in df.columns:
                df[col] = 0
        
        # 选择需要的列
        required_columns = ['date', 'snow_depth_mm', 'snow_fall_mm', 'snow_water_equivalent_mm', 
                           'day_of_year', 'month', 'year']
        
        df_fixed = df[required_columns].copy()
        
        # 保存修复后的数据
        df_fixed.to_csv(output_file, index=False)
        print(f"✅ 修复后的数据已保存: {output_file}")
        print(f"   记录数: {len(df_fixed)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 修复失败: {e}")
        return False

def create_sample_data():
    """创建示例数据用于测试"""
    print("\n🔧 创建示例数据")
    
    output_file = "data/processed/sample_training_data.csv"
    
    try:
        # 生成示例数据
        dates = pd.date_range('2000-01-01', '2024-12-31', freq='D')
        
        data = []
        for date in dates:
            # 模拟季节性雪数据
            day_of_year = date.timetuple().tm_yday
            seasonal_factor = 50 + 30 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            random_variation = np.random.normal(0, 10)
            
            snow_depth = max(0, seasonal_factor + random_variation)
            snow_fall = max(0, np.random.normal(20, 15))
            snow_we = max(0, snow_depth * 0.3 + np.random.normal(0, 5))
            
            data.append({
                'date': date,
                'snow_depth_mm': snow_depth,
                'snow_fall_mm': snow_fall,
                'snow_water_equivalent_mm': snow_we,
                'day_of_year': day_of_year,
                'month': date.month,
                'year': date.year
            })
        
        df = pd.DataFrame(data)
        
        # 保存示例数据
        df.to_csv(output_file, index=False)
        print(f"✅ 示例数据已创建: {output_file}")
        print(f"   记录数: {len(df)}")
        print(f"   时间范围: {df['date'].min()} - {df['date'].max()}")
        
        return True
        
    except Exception as e:
        print(f"❌ 创建示例数据失败: {e}")
        return False

def main():
    """主函数"""
    print("🚀 开始修复数据文件")
    print("=" * 50)
    
    # 修复现有数据
    success_count = 0
    total_tasks = 4
    
    if fix_eccc_data():
        success_count += 1
    
    if fix_hydat_data():
        success_count += 1
    
    if fix_comprehensive_data():
        success_count += 1
    
    if create_sample_data():
        success_count += 1
    
    print(f"\n📊 修复完成: {success_count}/{total_tasks} 成功")
    
    if success_count == total_tasks:
        print("🎉 所有数据文件修复成功！")
        print("✅ 现在可以重新运行数据扩展脚本")
        return True
    else:
        print("⚠️ 部分数据文件修复失败，需要进一步检查")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

