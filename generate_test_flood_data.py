#!/usr/bin/env python3
"""
生成测试洪水数据，解决风险概率为0的问题
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_test_flood_data():
    """生成测试洪水数据"""
    
    # 创建日期范围
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2024, 12, 31)
    dates = pd.date_range(start_date, end_date, freq='D')
    
    print(f"生成 {len(dates)} 天的测试数据...")
    
    # 创建基础数据
    data = []
    
    for i, date in enumerate(dates):
        # 基础天气数据
        month = date.month
        day_of_year = date.dayofyear
        
        # 季节性变化
        seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365)
        
        # 温度数据
        base_temp = 15 - 30 * np.cos(2 * np.pi * (month - 6) / 12)
        max_temp = base_temp + np.random.normal(5, 3)
        min_temp = base_temp + np.random.normal(-5, 3)
        mean_temp = (max_temp + min_temp) / 2
        
        # 降水数据
        rain_prob = 0.3 if month in [6, 7, 8] else 0.1  # 夏季降水概率高
        total_rain = np.random.exponential(10) if np.random.random() < rain_prob else 0
        
        # 积雪数据
        snow_prob = 0.4 if month in [12, 1, 2] else 0.05
        snow_on_ground = np.random.exponential(20) if np.random.random() < snow_prob else 0
        
        # 径流数据 - 基于降水和积雪
        base_flow = 100 + 50 * seasonal_factor
        flow_factor = 1 + (total_rain / 100) + (snow_on_ground / 50)
        flow_05OC001 = base_flow * flow_factor * np.random.uniform(0.8, 1.2)
        flow_05OC011 = flow_05OC001 * np.random.uniform(0.9, 1.1)
        flow_05OC012 = flow_05OC001 * np.random.uniform(0.8, 1.2)
        
        # 添加一些极端事件
        if np.random.random() < 0.05:  # 5%概率的极端事件
            flow_05OC001 *= np.random.uniform(3, 8)
            flow_05OC011 *= np.random.uniform(3, 8)
            flow_05OC012 *= np.random.uniform(3, 8)
            total_rain *= np.random.uniform(2, 5)
        
        # 创建数据行
        row = {
            'Date/Time': date,
            'Year': date.year,
            'Month': date.month,
            'Day': date.day,
            'Max Temp (°C)': max_temp,
            'Min Temp (°C)': min_temp,
            'Mean Temp (°C)': mean_temp,
            'Total Rain (mm)': total_rain,
            'Total Snow (cm)': snow_on_ground,
            'Snow on Grnd (cm)': snow_on_ground,
            '05OC001': flow_05OC001,
            '05OC011': flow_05OC011,
            '05OC012': flow_05OC012,
            'DayOfYear': day_of_year,
            'WeekOfYear': date.isocalendar()[1]
        }
        
        data.append(row)
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 添加特征工程
    print("添加特征工程...")
    
    # 时间特征
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    
    # 天气特征
    df['temp_anomaly'] = df['Mean Temp (°C)'] - df['Mean Temp (°C)'].rolling(30).mean()
    df['flow_change'] = df['05OC001'].pct_change()
    df['flow_volatility'] = df['05OC001'].rolling(7).std()
    df['flow_peak'] = df['05OC001'].rolling(7).max()
    
    # 相关性特征
    df['flow_corr_2_3'] = df['05OC001'].rolling(7).corr(df['05OC011'])
    df['flow_corr_2_4'] = df['05OC001'].rolling(7).corr(df['05OC012'])
    df['flow_corr_2_5'] = df['05OC001'].rolling(7).corr(df['Total Rain (mm)'])
    df['flow_corr_3_5'] = df['05OC011'].rolling(7).corr(df['Total Rain (mm)'])
    df['flow_corr_4_5'] = df['05OC012'].rolling(7).corr(df['Total Rain (mm)'])
    
    # 填充NaN值
    df = df.fillna(0)
    
    # 保存数据
    output_dir = "data/processed/flood_warning"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = f"{output_dir}/flood_warning_test_data.csv"
    df.to_csv(output_file, index=False)
    
    print(f"✅ 测试数据已保存: {output_file}")
    print(f"📊 数据形状: {df.shape}")
    print(f"📅 日期范围: {df['Date/Time'].min()} 到 {df['Date/Time'].max()}")
    print(f"🌊 径流范围: {df['05OC001'].min():.1f} - {df['05OC001'].max():.1f}")
    
    return df

if __name__ == "__main__":
    generate_test_flood_data()
