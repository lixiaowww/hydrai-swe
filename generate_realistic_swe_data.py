#!/usr/bin/env python3
"""
生成更符合真实数据特征的SWE数据
基于2010-2020年真实数据的模式
"""

import sqlite3
import numpy as np
from datetime import datetime, timedelta

def generate_realistic_swe_data():
    """基于2010-2020年真实数据模式生成2021-2024年数据"""
    
    # 连接数据库
    conn = sqlite3.connect('swe_data.db')
    cursor = conn.cursor()
    
    # 删除之前的模拟数据
    cursor.execute("DELETE FROM swe_data WHERE timestamp >= '2021-01-01'")
    
    # 获取2010-2020年的真实数据作为参考
    cursor.execute("SELECT timestamp, swe_mm FROM swe_data WHERE timestamp >= '2010-01-01' AND timestamp < '2021-01-01' ORDER BY timestamp")
    historical_data = cursor.fetchall()
    
    if not historical_data:
        print("没有找到2010-2020年的参考数据")
        return
    
    # 分析历史数据的模式
    swe_values = [row[1] for row in historical_data]
    dates = [datetime.strptime(row[0], '%Y-%m-%d') for row in historical_data]
    
    # 计算每年的平均SWE
    yearly_avg = {}
    for i, date in enumerate(dates):
        year = date.year
        if year not in yearly_avg:
            yearly_avg[year] = []
        yearly_avg[year].append(swe_values[i])
    
    # 计算每年的平均值
    for year in yearly_avg:
        yearly_avg[year] = np.mean(yearly_avg[year])
    
    print("2010-2020年各年平均SWE:")
    for year in sorted(yearly_avg.keys()):
        print(f"  {year}: {yearly_avg[year]:.2f}mm")
    
    # 计算整体趋势
    years = sorted(yearly_avg.keys())
    avg_values = [yearly_avg[year] for year in years]
    
    # 计算年际变化趋势
    if len(avg_values) > 1:
        trend = (avg_values[-1] - avg_values[0]) / (years[-1] - years[0])
    else:
        trend = 0
    
    print(f"年际变化趋势: {trend:.3f}mm/年")
    
    # 为2021-2024年生成数据
    for year in [2021, 2022, 2023, 2024]:
        # 基于趋势计算该年的基础平均值
        base_avg = yearly_avg[2020] + trend * (year - 2020)
        
        # 添加年际随机变化（±10%）
        year_variation = np.random.normal(0, 0.05)
        year_avg = base_avg * (1 + year_variation)
        
        print(f"生成{year}年数据，基础平均值: {year_avg:.2f}mm")
        
        # 生成该年的每日数据
        for month in range(1, 13):
            days_in_month = 31 if month in [1, 3, 5, 7, 8, 10, 12] else 30 if month in [4, 6, 9, 11] else 29 if year % 4 == 0 else 28
            
            for day in range(1, days_in_month + 1):
                date = datetime(year, month, day)
                date_str = date.strftime('%Y-%m-%d')
                
                # 基于历史数据中相同日期的模式
                historical_same_date = []
                for hist_date, hist_swe in historical_data:
                    hist_dt = datetime.strptime(hist_date, '%Y-%m-%d')
                    if hist_dt.month == month and hist_dt.day == day:
                        historical_same_date.append(hist_swe)
                
                if historical_same_date:
                    # 使用历史同一天的数据作为基础
                    base_swe = np.mean(historical_same_date)
                    # 调整到该年的平均水平
                    swe_value = base_swe * (year_avg / yearly_avg[2020])
                else:
                    # 如果没有历史同一天的数据，使用季节性模式
                    if month in [12, 1, 2]:  # 冬季
                        swe_value = year_avg * 1.2
                    elif month in [3, 4, 5]:  # 春季
                        swe_value = year_avg * (1.2 - (month - 3) * 0.2)
                    elif month in [6, 7, 8]:  # 夏季
                        swe_value = year_avg * 0.1
                    elif month in [9, 10, 11]:  # 秋季
                        swe_value = year_avg * (0.1 + (month - 9) * 0.1)
                    else:
                        swe_value = year_avg
                
                # 添加很小的随机变化（±2%），保持数据的平滑性
                random_factor = 1 + np.random.normal(0, 0.02)
                swe_value *= random_factor
                
                # 确保SWE值在合理范围内
                swe_value = max(0, min(swe_value, 100))
                
                cursor.execute(
                    "INSERT INTO swe_data (timestamp, swe_mm, data_source) VALUES (?, ?, ?)",
                    (date_str, round(swe_value, 1), f'realistic_{year}')
                )
    
    conn.commit()
    
    # 检查生成的数据
    cursor.execute("SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM swe_data")
    count, min_date, max_date = cursor.fetchone()
    
    print(f"\n生成数据后:")
    print(f"- 总记录数: {count}")
    print(f"- 时间范围: {min_date} 到 {max_date}")
    
    # 检查2021-2024年的数据
    cursor.execute("SELECT AVG(swe_mm), MIN(swe_mm), MAX(swe_mm) FROM swe_data WHERE timestamp >= '2021-01-01' AND timestamp < '2025-01-01'")
    avg_swe, min_swe, max_swe = cursor.fetchone()
    
    print(f"- 2021-2024年: 平均{avg_swe:.2f}mm, 最小{min_swe:.2f}mm, 最大{max_swe:.2f}mm")
    
    # 显示一些样本数据
    cursor.execute("SELECT timestamp, swe_mm FROM swe_data WHERE timestamp >= '2024-12-01' AND timestamp < '2024-12-31' ORDER BY timestamp LIMIT 10")
    samples = cursor.fetchall()
    print(f"\n2024年12月样本数据:")
    for date, swe in samples:
        print(f"  {date}: {swe}mm")
    
    conn.close()

if __name__ == "__main__":
    print("🎯 基于真实数据模式生成更合理的2021-2024年数据...")
    generate_realistic_swe_data()
    print("✅ 数据生成完成！")



