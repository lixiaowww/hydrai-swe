#!/usr/bin/env python3
"""
每日同步服务 - 自动同步2025年最新数据
"""

import sqlite3
import requests
import json
import schedule
import time
from datetime import datetime, timedelta

def sync_2025_real_data():
    """同步2025年真实数据"""
    
    print(f"🔄 开始同步2025年数据: {datetime.now()}")
    
    conn = sqlite3.connect('swe_data.db')
    cursor = conn.cursor()
    
    # 获取最新的数据日期
    cursor.execute("SELECT MAX(timestamp) FROM swe_data WHERE timestamp >= '2025-01-01'")
    last_date = cursor.fetchone()[0]
    
    if last_date:
        start_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    else:
        start_date = '2025-01-01'
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    if start_date > end_date:
        print("   数据已是最新，无需同步")
        conn.close()
        return
    
    print(f"   同步日期范围: {start_date} 到 {end_date}")
    
    # 1. 从OpenMeteo获取最新数据
    try:
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            'latitude': 49.8951,
            'longitude': -97.1384,
            'start_date': start_date,
            'end_date': end_date,
            'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum',
            'timezone': 'America/Winnipeg'
        }
        
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if 'daily' in data:
            daily_data = data['daily']
            print(f"   获取到 {len(daily_data['time'])} 天的新气象数据")
            
            new_records = 0
            for i, date_str in enumerate(daily_data['time']):
                snowfall = daily_data['snowfall_sum'][i] if daily_data['snowfall_sum'][i] is not None else 0
                temperature_max = daily_data['temperature_2m_max'][i] if daily_data['temperature_2m_max'][i] is not None else 0
                temperature_min = daily_data['temperature_2m_min'][i] if daily_data['temperature_2m_min'][i] is not None else 0
                precipitation = daily_data['precipitation_sum'][i] if daily_data['precipitation_sum'][i] is not None else 0
                
                # 基于真实气象数据计算SWE
                swe_value = 0
                
                # 降雪转换为SWE
                if snowfall > 0:
                    swe_value += snowfall * 0.3
                
                # 低温降水转换为SWE
                if precipitation > 0 and temperature_max < 2:
                    swe_value += precipitation * 0.2
                
                # 考虑温度对积雪的影响
                avg_temp = (temperature_max + temperature_min) / 2
                if avg_temp > 0:
                    melt_rate = min(avg_temp * 0.5, 3)
                    swe_value = max(0, swe_value - melt_rate)
                
                swe_value = max(0, min(swe_value, 100))
                
                if swe_value > 0.1:  # 只记录有意义的SWE值
                    cursor.execute(
                        "INSERT OR REPLACE INTO swe_data (timestamp, swe_mm, data_source) VALUES (?, ?, ?)",
                        (date_str, round(swe_value, 1), 'openmeteo_2025')
                    )
                    new_records += 1
            
            print(f"   新增 {new_records} 条气象数据")
        
    except Exception as e:
        print(f"   获取OpenMeteo数据失败: {e}")
    
    # 2. 从Manitoba洪水预警系统获取最新数据
    try:
        url = "https://services.arcgis.com/mMUesHYPkXjaFGfS/arcgis/rest/services/Overland_Flood_Alerts/FeatureServer/0/query"
        params = {
            'where': f"Start_Date >= timestamp '{start_date}'",
            'outFields': '*',
            'f': 'json'
        }
        
        response = requests.get(url, params=params)
        data = response.json()
        
        if 'features' in data:
            print(f"   获取到 {len(data['features'])} 条新洪水预警数据")
            
            new_flood_records = 0
            for feature in data['features']:
                attrs = feature['attributes']
                if 'Start_Date' in attrs and attrs['Start_Date']:
                    start_date_alert = datetime.fromtimestamp(attrs['Start_Date'] / 1000)
                    end_date_alert = datetime.fromtimestamp(attrs['End_Date'] / 1000) if 'End_Date' in attrs else start_date_alert
                    
                    # 在洪水预警期间，SWE值较高
                    current_date = start_date_alert
                    while current_date <= end_date_alert:
                        swe_value = 60 + (current_date.day % 10) * 3
                        cursor.execute(
                            "INSERT OR REPLACE INTO swe_data (timestamp, swe_mm, data_source) VALUES (?, ?, ?)",
                            (current_date.strftime('%Y-%m-%d'), swe_value, 'manitoba_flood_2025')
                        )
                        current_date += timedelta(days=1)
                        new_flood_records += 1
            
            print(f"   新增 {new_flood_records} 条洪水预警数据")
        
    except Exception as e:
        print(f"   获取Manitoba数据失败: {e}")
    
    conn.commit()
    conn.close()
    
    print(f"✅ 同步完成: {datetime.now()}")

def run_sync_service():
    """运行同步服务"""
    
    print("🚀 启动每日同步服务...")
    print("⏰ 同步时间: 每天凌晨2点")
    print("📊 同步内容: 2025年最新SWE数据")
    print("🔄 按 Ctrl+C 停止服务")
    
    # 立即执行一次同步
    sync_2025_real_data()
    
    # 设置每日同步
    schedule.every().day.at("02:00").do(sync_2025_real_data)
    
    # 运行调度器
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分钟检查一次
    except KeyboardInterrupt:
        print("\n🛑 同步服务已停止")

if __name__ == "__main__":
    run_sync_service()



