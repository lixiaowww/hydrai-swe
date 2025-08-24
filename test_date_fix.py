#!/usr/bin/env python3
"""
测试日期逻辑修复的脚本
"""

import requests
import json

def test_date_logic_fix():
    """测试日期逻辑修复"""
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 测试日期逻辑修复...")
    
    # 测试1: 情景模式下的日期逻辑
    print("\n1. 测试情景模式 (2023年) 的日期逻辑...")
    
    # 模拟前端的情景模式请求
    params = {
        "station_id": "05OC001",
        "start_date": "2023-03-15",  # 应该是2023年的日期
        "end_date": "2023-05-15",    # 应该是2023年的日期
        "mode": "scenario",
        "scenario_year": "2023"
    }
    
    try:
        response = requests.get(f"{base_url}/api/v1/runoff-forecast", params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 情景模式API调用成功")
            print(f"   请求参数: {json.dumps(params, indent=2, ensure_ascii=False)}")
            print(f"   返回数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            # 验证返回的日期是否在正确的年份范围内
            if "forecasts" in data and len(data["forecasts"]) > 0:
                dates = [f["date"] for f in data["forecasts"]]
                print(f"   预测日期范围: {min(dates)} 到 {max(dates)}")
                
                # 检查所有日期是否都在2023年
                all_2023 = all(date.startswith("2023") for date in dates)
                if all_2023:
                    print("✅ 所有预测日期都在2023年范围内")
                else:
                    print("❌ 存在非2023年的日期")
            else:
                print("⚠️ 返回数据中没有预测信息")
                
        else:
            print(f"❌ 情景模式API调用失败: {response.status_code}")
            print(f"   响应内容: {response.text}")
            
    except Exception as e:
        print(f"❌ 情景模式测试异常: {e}")
    
    # 测试2: 实时预测模式的日期逻辑
    print("\n2. 测试实时预测模式的日期逻辑...")
    
    from datetime import datetime, timedelta
    today = datetime.now()
    next_week = today + timedelta(days=7)
    
    params = {
        "station_id": "05OC001",
        "start_date": today.strftime("%Y-%m-%d"),
        "end_date": next_week.strftime("%Y-%m-%d"),
        "mode": "nowcast"
    }
    
    try:
        response = requests.get(f"{base_url}/api/v1/runoff-forecast", params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 实时预测模式API调用成功")
            print(f"   请求参数: {json.dumps(params, indent=2, ensure_ascii=False)}")
            print(f"   返回数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
        else:
            print(f"❌ 实时预测模式API调用失败: {response.status_code}")
            print(f"   响应内容: {response.text}")
            
    except Exception as e:
        print(f"❌ 实时预测模式测试异常: {e}")
    
    # 测试3: 验证前端页面逻辑
    print("\n3. 验证前端页面逻辑...")
    
    try:
        response = requests.get(f"{base_url}/ui", timeout=10)
        
        if response.status_code == 200:
            content = response.text
            
            # 检查关键函数是否存在
            if "updateDateFields" in content:
                print("✅ updateDateFields函数存在")
            else:
                print("❌ updateDateFields函数缺失")
                
            if "addScenarioYearListener" in content:
                print("✅ addScenarioYearListener函数存在")
            else:
                print("❌ addScenarioYearListener函数缺失")
                
            if "triggerInitialFetch" not in content:
                print("✅ 已移除triggerInitialFetch函数（性能优化）")
            else:
                print("⚠️ 仍存在triggerInitialFetch函数")
                
            # 检查日期字段的提示信息
            if "MM-DD format (year determined by scenario year)" in content:
                print("✅ 情景模式日期提示正确")
            else:
                print("❌ 情景模式日期提示缺失")
                
            if "YYYY-MM-DD format for future predictions" in content:
                print("✅ 实时预测模式日期提示正确")
            else:
                print("❌ 实时预测模式日期提示缺失")
                
        else:
            print(f"❌ 前端页面加载失败: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 前端页面验证异常: {e}")
    
    print("\n🎯 日期逻辑修复测试完成！")
    print("\n📱 现在您可以在浏览器中测试:")
    print(f"   主界面: {base_url}/ui")
    print("   1. 切换到'情景模拟'模式")
    print("   2. 设置情景年份为2023")
    print("   3. 验证开始和结束日期自动变为2023-03-15和2023-05-15")
    print("   4. 切换到'实时预测'模式，验证日期变为当前日期")

if __name__ == "__main__":
    test_date_logic_fix()
