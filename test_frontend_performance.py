#!/usr/bin/env python3
"""
前端性能测试脚本
测试优化后的前端是否正常工作
"""

import requests
import time
import json

def test_frontend_performance():
    """测试前端性能"""
    base_url = "http://127.0.0.1:8000"
    
    print("🧪 开始前端性能测试...")
    
    # 测试1: 主页面加载
    print("\n1. 测试主页面加载...")
    start_time = time.time()
    try:
        response = requests.get(f"{base_url}/ui", timeout=10)
        load_time = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ 主页面加载成功，耗时: {load_time:.2f}秒")
            print(f"   页面大小: {len(response.content) / 1024:.1f} KB")
            
            # 检查关键元素
            content = response.text
            if "Chart.js" in content:
                print("✅ Chart.js 库已加载")
            if "fetchForecast" in content:
                print("✅ 预测功能函数已加载")
            if "triggerInitialFetch" not in content:
                print("✅ 已移除自动数据获取（性能优化）")
            else:
                print("⚠️ 仍存在自动数据获取函数")
                
        else:
            print(f"❌ 主页面加载失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 主页面加载异常: {e}")
        return False
    
    # 测试2: API健康检查
    print("\n2. 测试API健康检查...")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ 根API端点正常")
        else:
            print(f"❌ 根API端点异常: {response.status_code}")
            
        response = requests.get(f"{base_url}/api/v1/cross-validation/health", timeout=5)
        if response.status_code == 200:
            print("✅ 交叉验证API健康检查正常")
        else:
            print(f"❌ 交叉验证API健康检查异常: {response.status_code}")
            
    except Exception as e:
        print(f"❌ API健康检查异常: {e}")
    
    # 测试3: 交叉验证快速接口
    print("\n3. 测试交叉验证快速接口...")
    try:
        payload = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",
            "station_id": "05OC001"
        }
        
        start_time = time.time()
        response = requests.post(
            f"{base_url}/api/v1/cross-validation/quick",
            json=payload,
            timeout=10
        )
        api_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 交叉验证快速接口正常，响应时间: {api_time:.2f}秒")
            print(f"   返回数据: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ 交叉验证快速接口异常: {response.status_code}")
            print(f"   响应内容: {response.text}")
            
    except Exception as e:
        print(f"❌ 交叉验证快速接口异常: {e}")
    
    # 测试4: 径流预测接口
    print("\n4. 测试径流预测接口...")
    try:
        params = {
            "station_id": "05OC001",
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",
            "mode": "nowcast"
        }
        
        start_time = time.time()
        response = requests.get(
            f"{base_url}/api/v1/runoff-forecast",
            params=params,
            timeout=10
        )
        api_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 径流预测接口正常，响应时间: {api_time:.2f}秒")
            if "forecasts" in data and len(data["forecasts"]) > 0:
                print(f"   返回预测数据: {len(data['forecasts'])} 条记录")
            else:
                print("   ⚠️ 返回数据为空")
        else:
            print(f"❌ 径流预测接口异常: {response.status_code}")
            print(f"   响应内容: {response.text}")
            
    except Exception as e:
        print(f"❌ 径流预测接口异常: {e}")
    
    print("\n🎯 前端性能测试完成！")
    print("\n📱 现在您可以在浏览器中访问:")
    print(f"   主界面: {base_url}/ui")
    print(f"   旧版本: {base_url}/ui/legacy")
    print(f"   API文档: {base_url}/docs")
    
    return True

if __name__ == "__main__":
    test_frontend_performance()
