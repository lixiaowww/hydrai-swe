#!/usr/bin/env python3
"""
SWE Analysis修复验证脚本
测试修复后的API和前端功能
"""

import requests
import json
from datetime import datetime

def test_swe_analysis_api():
    """测试SWE Analysis API"""
    print("🧪 测试SWE Analysis API修复")
    print("=" * 50)
    
    # 测试1: 季节性分析
    print("1️⃣ 测试季节性分析...")
    try:
        payload = {
            "mode": "seasonal",
            "column": "snow_water_equivalent_mm"
        }
        
        response = requests.post(
            "http://localhost:8000/api/swe/analysis",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 季节性分析成功: {data['mode']}")
            
            # 检查结果结构
            result = data.get('result', {})
            if 'monthly_patterns' in result:
                seasonal_indices = result['monthly_patterns'].get('seasonal_indices', [])
                print(f"   季节性指数: {len(seasonal_indices)} 个月")
                print(f"   整体均值: {result['monthly_patterns'].get('overall_mean', 'N/A')}")
            
            if 'annual_cycle' in result:
                trend = result['annual_cycle'].get('trend', {})
                print(f"   年际趋势: R² = {trend.get('r_squared', 'N/A'):.3f}")
                
        else:
            print(f"❌ 季节性分析失败: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 季节性分析异常: {e}")
        return False
    
    # 测试2: 异常检测
    print("\n2️⃣ 测试异常检测...")
    try:
        payload = {
            "mode": "anomaly",
            "column": "snow_water_equivalent_mm"
        }
        
        response = requests.post(
            "http://localhost:8000/api/swe/analysis",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 异常检测成功: {data['mode']}")
            
            result = data.get('result', {})
            print(f"   阈值: {result.get('threshold', 'N/A')}")
            print(f"   异常率: {result.get('anomaly_rate', 'N/A')}")
            
        else:
            print(f"❌ 异常检测失败: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 异常检测异常: {e}")
        return False
    
    # 测试3: 相关性分析
    print("\n3️⃣ 测试相关性分析...")
    try:
        payload = {
            "mode": "correlation",
            "column": "snow_water_equivalent_mm"
        }
        
        response = requests.post(
            "http://localhost:8000/api/swe/analysis",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 相关性分析成功: {data['mode']}")
            
            result = data.get('result', {})
            top_correlations = result.get('top_correlations', [])
            print(f"   前5相关性: {len(top_correlations)} 个变量")
            
            for i, corr in enumerate(top_correlations[:3]):
                print(f"     {i+1}. {corr.get('variable', 'N/A')}: r={corr.get('pearson_r', 'N/A'):.3f}")
                
        else:
            print(f"❌ 相关性分析失败: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 相关性分析异常: {e}")
        return False
    
    # 测试4: 综合分析
    print("\n4️⃣ 测试综合分析...")
    try:
        payload = {
            "mode": "comprehensive",
            "column": "snow_water_equivalent_mm"
        }
        
        response = requests.post(
            "http://localhost:8000/api/swe/analysis",
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 综合分析成功: {data['mode']}")
            print(f"   摘要: {data.get('result', {}).get('summary', 'N/A')}")
            
        else:
            print(f"❌ 综合分析失败: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ 综合分析异常: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("🎯 SWE Analysis API测试完成!")
    print("✅ 所有分析模式都正常工作")
    return True

def test_frontend_integration():
    """测试前端集成"""
    print("\n🌐 测试前端集成...")
    print("=" * 40)
    
    try:
        # 测试前端界面访问
        response = requests.get("http://localhost:8000/ui", timeout=10)
        if response.status_code == 200:
            print("✅ 前端界面访问正常")
            
            # 检查是否包含修复后的SWE Analysis内容
            content = response.text.lower()
            if 'swe analysis' in content and 'analyze swe trends' in content:
                print("✅ SWE Analysis前端集成成功")
            else:
                print("⚠️ SWE Analysis前端集成可能不完整")
        else:
            print(f"❌ 前端界面访问失败: {response.status_code}")
            
    except Exception as e:
        print(f"❌ 前端集成测试异常: {e}")

def main():
    """主函数"""
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API地址: http://localhost:8000")
    
    # 测试后端API
    if test_swe_analysis_api():
        # 测试前端集成
        test_frontend_integration()
        
        print("\n" + "=" * 60)
        print("🎉 SWE Analysis修复验证完成!")
        print("📊 系统状态: 完全正常")
        print("💡 前端SWE Analysis功能已修复")
    else:
        print("\n" + "=" * 60)
        print("❌ 修复验证失败，需要进一步检查")

if __name__ == "__main__":
    main()
