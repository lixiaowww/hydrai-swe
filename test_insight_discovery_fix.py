#!/usr/bin/env python3
"""
测试脚本：验证无监督探索模块的API修复
验证不同的分析模式是否正常工作
"""

import requests
import json
import time

def test_insight_discovery_api():
    """测试insight-discovery API端点的所有模式"""
    base_url = "http://localhost:8000/api/swe/insight-discovery"
    
    test_cases = [
        {
            "name": "异常检测模式",
            "payload": {
                "mode": "anomaly",
                "data_path": None,
                "target_column": "Snow on Grnd (cm)"
            }
        },
        {
            "name": "聚类分析模式",
            "payload": {
                "mode": "clustering", 
                "data_path": None,
                "target_column": "Snow on Grnd (cm)"
            }
        },
        {
            "name": "PCA降维模式",
            "payload": {
                "mode": "pca",
                "data_path": None,
                "target_column": "Snow on Grnd (cm)"
            }
        },
        {
            "name": "时间模式分析",
            "payload": {
                "mode": "time_patterns",
                "data_path": None,
                "target_column": "Snow on Grnd (cm)"
            }
        },
        {
            "name": "综合分析模式",
            "payload": {
                "mode": "comprehensive",
                "data_path": None,
                "target_column": "Snow on Grnd (cm)"
            }
        }
    ]
    
    print("🔍 开始测试无监督探索模块API...")
    print("=" * 60)
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}️⃣ 测试: {test_case['name']}")
        print(f"   模式: {test_case['payload']['mode']}")
        
        try:
            # 发送POST请求
            response = requests.post(
                base_url, 
                json=test_case['payload'],
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ 成功 (状态码: {response.status_code})")
                print(f"   📊 模式: {result.get('mode', 'N/A')}")
                print(f"   📁 数据源: {result.get('data_source', 'N/A')}")
                print(f"   🎯 目标列: {result.get('target_column', 'N/A')}")
                print(f"   ⏱️  执行时间: {result.get('execution_time', 'N/A')}")
                
                # 检查特定模式的结果
                if test_case['payload']['mode'] == 'anomaly' and 'anomaly_detection' in result:
                    anomaly_info = result['anomaly_detection']
                    print(f"   🚨 异常检测: {anomaly_info.get('total_anomalies', 0)} 个异常")
                    print(f"   📈 异常率: {anomaly_info.get('anomaly_rate', 0):.2%}")
                    
                elif test_case['payload']['mode'] == 'clustering' and 'clustering' in result:
                    cluster_info = result['clustering']
                    print(f"   🗂️  聚类数: {cluster_info.get('n_clusters', 'N/A')}")
                    print(f"   📊 轮廓系数: {cluster_info.get('silhouette_score', 0):.3f}")
                    
                elif test_case['payload']['mode'] == 'pca' and 'pca_analysis' in result:
                    pca_info = result['pca_analysis']
                    print(f"   🔢 主成分数: {pca_info.get('n_components', 'N/A')}")
                    variance_ratio = pca_info.get('explained_variance_ratio', [])
                    if variance_ratio:
                        print(f"   📈 解释方差比: {variance_ratio[:3]}...")  # 只显示前3个
                        
                elif test_case['payload']['mode'] == 'comprehensive' and 'insights' in result:
                    insights = result['insights']
                    if 'summary' in insights:
                        summary = insights['summary']
                        print(f"   🔍 洞察数量: {summary.get('total_insights', 0)}")
                        print(f"   ⚠️  风险评估: {summary.get('risk_assessment', 'N/A')}")
                        key_findings = summary.get('key_findings', [])[:2]  # 只显示前2个发现
                        for finding in key_findings:
                            print(f"   💡 发现: {finding}")
                
                results.append((test_case['name'], True, response.status_code))
                
            else:
                print(f"   ❌ 失败 (状态码: {response.status_code})")
                print(f"   📄 响应: {response.text[:200]}...")
                results.append((test_case['name'], False, response.status_code))
                
        except Exception as e:
            print(f"   ❌ 异常: {e}")
            results.append((test_case['name'], False, "Exception"))
            
        # 稍微等待一下避免请求过快
        time.sleep(0.5)
    
    # 打印总结
    print("\n" + "=" * 60)
    print("📊 测试总结:")
    
    success_count = sum(1 for _, success, _ in results if success)
    total_count = len(results)
    
    for name, success, status in results:
        status_icon = "✅" if success else "❌"
        print(f"   {status_icon} {name}: {status}")
    
    print(f"\n🎯 成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    
    if success_count == total_count:
        print("\n🎉 所有测试通过！无监督探索模块API修复成功！")
        print("\n💡 使用说明:")
        print("   • POST /api/swe/insight-discovery")
        print("   • 支持模式: anomaly, clustering, pca, time_patterns, comprehensive")
        print("   • 自动选择数据源和目标列")
        print("   • 返回结构化的分析结果")
    else:
        print(f"\n⚠️  部分测试失败 ({total_count - success_count} 个)")
        print("需要进一步检查API实现")
        
    return success_count == total_count

def test_method_not_allowed_fix():
    """测试之前的Method Not Allowed错误是否已修复"""
    print("\n🔧 测试Method Not Allowed错误修复...")
    
    # 测试GET请求 (应该返回Method Not Allowed)
    try:
        response = requests.get("http://localhost:8000/api/swe/insight-discovery", timeout=5)
        if response.status_code == 405:  # Method Not Allowed
            print("   ✅ GET请求正确返回405 Method Not Allowed")
        else:
            print(f"   ⚠️  GET请求返回: {response.status_code}")
    except Exception as e:
        print(f"   ❌ GET请求异常: {e}")
    
    # 测试POST请求 (应该成功)
    try:
        response = requests.post(
            "http://localhost:8000/api/swe/insight-discovery",
            json={"mode": "anomaly"},
            timeout=10
        )
        if response.status_code == 200:
            print("   ✅ POST请求正常工作")
        else:
            print(f"   ⚠️  POST请求返回: {response.status_code}")
    except Exception as e:
        print(f"   ❌ POST请求异常: {e}")

def main():
    print("🧪 无监督探索模块API修复验证")
    print("=" * 60)
    print("⏰ 测试开始时间:", time.strftime('%Y-%m-%d %H:%M:%S'))
    
    # 测试Method Not Allowed修复
    test_method_not_allowed_fix()
    
    # 测试所有API模式
    success = test_insight_discovery_api()
    
    print("\n" + "=" * 60)
    print("⏰ 测试结束时间:", time.strftime('%Y-%m-%d %H:%M:%S'))
    
    if success:
        print("🎉 所有测试通过！API修复完成！")
        return True
    else:
        print("❌ 部分测试失败，需要进一步修复")
        return False

if __name__ == "__main__":
    exit(0 if main() else 1)
