#!/usr/bin/env python3
"""
简单的NASA API测试脚本
验证API连接和基本功能
"""

import requests
import json
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv('config/credentials.env')

def test_nasa_cmr_api():
    """测试NASA CMR API基本功能"""
    print("🧪 测试NASA CMR API")
    print("=" * 50)
    
    # 测试1: 基本连接
    print("1️⃣ 测试基本连接...")
    try:
        response = requests.get("https://cmr.earthdata.nasa.gov/search/collections.json", timeout=10)
        print(f"✅ 连接成功: {response.status_code}")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return False
    
    # 测试2: 搜索SMAP集合
    print("\n2️⃣ 搜索SMAP集合...")
    try:
        params = {
            'keyword': 'SMAP',
            'page_size': 5
        }
        response = requests.get("https://cmr.earthdata.nasa.gov/search/collections.json", params=params, timeout=10)
        data = response.json()
        
        collections = data.get('feed', {}).get('entry', [])
        print(f"✅ 找到 {len(collections)} 个SMAP集合")
        
        for i, collection in enumerate(collections[:3]):
            title = collection.get('title', 'Unknown')
            concept_id = collection.get('id', 'Unknown')
            print(f"   {i+1}. {title}")
            print(f"      ID: {concept_id}")
            
    except Exception as e:
        print(f"❌ 搜索失败: {e}")
        return False
    
    # 测试3: 测试特定集合
    print("\n3️⃣ 测试特定SMAP集合...")
    test_collection = 'C1940468260-POCLOUD'  # SMAP L3
    
    try:
        params = {
            'collection_concept_id': test_collection,
            'page_size': 1
        }
        response = requests.get("https://cmr.earthdata.nasa.gov/search/granules.umm_json", params=params, timeout=10)
        data = response.json()
        
        print(f"✅ 集合 {test_collection} 响应成功")
        print(f"响应键: {list(data.keys())}")
        
        if 'hits' in data:
            hits = data['hits']
            print(f"Hits类型: {type(hits)}")
            if isinstance(hits, int):
                print(f"总文件数: {hits}")
            elif isinstance(hits, dict):
                print(f"Hits键: {list(hits.keys())}")
        
        if 'items' in data:
            items = data['items']
            print(f"Items类型: {type(items)}")
            print(f"Items数量: {len(items) if isinstance(items, list) else 'Not a list'}")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    # 测试4: 测试认证
    print("\n4️⃣ 测试认证...")
    username = os.getenv('NASA_EARTHDATA_USERNAME')
    password = os.getenv('NASA_EARTHDATA_PASSWORD')
    
    if username and password:
        print(f"✅ 凭据已配置: {username}")
        
        # 尝试访问需要认证的端点
        try:
            session = requests.Session()
            session.auth = (username, password)
            
            # 测试一个简单的认证端点
            response = session.get("https://urs.earthdata.nasa.gov/profile", timeout=10)
            if response.status_code == 200:
                print("✅ 认证成功")
            else:
                print(f"⚠️ 认证状态: {response.status_code}")
                
        except Exception as e:
            print(f"⚠️ 认证测试失败: {e}")
    else:
        print("❌ 凭据未配置")
        return False
    
    print("\n" + "=" * 50)
    print("🎯 测试完成!")
    return True

def test_smap_data_availability():
    """测试SMAP数据可用性"""
    print("\n🔍 测试SMAP数据可用性")
    print("=" * 50)
    
    # 测试不同的时间范围
    time_ranges = [
        ("2024-01-01", "2024-01-31"),  # 2024年1月
        ("2024-06-01", "2024-06-30"),  # 2024年6月
        ("2024-12-01", "2024-12-31"),  # 2024年12月
        ("2025-01-01", "2025-01-31"),  # 2025年1月
    ]
    
    # 测试不同的地理范围
    regions = [
        ("Manitoba", [-102.0, 49.0, -95.0, 53.0]),
        ("Red River Basin", [-97.5, 49.0, -96.5, 50.5]),
        ("Winnipeg", [-97.2, 49.8, -97.0, 50.0]),
    ]
    
    collection_id = 'C1940468260-POCLOUD'  # SMAP L3
    
    for region_name, bbox in regions:
        print(f"\n📍 测试区域: {region_name}")
        print(f"边界框: {bbox}")
        
        for start_date, end_date in time_ranges:
            print(f"   📅 {start_date} 到 {end_date}")
            
            try:
                params = {
                    'collection_concept_id': collection_id,
                    'temporal': f"{start_date}T00:00:00Z,{end_date}T23:59:59Z",
                    'bounding_box': f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                    'page_size': 1
                }
                
                response = requests.get("https://cmr.earthdata.nasa.gov/search/granules.umm_json", params=params, timeout=10)
                data = response.json()
                
                if 'hits' in data:
                    hits = data['hits']
                    if isinstance(hits, int) and hits > 0:
                        print(f"      ✅ 找到 {hits} 个文件")
                    elif isinstance(hits, dict):
                        file_count = hits.get('hits', [])
                        print(f"      ✅ 找到 {len(file_count)} 个文件")
                    else:
                        print(f"      ❌ 无数据")
                else:
                    print(f"      ❌ 响应格式异常")
                    
            except Exception as e:
                print(f"      ❌ 查询失败: {e}")

def main():
    """主函数"""
    print("🚀 NASA API 连接测试")
    print("=" * 60)
    
    # 基本API测试
    if not test_nasa_cmr_api():
        print("❌ 基本API测试失败，停止后续测试")
        return
    
    # SMAP数据可用性测试
    test_smap_data_availability()
    
    print("\n" + "=" * 60)
    print("🎯 所有测试完成!")

if __name__ == "__main__":
    main()
