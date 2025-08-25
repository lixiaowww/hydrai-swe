#!/usr/bin/env python3
"""
测试 HydrAI-SWE 数据管道功能
验证备用数据源接管和真实状态反馈
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000/api/v1/pipeline"

def test_pipeline_status():
    """测试管道状态查询"""
    print("🔍 测试管道状态查询...")
    
    try:
        response = requests.get(f"{BASE_URL}/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ 管道状态查询成功")
            print(f"📊 数据源数量: {len(data['sources'])}")
            
            for source, info in data['sources'].items():
                status_icon = "🟢" if "Active" in info['status'] else "🔴" if "Idle" in info['status'] else "🟡"
                print(f"  {status_icon} {source}: {info['status']} ({info['records']} records)")
        else:
            print(f"❌ 管道状态查询失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 管道状态查询异常: {e}")
        return False
    
    return True

def test_backup_status():
    """测试备用数据源状态"""
    print("\n🔍 测试备用数据源状态...")
    
    try:
        response = requests.get(f"{BASE_URL}/backup/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ 备用数据源状态查询成功")
            
            for source, info in data['backup_sources'].items():
                backup_icon = "✅" if info['backup_available'] else "❌"
                print(f"  {backup_icon} {source}: 备用源 {', '.join(info['backups'])}")
        else:
            print(f"❌ 备用数据源状态查询失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 备用数据源状态查询异常: {e}")
        return False
    
    return True

def test_credentials_status():
    """测试凭据状态"""
    print("\n🔍 测试Earthdata凭据状态...")
    
    try:
        response = requests.get(f"{BASE_URL}/credentials/status")
        if response.status_code == 200:
            data = response.json()
            print("✅ 凭据状态查询成功")
            
            cred_icons = {
                'netrc_exists': "✅" if data['netrc_exists'] else "❌",
                'bearer_token': "✅" if data['bearer_token'] else "❌",
                'earthaccess_installed': "✅" if data['earthaccess_installed'] else "❌"
            }
            
            print(f"  {cred_icons['netrc_exists']} ~/.netrc: {'存在' if data['netrc_exists'] else '不存在'}")
            print(f"  {cred_icons['bearer_token']} EARTHDATA_BEARER: {'已设置' if data['bearer_token'] else '未设置'}")
            print(f"  {cred_icons['earthaccess_installed']} earthaccess包: {'已安装' if data['earthaccess_installed'] else '未安装'}")
        else:
            print(f"❌ 凭据状态查询失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 凭据状态查询异常: {e}")
        return False
    
    return True

def test_satellite_sync_with_fallback():
    """测试卫星数据同步（包含备用数据源回退）"""
    print("\n🔍 测试卫星数据同步（备用数据源回退）...")
    
    # 测试MODIS
    print("  📡 测试MODIS同步...")
    try:
        response = requests.post(f"{BASE_URL}/sync?source=modis")
        if response.status_code == 200:
            data = response.json()
            job_id = data['job_id']
            print(f"    ✅ 作业已创建: {job_id}")
            
            # 等待作业完成
            print("    ⏳ 等待作业完成...")
            for i in range(10):  # 最多等待10秒
                time.sleep(1)
                job_response = requests.get(f"{BASE_URL}/job/{job_id}")
                if job_response.status_code == 200:
                    job_data = job_response.json()
                    if job_data['status'] in ['succeeded', 'failed']:
                        print(f"    ✅ 作业完成: {job_data['status']}")
                        print(f"    📝 消息: {job_data['message']}")
                        print(f"    📊 记录数: {job_data['records']}")
                        break
                print(f"    ⏳ 作业状态: {job_data.get('status', 'unknown')}")
            else:
                print("    ⏰ 作业超时")
        else:
            print(f"    ❌ MODIS同步启动失败: {response.status_code}")
    except Exception as e:
        print(f"    ❌ MODIS同步异常: {e}")
    
    # 测试Sentinel-2
    print("  🛰️ 测试Sentinel-2同步...")
    try:
        response = requests.post(f"{BASE_URL}/sync?source=sentinel2")
        if response.status_code == 200:
            data = response.json()
            job_id = data['job_id']
            print(f"    ✅ 作业已创建: {job_id}")
            
            # 等待作业完成
            print("    ⏳ 等待作业完成...")
            for i in range(10):  # 最多等待10秒
                time.sleep(1)
                job_response = requests.get(f"{BASE_URL}/job/{job_id}")
                if job_response.status_code == 200:
                    job_data = job_response.json()
                    if job_data['status'] in ['succeeded', 'failed']:
                        print(f"    ✅ 作业完成: {job_data['status']}")
                        print(f"    📝 消息: {job_data['message']}")
                        print(f"    📊 记录数: {job_data['records']}")
                        break
                print(f"    ⏳ 作业状态: {job_data.get('status', 'unknown')}")
            else:
                print("    ⏰ 作业超时")
        else:
            print(f"    ❌ Sentinel-2同步启动失败: {response.status_code}")
    except Exception as e:
        print(f"    ❌ Sentinel-2同步异常: {e}")

def test_terrestrial_sync():
    """测试地面数据同步"""
    print("\n🔍 测试地面数据同步...")
    
    # 测试ECCC
    print("  🌤️ 测试ECCC同步...")
    try:
        response = requests.post(f"{BASE_URL}/sync?source=eccc")
        if response.status_code == 200:
            data = response.json()
            job_id = data['job_id']
            print(f"    ✅ 作业已创建: {job_id}")
            
            # 等待作业完成
            print("    ⏳ 等待作业完成...")
            for i in range(5):  # 最多等待5秒
                time.sleep(1)
                job_response = requests.get(f"{BASE_URL}/job/{job_id}")
                if job_response.status_code == 200:
                    job_data = job_response.json()
                    if job_data['status'] in ['succeeded', 'failed']:
                        print(f"    ✅ 作业完成: {job_data['status']}")
                        print(f"    📝 消息: {job_data['message']}")
                        print(f"    📊 记录数: {job_data['records']}")
                        break
                print(f"    ⏳ 作业状态: {job_data.get('status', 'unknown')}")
            else:
                print("    ⏰ 作业超时")
        else:
            print(f"    ❌ ECCC同步启动失败: {response.status_code}")
    except Exception as e:
        print(f"    ❌ ECCC同步异常: {e}")

def main():
    """主测试函数"""
    print("🚀 HydrAI-SWE 数据管道功能测试")
    print("=" * 60)
    print(f"⏰ 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试基本功能
    if not test_pipeline_status():
        print("❌ 基本功能测试失败，退出")
        return
    
    if not test_backup_status():
        print("❌ 备用数据源状态测试失败，退出")
        return
    
    if not test_credentials_status():
        print("❌ 凭据状态测试失败，退出")
        return
    
    # 测试数据同步
    test_satellite_sync_with_fallback()
    test_terrestrial_sync()
    
    # 最终状态检查
    print("\n🔍 最终状态检查...")
    time.sleep(2)
    test_pipeline_status()
    
    print("\n✅ 测试完成!")
    print(f"⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
