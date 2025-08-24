#!/usr/bin/env python3
"""
交叉验证系统测试脚本
测试历史数据交叉验证功能的完整流程
"""

import os
import sys
import json
import time
from pathlib import Path
import requests
import pandas as pd

# 添加src到路径
sys.path.append('src')

from src.models.flood_risk_cross_validation import FloodRiskCrossValidator
from src.models.flood_risk_assessment import FloodRiskAssessment

def test_cross_validation_model():
    """测试交叉验证模型"""
    print("🧪 测试交叉验证模型...")
    
    try:
        # 创建交叉验证实例（不需要参数）
        cv_validator = FloodRiskCrossValidator()
        
        print("✅ 交叉验证模型初始化成功")
        return cv_validator
        
    except Exception as e:
        print(f"❌ 交叉验证模型初始化失败: {e}")
        return None

def test_historical_data_loading():
    """测试历史数据加载"""
    print("\n📊 测试历史数据加载...")
    
    try:
        # 检查是否有可用的历史数据
        data_files = [
            "data/processed/hydat_streamflow_processed.csv",
            "data/raw/era5_land/era5_land_soil_moisture_sample_2024-01-01_2024-01-07.csv",
            "data/raw/smap/smap_soil_moisture_sample_2024-01-01_2024-01-07.csv"
        ]
        
        available_data = []
        for file_path in data_files:
            if Path(file_path).exists():
                available_data.append(file_path)
                print(f"✅ 找到数据文件: {file_path}")
            else:
                print(f"⚠️ 数据文件不存在: {file_path}")
        
        if not available_data:
            print("❌ 没有找到可用的历史数据文件")
            return False
        
        print(f"✅ 找到 {len(available_data)} 个数据文件")
        return True
        
    except Exception as e:
        print(f"❌ 历史数据加载测试失败: {e}")
        return False

def test_cross_validation_execution(cv_validator):
    """测试交叉验证执行"""
    print("\n🚀 测试交叉验证执行...")
    
    try:
        # 测试参数
        start_date = "2024-01-01"
        end_date = "2024-01-07"
        station_id = "05OC001"
        
        # 执行交叉验证
        print(f"执行交叉验证: 站点: {station_id}")
        
        result = cv_validator.run_cross_validation(
            data_path="data/processed/hydat_streamflow_processed.csv",
            stations=[station_id],
            validation_windows=5,
            forecast_horizon=7
        )
        
        if result:
            print("✅ 交叉验证执行成功")
            print(f"结果文件: {result}")
            return result
        else:
            print("❌ 交叉验证执行失败")
            return None
            
    except Exception as e:
        print(f"❌ 交叉验证执行测试失败: {e}")
        return None

def test_api_endpoints():
    """测试API端点"""
    print("\n🌐 测试API端点...")
    
    base_url = "http://localhost:8000"
    
    try:
        # 测试健康检查
        response = requests.get(f"{base_url}/api/v1/cross-validation/health", timeout=10)
        if response.status_code == 200:
            print("✅ 健康检查端点正常")
        else:
            print(f"⚠️ 健康检查端点异常: {response.status_code}")
        
        # 测试可用数据端点
        response = requests.get(f"{base_url}/api/v1/cross-validation/available-data", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 可用数据端点正常: {len(data.get('files', []))} 个文件")
        else:
            print(f"⚠️ 可用数据端点异常: {response.status_code}")
        
        # 测试快速验证端点
        quick_data = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-07",
            "station_id": "05OC001"
        }
        
        response = requests.post(
            f"{base_url}/api/v1/cross-validation/quick",
            json=quick_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ 快速验证端点正常")
            print(f"验证结果: {result.get('message', 'N/A')}")
        else:
            print(f"⚠️ 快速验证端点异常: {response.status_code}")
            if response.status_code != 500:  # 500可能是服务器内部错误
                print(f"错误详情: {response.text}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ 无法连接到API服务器，请确保服务器正在运行")
        return False
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False

def test_soil_moisture_integration():
    """测试土壤湿度数据集成"""
    print("\n🌱 测试土壤湿度数据集成...")
    
    try:
        # 导入土壤湿度模块
        from src.data.soil_moisture import SoilMoistureIntegrator
        
        # 创建集成器实例
        integrator = SoilMoistureIntegrator()
        
        # 测试参数
        start_date = "2024-01-01"
        end_date = "2024-01-07"
        bounding_box = (-97.5, 49.0, -96.5, 50.5)  # 红河流域
        
        # 下载数据
        print("下载土壤湿度数据...")
        download_results = integrator.download_all_soil_moisture_data(
            start_date, end_date, bounding_box
        )
        print(f"下载结果: {download_results}")
        
        # 集成数据
        print("集成土壤湿度数据...")
        integrated_data = integrator.integrate_soil_moisture_data(
            start_date, end_date, bounding_box
        )
        
        if not integrated_data.empty:
            print(f"✅ 土壤湿度数据集成成功: {len(integrated_data)} 条记录")
            
            # 生成摘要
            summary = integrator.generate_soil_moisture_summary(integrated_data)
            print("数据摘要:")
            print(json.dumps(summary, indent=2, ensure_ascii=False))
            
            return True
        else:
            print("❌ 土壤湿度数据集成失败")
            return False
            
    except Exception as e:
        print(f"❌ 土壤湿度数据集成测试失败: {e}")
        return False

def run_full_test():
    """运行完整测试"""
    print("🚀 开始交叉验证系统完整测试")
    print("=" * 50)
    
    # 测试1: 交叉验证模型
    cv_validator = test_cross_validation_model()
    if not cv_validator:
        print("❌ 模型测试失败，停止测试")
        return False
    
    # 测试2: 历史数据加载
    if not test_historical_data_loading():
        print("⚠️ 历史数据加载测试失败，但继续测试")
    
    # 测试3: 交叉验证执行
    cv_result = test_cross_validation_execution(cv_validator)
    if not cv_result:
        print("⚠️ 交叉验证执行测试失败，但继续测试")
    
    # 测试4: API端点
    api_success = test_api_endpoints()
    
    # 测试5: 土壤湿度数据集成
    soil_success = test_soil_moisture_integration()
    
    # 测试总结
    print("\n" + "=" * 50)
    print("📊 测试总结")
    print("=" * 50)
    
    test_results = {
        "交叉验证模型": "✅ 通过" if cv_validator else "❌ 失败",
        "历史数据加载": "✅ 通过" if test_historical_data_loading() else "⚠️ 部分通过",
        "交叉验证执行": "✅ 通过" if cv_result else "⚠️ 部分通过",
        "API端点": "✅ 通过" if api_success else "❌ 失败",
        "土壤湿度集成": "✅ 通过" if soil_success else "❌ 失败"
    }
    
    for test_name, result in test_results.items():
        print(f"{test_name}: {result}")
    
    # 计算成功率
    passed_tests = sum(1 for result in test_results.values() if "✅" in result)
    total_tests = len(test_results)
    success_rate = (passed_tests / total_tests) * 100
    
    print(f"\n总体成功率: {success_rate:.1f}% ({passed_tests}/{total_tests})")
    
    if success_rate >= 80:
        print("🎉 测试总体成功！系统基本可用")
        return True
    elif success_rate >= 60:
        print("⚠️ 测试部分成功，系统需要改进")
        return False
    else:
        print("❌ 测试失败较多，系统需要重大修复")
        return False

def main():
    """主函数"""
    print("HydrAI-SWE 交叉验证系统测试")
    print("=" * 50)
    
    # 检查工作目录
    if not Path("src").exists():
        print("❌ 请在项目根目录运行此脚本")
        return
    
    # 运行完整测试
    success = run_full_test()
    
    if success:
        print("\n🎯 建议下一步:")
        print("1. 启动API服务器: uvicorn src.api.main:app --reload")
        print("2. 访问Web界面: http://localhost:8000/ui")
        print("3. 测试交叉验证功能")
    else:
        print("\n🔧 需要修复的问题:")
        print("1. 检查依赖包安装")
        print("2. 检查数据文件")
        print("3. 检查API服务器状态")

if __name__ == "__main__":
    main()
