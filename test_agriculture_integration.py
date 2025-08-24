#!/usr/bin/env python3
"""
HydrAI-SWE 农业模块集成测试脚本
测试从GitHub集成的农业AI功能
"""

import requests
import json
import time
from datetime import datetime

# API基础URL
BASE_URL = "http://localhost:8000/api/v1/agriculture"

def test_agriculture_health():
    """测试农业模块健康检查"""
    print("🔍 测试农业模块健康检查...")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查通过: {data['status']}")
            print(f"📊 可用功能: {', '.join(data['features'])}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def test_soil_moisture_prediction():
    """测试土壤水分预测"""
    print("\n🌱 测试土壤水分预测...")
    
    try:
        payload = {
            "location": "red-river-basin",
            "start_date": "2024-01-01",
            "end_date": "2024-08-20"
        }
        
        response = requests.post(f"{BASE_URL}/soil-moisture/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 土壤水分预测成功")
            print(f"📊 预测统计: 均值={data['prediction_stats']['mean']:.2f}%, 范围={data['prediction_stats']['min']:.2f}%-{data['prediction_stats']['max']:.2f}%")
            print(f"🤖 模型信息: {data['model_info']['type']} ({data['model_info']['layers']}层)")
            return True
        else:
            print(f"❌ 土壤水分预测失败: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 土壤水分预测异常: {e}")
        return False

def test_crop_recommendation():
    """测试作物推荐"""
    print("\n🌾 测试作物推荐...")
    
    try:
        payload = {
            "location": "manitoba-province",
            "temperature": 20.0,
            "precipitation": 300.0,
            "soil_moisture": 25.0,
            "soil_type": "loam"
        }
        
        response = requests.post(f"{BASE_URL}/crop/recommend", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 作物推荐成功")
            print(f"🌡️ 环境条件: {data['environmental_conditions']['temperature']}°C, {data['environmental_conditions']['precipitation']}mm, {data['environmental_conditions']['soil_moisture']}%")
            print(f"🥇 高度适宜: {', '.join(data['recommendations']['highly_suitable'])}")
            print(f"🥈 适宜: {', '.join(data['recommendations']['suitable'])}")
            return True
        else:
            print(f"❌ 作物推荐失败: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 作物推荐异常: {e}")
        return False

def test_yield_prediction():
    """测试产量预测"""
    print("\n📈 测试产量预测...")
    
    try:
        payload = {
            "crop_type": "corn",
            "location": "red-river-basin",
            "planting_date": "2024-05-01",
            "weather_conditions": {
                "temperature": 22.0,
                "precipitation": 400.0,
                "soil_moisture": 30.0
            }
        }
        
        response = requests.post(f"{BASE_URL}/yield/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 产量预测成功")
            print(f"🌽 作物: {data['crop_type']}")
            print(f"📊 预测产量: {data['predicted_yield']['value']} {data['predicted_yield']['unit']}")
            print(f"🎯 置信区间: {data['predicted_yield']['confidence_interval'][0]}-{data['predicted_yield']['confidence_interval'][1]} {data['predicted_yield']['unit']}")
            print(f"❓ 不确定性: ±{data['predicted_yield']['uncertainty']} {data['predicted_yield']['unit']}")
            return True
        else:
            print(f"❌ 产量预测失败: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 产量预测异常: {e}")
        return False

def test_available_features():
    """测试可用特征获取"""
    print("\n🔍 测试可用特征获取...")
    
    try:
        response = requests.get(f"{BASE_URL}/data/available-features")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 特征获取成功")
            print(f"📊 总特征数: {data['total_features']}")
            print(f"🌤️ 天气特征: {', '.join(data['feature_categories']['weather'][:5])}...")
            print(f"❄️ 雪特征: {', '.join(data['feature_categories']['snow'])}")
            print(f"⏰ 时间特征: {', '.join(data['feature_categories']['temporal'])}")
            return True
        else:
            print(f"❌ 特征获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 特征获取异常: {e}")
        return False

def test_model_status():
    """测试模型状态"""
    print("\n🤖 测试模型状态...")
    
    try:
        response = requests.get(f"{BASE_URL}/models/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 模型状态获取成功")
            print(f"🌱 土壤水分预测器: {data['models']['soil_moisture_predictor']['status']}")
            if data['models']['soil_moisture_predictor']['config']:
                config = data['models']['soil_moisture_predictor']['config']
                print(f"⚙️ 配置: {config['hidden_size']}隐藏单元, {config['num_layers']}层, {config['dropout']}dropout")
            return True
        else:
            print(f"❌ 模型状态获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 模型状态获取异常: {e}")
        return False

def test_model_training():
    """测试模型训练"""
    print("\n🚀 测试模型训练...")
    
    try:
        response = requests.post(f"{BASE_URL}/models/train")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 模型训练成功")
            print(f"📊 训练结果: 训练损失={data['training_results']['final_train_loss']:.4f}, 验证损失={data['training_results']['final_val_loss']:.4f}")
            print(f"🎯 测试指标: RMSE={data['training_results']['test_metrics']['rmse']:.4f}, R²={data['training_results']['test_metrics']['r2']:.4f}")
            return True
        else:
            print(f"❌ 模型训练失败: {response.status_code}")
            print(f"错误详情: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 模型训练异常: {e}")
        return False

def main():
    """主测试函数"""
    print("🚀 HydrAI-SWE 农业模块集成测试")
    print("=" * 60)
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API地址: {BASE_URL}")
    
    # 测试结果统计
    test_results = []
    
    # 执行测试
    tests = [
        ("健康检查", test_agriculture_health),
        ("土壤水分预测", test_soil_moisture_prediction),
        ("作物推荐", test_crop_recommendation),
        ("产量预测", test_yield_prediction),
        ("可用特征", test_available_features),
        ("模型状态", test_model_status),
        ("模型训练", test_model_training)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
            time.sleep(1)  # 避免API过载
        except Exception as e:
            print(f"❌ {test_name}测试异常: {e}")
            test_results.append((test_name, False))
    
    # 输出测试总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！农业模块集成成功！")
    else:
        print("⚠️ 部分测试失败，请检查相关功能")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
