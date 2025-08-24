#!/usr/bin/env python3
"""
优化后的农业模块测试脚本
测试改进的LSTM模型训练配置
"""

import requests
import json
import time
from datetime import datetime

# API基础URL
BASE_URL = "http://localhost:8000/api/v1/agriculture"

def test_optimized_training():
    """测试优化后的模型训练"""
    print("🚀 测试优化后的农业模块训练")
    print("=" * 60)
    
    # 1. 检查模型状态
    print("1️⃣ 检查当前模型状态...")
    try:
        response = requests.get(f"{BASE_URL}/models/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 模型状态: {data['models']['soil_moisture_predictor']['status']}")
        else:
            print(f"❌ 状态检查失败: {response.status_code}")
    except Exception as e:
        print(f"❌ 状态检查异常: {e}")
    
    # 2. 开始优化训练
    print("\n2️⃣ 开始优化后的模型训练...")
    try:
        response = requests.post(f"{BASE_URL}/models/train")
        if response.status_code == 200:
            data = response.json()
            print("✅ 训练成功启动!")
            print(f"📊 训练结果:")
            print(f"   - 最终训练损失: {data['training_results']['final_train_loss']:.6f}")
            print(f"   - 最终验证损失: {data['training_results']['final_val_loss']:.6f}")
            print(f"   - 测试指标: RMSE={data['training_results']['test_metrics']['rmse']:.4f}")
            print(f"   - 测试指标: R²={data['training_results']['test_metrics']['r2']:.4f}")
        else:
            print(f"❌ 训练失败: {response.status_code}")
            print(f"错误详情: {response.text}")
    except Exception as e:
        print(f"❌ 训练异常: {e}")
    
    # 3. 测试土壤水分预测
    print("\n3️⃣ 测试优化后的土壤水分预测...")
    try:
        payload = {
            "location": "red-river-basin",
            "start_date": "2024-01-01",
            "end_date": "2024-08-20"
        }
        
        response = requests.post(f"{BASE_URL}/soil-moisture/predict", json=payload)
        if response.status_code == 200:
            data = response.json()
            print("✅ 预测成功!")
            print(f"📊 预测统计:")
            print(f"   - 均值: {data['prediction_stats']['mean']:.2f}%")
            print(f"   - 标准差: {data['prediction_stats']['std']:.2f}%")
            print(f"   - 范围: {data['prediction_stats']['min']:.2f}% - {data['prediction_stats']['max']:.2f}%")
            print(f"   - 模型: {data['model_info']['type']} ({data['model_info']['layers']}层)")
        else:
            print(f"❌ 预测失败: {response.status_code}")
            print(f"错误详情: {response.text}")
    except Exception as e:
        print(f"❌ 预测异常: {e}")
    
    # 4. 性能对比
    print("\n4️⃣ 性能对比分析...")
    print("📈 优化前 vs 优化后:")
    print("   - 学习率: 0.001 → 0.0005 (更稳定)")
    print("   - 隐藏层: 128 → 64 (减少过拟合)")
    print("   - 网络层数: 2 → 1 (简化模型)")
    print("   - Dropout: 0.2 → 0.1 (提高训练稳定性)")
    print("   - Batch Size: 32 → 64 (提高训练稳定性)")
    print("   - 新增功能: 学习率调度器 + 早停机制")

def main():
    """主函数"""
    print(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🌐 API地址: {BASE_URL}")
    
    test_optimized_training()
    
    print("\n" + "=" * 60)
    print("🎯 测试完成!")
    print("💡 如果loss仍然很高，可能需要:")
    print("   1. 检查数据质量和标准化")
    print("   2. 调整特征工程")
    print("   3. 使用更简单的模型架构")

if __name__ == "__main__":
    main()
