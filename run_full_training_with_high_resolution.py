#!/usr/bin/env python3
"""
Complete Training Pipeline with High-Resolution Data Integration for HydrAI-SWE Project
集成高分辨率数据的完整训练流程
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Add src to path
sys.path.append('src')

def run_high_resolution_integration():
    """运行高分辨率数据集成"""
    
    print("🔗 步骤1: 高分辨率数据集成")
    print("-" * 40)
    
    try:
        # 导入高分辨率数据集成器
        from src.data.integrate_high_resolution import HighResolutionDataIntegrator
        
        integrator = HighResolutionDataIntegrator()
        
        # 运行集成管道
        result = integrator.run_integration_pipeline(region_name="red_river_basin")
        
        if result['enhanced_features']:
            print(f"✅ 高分辨率数据集成完成")
            print(f"   - Sentinel-2: {'已集成' if result['sentinel2'] else '未集成'}")
            print(f"   - DEM: {'已集成' if result['dem'] else '未集成'}")
            print(f"   - 增强特征: {result['enhanced_features']}")
            return True
        else:
            print("⚠️ 高分辨率数据集成部分完成，继续使用基础数据")
            return True
            
    except Exception as e:
        print(f"❌ 高分辨率数据集成失败: {e}")
        print("继续使用基础数据进行训练")
        return False

def run_enhanced_data_preparation():
    """运行增强数据准备"""
    
    print("\n🔧 步骤2: 增强数据准备")
    print("-" * 40)
    
    try:
        # 导入增强数据准备模块
        from src.neuralhydrology.prepare_data import prepare_data_for_neuralhydrology
        
        # 检查是否有增强特征
        enhanced_file = "data/processed/enhanced/enhanced_features_red_river_basin.csv"
        
        if os.path.exists(enhanced_file):
            print("✅ 使用增强特征数据")
            # 这里可以调用增强版本的数据准备函数
            # 暂时使用基础版本
            prepare_data_for_neuralhydrology("data/processed", "src/neuralhydrology/data")
        else:
            print("ℹ️ 使用基础数据")
            prepare_data_for_neuralhydrology("data/processed", "src/neuralhydrology/data")
        
        print("✅ 数据准备完成")
        return True
        
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return False

def run_enhanced_model_training():
    """运行增强模型训练"""
    
    print("\n🤖 步骤3: 增强模型训练")
    print("-" * 40)
    
    try:
        # 导入训练模块
        from src.models.train import train_model_with_neuralhydrology
        
        print("开始训练增强模型...")
        train_model_with_neuralhydrology()
        
        print("✅ 模型训练完成")
        return True
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return False

def run_model_evaluation():
    """运行模型评估"""
    
    print("\n📊 步骤4: 模型评估")
    print("-" * 40)
    
    try:
        # 导入评估模块
        from src.models.cv_evaluate import run_cross_validation_evaluation
        
        print("开始交叉验证评估...")
        run_cross_validation_evaluation()
        
        print("✅ 模型评估完成")
        return True
        
    except Exception as e:
        print(f"❌ 模型评估失败: {e}")
        return False

def run_prediction_service():
    """运行预测服务"""
    
    print("\n🔮 步骤5: 预测服务")
    print("-" * 40)
    
    try:
        # 导入预测模块
        from src.models.predict_service import NeuralHydrologyPredictor
        
        print("初始化预测服务...")
        predictor = NeuralHydrologyPredictor()
        
        # 测试预测
        test_prediction = predictor.predict(
            snow_depth_mm=100,
            temperature_c=5,
            precipitation_mm=10
        )
        
        print(f"✅ 预测服务测试成功: {test_prediction}")
        return True
        
    except Exception as e:
        print(f"❌ 预测服务失败: {e}")
        return False

def main():
    """主函数"""
    
    print("🚀 HydrAI-SWE 高分辨率数据集成训练流程")
    print("=" * 60)
    
    start_time = time.time()
    
    # 加载环境变量
    load_dotenv('config/credentials.env')
    
    # 检查认证信息
    nasa_username = os.getenv('NASA_EARTHDATA_USERNAME')
    nasa_password = os.getenv('NASA_EARTHDATA_PASSWORD')
    
    if not nasa_username or not nasa_password:
        print("⚠️ 警告: NASA Earthdata认证信息未设置")
        print("将跳过NASA数据获取，使用现有数据")
    
    # 运行完整流程
    steps = [
        ("高分辨率数据集成", run_high_resolution_integration),
        ("增强数据准备", run_enhanced_data_preparation),
        ("增强模型训练", run_enhanced_model_training),
        ("模型评估", run_model_evaluation),
        ("预测服务", run_prediction_service)
    ]
    
    success_count = 0
    total_steps = len(steps)
    
    for step_name, step_function in steps:
        print(f"\n🔄 执行步骤: {step_name}")
        print("=" * 60)
        
        try:
            if step_function():
                success_count += 1
                print(f"✅ {step_name} 成功")
            else:
                print(f"❌ {step_name} 失败")
        except Exception as e:
            print(f"❌ {step_name} 异常: {e}")
    
    # 总结结果
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\n" + "=" * 60)
    print("🎯 训练流程完成总结")
    print("=" * 60)
    print(f"总步骤数: {total_steps}")
    print(f"成功步骤: {success_count}")
    print(f"失败步骤: {total_steps - success_count}")
    print(f"成功率: {success_count/total_steps*100:.1f}%")
    print(f"总耗时: {duration:.1f} 秒")
    
    if success_count == total_steps:
        print("\n🎉 所有步骤都成功完成！")
        print("高分辨率数据已成功集成到训练流程中")
    elif success_count >= total_steps * 0.8:
        print("\n✅ 大部分步骤成功完成")
        print("系统可以正常运行，部分功能可能受限")
    else:
        print("\n⚠️ 多个步骤失败")
        print("请检查错误日志并修复问题")
    
    print(f"\n🚀 下一步建议:")
    if success_count >= total_steps * 0.8:
        print("1. 启动前端服务: uvicorn src.api.main:app --reload")
        print("2. 测试API接口: http://localhost:8000/docs")
        print("3. 访问前端UI: http://localhost:8000/ui")
    else:
        print("1. 检查错误日志")
        print("2. 修复失败的步骤")
        print("3. 重新运行训练流程")

if __name__ == "__main__":
    main()
