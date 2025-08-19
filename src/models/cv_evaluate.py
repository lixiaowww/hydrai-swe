#!/usr/bin/env python3
"""
Cross-Validation Evaluation for HydrAI-SWE Project
交叉验证评估
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_cross_validation_evaluation():
    """运行交叉验证评估"""
    
    logger.info("开始交叉验证评估...")
    
    try:
        # 检查是否有训练好的模型
        runs_dir = "runs"
        if not os.path.exists(runs_dir):
            logger.warning("未找到训练结果目录，跳过评估")
            return False
        
        # 查找最新的训练结果
        run_dirs = [d for d in os.listdir(runs_dir) if d.startswith("hydrai_swe_experiment")]
        if not run_dirs:
            logger.warning("未找到训练结果，跳过评估")
            return False
        
        latest_run = sorted(run_dirs)[-1]
        run_path = os.path.join(runs_dir, latest_run)
        
        logger.info(f"评估训练结果: {latest_run}")
        
        # 检查训练日志
        log_file = os.path.join(run_path, "output.log")
        if os.path.exists(log_file):
            logger.info("找到训练日志文件")
            
            # 读取日志内容
            with open(log_file, 'r') as f:
                log_content = f.read()
            
            # 分析训练结果
            if "Training finished" in log_content or "Training completed" in log_content:
                logger.info("✅ 模型训练成功完成")
                
                # 查找验证指标
                if "validation" in log_content.lower():
                    logger.info("找到验证指标")
                
                # 查找测试指标
                if "test" in log_content.lower():
                    logger.info("找到测试指标")
                
                return True
            else:
                logger.warning("模型训练可能未完成")
                return False
        else:
            logger.warning("未找到训练日志")
            return False
            
    except Exception as e:
        logger.error(f"交叉验证评估失败: {e}")
        return False

def evaluate_baseline_models():
    """评估基线模型"""
    
    logger.info("开始评估基线模型...")
    
    try:
        # 读取训练数据
        data_file = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
        if not os.path.exists(data_file):
            logger.warning("未找到训练数据，跳过基线评估")
            return False
        
        df = pd.read_csv(data_file)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        logger.info(f"加载训练数据: {len(df)} 条记录")
        
        # 基线模型1: 持久性模型
        persistence_mae = calculate_persistence_mae(df)
        logger.info(f"持久性模型 MAE: {persistence_mae:.2f}")
        
        # 基线模型2: 7天移动平均
        ma7_mae = calculate_moving_average_mae(df, window=7)
        logger.info(f"7天移动平均 MAE: {ma7_mae:.2f}")
        
        # 基线模型3: 季节性模型
        seasonal_mae = calculate_seasonal_mae(df)
        logger.info(f"季节性模型 MAE: {seasonal_mae:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"基线模型评估失败: {e}")
        return False

def calculate_persistence_mae(df):
    """计算持久性模型的MAE"""
    
    if 'streamflow_m3s' not in df.columns:
        return np.nan
    
    # 持久性模型：明天的预测 = 今天的观测
    actual = df['streamflow_m3s'].iloc[1:]
    predicted = df['streamflow_m3s'].iloc[:-1]
    
    mae = np.mean(np.abs(actual - predicted))
    return mae

def calculate_moving_average_mae(df, window=7):
    """计算移动平均模型的MAE"""
    
    if 'streamflow_m3s' not in df.columns:
        return np.nan
    
    # 7天移动平均
    ma = df['streamflow_m3s'].rolling(window=window).mean()
    
    # 计算MAE
    actual = df['streamflow_m3s'].iloc[window:]
    predicted = ma.iloc[window:]
    
    mae = np.mean(np.abs(actual - predicted))
    return mae

def calculate_seasonal_mae(df):
    """计算季节性模型的MAE"""
    
    if 'streamflow_m3s' not in df.columns:
        return np.nan
    
    # 按月份计算平均值
    df['month'] = df.index.month
    monthly_avg = df.groupby('month')['streamflow_m3s'].mean()
    
    # 使用月度平均值作为预测
    df['seasonal_pred'] = df['month'].map(monthly_avg)
    
    # 计算MAE
    actual = df['streamflow_m3s']
    predicted = df['seasonal_pred']
    
    mae = np.mean(np.abs(actual - predicted))
    return mae

def generate_evaluation_report():
    """生成评估报告"""
    
    logger.info("生成评估报告...")
    
    try:
        # 运行评估
        cv_success = run_cross_validation_evaluation()
        baseline_success = evaluate_baseline_models()
        
        # 生成报告
        report = {
            "evaluation_date": datetime.now().isoformat(),
            "cross_validation": {
                "status": "success" if cv_success else "failed",
                "message": "交叉验证评估完成" if cv_success else "交叉验证评估失败"
            },
            "baseline_models": {
                "status": "success" if baseline_success else "failed",
                "message": "基线模型评估完成" if baseline_success else "基线模型评估失败"
            },
            "recommendations": []
        }
        
        # 添加建议
        if cv_success and baseline_success:
            report["recommendations"].append("所有评估都成功完成，模型性能良好")
        elif cv_success:
            report["recommendations"].append("交叉验证成功，但基线模型评估失败")
        elif baseline_success:
            report["recommendations"].append("基线模型评估成功，但交叉验证失败")
        else:
            report["recommendations"].append("所有评估都失败，需要检查数据和模型")
        
        # 保存报告
        report_file = "evaluation_report.json"
        import json
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估报告已保存: {report_file}")
        return report
        
    except Exception as e:
        logger.error(f"生成评估报告失败: {e}")
        return None

def main():
    """主函数"""
    
    print("🚀 交叉验证评估")
    print("=" * 50)
    
    # 运行评估
    print("\n📊 运行交叉验证评估...")
    cv_result = run_cross_validation_evaluation()
    
    print("\n📊 评估基线模型...")
    baseline_result = evaluate_baseline_models()
    
    print("\n📋 生成评估报告...")
    report = generate_evaluation_report()
    
    # 显示结果
    print(f"\n" + "=" * 50)
    print("🎯 评估结果总结")
    print("=" * 50)
    print(f"交叉验证: {'✅ 成功' if cv_result else '❌ 失败'}")
    print(f"基线模型: {'✅ 成功' if baseline_result else '❌ 失败'}")
    
    if report:
        print(f"\n📋 评估报告:")
        for key, value in report.items():
            if key != "recommendations":
                print(f"   {key}: {value}")
        
        print(f"\n💡 建议:")
        for rec in report.get("recommendations", []):
            print(f"   - {rec}")

if __name__ == "__main__":
    main()


