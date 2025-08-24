#!/usr/bin/env python3
"""
使用修复后的数据重新训练模型
验证数据质量修复是否解决了R²为负值的问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging
from datetime import datetime
import json

# 导入ERA5土壤湿度预测器
from src.models.agriculture.era5_soil_moisture_predictor import ERA5SoilMoisturePredictor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_with_fixed_data():
    """使用修复后的数据重新训练模型"""
    try:
        logger.info("🚀 开始使用修复后的数据重新训练模型...")
        
        # 步骤1: 加载修复后的数据
        logger.info("📊 步骤1: 加载修复后的数据...")
        fixed_data_dir = "data/processed/era5_fixed"
        
        if not os.path.exists(fixed_data_dir):
            logger.error(f"❌ 修复后的数据目录不存在: {fixed_data_dir}")
            return {'status': 'error', 'error': '修复后的数据不存在'}
        
        # 加载修复后的数据
        X_train_fixed = np.load(os.path.join(fixed_data_dir, 'X_train_fixed.npy'))
        y_train_fixed = np.load(os.path.join(fixed_data_dir, 'y_train_fixed.npy'))
        
        logger.info(f"✅ 成功加载修复后的数据: X={X_train_fixed.shape}, y={y_train_fixed.shape}")
        
        # 步骤2: 重新构建数据
        logger.info("🔧 步骤2: 重新构建数据...")
        
        # 将展平的数据重新组织为序列格式
        sequence_length = 7  # 原始序列长度
        n_samples = len(y_train_fixed) // sequence_length
        
        # 重新组织为序列格式
        X_train_reshaped = X_train_fixed[:n_samples * sequence_length].reshape(n_samples, sequence_length, -1)
        y_train_reshaped = y_train_fixed[:n_samples * sequence_length:sequence_length]  # 每个序列取一个y值
        
        logger.info(f"✅ 数据重新组织完成: X={X_train_reshaped.shape}, y={y_train_reshaped.shape}")
        
        # 步骤3: 创建新的数据加载器
        logger.info("🔧 步骤3: 创建数据加载器...")
        
        # 分割训练集和验证集
        train_size = int(0.8 * len(X_train_reshaped))
        X_train = X_train_reshaped[:train_size]
        y_train = y_train_reshaped[:train_size]
        X_val = X_train_reshaped[train_size:]
        y_val = y_train_reshaped[train_size:]
        
        logger.info(f"📊 数据分割完成: 训练集 {X_train.shape}, 验证集 {X_val.shape}")
        
        # 创建数据加载器
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train), 
            torch.FloatTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val), 
            torch.FloatTensor(y_val)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=len(X_val))
        
        data_loaders = {
            'train': train_loader,
            'val': val_loader,
            'test': val_loader  # 暂时用验证集作为测试集
        }
        
        logger.info("✅ 数据加载器创建完成")
        
        # 步骤4: 创建并训练新模型
        logger.info("🔧 步骤4: 创建并训练新模型...")
        
        # 创建新的预测器
        predictor = ERA5SoilMoisturePredictor()
        
        # 构建模型（使用修复后的特征数量）
        input_size = X_train.shape[-1]
        predictor.build_model(input_size)
        
        logger.info(f"✅ 新模型构建完成，输入特征数: {input_size}")
        
        # 训练模型
        training_result = predictor.train_model(data_loaders)
        
        if training_result['status'] != 'success':
            logger.error(f"❌ 模型训练失败: {training_result}")
            return {'status': 'error', 'error': '模型训练失败'}
        
        logger.info("✅ 新模型训练完成")
        
        # 步骤5: 评估新模型性能
        logger.info("📊 步骤5: 评估新模型性能...")
        
        # 使用验证集评估
        predictor.model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loaders['val']:
                outputs = predictor.model(batch_X)
                val_predictions.extend(outputs.squeeze().cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # 计算性能指标
        val_predictions = np.array(val_predictions)
        val_targets = np.array(val_targets)
        
        # R²
        ss_res = np.sum((val_targets - val_predictions) ** 2)
        ss_tot = np.sum((val_targets - np.mean(val_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAE
        mae = np.mean(np.abs(val_targets - val_predictions))
        
        # RMSE
        rmse = np.sqrt(np.mean((val_targets - val_predictions) ** 2))
        
        performance = {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'status': 'overfitting' if r2 < 0 else 'normal',
            'val_samples': len(val_targets)
        }
        
        logger.info(f"📊 新模型性能评估完成:")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  状态: {'过拟合' if r2 < 0 else '正常'}")
        
        # 保存新模型
        predictor.save_model('model_with_fixed_data.pth')
        logger.info("✅ 新模型已保存")
        
        # 生成训练报告
        report = {
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'data_quality_improvement': {
                'original_features': 35,
                'fixed_features': input_size,
                'feature_reduction': f"{((35 - input_size) / 35 * 100):.1f}%"
            },
            'model_performance': performance,
            'training_summary': training_result,
            'data_shape': {
                'X_train': X_train.shape,
                'y_train': y_train.shape,
                'X_val': X_val.shape,
                'y_val': y_val.shape
            }
        }
        
        # 保存报告
        report_file = f"training_with_fixed_data_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ 训练报告已保存: {report_file}")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ 使用修复后的数据训练模型失败: {e}")
        return {'status': 'error', 'error': str(e)}

def main():
    """主函数"""
    try:
        logger.info("🚀 启动使用修复后的数据重新训练模型...")
        
        # 训练模型
        result = train_with_fixed_data()
        
        if result['status'] == 'success':
            logger.info("🎉 使用修复后的数据训练模型成功！")
            
            # 显示性能结果
            performance = result['model_performance']
            logger.info(f"🎯 最终性能: R² = {performance['r2_score']:.4f}")
            
            if performance['r2_score'] > 0:
                logger.info("✅ 成功！R²已转为正值，数据质量修复有效！")
            else:
                logger.info("⚠️ R²仍为负值，可能需要进一步优化")
            
            # 显示数据质量改进
            data_improvement = result['data_quality_improvement']
            logger.info(f"📊 数据质量改进:")
            logger.info(f"  特征数量: {data_improvement['original_features']} -> {data_improvement['fixed_features']}")
            logger.info(f"  特征减少: {data_improvement['feature_reduction']}")
            
        else:
            logger.error(f"❌ 训练失败: {result.get('error', '未知错误')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    main()
