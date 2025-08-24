#!/usr/bin/env python3
"""
修复真实ERA5土壤湿度预测模型
应用防过拟合系统解决R²为负值问题
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

# 导入防过拟合系统
from src.models.anti_overfitting_core import AntiOverfittingCore
from src.data.data_quality_detector import DataQualityDetector
from src.models.training_fixer import TrainingFixer

# 导入ERA5土壤湿度预测器
from src.models.agriculture.era5_soil_moisture_predictor import ERA5SoilMoisturePredictor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_real_soil_moisture_model():
    """修复真实ERA5土壤湿度预测模型"""
    try:
        logger.info("🚀 开始修复真实ERA5土壤湿度预测模型...")
        
        # 步骤1: 初始化ERA5土壤湿度预测器
        logger.info("🔧 步骤1: 初始化ERA5土壤湿度预测器...")
        predictor = ERA5SoilMoisturePredictor()
        
        # 步骤2: 获取训练数据
        logger.info("📊 步骤2: 获取训练数据...")
        try:
            # 加载数据
            data = predictor.load_data()
            logger.info("✅ 成功加载数据")
            
            # 创建数据加载器
            data_loaders = predictor.create_data_loaders(data)
            logger.info("✅ 成功创建数据加载器")
            
        except Exception as e:
            logger.error(f"❌ 数据准备失败: {e}")
            raise
        
        # 步骤3: 构建模型
        logger.info("🔧 步骤3: 构建模型...")
        input_size = data['X_train'].shape[-1]  # 获取特征数量
        predictor.build_model(input_size)
        logger.info("✅ 模型构建完成")
        
        # 步骤4: 初始化防过拟合系统
        logger.info("🔍 步骤3: 初始化防过拟合系统...")
        anti_overfitting = AntiOverfittingCore()
        data_quality = DataQualityDetector()
        training_fixer = TrainingFixer()
        
        # 步骤5: 训练原始模型并检测过拟合
        logger.info("🚀 步骤4: 训练原始模型...")
        training_result = predictor.train_model(data_loaders)
        
        if training_result['status'] != 'success':
            logger.error(f"❌ 模型训练失败: {training_result}")
            return {'status': 'error', 'error': '模型训练失败'}
        
        logger.info("✅ 原始模型训练完成")
        
        # 步骤6: 检测过拟合
        logger.info("🔍 步骤5: 检测过拟合...")
        if 'training_history' in training_result:
            train_losses = training_result['training_history'].get('train_loss', [])
            val_losses = training_result['training_history'].get('val_loss', [])
            
            if len(train_losses) >= 5 and len(val_losses) >= 5:
                overfitting_result = anti_overfitting.detect_overfitting(train_losses, val_losses)
                
                if overfitting_result['status'] == 'success':
                    logger.info(f"过拟合检测结果: {'是' if overfitting_result['overfitting'] else '否'}")
                    if overfitting_result['overfitting']:
                        logger.info(f"严重程度: {overfitting_result['severity']:.3f}")
                        logger.info(f"建议: {overfitting_result['recommendation']}")
                        
                        # 步骤7: 应用修复
                        logger.info("🔧 步骤6: 应用过拟合修复...")
                        fix_result = anti_overfitting.fix_overfitting(
                            predictor.model, 
                            data_loaders['train'].dataset.tensors[0].numpy(),
                            data_loaders['train'].dataset.tensors[1].numpy(),
                            data_loaders['val'].dataset.tensors[0].numpy(),
                            data_loaders['val'].dataset.tensors[1].numpy()
                        )
                        
                        if fix_result['status'] == 'success':
                            logger.info("✅ 过拟合修复完成")
                            
                            # 保存修复后的模型
                            predictor.save_model('fixed_soil_moisture_model.pth')
                            logger.info("✅ 修复后的模型已保存")
                            
                            return {
                                'status': 'success',
                                'message': '过拟合问题已修复',
                                'fix_result': fix_result,
                                'model_saved': 'fixed_soil_moisture_model.pth'
                            }
                        else:
                            logger.error(f"❌ 过拟合修复失败: {fix_result}")
                            return {'status': 'error', 'error': '过拟合修复失败'}
                    else:
                        logger.info("✅ 未检测到过拟合，模型训练正常")
                        return {
                            'status': 'success',
                            'message': '模型训练正常，无需修复',
                            'overfitting_detected': False
                        }
                else:
                    logger.warning(f"⚠️ 过拟合检测失败: {overfitting_result}")
            else:
                logger.warning("⚠️ 训练历史数据不足，无法检测过拟合")
        else:
            logger.warning("⚠️ 训练结果中缺少训练历史数据")
        
        # 如果没有过拟合，保存当前模型
        predictor.save_model('current_soil_moisture_model.pth')
        logger.info("✅ 当前模型已保存")
        
        return {
            'status': 'success',
            'message': '模型训练完成，已保存',
            'model_saved': 'current_soil_moisture_model.pth'
        }
        
    except Exception as e:
        logger.error(f"❌ 修复真实土壤湿度预测模型失败: {e}")
        return {'status': 'error', 'error': str(e)}

def evaluate_model_performance():
    """评估模型性能"""
    try:
        logger.info("📊 开始评估模型性能...")
        
        # 加载模型
        predictor = ERA5SoilMoisturePredictor()
        
        # 尝试加载已保存的模型
        model_files = ['current_soil_moisture_model.pth', 'best_model.pth']
        model_loaded = False
        
        for model_file in model_files:
            try:
                predictor.load_model(model_file)
                logger.info(f"✅ 成功加载模型: {model_file}")
                model_loaded = True
                break
            except Exception as e:
                logger.warning(f"⚠️ 加载模型 {model_file} 失败: {e}")
                continue
        
        if not model_loaded:
            logger.error("❌ 无法加载任何模型")
            return {'status': 'error', 'error': '无法加载模型'}
        
        # 准备测试数据
        data = predictor.load_data()
        data_loaders = predictor.create_data_loaders(data)
        
        # 评估模型
        predictor.model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for batch_X, batch_y in data_loaders['test']:
                outputs = predictor.model(batch_X)
                test_predictions.extend(outputs.squeeze().cpu().numpy())
                test_targets.extend(batch_y.cpu().numpy())
        
        # 计算性能指标
        test_predictions = np.array(test_predictions)
        test_targets = np.array(test_targets)
        
        # R²
        ss_res = np.sum((test_targets - test_predictions) ** 2)
        ss_tot = np.sum((test_targets - np.mean(test_targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # MAE
        mae = np.mean(np.abs(test_targets - test_predictions))
        
        # RMSE
        rmse = np.sqrt(np.mean((test_targets - test_predictions) ** 2))
        
        performance = {
            'r2_score': r2,
            'mae': mae,
            'rmse': rmse,
            'status': 'overfitting' if r2 < 0 else 'normal',
            'test_samples': len(test_targets)
        }
        
        logger.info(f"📊 模型性能评估完成:")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  状态: {'过拟合' if r2 < 0 else '正常'}")
        
        return {
            'status': 'success',
            'performance': performance
        }
        
    except Exception as e:
        logger.error(f"❌ 评估模型性能失败: {e}")
        return {'status': 'error', 'error': str(e)}

def main():
    """主函数"""
    try:
        logger.info("🚀 启动ERA5土壤湿度预测模型修复...")
        
        # 修复模型
        fix_result = fix_real_soil_moisture_model()
        
        if fix_result['status'] == 'success':
            logger.info("🎉 模型修复成功！")
            logger.info(f"📝 {fix_result['message']}")
            
            if 'model_saved' in fix_result:
                logger.info(f"💾 模型已保存: {fix_result['model_saved']}")
            
            # 评估修复后的性能
            logger.info("\n📊 评估修复后的模型性能...")
            performance_result = evaluate_model_performance()
            
            if performance_result['status'] == 'success':
                performance = performance_result['performance']
                logger.info(f"🎯 最终性能: R² = {performance['r2_score']:.4f}")
                
                if performance['r2_score'] < 0:
                    logger.warning("⚠️ R²仍为负值，可能需要进一步优化")
                else:
                    logger.info("✅ R²已转为正值，过拟合问题得到改善")
            else:
                logger.error(f"❌ 性能评估失败: {performance_result['error']}")
        else:
            logger.error(f"❌ 模型修复失败: {fix_result['error']}")
        
        return fix_result
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    main()
