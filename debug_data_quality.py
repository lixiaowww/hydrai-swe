#!/usr/bin/env python3
"""
调试数据质量问题
详细分析为什么数据质量得分只有0.247
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from src.data.data_quality_detector import DataQualityDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_data_quality():
    """调试数据质量问题"""
    try:
        logger.info("🔍 开始调试数据质量问题...")
        
        # 加载ERA5数据
        from src.models.agriculture.era5_soil_moisture_predictor import ERA5SoilMoisturePredictor
        
        predictor = ERA5SoilMoisturePredictor()
        data = predictor.load_data()
        
        logger.info("📊 数据形状分析:")
        logger.info(f"  X_train: {data['X_train'].shape}")
        logger.info(f"  y_train: {data['y_train'].shape}")
        logger.info(f"  X_val: {data['X_val'].shape}")
        logger.info(f"  y_val: {data['y_val'].shape}")
        logger.info(f"  X_test: {data['X_test'].shape}")
        logger.info(f"  y_test: {data['y_test'].shape}")
        
        # 展平数据进行分析
        X_train_flat = data['X_train'].reshape(-1, data['X_train'].shape[-1])
        y_train_flat = data['y_train']
        
        logger.info(f"📊 展平后数据形状:")
        logger.info(f"  X_train_flat: {X_train_flat.shape}")
        logger.info(f"  y_train_flat: {y_train_flat.shape}")
        
        # 检查数据统计信息
        logger.info("📊 数据统计信息:")
        logger.info(f"  X_train 均值: {np.mean(X_train_flat, axis=0)[:5]}...")  # 显示前5个特征
        logger.info(f"  X_train 标准差: {np.std(X_train_flat, axis=0)[:5]}...")
        logger.info(f"  y_train 均值: {np.mean(y_train_flat):.6f}")
        logger.info(f"  y_train 标准差: {np.std(y_train_flat):.6f}")
        
        # 检查缺失值
        logger.info("📊 缺失值检查:")
        logger.info(f"  X_train NaN数量: {np.isnan(X_train_flat).sum()}")
        logger.info(f"  y_train NaN数量: {np.isnan(y_train_flat).sum()}")
        
        # 检查无穷值
        logger.info("📊 无穷值检查:")
        logger.info(f"  X_train Inf数量: {np.isinf(X_train_flat).sum()}")
        logger.info(f"  y_train Inf数量: {np.isinf(y_train_flat).sum()}")
        
        # 检查异常值
        logger.info("📊 异常值检查:")
        for i in range(min(5, X_train_flat.shape[1])):  # 检查前5个特征
            feature_data = X_train_flat[:, i]
            Q1 = np.percentile(feature_data, 25)
            Q3 = np.percentile(feature_data, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = np.sum((feature_data < lower_bound) | (feature_data > upper_bound))
            logger.info(f"  特征 {i}: 异常值数量 {outliers}/{len(feature_data)} ({outliers/len(feature_data)*100:.1f}%)")
        
        # 运行数据质量检测器
        logger.info("\n🔍 运行数据质量检测器...")
        detector = DataQualityDetector()
        quality_result = detector.detect_data_issues(X_train_flat, y_train_flat)
        
        if quality_result['status'] == 'success':
            logger.info(f"📊 数据质量检测结果:")
            logger.info(f"  质量得分: {quality_result['quality_score']:.3f}")
            logger.info(f"  问题总数: {quality_result['total_issues']}")
            logger.info(f"  严重问题: {quality_result['critical_issues']}")
            logger.info(f"  高严重性: {quality_result['high_issues']}")
            logger.info(f"  中等严重性: {quality_result['medium_issues']}")
            
            logger.info("\n📋 详细问题列表:")
            for i, issue in enumerate(quality_result['issues']):
                logger.info(f"  问题 {i+1}:")
                logger.info(f"    类型: {issue['type']}")
                logger.info(f"    严重性: {issue['severity']}")
                logger.info(f"    描述: {issue['description']}")
                logger.info(f"    建议: {issue['recommendation']}")
                logger.info("")
            
            logger.info("\n💡 改进建议:")
            for rec in quality_result['recommendations']:
                logger.info(f"  - {rec}")
        
        return quality_result
        
    except Exception as e:
        logger.error(f"❌ 调试数据质量失败: {e}")
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    debug_data_quality()
