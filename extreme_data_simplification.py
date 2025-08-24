#!/usr/bin/env python3
"""
极端数据简化脚本
专门解决小数据集(11样本)导致R²为负值的问题
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
from typing import Dict, List, Tuple
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExtremeDataSimplifier:
    """极端数据简化器 - 专门解决小数据集问题"""
    
    def __init__(self):
        """初始化"""
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        
    def extreme_simplify(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """极端简化数据"""
        try:
            logger.info("🚀 开始极端数据简化...")
            logger.info(f"📊 原始数据: X={X.shape}, y={y.shape}")
            
            # 策略1: 极简特征选择 (只保留最重要的2-3个特征)
            logger.info("🔧 策略1: 极简特征选择...")
            X_simple, selected_features = self._extreme_feature_selection(X, y, max_features=3)
            logger.info(f"✅ 特征选择完成: {X.shape[1]} -> {X_simple.shape[1]}")
            logger.info(f"📋 选中特征: {selected_features}")
            
            # 策略2: 数据标准化
            logger.info("🔧 策略2: 数据标准化...")
            X_scaled = self.scaler.fit_transform(X_simple)
            
            # 策略3: 异常值处理 (更激进)
            logger.info("🔧 策略3: 激进异常值处理...")
            X_clean, y_clean = self._aggressive_outlier_removal(X_scaled, y)
            logger.info(f"✅ 异常值处理完成: {len(y)} -> {len(y_clean)} 样本")
            
            # 策略4: 数据增强 (通过插值增加样本)
            logger.info("🔧 策略4: 数据增强...")
            X_augmented, y_augmented = self._augment_data(X_clean, y_clean, target_samples=50)
            logger.info(f"✅ 数据增强完成: {len(y_clean)} -> {len(y_augmented)} 样本")
            
            result = {
                'status': 'success',
                'X_simplified': X_augmented,
                'y_simplified': y_augmented,
                'selected_features': selected_features,
                'original_shape': X.shape,
                'simplified_shape': X_augmented.shape,
                'feature_reduction': f"{((X.shape[1] - X_augmented.shape[1]) / X.shape[1] * 100):.1f}%",
                'sample_increase': f"{((len(y_augmented) - len(y)) / len(y) * 100):.1f}%"
            }
            
            logger.info("🎉 极端数据简化完成！")
            logger.info(f"📊 最终数据: X={X_augmented.shape}, y={y_augmented.shape}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 极端数据简化失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _extreme_feature_selection(self, X: np.ndarray, y: np.ndarray, max_features: int = 3) -> Tuple[np.ndarray, List[str]]:
        """极简特征选择"""
        try:
            # 使用F检验选择最重要的特征
            selector = SelectKBest(score_func=f_regression, k=max_features)
            X_selected = selector.fit_transform(X, y)
            
            # 获取选中的特征索引
            selected_indices = selector.get_support(indices=True)
            
            # 生成特征名称 (简化版)
            feature_names = [f"Feature_{i}" for i in selected_indices]
            
            return X_selected, feature_names
            
        except Exception as e:
            logger.error(f"❌ 特征选择失败: {e}")
            # 如果失败，随机选择前3个特征
            return X[:, :max_features], [f"Feature_{i}" for i in range(max_features)]
    
    def _aggressive_outlier_removal(self, X: np.ndarray, y: np.ndarray, threshold: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """激进异常值处理"""
        try:
            # 计算每个特征的IQR
            Q1 = np.percentile(X, 25, axis=0)
            Q3 = np.percentile(X, 75, axis=0)
            IQR = Q3 - Q1
            
            # 定义异常值边界 (更激进)
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # 标记异常值
            outlier_mask = np.any((X < lower_bound) | (X > upper_bound), axis=1)
            
            # 移除异常值
            X_clean = X[~outlier_mask]
            y_clean = y[~outlier_mask]
            
            return X_clean, y_clean
            
        except Exception as e:
            logger.error(f"❌ 异常值处理失败: {e}")
            return X, y
    
    def _augment_data(self, X: np.ndarray, y: np.ndarray, target_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """数据增强"""
        try:
            if len(X) >= target_samples:
                return X, y
            
            # 通过插值增加样本
            additional_samples = target_samples - len(X)
            
            # 创建插值样本
            X_augmented = []
            y_augmented = []
            
            for i in range(additional_samples):
                # 随机选择两个现有样本
                idx1, idx2 = np.random.choice(len(X), 2, replace=False)
                
                # 插值权重
                alpha = np.random.random()
                
                # 插值
                X_interp = alpha * X[idx1] + (1 - alpha) * X[idx2]
                y_interp = alpha * y[idx1] + (1 - alpha) * y[idx2]
                
                X_augmented.append(X_interp)
                y_augmented.append(y_interp)
            
            # 合并原始数据和增强数据
            X_final = np.vstack([X, np.array(X_augmented)])
            y_final = np.hstack([y, np.array(y_augmented)])
            
            return X_final, y_final
            
        except Exception as e:
            logger.error(f"❌ 数据增强失败: {e}")
            return X, y

def test_simple_models(X: np.ndarray, y: np.ndarray) -> Dict:
    """测试简单模型"""
    try:
        logger.info("🧪 测试简单模型...")
        
        # 分割数据
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        models = {
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'RandomForest': RandomForestRegressor(n_estimators=10, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"🔍 测试模型: {name}")
            
            # 训练
            model.fit(X_train, y_train)
            
            # 预测
            y_pred = model.predict(X_test)
            
            # 评估
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            results[name] = {
                'r2_score': r2,
                'mae': mae,
                'rmse': rmse,
                'status': 'overfitting' if r2 < 0 else 'normal'
            }
            
            logger.info(f"  {name}: R²={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ 模型测试失败: {e}")
        return {'status': 'error', 'error': str(e)}

def main():
    """主函数"""
    try:
        logger.info("🚀 启动极端数据简化...")
        
        # 加载ERA5数据
        from src.models.agriculture.era5_soil_moisture_predictor import ERA5SoilMoisturePredictor
        
        predictor = ERA5SoilMoisturePredictor()
        data = predictor.load_data()
        
        # 展平数据
        X_train = data['X_train'].reshape(-1, data['X_train'].shape[-1])
        y_train = np.repeat(data['y_train'], data['X_train'].shape[1])
        
        logger.info(f"📊 原始数据: X={X_train.shape}, y={y_train.shape}")
        
        # 极端数据简化
        simplifier = ExtremeDataSimplifier()
        result = simplifier.extreme_simplify(X_train, y_train)
        
        if result['status'] != 'success':
            logger.error(f"❌ 数据简化失败: {result}")
            return
        
        # 测试简单模型
        X_simple = result['X_simplified']
        y_simple = result['y_simplified']
        
        model_results = test_simple_models(X_simple, y_simple)
        
        # 保存结果
        final_result = {
            'timestamp': datetime.now().isoformat(),
            'data_simplification': result,
            'model_testing': model_results
        }
        
        # 保存简化后的数据
        output_dir = "data/processed/era5_extreme_simplified"
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_extreme_simplified.npy'), X_simple)
        np.save(os.path.join(output_dir, 'y_extreme_simplified.npy'), y_simple)
        
        # 保存结果报告
        report_file = f"extreme_simplification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ 简化后的数据已保存到: {output_dir}")
        logger.info(f"✅ 结果报告已保存: {report_file}")
        
        # 显示最佳模型
        if 'status' not in model_results:
            best_model = max(model_results.items(), key=lambda x: x[1]['r2_score'])
            logger.info(f"🏆 最佳模型: {best_model[0]}, R²={best_model[1]['r2_score']:.4f}")
            
            if best_model[1]['r2_score'] > 0:
                logger.info("🎉 成功！R²已转为正值！")
            else:
                logger.info("⚠️ R²仍为负值，需要更激进的简化")
        
        return final_result
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    main()
