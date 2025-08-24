#!/usr/bin/env python3
"""
数据质量根本性修复脚本
解决数据分布、相关性和根本质量问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FundamentalDataFixer:
    """数据质量根本性修复器"""
    
    def __init__(self):
        """初始化"""
        self.scaler = StandardScaler()
        self.pca = None
        self.feature_importance = None
        
    def fundamental_fix(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """根本性修复数据质量"""
        try:
            logger.info("🚀 开始数据质量根本性修复...")
            logger.info(f"📊 原始数据: X={X.shape}, y={y.shape}")
            
            # 步骤1: 数据分布诊断
            logger.info("🔍 步骤1: 数据分布诊断...")
            distribution_analysis = self._analyze_distributions(X, y)
            
            # 步骤2: 特征相关性分析
            logger.info("🔍 步骤2: 特征相关性分析...")
            correlation_analysis = self._analyze_correlations(X, y)
            
            # 步骤3: 特征重要性分析
            logger.info("🔍 步骤3: 特征重要性分析...")
            importance_analysis = self._analyze_feature_importance(X, y)
            
            # 步骤4: 数据重构
            logger.info("🔧 步骤4: 数据重构...")
            X_reconstructed, y_reconstructed = self._reconstruct_data(X, y, importance_analysis)
            
            # 步骤5: 验证修复效果
            logger.info("🔍 步骤5: 验证修复效果...")
            validation_result = self._validate_fix(X_reconstructed, y_reconstructed)
            
            result = {
                'status': 'success',
                'X_fixed': X_reconstructed,
                'y_fixed': y_reconstructed,
                'distribution_analysis': distribution_analysis,
                'correlation_analysis': correlation_analysis,
                'importance_analysis': importance_analysis,
                'validation_result': validation_result,
                'original_shape': X.shape,
                'fixed_shape': X_reconstructed.shape
            }
            
            logger.info("🎉 数据质量根本性修复完成！")
            return result
            
        except Exception as e:
            logger.error(f"❌ 数据质量根本性修复失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_distributions(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """分析数据分布"""
        try:
            analysis = {}
            
            # 目标变量分布
            y_stats = {
                'mean': np.mean(y),
                'std': np.std(y),
                'min': np.min(y),
                'max': np.max(y),
                'skewness': stats.skew(y),
                'kurtosis': stats.kurtosis(y)
            }
            analysis['target_distribution'] = y_stats
            
            # 特征分布统计
            feature_stats = []
            for i in range(X.shape[1]):
                feature_data = X[:, i]
                feature_stats.append({
                    'feature_id': i,
                    'mean': np.mean(feature_data),
                    'std': np.std(feature_data),
                    'min': np.min(feature_data),
                    'max': np.max(feature_data),
                    'skewness': stats.skew(feature_data),
                    'kurtosis': stats.kurtosis(feature_data),
                    'missing_ratio': np.sum(np.isnan(feature_data)) / len(feature_data)
                })
            
            analysis['feature_distributions'] = feature_stats
            
            # 数据质量问题识别
            issues = []
            
            # 检查目标变量方差
            if np.var(y) < 1e-6:
                issues.append({
                    'type': 'low_target_variance',
                    'severity': 'critical',
                    'description': f'目标变量方差过低: {np.var(y):.6f}',
                    'recommendation': '目标变量缺乏变化，无法进行有效预测'
                })
            
            # 检查特征方差
            low_variance_features = [i for i, stats in enumerate(feature_stats) if stats['std'] < 1e-6]
            if low_variance_features:
                issues.append({
                    'type': 'low_feature_variance',
                    'severity': 'high',
                    'description': f'发现 {len(low_variance_features)} 个低方差特征',
                    'features': low_variance_features,
                    'recommendation': '移除低方差特征'
                })
            
            analysis['issues'] = issues
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 分布分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_correlations(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """分析特征相关性"""
        try:
            analysis = {}
            
            # 特征间相关性
            feature_corr = np.corrcoef(X.T)
            analysis['feature_correlation_matrix'] = feature_corr.tolist()
            
            # 特征与目标的相关性
            target_correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                if not np.isnan(corr):
                    target_correlations.append({
                        'feature_id': i,
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
            
            # 按相关性排序
            target_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
            analysis['target_correlations'] = target_correlations
            
            # 高相关特征对
            high_corr_pairs = []
            for i in range(X.shape[1]):
                for j in range(i+1, X.shape[1]):
                    corr = feature_corr[i, j]
                    if abs(corr) > 0.8:
                        high_corr_pairs.append({
                            'feature1': i,
                            'feature2': j,
                            'correlation': corr
                        })
            
            analysis['high_correlation_pairs'] = high_corr_pairs
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 相关性分析失败: {e}")
            return {'error': str(e)}
    
    def _analyze_feature_importance(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """分析特征重要性"""
        try:
            analysis = {}
            
            # 使用互信息
            try:
                mi_scores = mutual_info_regression(X, y, random_state=42)
                mi_features = [(i, score) for i, score in enumerate(mi_scores)]
                mi_features.sort(key=lambda x: x[1], reverse=True)
                analysis['mutual_info_scores'] = mi_features
            except:
                analysis['mutual_info_scores'] = []
            
            # 使用F检验
            try:
                f_scores, _ = f_regression(X, y)
                f_features = [(i, score) for i, score in enumerate(f_scores)]
                f_features.sort(key=lambda x: x[1], reverse=True)
                analysis['f_scores'] = f_features
            except:
                analysis['f_features'] = []
            
            # 综合重要性评分
            if analysis['mutual_info_scores'] and analysis['f_scores']:
                # 归一化分数
                mi_max = max(score for _, score in analysis['mutual_info_scores'])
                f_max = max(score for _, score in analysis['f_scores'])
                
                combined_scores = []
                for i in range(X.shape[1]):
                    mi_score = next(score for feat_id, score in analysis['mutual_info_scores'] if feat_id == i)
                    f_score = next(score for feat_id, score in analysis['f_scores'] if feat_id == i)
                    
                    # 归一化并组合
                    mi_norm = mi_score / mi_max if mi_max > 0 else 0
                    f_norm = f_score / f_max if f_max > 0 else 0
                    combined = (mi_norm + f_norm) / 2
                    
                    combined_scores.append((i, combined))
                
                combined_scores.sort(key=lambda x: x[1], reverse=True)
                analysis['combined_importance'] = combined_scores
            else:
                analysis['combined_importance'] = []
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ 特征重要性分析失败: {e}")
            return {'error': str(e)}
    
    def _reconstruct_data(self, X: np.ndarray, y: np.ndarray, importance_analysis: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """重构数据"""
        try:
            # 基于重要性选择特征
            if importance_analysis.get('combined_importance'):
                # 选择前5个最重要的特征
                top_features = [feat_id for feat_id, _ in importance_analysis['combined_importance'][:5]]
                X_selected = X[:, top_features]
                logger.info(f"✅ 基于重要性选择特征: {len(top_features)} 个")
            else:
                # 如果无法计算重要性，使用相关性
                target_corr = importance_analysis.get('target_correlations', [])
                if target_corr:
                    top_features = [feat['feature_id'] for feat in target_corr[:5]]
                    X_selected = X[:, top_features]
                    logger.info(f"✅ 基于相关性选择特征: {len(top_features)} 个")
                else:
                    # 最后手段：随机选择
                    top_features = list(range(min(5, X.shape[1])))
                    X_selected = X[:, top_features]
                    logger.info(f"⚠️ 随机选择特征: {len(top_features)} 个")
            
            # 数据标准化
            X_scaled = self.scaler.fit_transform(X_selected)
            
            # 异常值处理
            X_clean, y_clean = self._remove_outliers(X_scaled, y)
            
            # 数据增强
            X_augmented, y_augmented = self._augment_data(X_clean, y_clean, target_samples=100)
            
            return X_augmented, y_augmented
            
        except Exception as e:
            logger.error(f"❌ 数据重构失败: {e}")
            return X, y
    
    def _remove_outliers(self, X: np.ndarray, y: np.ndarray, threshold: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """移除异常值"""
        try:
            # 使用Z-score方法
            z_scores = np.abs(stats.zscore(X))
            outlier_mask = np.any(z_scores > threshold, axis=1)
            
            X_clean = X[~outlier_mask]
            y_clean = y[~outlier_mask]
            
            logger.info(f"✅ 异常值处理: {len(y)} -> {len(y_clean)} 样本")
            return X_clean, y_clean
            
        except Exception as e:
            logger.error(f"❌ 异常值处理失败: {e}")
            return X, y
    
    def _augment_data(self, X: np.ndarray, y: np.ndarray, target_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """数据增强"""
        try:
            if len(X) >= target_samples:
                return X, y
            
            # 通过插值和噪声增加样本
            additional_samples = target_samples - len(X)
            
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
                
                # 添加少量噪声
                noise_scale = 0.01
                X_noise = X_interp + np.random.normal(0, noise_scale, X_interp.shape)
                y_noise = y_interp + np.random.normal(0, noise_scale)
                
                X_augmented.append(X_noise)
                y_augmented.append(y_noise)
            
            # 合并数据
            X_final = np.vstack([X, np.array(X_augmented)])
            y_final = np.hstack([y, np.array(y_augmented)])
            
            logger.info(f"✅ 数据增强: {len(y)} -> {len(y_final)} 样本")
            return X_final, y_final
            
        except Exception as e:
            logger.error(f"❌ 数据增强失败: {e}")
            return X, y
    
    def _validate_fix(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """验证修复效果"""
        try:
            validation = {}
            
            # 数据质量指标
            validation['feature_count'] = X.shape[1]
            validation['sample_count'] = len(y)
            validation['target_variance'] = np.var(y)
            validation['feature_variance'] = [np.var(X[:, i]) for i in range(X.shape[1])]
            
            # 相关性检查
            target_correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], y)[0, 1]
                if not np.isnan(corr):
                    target_correlations.append(abs(corr))
            
            validation['avg_target_correlation'] = np.mean(target_correlations) if target_correlations else 0
            validation['max_target_correlation'] = np.max(target_correlations) if target_correlations else 0
            
            # 质量评分
            quality_score = 0
            if validation['target_variance'] > 1e-6:
                quality_score += 0.3
            if validation['avg_target_correlation'] > 0.1:
                quality_score += 0.3
            if validation['sample_count'] >= 50:
                quality_score += 0.2
            if validation['feature_count'] <= 10:
                quality_score += 0.2
            
            validation['quality_score'] = quality_score
            
            return validation
            
        except Exception as e:
            logger.error(f"❌ 验证失败: {e}")
            return {'error': str(e)}

def main():
    """主函数"""
    try:
        logger.info("🚀 启动数据质量根本性修复...")
        
        # 加载ERA5数据
        from src.models.agriculture.era5_soil_moisture_predictor import ERA5SoilMoisturePredictor
        
        predictor = ERA5SoilMoisturePredictor()
        data = predictor.load_data()
        
        # 展平数据
        X_train = data['X_train'].reshape(-1, data['X_train'].shape[-1])
        y_train = np.repeat(data['y_train'], data['X_train'].shape[1])
        
        logger.info(f"📊 原始数据: X={X_train.shape}, y={y_train.shape}")
        
        # 根本性修复
        fixer = FundamentalDataFixer()
        result = fixer.fundamental_fix(X_train, y_train)
        
        if result['status'] != 'success':
            logger.error(f"❌ 根本性修复失败: {result}")
            return
        
        # 保存修复后的数据
        output_dir = "data/processed/era5_fundamentally_fixed"
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_fundamentally_fixed.npy'), result['X_fixed'])
        np.save(os.path.join(output_dir, 'y_fundamentally_fixed.npy'), result['y_fixed'])
        
        # 保存分析报告
        report_file = f"fundamental_fix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ 修复后的数据已保存到: {output_dir}")
        logger.info(f"✅ 分析报告已保存: {report_file}")
        
        # 显示关键结果
        validation = result['validation_result']
        logger.info(f"📊 修复效果:")
        logger.info(f"  特征数量: {result['original_shape'][1]} -> {result['fixed_shape'][1]}")
        logger.info(f"  样本数量: {result['original_shape'][0]} -> {result['fixed_shape'][0]}")
        logger.info(f"  目标方差: {validation['target_variance']:.6f}")
        logger.info(f"  平均相关性: {validation['avg_target_correlation']:.4f}")
        logger.info(f"  质量评分: {validation['quality_score']:.2f}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    main()
