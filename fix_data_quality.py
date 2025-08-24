#!/usr/bin/env python3
"""
数据质量修复器
解决特征过多、异常值过多、高相关特征等问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 导入数据质量检测器
from src.data.data_quality_detector import DataQualityDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataQualityFixer:
    """数据质量修复器"""
    
    def __init__(self):
        """初始化数据质量修复器"""
        self.quality_detector = DataQualityDetector()
        self.fix_history = []
        
        logger.info("✅ 数据质量修复器初始化完成")
    
    def fix_data_quality(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """修复数据质量问题"""
        try:
            logger.info("🔧 开始修复数据质量问题...")
            
            # 步骤1: 检测数据质量问题
            logger.info("🔍 步骤1: 检测数据质量问题...")
            quality_result = self.quality_detector.detect_data_issues(X, y)
            
            if quality_result['status'] != 'success':
                logger.error("❌ 数据质量检测失败")
                return {'status': 'error', 'error': '数据质量检测失败'}
            
            logger.info(f"📊 检测到 {quality_result['total_issues']} 个问题")
            
            # 步骤2: 应用修复策略
            logger.info("🔧 步骤2: 应用修复策略...")
            fixed_data = self._apply_fixes(X, y, quality_result)
            
            # 步骤3: 验证修复效果
            logger.info("🔍 步骤3: 验证修复效果...")
            validation_result = self.quality_detector.detect_data_issues(fixed_data['X_fixed'], y)
            
            # 生成修复报告
            fix_report = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'original_quality_score': quality_result['quality_score'],
                'fixed_quality_score': validation_result['quality_score'],
                'improvement': validation_result['quality_score'] - quality_result['quality_score'],
                'fixes_applied': fixed_data['fixes_applied'],
                'original_shape': X.shape,
                'fixed_shape': fixed_data['X_fixed'].shape,
                'X_fixed': fixed_data['X_fixed'],
                'y_fixed': fixed_data['y_fixed'],
                'details': {
                    'original_issues': quality_result,
                    'fixed_issues': validation_result
                }
            }
            
            self.fix_history.append(fix_report)
            
            logger.info("✅ 数据质量修复完成")
            logger.info(f"📊 质量得分提升: {quality_result['quality_score']:.3f} -> {validation_result['quality_score']:.3f}")
            
            return fix_report
            
        except Exception as e:
            logger.error(f"❌ 修复数据质量失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _apply_fixes(self, X: np.ndarray, y: np.ndarray, quality_result: Dict) -> Dict:
        """应用修复策略"""
        try:
            X_fixed = X.copy()
            y_fixed = y.copy()
            fixes_applied = []
            
            # 修复1: 处理异常值
            if quality_result.get('high_issues', 0) > 0:
                logger.info("🔧 修复1: 处理异常值...")
                X_fixed, outlier_fixes = self._fix_outliers(X_fixed)
                fixes_applied.extend(outlier_fixes)
            
            # 修复2: 特征选择（减少特征数量）
            if any(issue['type'] == 'too_many_features' for issue in quality_result.get('issues', [])):
                logger.info("🔧 修复2: 特征选择...")
                X_fixed, feature_fixes = self._select_features(X_fixed, y_fixed)
                fixes_applied.extend(feature_fixes)
            
            # 修复3: 移除高相关特征
            if any(issue['type'] == 'high_feature_correlation' for issue in quality_result.get('issues', [])):
                logger.info("🔧 修复3: 移除高相关特征...")
                X_fixed, correlation_fixes = self._remove_correlated_features(X_fixed)
                fixes_applied.extend(correlation_fixes)
            
            # 修复4: 数据标准化
            logger.info("🔧 修复4: 数据标准化...")
            X_fixed, scaling_fixes = self._standardize_features(X_fixed)
            fixes_applied.extend(scaling_fixes)
            
            return {
                'X_fixed': X_fixed,
                'y_fixed': y_fixed,
                'fixes_applied': fixes_applied
            }
            
        except Exception as e:
            logger.error(f"❌ 应用修复策略失败: {e}")
            raise
    
    def _fix_outliers(self, X: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """修复异常值"""
        try:
            fixes_applied = []
            X_fixed = X.copy()
            
            # 使用IQR方法处理异常值
            for col in range(X.shape[1]):
                feature_data = X[:, col]
                Q1 = np.percentile(feature_data, 25)
                Q3 = np.percentile(feature_data, 75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 统计异常值
                outliers_mask = (feature_data < lower_bound) | (feature_data > upper_bound)
                outlier_count = np.sum(outliers_mask)
                
                if outlier_count > 0:
                    # 将异常值替换为边界值
                    X_fixed[outliers_mask, col] = np.clip(
                        X_fixed[outliers_mask, col], 
                        lower_bound, 
                        upper_bound
                    )
                    
                    fixes_applied.append({
                        'type': 'outlier_fix',
                        'feature': col,
                        'outliers_fixed': outlier_count,
                        'method': 'IQR_clipping'
                    })
            
            logger.info(f"✅ 异常值修复完成，修复了 {len(fixes_applied)} 个特征")
            return X_fixed, fixes_applied
            
        except Exception as e:
            logger.error(f"❌ 修复异常值失败: {e}")
            return X, []
    
    def _select_features(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """特征选择"""
        try:
            fixes_applied = []
            
            # 计算合适的特征数量（样本数的1/3）
            max_features = max(5, len(X) // 3)
            
            if X.shape[1] > max_features:
                logger.info(f"🔧 从 {X.shape[1]} 个特征中选择 {max_features} 个最重要的特征")
                
                # 使用F检验选择特征
                selector = SelectKBest(score_func=f_regression, k=max_features)
                X_selected = selector.fit_transform(X, y)
                
                # 获取选中的特征索引
                selected_features = selector.get_support(indices=True)
                
                fixes_applied.append({
                    'type': 'feature_selection',
                    'original_features': X.shape[1],
                    'selected_features': max_features,
                    'method': 'F_test',
                    'selected_indices': selected_features.tolist()
                })
                
                logger.info(f"✅ 特征选择完成: {X.shape[1]} -> {X_selected.shape[1]}")
                return X_selected, fixes_applied
            else:
                logger.info("✅ 特征数量合适，无需选择")
                return X, fixes_applied
                
        except Exception as e:
            logger.error(f"❌ 特征选择失败: {e}")
            return X, []
    
    def _remove_correlated_features(self, X: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """移除高相关特征"""
        try:
            fixes_applied = []
            
            if X.shape[1] <= 1:
                return X, fixes_applied
            
            # 计算特征相关性矩阵
            corr_matrix = np.corrcoef(X.T)
            
            # 找到高相关特征对（相关系数 > 0.95）
            high_corr_pairs = []
            for i in range(corr_matrix.shape[0]):
                for j in range(i + 1, corr_matrix.shape[1]):
                    if abs(corr_matrix[i, j]) > 0.95:
                        high_corr_pairs.append((i, j))
            
            if high_corr_pairs:
                logger.info(f"🔧 发现 {len(high_corr_pairs)} 对高相关特征")
                
                # 移除高相关特征（保留第一个）
                features_to_remove = set()
                for i, j in high_corr_pairs:
                    features_to_remove.add(j)  # 移除第二个特征
                
                # 保留要保留的特征
                features_to_keep = [i for i in range(X.shape[1]) if i not in features_to_remove]
                X_uncorr = X[:, features_to_keep]
                
                fixes_applied.append({
                    'type': 'correlation_fix',
                    'high_corr_pairs': len(high_corr_pairs),
                    'features_removed': len(features_to_remove),
                    'features_kept': len(features_to_keep),
                    'method': 'high_correlation_removal'
                })
                
                logger.info(f"✅ 高相关特征移除完成: {X.shape[1]} -> {X_uncorr.shape[1]}")
                return X_uncorr, fixes_applied
            else:
                logger.info("✅ 没有发现高相关特征")
                return X, fixes_applied
                
        except Exception as e:
            logger.error(f"❌ 移除高相关特征失败: {e}")
            return X, []
    
    def _standardize_features(self, X: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """特征标准化"""
        try:
            fixes_applied = []
            
            # 使用Z-score标准化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            fixes_applied.append({
                'type': 'standardization',
                'method': 'Z_score',
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            })
            
            logger.info("✅ 特征标准化完成")
            return X_scaled, fixes_applied
            
        except Exception as e:
            logger.error(f"❌ 特征标准化失败: {e}")
            return X, []
    
    def get_fix_summary(self) -> Dict:
        """获取修复摘要"""
        if not self.fix_history:
            return {"message": "暂无修复记录"}
        
        total_fixes = len(self.fix_history)
        successful_fixes = sum(1 for r in self.fix_history if r['status'] == 'success')
        
        # 计算平均改进
        improvements = [r['improvement'] for r in self.fix_history if r['status'] == 'success']
        avg_improvement = np.mean(improvements) if improvements else 0
        
        return {
            "total_fixes": total_fixes,
            "successful_fixes": successful_fixes,
            "success_rate": f"{successful_fixes/total_fixes*100:.1f}%",
            "average_improvement": f"{avg_improvement:.3f}",
            "last_fix": self.fix_history[-1].get('timestamp', 'Unknown')
        }

def main():
    """主函数"""
    try:
        logger.info("🚀 启动数据质量修复...")
        
        # 加载ERA5数据
        from src.models.agriculture.era5_soil_moisture_predictor import ERA5SoilMoisturePredictor
        
        predictor = ERA5SoilMoisturePredictor()
        data = predictor.load_data()
        
        # 展平数据 - 正确处理序列数据
        X_train = data['X_train'].reshape(-1, data['X_train'].shape[-1])  # (11*7, 35) = (77, 35)
        y_train = np.repeat(data['y_train'], data['X_train'].shape[1])    # 重复y值以匹配展平后的X
        
        logger.info(f"📊 原始数据形状: X={X_train.shape}, y={y_train.shape}")
        
        # 创建数据质量修复器
        fixer = DataQualityFixer()
        
        # 修复数据质量
        result = fixer.fix_data_quality(X_train, y_train)
        
        if result['status'] == 'success':
            logger.info("🎉 数据质量修复成功！")
            logger.info(f"📊 质量得分提升: {result['improvement']:.3f}")
            logger.info(f"🔧 应用修复: {len(result['fixes_applied'])} 个")
            logger.info(f"📐 数据形状: {result['original_shape']} -> {result['fixed_shape']}")
            
            # 保存修复后的数据
            X_fixed = result.get('X_fixed')
            y_fixed = result.get('y_fixed')
            
            if X_fixed is None or y_fixed is None:
                logger.error("❌ 修复后的数据为空")
                return result
            
            # 保存到文件
            output_dir = "data/processed/era5_fixed"
            os.makedirs(output_dir, exist_ok=True)
            
            np.save(os.path.join(output_dir, 'X_train_fixed.npy'), X_fixed)
            np.save(os.path.join(output_dir, 'y_train_fixed.npy'), y_fixed)
            
            logger.info(f"✅ 修复后的数据已保存到: {output_dir}")
            
            # 保存修复报告
            report_file = f"data_quality_fix_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ 修复报告已保存: {report_file}")
            
        else:
            logger.error(f"❌ 数据质量修复失败: {result.get('error', '未知错误')}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    main()
