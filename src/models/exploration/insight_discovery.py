#!/usr/bin/env python3
"""
无监督探索模块 - 发现问题背后的模式
定位：探索 + 解释，补充预测的可信度和理解
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    # 在logger定义之前不能使用logger
    PLOTTING_AVAILABLE = False
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InsightDiscoveryModule:
    """无监督探索模块 - 发现问题背后的模式"""
    
    def __init__(self):
        """初始化探索模块"""
        self.scaler = StandardScaler()
        self.pca = None
        self.clusters = None
        self.anomalies = None
        self.insights = {}
        
        logger.info("🔍 无监督探索模块初始化完成")
    
    def discover_patterns(self, data: pd.DataFrame, target_col: str = 'estimated_soil_moisture') -> Dict:
        """发现数据背后的模式"""
        try:
            logger.info("🚀 开始无监督模式发现...")
            
            # 步骤1: 数据预处理
            logger.info("🔧 步骤1: 数据预处理...")
            X_processed = self._preprocess_data(data)
            
            # 步骤2: 异常检测
            logger.info("🔍 步骤2: 异常检测...")
            anomaly_insights = self._detect_anomalies(X_processed, data)
            
            # 步骤3: 聚类分析
            logger.info("🔍 步骤3: 聚类分析...")
            cluster_insights = self._cluster_analysis(X_processed, data)
            
            # 步骤4: 降维可视化
            logger.info("🔍 步骤4: 降维可视化...")
            dimension_insights = self._dimension_reduction(X_processed, data)
            
            # 步骤5: 时间模式分析
            logger.info("🔍 步骤5: 时间模式分析...")
            temporal_insights = self._temporal_patterns(data, target_col)
            
            # 步骤6: 风险机制识别
            logger.info("🔍 步骤6: 风险机制识别...")
            risk_insights = self._identify_risk_mechanisms(data, target_col)
            
            # 步骤7: 重要影响因素发现
            logger.info("🔍 步骤7: 重要影响因素发现...")
            factor_insights = self._discover_important_factors(data, target_col)
            
            # 步骤8: 相关性网络分析
            logger.info("🔍 步骤8: 相关性网络分析...")
            correlation_insights = self._analyze_correlation_network(data, target_col)
            
            # 步骤9: SWE冷门因素发现
            logger.info("🔍 步骤9: SWE冷门因素发现...")
            swe_cold_factors = self._discover_swe_cold_factors(data, target_col)
            
            # 整合所有洞察
            self.insights = {
                'timestamp': datetime.now().isoformat(),
                'anomalies': anomaly_insights,
                'clusters': cluster_insights,
                'dimensions': dimension_insights,
                'temporal': temporal_insights,
                'risk_mechanisms': risk_insights,
                'important_factors': factor_insights,
                'correlation_network': correlation_insights,
                'swe_cold_factors': swe_cold_factors
            }
            
            # 步骤10: 生成摘要 (在所有洞察构建完成后)
            logger.info("🔍 步骤10: 生成摘要...")
            summary_insights = self._generate_summary()
            self.insights['summary'] = summary_insights
            
            logger.info("🎉 无监督模式发现完成！")
            return self.insights
            
        except Exception as e:
            logger.error(f"❌ 模式发现失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """改进的数据预处理 - 专门处理高缺失率数据"""
        try:
            # 选择数值特征
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # 移除目标变量
            if 'estimated_soil_moisture' in numeric_cols:
                numeric_cols.remove('estimated_soil_moisture')
            
            # 分析缺失值模式
            missing_rates = data[numeric_cols].isnull().sum() / len(data)
            logger.info(f"📊 缺失值分析: 总特征数 {len(numeric_cols)}")
            
            # 只保留缺失率 < 50% 的特征
            valid_features = missing_rates[missing_rates < 0.5].index.tolist()
            high_missing_features = missing_rates[missing_rates >= 0.5].index.tolist()
            
            logger.info(f"✅ 有效特征: {len(valid_features)} 个 (缺失率 < 50%)")
            logger.info(f"⚠️ 高缺失特征: {len(high_missing_features)} 个 (缺失率 >= 50%)")
            
            if len(valid_features) == 0:
                logger.warning("⚠️ 没有有效特征，尝试放宽标准到80%缺失率")
                valid_features = missing_rates[missing_rates < 0.8].index.tolist()
                if len(valid_features) == 0:
                    logger.error("❌ 所有特征缺失率都过高，无法进行有效分析")
                    return np.array([])
            
            # 使用更智能的缺失值填充
            X = data[valid_features].copy()
            
            # 对于连续变量，使用中位数填充
            for col in X.columns:
                if X[col].dtype in ['float64', 'int64']:
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        # 如果中位数也是NaN，使用0填充
                        X[col] = X[col].fillna(0)
                    else:
                        X[col] = X[col].fillna(median_val)
            
            # 检查是否还有NaN值
            remaining_nans = X.isnull().sum().sum()
            if remaining_nans > 0:
                logger.warning(f"⚠️ 仍有 {remaining_nans} 个NaN值，使用0填充")
                X = X.fillna(0)
            
            # 标准化
            X_scaled = self.scaler.fit_transform(X)
            
            logger.info(f"✅ 改进的数据预处理完成: {X.shape[1]} 个有效特征")
            logger.info(f"📊 数据形状: {X_scaled.shape}")
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"❌ 改进的数据预处理失败: {e}")
            return np.array([])
    
    def _detect_anomalies(self, X: np.ndarray, data: pd.DataFrame) -> Dict:
        """异常检测"""
        try:
            # 使用Isolation Forest检测异常
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(X)
            
            # 统计异常
            anomaly_count = np.sum(anomaly_labels == -1)
            total_count = len(anomaly_labels)
            anomaly_rate = anomaly_count / total_count
            
            # 分析异常特征
            anomaly_data = data[anomaly_labels == -1]
            normal_data = data[anomaly_labels == 1]
            
            anomaly_insights = {
                'anomaly_count': int(anomaly_count),
                'anomaly_rate': float(anomaly_rate),
                'total_count': int(total_count),
                'anomaly_features': self._analyze_anomaly_features(anomaly_data, normal_data),
                'anomaly_timestamps': anomaly_data['timestamp'].tolist() if 'timestamp' in anomaly_data.columns else []
            }
            
            logger.info(f"✅ 异常检测完成: 发现 {anomaly_count} 个异常 ({anomaly_rate:.1%})")
            return anomaly_insights
            
        except Exception as e:
            logger.error(f"❌ 异常检测失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_anomaly_features(self, anomaly_data: pd.DataFrame, normal_data: pd.DataFrame) -> Dict:
        """分析异常特征"""
        try:
            numeric_cols = anomaly_data.select_dtypes(include=[np.number]).columns.tolist()
            
            feature_analysis = {}
            for col in numeric_cols:
                if col in normal_data.columns:
                    anomaly_mean = anomaly_data[col].mean()
                    normal_mean = normal_data[col].mean()
                    difference = anomaly_mean - normal_mean
                    
                    feature_analysis[col] = {
                        'anomaly_mean': float(anomaly_mean),
                        'normal_mean': float(normal_mean),
                        'difference': float(difference),
                        'deviation': float(abs(difference) / normal_mean) if normal_mean != 0 else 0
                    }
            
            return feature_analysis
            
        except Exception as e:
            logger.error(f"❌ 异常特征分析失败: {e}")
            return {}
    
    def _cluster_analysis(self, X: np.ndarray, data: pd.DataFrame) -> Dict:
        """改进的聚类分析 - 增加缺失值检查"""
        try:
            # 检查输入数据是否包含NaN
            if np.isnan(X).any():
                logger.error("❌ 输入数据包含NaN值，无法进行聚类分析")
                return {
                    'status': 'error', 
                    'error': '输入数据包含NaN值，请先完成数据预处理'
                }
            
            # 检查数据是否为空
            if X.size == 0:
                logger.error("❌ 输入数据为空，无法进行聚类分析")
                return {
                    'status': 'error', 
                    'error': '输入数据为空，请检查数据预处理步骤'
                }
            
            # 确定最佳聚类数
            silhouette_scores = []
            K_range = range(2, min(11, len(X) // 10 + 1))
            
            if len(K_range) == 0:
                logger.warning("⚠️ 数据量过少，使用默认聚类数2")
                K_range = [2]
            
            for k in K_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(X)
                    score = silhouette_score(X, cluster_labels)
                    silhouette_scores.append(score)
                except Exception as e:
                    logger.warning(f"⚠️ 聚类数 {k} 失败: {e}")
                    silhouette_scores.append(-1)
            
            if not silhouette_scores or max(silhouette_scores) == -1:
                logger.warning("⚠️ 所有聚类数都失败，使用默认聚类数2")
                best_k = 2
                best_score = 0
            else:
                best_k = K_range[np.argmax(silhouette_scores)]
                best_score = max(silhouette_scores)
            
            # 执行最佳聚类
            try:
                kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X)
                
                # 分析聚类特征
                data_with_clusters = data.copy()
                data_with_clusters['cluster'] = cluster_labels
                
                cluster_insights = {
                    'optimal_clusters': int(best_k),
                    'silhouette_score': float(best_score),
                    'cluster_sizes': data_with_clusters['cluster'].value_counts().to_dict(),
                    'cluster_characteristics': self._analyze_cluster_characteristics(data_with_clusters)
                }
                
                self.clusters = cluster_labels
                
                logger.info(f"✅ 聚类分析完成: 最佳聚类数 {best_k}, 轮廓系数 {best_score:.3f}")
                return cluster_insights
                
            except Exception as e:
                logger.error(f"❌ 最终聚类失败: {e}")
                return {'status': 'error', 'error': str(e)}
            
        except Exception as e:
            logger.error(f"❌ 聚类分析失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_cluster_characteristics(self, data: pd.DataFrame) -> Dict:
        """分析聚类特征"""
        try:
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if 'cluster' in numeric_cols:
                numeric_cols.remove('cluster')
            
            cluster_chars = {}
            for cluster_id in data['cluster'].unique():
                cluster_data = data[data['cluster'] == cluster_id]
                
                cluster_profile = {}
                for col in numeric_cols:
                    cluster_profile[col] = {
                        'mean': float(cluster_data[col].mean()),
                        'std': float(cluster_data[col].std()),
                        'min': float(cluster_data[col].min()),
                        'max': float(cluster_data[col].max())
                    }
                
                cluster_chars[f'cluster_{cluster_id}'] = {
                    'size': int(len(cluster_data)),
                    'profile': cluster_profile
                }
            
            return cluster_chars
            
        except Exception as e:
            logger.error(f"❌ 聚类特征分析失败: {e}")
            return {}
    
    def _dimension_reduction(self, X: np.ndarray, data: pd.DataFrame) -> Dict:
        """改进的降维分析 - 增加缺失值检查"""
        try:
            # 检查输入数据是否包含NaN
            if np.isnan(X).any():
                logger.error("❌ 输入数据包含NaN值，无法进行降维分析")
                return {
                    'status': 'error', 
                    'error': '输入数据包含NaN值，请先完成数据预处理'
                }
            
            # 检查数据是否为空
            if X.size == 0:
                logger.error("❌ 输入数据为空，无法进行降维分析")
                return {
                    'status': 'error', 
                    'error': '输入数据为空，请检查数据预处理步骤'
                }
            
            # 检查特征数量
            if X.shape[1] < 2:
                logger.warning("⚠️ 特征数量少于2，无法进行PCA降维")
                return {
                    'status': 'warning',
                    'message': '特征数量不足，跳过PCA降维'
                }
            
            # PCA降维
            n_components = min(3, X.shape[1])
            self.pca = PCA(n_components=n_components)
            
            try:
                X_pca = self.pca.fit_transform(X)
                
                # 分析主成分
                explained_variance = self.pca.explained_variance_ratio_
                cumulative_variance = np.cumsum(explained_variance)
                
                # 特征重要性
                feature_importance = {}
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                
                for i, component in enumerate(self.pca.components_):
                    for j, importance in enumerate(component):
                        if j < len(numeric_cols):
                            col_name = numeric_cols[j]
                            if col_name not in feature_importance:
                                feature_importance[col_name] = []
                            feature_importance[col_name].append(float(importance))
                
                dimension_insights = {
                    'n_components': int(n_components),
                    'explained_variance': explained_variance.tolist(),
                    'cumulative_variance': cumulative_variance.tolist(),
                    'feature_importance': feature_importance,
                    'pca_data': X_pca.tolist()
                }
                
                logger.info(f"✅ 降维分析完成: 保留 {n_components} 个主成分")
                logger.info(f"📊 解释方差: {explained_variance}")
                logger.info(f"📊 累积方差: {cumulative_variance}")
                
                return dimension_insights
                
            except Exception as e:
                logger.error(f"❌ PCA计算失败: {e}")
                return {'status': 'error', 'error': str(e)}
            
        except Exception as e:
            logger.error(f"❌ 降维分析失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _temporal_patterns(self, data: pd.DataFrame, target_col: str) -> Dict:
        """改进的时间模式分析 - 增加数据列检查"""
        try:
            # 检查是否有时间相关列
            time_columns = []
            if 'Date/Time' in data.columns:
                time_columns.append('Date/Time')
            if 'Year' in data.columns:
                time_columns.append('Year')
            if 'Month' in data.columns:
                time_columns.append('Month')
            if 'Day' in data.columns:
                time_columns.append('Day')
            
            if not time_columns:
                logger.warning("⚠️ 未找到时间相关列，跳过时间模式分析")
                return {
                    'status': 'warning',
                    'message': '未找到时间相关列，跳过时间模式分析'
                }
            
            # 检查目标列
            if target_col not in data.columns:
                logger.warning(f"⚠️ 目标列 '{target_col}' 不存在，使用第一个数值列")
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    target_col = numeric_cols[0]
                else:
                    logger.error("❌ 没有找到数值列，无法进行时间模式分析")
                    return {
                        'status': 'error',
                        'error': '没有找到数值列，无法进行时间模式分析'
                    }
            
            # 时间模式分析
            temporal_insights = {
                'time_columns_found': time_columns,
                'target_column': target_col,
                'patterns': {}
            }
            
            # 年度模式
            if 'Year' in data.columns:
                yearly_stats = data.groupby('Year')[target_col].agg(['mean', 'std', 'min', 'max']).reset_index()
                temporal_insights['patterns']['yearly'] = {
                    'yearly_stats': yearly_stats.to_dict('records'),
                    'yearly_trend': 'stable'  # 简化处理
                }
            
            # 月度模式
            if 'Month' in data.columns:
                monthly_stats = data.groupby('Month')[target_col].agg(['mean', 'std', 'min', 'max']).reset_index()
                temporal_insights['patterns']['monthly'] = {
                    'monthly_stats': monthly_stats.to_dict('records'),
                    'seasonal_pattern': 'detected'  # 简化处理
                }
            
            # 日模式
            if 'Day' in data.columns:
                daily_stats = data.groupby('Day')[target_col].agg(['mean', 'std', 'min', 'max']).reset_index()
                temporal_insights['patterns']['daily'] = {
                    'daily_stats': daily_stats.to_dict('records')
                }
            
            logger.info(f"✅ 时间模式分析完成: 分析了 {len(time_columns)} 个时间列")
            return temporal_insights
            
        except Exception as e:
            logger.error(f"❌ 时间模式分析失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_seasonal_trends(self, data: pd.DataFrame, target_col: str) -> Dict:
        """分析季节性趋势"""
        try:
            # 计算移动平均
            data_sorted = data.sort_values('timestamp')
            data_sorted[f'{target_col}_ma7'] = data_sorted[target_col].rolling(window=7).mean()
            data_sorted[f'{target_col}_ma30'] = data_sorted[target_col].rolling(window=30).mean()
            
            # 季节性统计
            seasonal_stats = data_sorted.groupby('month')[target_col].agg(['mean', 'std', 'min', 'max']).to_dict()
            
            return {
                'seasonal_stats': seasonal_stats,
                'trend_data': {
                    'ma7': data_sorted[f'{target_col}_ma7'].dropna().tolist(),
                    'ma30': data_sorted[f'{target_col}_ma30'].dropna().tolist()
                }
            }
            
        except Exception as e:
            logger.error(f"❌ 季节性趋势分析失败: {e}")
            return {}
    
    def _identify_risk_mechanisms(self, data: pd.DataFrame, target_col: str) -> Dict:
        """识别风险机制"""
        try:
            risk_mechanisms = {}
            
            # 1. 极端值风险
            if target_col in data.columns:
                target_data = data[target_col].dropna()
                q1, q3 = target_data.quantile([0.25, 0.75])
                iqr = q3 - q1
                extreme_threshold = 1.5 * iqr
                
                extreme_low = q1 - extreme_threshold
                extreme_high = q3 + extreme_threshold
                
                extreme_events = data[
                    (data[target_col] < extreme_low) | 
                    (data[target_col] > extreme_high)
                ]
                
                risk_mechanisms['extreme_values'] = {
                    'threshold_low': float(extreme_low),
                    'threshold_high': float(extreme_high),
                    'extreme_count': int(len(extreme_events)),
                    'risk_level': 'high' if len(extreme_events) > len(data) * 0.1 else 'medium'
                }
            
            # 2. 数据质量风险
            missing_rates = data.isnull().sum() / len(data)
            high_missing_features = missing_rates[missing_rates > 0.1].index.tolist()
            
            risk_mechanisms['data_quality'] = {
                'high_missing_features': high_missing_features,
                'overall_missing_rate': float(data.isnull().sum().sum() / (len(data) * len(data.columns))),
                'risk_level': 'high' if len(high_missing_features) > 0 else 'low'
            }
            
            # 3. 时间连续性风险
            if 'timestamp' in data.columns:
                data_sorted = data.sort_values('timestamp')
                time_gaps = data_sorted['timestamp'].diff().dt.total_seconds() / 3600  # 小时
                large_gaps = time_gaps[time_gaps > 24]  # 超过24小时的间隔
                
                risk_mechanisms['temporal_continuity'] = {
                    'large_gaps_count': int(len(large_gaps)),
                    'max_gap_hours': float(large_gaps.max()) if len(large_gaps) > 0 else 0,
                    'risk_level': 'high' if len(large_gaps) > 0 else 'low'
                }
            
            logger.info("✅ 风险机制识别完成")
            return risk_mechanisms
            
        except Exception as e:
            logger.error(f"❌ 风险机制识别失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _generate_summary(self) -> Dict:
        """改进的摘要生成 - 处理新的分析结果格式"""
        try:
            summary = {
                'total_insights': 0,
                'key_findings': [],
                'risk_assessment': 'unknown',
                'recommendations': []
            }
            
            # 统计洞察数量
            insight_count = 0
            
            # 异常检测洞察
            if 'anomalies' in self.insights and 'anomaly_count' in self.insights['anomalies']:
                insight_count += 1
                anomaly_rate = self.insights['anomalies'].get('anomaly_rate', 0)
                if anomaly_rate > 0.1:
                    summary['key_findings'].append(f"发现异常数据比例较高: {anomaly_rate:.1%}")
                else:
                    summary['key_findings'].append(f"异常检测正常: {anomaly_rate:.1%}")
            
            # 聚类分析洞察
            if 'clusters' in self.insights and 'optimal_clusters' in self.insights['clusters']:
                insight_count += 1
                optimal_clusters = self.insights['clusters'].get('optimal_clusters', 0)
                silhouette_score = self.insights['clusters'].get('silhouette_score', 0)
                summary['key_findings'].append(f"聚类分析完成: 最佳聚类数 {optimal_clusters}, 轮廓系数 {silhouette_score:.3f}")
            elif 'clusters' in self.insights and self.insights['clusters'].get('status') == 'warning':
                summary['key_findings'].append(f"聚类分析: {self.insights['clusters'].get('message', '警告')}")
            
            # 降维分析洞察
            if 'dimensions' in self.insights and 'n_components' in self.insights['dimensions']:
                insight_count += 1
                n_components = self.insights['dimensions'].get('n_components', 0)
                cumulative_variance = self.insights['dimensions'].get('cumulative_variance', [])
                if cumulative_variance:
                    total_variance = cumulative_variance[-1] if cumulative_variance else 0
                    summary['key_findings'].append(f"降维分析完成: {n_components} 个主成分解释 {total_variance:.1%} 的方差")
            elif 'dimensions' in self.insights and self.insights['dimensions'].get('status') == 'warning':
                summary['key_findings'].append(f"降维分析: {self.insights['dimensions'].get('message', '警告')}")
            
            # 时间模式洞察
            if 'temporal' in self.insights and 'time_columns_found' in self.insights['temporal']:
                insight_count += 1
                time_columns = self.insights['temporal'].get('time_columns_found', [])
                summary['key_findings'].append(f"时间模式分析完成: 分析了 {len(time_columns)} 个时间维度")
            elif 'temporal' in self.insights and self.insights['temporal'].get('status') == 'warning':
                summary['key_findings'].append(f"时间模式分析: {self.insights['temporal'].get('message', '警告')}")
            
            # 风险机制洞察
            if 'risk_mechanisms' in self.insights:
                data_quality = self.insights['risk_mechanisms'].get('data_quality', {})
                missing_rate = data_quality.get('overall_missing_rate', 0)
                risk_level = data_quality.get('risk_level', 'unknown')
                
                summary['key_findings'].append(f"数据缺失率: {missing_rate:.1%}")
                summary['risk_assessment'] = risk_level
                
                if missing_rate > 0.5:
                    summary['key_findings'].append("数据质量风险较高，建议改善数据收集")
                    summary['recommendations'].append("建议立即检查数据质量和异常值")
                    summary['recommendations'].append("考虑增加数据验证机制")
                elif missing_rate > 0.2:
                    summary['key_findings'].append("数据质量中等，需要关注")
                    summary['recommendations'].append("建议定期监控数据质量")
                    summary['recommendations'].append("优化数据收集流程")
                else:
                    summary['key_findings'].append("数据质量良好")
                    summary['recommendations'].append("数据质量良好，可继续现有流程")
                    summary['recommendations'].append("建议定期进行模式分析")
            
            # 重要影响因素洞察
            if 'important_factors' in self.insights and 'new_discoveries' in self.insights['important_factors']:
                insight_count += 1
                new_discoveries = self.insights['important_factors']['new_discoveries']
                for discovery in new_discoveries[:2]:  # 显示前2个重要发现
                    summary['key_findings'].append(f"重要发现: {discovery}")
                
                # 添加基于发现的建议
                if any("交互效应" in discovery for discovery in new_discoveries):
                    summary['recommendations'].append("发现显著交互效应，建议在预测模型中考虑特征交互项")
                if any("季节性" in discovery for discovery in new_discoveries):
                    summary['recommendations'].append("发现强季节性特征，建议建立季节性预测模型")
            
            # 相关性网络洞察
            if 'correlation_network' in self.insights and 'central_features' in self.insights['correlation_network']:
                insight_count += 1
                central_features = self.insights['correlation_network']['central_features']
                if central_features:
                    top_central = central_features[0]
                    summary['key_findings'].append(f"网络中心特征: {top_central['feature']} (中心性得分: {top_central['centrality_score']:.3f})")
                    summary['recommendations'].append(f"重点关注 {top_central['feature']} 作为关键影响因素")
            
            # SWE冷门因素洞察
            if 'swe_cold_factors' in self.insights and 'potential_discoveries' in self.insights['swe_cold_factors']:
                insight_count += 1
                potential_discoveries = self.insights['swe_cold_factors']['potential_discoveries']
                
                if potential_discoveries:
                    # 显示前2个最重要的发现
                    for discovery in potential_discoveries[:2]:
                        if discovery.get('type') == 'hidden_effect':
                            summary['key_findings'].append(f"隐藏效应发现: {discovery['feature']} (残差相关性: {discovery['residual_correlation']:.3f})")
                        elif discovery.get('type') == 'nonlinear_interaction':
                            summary['key_findings'].append(f"非线性交互: {discovery['cold_feature']} × {discovery['main_feature']} (强度: {discovery['interaction_strength']:.3f})")
                        else:
                            summary['key_findings'].append(f"冷门因素: {discovery.get('description', '未知')} (重要性: {discovery.get('potential_importance', 0):.3f})")
                    
                    # 添加基于冷门因素的建议
                    summary['recommendations'].append("发现潜在冷门因素，建议在SWE预测模型中考虑土壤湿度、空间变异性等被忽视的因素")
                    summary['recommendations'].append("建议进行标准化分析，去除已知主效应以发现隐藏的影响因素")
                
                # 添加研究洞察
                if 'research_insights' in self.insights['swe_cold_factors']:
                    research_insights = self.insights['swe_cold_factors']['research_insights']
                    for insight in research_insights[:2]:  # 显示前2个研究洞察
                        summary['recommendations'].append(f"研究建议: {insight}")
            
            summary['total_insights'] = insight_count
            
            # 如果没有关键发现，添加默认信息
            if not summary['key_findings']:
                summary['key_findings'].append("数据探索完成，但发现有限")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 生成摘要失败: {e}")
            return {
                'status': 'error', 
                'error': str(e),
                'total_insights': 0,
                'key_findings': ['摘要生成失败'],
                'risk_assessment': 'unknown',
                'recommendations': ['请检查系统状态']
            }
    
    def _discover_important_factors(self, data: pd.DataFrame, target_col: str) -> Dict:
        """发现重要影响因素 - 核心功能：解释数据关系"""
        try:
            logger.info("🔍 开始重要影响因素发现...")
            
            # 获取数值特征
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            # 移除缺失率过高的特征
            missing_rates = data[numeric_cols].isnull().sum() / len(data)
            valid_features = missing_rates[missing_rates < 0.5].index.tolist()
            
            if len(valid_features) == 0:
                return {'status': 'warning', 'message': '没有足够的有效特征进行影响因素分析'}
            
            # 准备数据
            X = data[valid_features].fillna(data[valid_features].median())
            y = data[target_col].fillna(data[target_col].median()) if target_col in data.columns else None
            
            factor_insights = {
                'total_features_analyzed': len(valid_features),
                'feature_importance': {},
                'correlation_analysis': {},
                'interaction_effects': {},
                'seasonal_factors': {},
                'new_discoveries': []
            }
            
            # 1. 特征重要性分析 (基于方差和相关性)
            feature_importance = {}
            for col in valid_features:
                if col in X.columns:
                    # 计算方差 (高方差 = 高影响潜力)
                    variance = X[col].var()
                    
                    # 计算与目标的相关性
                    if y is not None:
                        correlation = X[col].corr(y)
                    else:
                        correlation = 0
                    
                    # 计算变异系数 (稳定性指标)
                    cv = X[col].std() / X[col].mean() if X[col].mean() != 0 else 0
                    
                    feature_importance[col] = {
                        'variance': float(variance),
                        'correlation_with_target': float(correlation) if not pd.isna(correlation) else 0,
                        'coefficient_of_variation': float(cv),
                        'importance_score': float(abs(correlation) * variance) if not pd.isna(correlation) else 0
                    }
            
            # 按重要性排序
            sorted_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1]['importance_score'], reverse=True)
            
            factor_insights['feature_importance'] = dict(sorted_features)
            
            # 2. 相关性网络分析
            correlation_matrix = X.corr()
            strong_correlations = []
            
            for i, col1 in enumerate(valid_features):
                for j, col2 in enumerate(valid_features[i+1:], i+1):
                    corr_value = correlation_matrix.loc[col1, col2]
                    if abs(corr_value) > 0.7:  # 强相关
                        strong_correlations.append({
                            'feature1': col1,
                            'feature2': col2,
                            'correlation': float(corr_value),
                            'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                        })
            
            factor_insights['correlation_analysis'] = {
                'strong_correlations': strong_correlations,
                'correlation_matrix': correlation_matrix.to_dict()
            }
            
            # 3. 交互效应发现
            interaction_effects = []
            top_features = [f[0] for f in sorted_features[:5]]  # 前5个重要特征
            
            for i, feat1 in enumerate(top_features):
                for feat2 in top_features[i+1:]:
                    if feat1 in X.columns and feat2 in X.columns:
                        # 计算交互项
                        interaction = X[feat1] * X[feat2]
                        if y is not None:
                            interaction_corr = interaction.corr(y)
                            if abs(interaction_corr) > 0.3:  # 显著交互效应
                                interaction_effects.append({
                                    'feature1': feat1,
                                    'feature2': feat2,
                                    'interaction_correlation': float(interaction_corr),
                                    'interpretation': f"{feat1} 和 {feat2} 的交互效应显著"
                                })
            
            factor_insights['interaction_effects'] = interaction_effects
            
            # 4. 季节性因素分析
            seasonal_factors = {}
            if 'Month' in data.columns:
                monthly_stats = data.groupby('Month')[valid_features].mean()
                seasonal_variation = monthly_stats.std() / monthly_stats.mean()
                
                seasonal_factors = {
                    'monthly_variation': seasonal_variation.to_dict(),
                    'most_seasonal_features': seasonal_variation.nlargest(3).to_dict()
                }
            
            factor_insights['seasonal_factors'] = seasonal_factors
            
            # 5. 新发现总结
            new_discoveries = []
            
            # 发现最重要的影响因素
            if sorted_features:
                top_factor = sorted_features[0]
                new_discoveries.append(f"最重要的影响因素: {top_factor[0]} (重要性得分: {top_factor[1]['importance_score']:.3f})")
            
            # 发现强相关特征对
            if strong_correlations:
                strongest_corr = max(strong_correlations, key=lambda x: abs(x['correlation']))
                new_discoveries.append(f"最强相关特征对: {strongest_corr['feature1']} ↔ {strongest_corr['feature2']} (相关系数: {strongest_corr['correlation']:.3f})")
            
            # 发现显著交互效应
            if interaction_effects:
                strongest_interaction = max(interaction_effects, key=lambda x: abs(x['interaction_correlation']))
                new_discoveries.append(f"显著交互效应: {strongest_interaction['feature1']} × {strongest_interaction['feature2']} (交互相关系数: {strongest_interaction['interaction_correlation']:.3f})")
            
            # 发现季节性特征
            if seasonal_factors and 'most_seasonal_features' in seasonal_factors:
                most_seasonal = max(seasonal_factors['most_seasonal_features'].items(), key=lambda x: x[1])
                new_discoveries.append(f"最强季节性特征: {most_seasonal[0]} (季节性变异系数: {most_seasonal[1]:.3f})")
            
            factor_insights['new_discoveries'] = new_discoveries
            
            logger.info(f"✅ 重要影响因素发现完成: 分析了 {len(valid_features)} 个特征")
            logger.info(f"🔍 发现 {len(new_discoveries)} 个重要洞察")
            
            return factor_insights
            
        except Exception as e:
            logger.error(f"❌ 重要影响因素发现失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _analyze_correlation_network(self, data: pd.DataFrame, target_col: str) -> Dict:
        """相关性网络分析 - 发现特征间的复杂关系"""
        try:
            logger.info("🔍 开始相关性网络分析...")
            
            # 获取数值特征
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            # 移除缺失率过高的特征
            missing_rates = data[numeric_cols].isnull().sum() / len(data)
            valid_features = missing_rates[missing_rates < 0.5].index.tolist()
            
            if len(valid_features) < 3:
                return {'status': 'warning', 'message': '特征数量不足，无法进行网络分析'}
            
            # 准备数据 - 确保没有NaN值
            X = data[valid_features].fillna(data[valid_features].median())
            
            # 再次检查并处理任何剩余的NaN值
            X = X.fillna(0)
            
            # 计算相关性矩阵
            correlation_matrix = X.corr()
            
            network_insights = {
                'network_statistics': {},
                'central_features': [],
                'feature_clusters': [],
                'influence_paths': [],
                'network_visualization': {}
            }
            
            # 1. 网络统计
            total_connections = 0
            strong_connections = 0
            moderate_connections = 0
            
            for i, col1 in enumerate(valid_features):
                for j, col2 in enumerate(valid_features[i+1:], i+1):
                    corr_value = abs(correlation_matrix.loc[col1, col2])
                    if corr_value > 0.3:  # 有意义的连接
                        total_connections += 1
                        if corr_value > 0.7:
                            strong_connections += 1
                        elif corr_value > 0.5:
                            moderate_connections += 1
            
            network_insights['network_statistics'] = {
                'total_features': len(valid_features),
                'total_connections': total_connections,
                'strong_connections': strong_connections,
                'moderate_connections': moderate_connections,
                'network_density': total_connections / (len(valid_features) * (len(valid_features) - 1) / 2)
            }
            
            # 2. 中心性特征 (与其他特征相关性最多的特征)
            centrality_scores = {}
            for col in valid_features:
                connections = 0
                total_corr = 0
                for other_col in valid_features:
                    if col != other_col:
                        corr_value = abs(correlation_matrix.loc[col, other_col])
                        if corr_value > 0.3:
                            connections += 1
                            total_corr += corr_value
                
                centrality_scores[col] = {
                    'connection_count': connections,
                    'average_correlation': total_corr / connections if connections > 0 else 0,
                    'centrality_score': connections * (total_corr / connections if connections > 0 else 0)
                }
            
            # 按中心性排序
            central_features = sorted(centrality_scores.items(), 
                                    key=lambda x: x[1]['centrality_score'], reverse=True)[:5]
            
            network_insights['central_features'] = [
                {
                    'feature': feat,
                    'centrality_score': score['centrality_score'],
                    'connection_count': score['connection_count'],
                    'average_correlation': score['average_correlation']
                }
                for feat, score in central_features
            ]
            
            # 3. 特征聚类 (基于相关性)
            from sklearn.cluster import AgglomerativeClustering
            
            # 使用1-|correlation|作为距离
            distance_matrix = 1 - abs(correlation_matrix)
            
            # 聚类
            clustering = AgglomerativeClustering(n_clusters=min(3, len(valid_features)//2), 
                                               metric='precomputed', linkage='average')
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # 组织聚类结果
            feature_clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in feature_clusters:
                    feature_clusters[label] = []
                feature_clusters[label].append(valid_features[i])
            
            network_insights['feature_clusters'] = [
                {
                    'cluster_id': cluster_id,
                    'features': features,
                    'cluster_size': len(features),
                    'intra_cluster_correlation': self._calculate_intra_cluster_correlation(features, correlation_matrix)
                }
                for cluster_id, features in feature_clusters.items()
            ]
            
            # 4. 影响路径分析
            influence_paths = []
            for central_feat, _ in central_features[:3]:  # 前3个中心特征
                paths = self._find_influence_paths(central_feat, valid_features, correlation_matrix)
                influence_paths.extend(paths)
            
            network_insights['influence_paths'] = influence_paths
            
            # 5. 网络可视化数据
            network_insights['network_visualization'] = {
                'nodes': [
                    {
                        'id': feat,
                        'centrality': centrality_scores[feat]['centrality_score'],
                        'cluster': cluster_labels[valid_features.index(feat)]
                    }
                    for feat in valid_features
                ],
                'edges': [
                    {
                        'source': valid_features[i],
                        'target': valid_features[j],
                        'weight': abs(correlation_matrix.loc[valid_features[i], valid_features[j]]),
                        'correlation': correlation_matrix.loc[valid_features[i], valid_features[j]]
                    }
                    for i in range(len(valid_features))
                    for j in range(i+1, len(valid_features))
                    if abs(correlation_matrix.loc[valid_features[i], valid_features[j]]) > 0.3
                ]
            }
            
            logger.info(f"✅ 相关性网络分析完成: {len(valid_features)} 个特征, {total_connections} 个连接")
            
            return network_insights
            
        except Exception as e:
            logger.error(f"❌ 相关性网络分析失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_intra_cluster_correlation(self, features: List[str], correlation_matrix: pd.DataFrame) -> float:
        """计算聚类内平均相关性"""
        if len(features) < 2:
            return 0.0
        
        total_corr = 0
        count = 0
        for i, feat1 in enumerate(features):
            for feat2 in features[i+1:]:
                if feat1 in correlation_matrix.columns and feat2 in correlation_matrix.columns:
                    total_corr += abs(correlation_matrix.loc[feat1, feat2])
                    count += 1
        
        return total_corr / count if count > 0 else 0.0
    
    def _find_influence_paths(self, central_feature: str, all_features: List[str], 
                            correlation_matrix: pd.DataFrame, max_depth: int = 2) -> List[Dict]:
        """发现影响路径"""
        paths = []
        
        # 找到与中心特征强相关的特征
        strong_connections = []
        for feat in all_features:
            if feat != central_feature:
                corr = abs(correlation_matrix.loc[central_feature, feat])
                if corr > 0.5:
                    strong_connections.append((feat, corr))
        
        # 按相关性排序
        strong_connections.sort(key=lambda x: x[1], reverse=True)
        
        # 构建影响路径
        for connected_feat, corr in strong_connections[:3]:  # 前3个强连接
            paths.append({
                'central_feature': central_feature,
                'connected_feature': connected_feat,
                'correlation_strength': float(corr),
                'path_type': 'direct_influence',
                'interpretation': f"{central_feature} 直接影响 {connected_feat} (相关系数: {corr:.3f})"
            })
        
        return paths
    
    def _discover_swe_cold_factors(self, data: pd.DataFrame, target_col: str) -> Dict:
        """发现SWE冷门因素 - 基于研究文献的潜在重要影响因素"""
        try:
            logger.info("🔍 开始SWE冷门因素发现...")
            
            # 获取数值特征
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_cols:
                numeric_cols.remove(target_col)
            
            # 移除缺失率过高的特征
            missing_rates = data[numeric_cols].isnull().sum() / len(data)
            valid_features = missing_rates[missing_rates < 0.5].index.tolist()
            
            if len(valid_features) == 0:
                return {'status': 'warning', 'message': '没有足够的有效特征进行SWE冷门因素分析'}
            
            # 准备数据
            X = data[valid_features].fillna(data[valid_features].median())
            y = data[target_col].fillna(data[target_col].median()) if target_col in data.columns else None
            
            swe_cold_factors = {
                'target_analysis': target_col,
                'candidate_cold_factors': {},
                'standardized_analysis': {},
                'nonlinear_interactions': {},
                'potential_discoveries': [],
                'research_insights': []
            }
            
            # 1. 定义SWE冷门因素候选特征
            cold_factor_candidates = {
                'soil_moisture_related': {
                    'description': '土壤湿度相关因素',
                    'research_basis': '土壤湿度对雪盖动态具有重要影响，特别是在复杂地形区域',
                    'candidate_features': [col for col in valid_features if any(keyword in col.lower() 
                                    for keyword in ['moisture', 'humidity', 'precip', 'rain'])]
                },
                'spatial_variability': {
                    'description': '空间变异性因素',
                    'research_basis': '雪密度和SWE的空间变异性对估计准确性有显著影响',
                    'candidate_features': [col for col in valid_features if any(keyword in col.lower() 
                                    for keyword in ['lat', 'lon', 'longitude', 'latitude', 'elevation', 'altitude'])]
                },
                'forest_cover_impact': {
                    'description': '森林覆盖影响',
                    'research_basis': '森林覆盖影响遥感数据对雪盖的测量精度',
                    'candidate_features': [col for col in valid_features if any(keyword in col.lower() 
                                    for keyword in ['forest', 'tree', 'vegetation', 'cover'])]
                },
                'snow_type_changes': {
                    'description': '雪盖类型变化',
                    'research_basis': '雪盖类型变化可能导致SWE变化，特别是在气候变化影响下',
                    'candidate_features': [col for col in valid_features if any(keyword in col.lower() 
                                    for keyword in ['snow', 'ice', 'density', 'type'])]
                },
                'temporal_changes': {
                    'description': '时间变化因素',
                    'research_basis': 'SWE的时间变化影响水资源可用性，北半球SWE在1951-2022年期间下降',
                    'candidate_features': [col for col in valid_features if any(keyword in col.lower() 
                                    for keyword in ['time', 'date', 'year', 'month', 'day', 'season'])]
                }
            }
            
            # 2. 分析每个冷门因素类别
            for category, info in cold_factor_candidates.items():
                candidate_features = info['candidate_features']
                
                if candidate_features:
                    category_analysis = {
                        'description': info['description'],
                        'research_basis': info['research_basis'],
                        'candidate_features': candidate_features,
                        'feature_analysis': {},
                        'potential_importance': 0.0
                    }
                    
                    # 分析每个候选特征
                    for feature in candidate_features:
                        if feature in X.columns:
                            # 计算特征统计
                            feature_stats = {
                                'variance': float(X[feature].var()),
                                'mean': float(X[feature].mean()),
                                'std': float(X[feature].std()),
                                'skewness': float(X[feature].skew()),
                                'kurtosis': float(X[feature].kurtosis())
                            }
                            
                            # 计算与目标的相关性（如果有目标变量）
                            if y is not None:
                                correlation = X[feature].corr(y)
                                feature_stats['correlation_with_target'] = float(correlation) if not pd.isna(correlation) else 0.0
                            else:
                                feature_stats['correlation_with_target'] = 0.0
                            
                            # 计算变异系数
                            cv = X[feature].std() / X[feature].mean() if X[feature].mean() != 0 else 0
                            feature_stats['coefficient_of_variation'] = float(cv)
                            
                            # 计算潜在重要性得分
                            importance_score = abs(feature_stats['correlation_with_target']) * feature_stats['variance'] * (1 + abs(cv))
                            feature_stats['potential_importance'] = float(importance_score)
                            
                            category_analysis['feature_analysis'][feature] = feature_stats
                            category_analysis['potential_importance'] += importance_score
                    
                    swe_cold_factors['candidate_cold_factors'][category] = category_analysis
            
            # 3. 标准化分析 - 去除已知主效应
            logger.info("🔍 进行标准化分析，去除已知主效应...")
            
            # 识别主要影响因素（温度、降水等）
            main_effects = []
            for col in valid_features:
                if any(keyword in col.lower() for keyword in ['temp', 'precip', 'rain', 'snow']):
                    main_effects.append(col)
            
            if main_effects and y is not None:
                # 计算主效应的线性组合
                main_effect_values = X[main_effects].mean(axis=1)
                
                # 从目标变量中去除主效应
                y_residual = y - main_effect_values.corr(y) * main_effect_values if len(main_effects) > 0 else y
                
                # 重新计算冷门因素与残差的相关性
                standardized_analysis = {}
                for category, info in swe_cold_factors['candidate_cold_factors'].items():
                    if 'feature_analysis' in info:
                        standardized_features = {}
                        for feature, stats in info['feature_analysis'].items():
                            if feature in X.columns:
                                # 计算与残差的相关性
                                residual_correlation = X[feature].corr(y_residual)
                                standardized_features[feature] = {
                                    'original_correlation': stats['correlation_with_target'],
                                    'residual_correlation': float(residual_correlation) if not pd.isna(residual_correlation) else 0.0,
                                    'correlation_change': float(residual_correlation - stats['correlation_with_target']) if not pd.isna(residual_correlation) else 0.0,
                                    'potential_hidden_effect': abs(residual_correlation) > abs(stats['correlation_with_target']) if not pd.isna(residual_correlation) else False
                                }
                        standardized_analysis[category] = standardized_features
                
                swe_cold_factors['standardized_analysis'] = standardized_analysis
            
            # 4. 非线性交互效应发现
            logger.info("🔍 发现非线性交互效应...")
            
            nonlinear_interactions = {}
            top_cold_features = []
            
            # 收集所有冷门特征的潜在重要性
            for category, info in swe_cold_factors['candidate_cold_factors'].items():
                if 'feature_analysis' in info:
                    for feature, stats in info['feature_analysis'].items():
                        if stats['potential_importance'] > 0:
                            top_cold_features.append((feature, stats['potential_importance']))
            
            # 按重要性排序
            top_cold_features.sort(key=lambda x: x[1], reverse=True)
            top_cold_features = [f[0] for f in top_cold_features[:5]]  # 前5个冷门特征
            
            # 分析冷门特征与主要特征的交互效应
            for cold_feature in top_cold_features:
                if cold_feature in X.columns:
                    interactions = {}
                    for main_feature in main_effects[:3]:  # 前3个主要特征
                        if main_feature in X.columns:
                            # 计算交互项
                            interaction_term = X[cold_feature] * X[main_feature]
                            
                            # 计算交互项与目标的相关性
                            if y is not None:
                                interaction_correlation = interaction_term.corr(y)
                                interactions[main_feature] = {
                                    'interaction_correlation': float(interaction_correlation) if not pd.isna(interaction_correlation) else 0.0,
                                    'interaction_strength': abs(interaction_correlation) if not pd.isna(interaction_correlation) else 0.0,
                                    'interpretation': f"{cold_feature} × {main_feature} 的交互效应"
                                }
                    
                    if interactions:
                        nonlinear_interactions[cold_feature] = interactions
            
            swe_cold_factors['nonlinear_interactions'] = nonlinear_interactions
            
            # 5. 潜在发现总结
            potential_discoveries = []
            
            # 发现高潜在重要性的冷门因素
            for category, info in swe_cold_factors['candidate_cold_factors'].items():
                if info['potential_importance'] > 0:
                    potential_discoveries.append({
                        'category': category,
                        'description': info['description'],
                        'potential_importance': info['potential_importance'],
                        'research_basis': info['research_basis']
                    })
            
            # 发现标准化后的隐藏效应
            if 'standardized_analysis' in swe_cold_factors:
                for category, features in swe_cold_factors['standardized_analysis'].items():
                    for feature, analysis in features.items():
                        if analysis['potential_hidden_effect']:
                            potential_discoveries.append({
                                'type': 'hidden_effect',
                                'feature': feature,
                                'category': category,
                                'original_correlation': analysis['original_correlation'],
                                'residual_correlation': analysis['residual_correlation'],
                                'interpretation': f"{feature} 在去除主效应后显示出隐藏的影响"
                            })
            
            # 发现显著的非线性交互
            for cold_feature, interactions in nonlinear_interactions.items():
                for main_feature, interaction in interactions.items():
                    if interaction['interaction_strength'] > 0.3:  # 显著交互
                        potential_discoveries.append({
                            'type': 'nonlinear_interaction',
                            'cold_feature': cold_feature,
                            'main_feature': main_feature,
                            'interaction_strength': interaction['interaction_strength'],
                            'interpretation': interaction['interpretation']
                        })
            
            swe_cold_factors['potential_discoveries'] = potential_discoveries
            
            # 6. 研究洞察总结
            research_insights = [
                "基于文献研究，土壤湿度、雪密度空间变异性、森林覆盖影响等冷门因素可能对SWE估计有重要影响",
                "通过标准化分析去除已知主效应，可以更敏感地发现次要或潜在因素",
                "冷门因素可能与其他特征存在非线性组合效应，需要完整特征集才能捕捉",
                "建议采用广泛候选特征→标准化→降维→潜在贡献分析的策略"
            ]
            
            swe_cold_factors['research_insights'] = research_insights
            
            logger.info(f"✅ SWE冷门因素发现完成: 分析了 {len(cold_factor_candidates)} 个因素类别")
            logger.info(f"🔍 发现 {len(potential_discoveries)} 个潜在重要发现")
            
            return swe_cold_factors
            
        except Exception as e:
            logger.error(f"❌ SWE冷门因素发现失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def interpret_insights(self, insights: Dict = None) -> Dict:
        """解读洞察结果 - 将技术分析转化为用户可理解的洞察"""
        try:
            logger.info("🔍 开始解读洞察结果...")
            
            if insights is None:
                insights = self.insights
            
            if not insights:
                return {'status': 'error', 'error': '没有可解读的洞察结果'}
            
            interpretation = {
                'timestamp': datetime.now().isoformat(),
                'executive_summary': {},
                'business_insights': {},
                'technical_explanations': {},
                'actionable_recommendations': {},
                'risk_assessment': {},
                'data_quality_insights': {}
            }
            
            # 1. 执行摘要解读
            if 'summary' in insights:
                summary = insights['summary']
                interpretation['executive_summary'] = {
                    'total_discoveries': summary.get('total_insights', 0),
                    'key_message': self._generate_key_message(summary),
                    'business_impact': self._assess_business_impact(summary),
                    'urgency_level': self._assess_urgency(summary)
                }
            
            # 2. 异常检测解读
            if 'anomalies' in insights and 'anomaly_count' in insights['anomalies']:
                interpretation['business_insights']['anomaly_analysis'] = self._interpret_anomalies(insights['anomalies'])
            
            # 3. 聚类分析解读
            if 'clusters' in insights and 'optimal_clusters' in insights['clusters']:
                interpretation['business_insights']['clustering_analysis'] = self._interpret_clusters(insights['clusters'])
            
            # 4. 降维分析解读
            if 'dimensions' in insights and 'n_components' in insights['dimensions']:
                interpretation['business_insights']['dimension_analysis'] = self._interpret_dimensions(insights['dimensions'])
            
            # 5. 时间模式解读
            if 'temporal' in insights and 'time_columns_found' in insights['temporal']:
                interpretation['business_insights']['temporal_analysis'] = self._interpret_temporal_patterns(insights['temporal'])
            
            # 6. 风险机制解读
            if 'risk_mechanisms' in insights:
                interpretation['risk_assessment'] = self._interpret_risk_mechanisms(insights['risk_mechanisms'])
            
            # 7. 重要影响因素解读
            if 'important_factors' in insights and 'new_discoveries' in insights['important_factors']:
                interpretation['business_insights']['factor_analysis'] = self._interpret_important_factors(insights['important_factors'])
            
            # 8. 相关性网络解读
            if 'correlation_network' in insights and 'central_features' in insights['correlation_network']:
                interpretation['business_insights']['network_analysis'] = self._interpret_correlation_network(insights['correlation_network'])
            
            # 9. SWE冷门因素解读
            if 'swe_cold_factors' in insights and 'potential_discoveries' in insights['swe_cold_factors']:
                interpretation['business_insights']['cold_factors_analysis'] = self._interpret_swe_cold_factors(insights['swe_cold_factors'])
            
            # 10. 数据质量洞察
            interpretation['data_quality_insights'] = self._interpret_data_quality(insights)
            
            # 11. 可操作建议
            interpretation['actionable_recommendations'] = self._generate_actionable_recommendations(interpretation)
            
            logger.info("✅ 洞察结果解读完成")
            return interpretation
            
        except Exception as e:
            logger.error(f"❌ 洞察结果解读失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _generate_key_message(self, summary: Dict) -> str:
        """生成关键信息"""
        try:
            risk_level = summary.get('risk_assessment', 'unknown')
            total_insights = summary.get('total_insights', 0)
            
            if risk_level == 'high':
                return f"⚠️ 数据质量风险较高，发现 {total_insights} 个重要洞察，建议立即关注"
            elif risk_level == 'medium':
                return f"📊 数据质量中等，发现 {total_insights} 个重要洞察，需要持续监控"
            else:
                return f"✅ 数据质量良好，发现 {total_insights} 个重要洞察，系统运行正常"
        except Exception as e:
            return "数据探索完成，发现有限"
    
    def _assess_business_impact(self, summary: Dict) -> str:
        """评估业务影响"""
        try:
            risk_level = summary.get('risk_assessment', 'unknown')
            
            if risk_level == 'high':
                return "高影响 - 数据质量问题可能影响预测准确性，建议立即处理"
            elif risk_level == 'medium':
                return "中等影响 - 数据质量需要关注，可能影响长期预测效果"
            else:
                return "低影响 - 数据质量良好，系统运行稳定"
        except Exception as e:
            return "影响程度待评估"
    
    def _assess_urgency(self, summary: Dict) -> str:
        """评估紧急程度"""
        try:
            risk_level = summary.get('risk_assessment', 'unknown')
            
            if risk_level == 'high':
                return "高紧急 - 建议24小时内处理"
            elif risk_level == 'medium':
                return "中等紧急 - 建议1周内处理"
            else:
                return "低紧急 - 可定期监控"
        except Exception as e:
            return "紧急程度待评估"
    
    def _interpret_anomalies(self, anomalies: Dict) -> Dict:
        """解读异常检测结果"""
        try:
            anomaly_rate = anomalies.get('anomaly_rate', 0)
            anomaly_count = anomalies.get('anomaly_count', 0)
            
            interpretation = {
                'anomaly_rate_interpretation': '',
                'business_implications': '',
                'recommended_actions': []
            }
            
            # 异常率解读
            if anomaly_rate > 0.15:
                interpretation['anomaly_rate_interpretation'] = f"异常数据比例过高 ({anomaly_rate:.1%})，表明数据质量存在严重问题"
                interpretation['business_implications'] = "高异常率可能影响模型训练效果和预测准确性"
                interpretation['recommended_actions'] = [
                    "立即检查数据源和数据收集流程",
                    "验证传感器和设备状态",
                    "暂停使用有问题的数据进行模型训练"
                ]
            elif anomaly_rate > 0.05:
                interpretation['anomaly_rate_interpretation'] = f"异常数据比例较高 ({anomaly_rate:.1%})，需要关注数据质量"
                interpretation['business_implications'] = "中等异常率可能影响模型性能，需要监控"
                interpretation['recommended_actions'] = [
                    "定期检查数据质量",
                    "监控异常数据趋势",
                    "优化数据预处理流程"
                ]
            else:
                interpretation['anomaly_rate_interpretation'] = f"异常数据比例正常 ({anomaly_rate:.1%})，数据质量良好"
                interpretation['business_implications'] = "低异常率表明数据质量稳定，可以继续使用"
                interpretation['recommended_actions'] = [
                    "继续监控数据质量",
                    "定期进行异常检测",
                    "保持现有数据收集流程"
                ]
            
            return interpretation
            
        except Exception as e:
            return {'error': f'异常检测解读失败: {e}'}
    
    def _interpret_clusters(self, clusters: Dict) -> Dict:
        """解读聚类分析结果"""
        try:
            optimal_clusters = clusters.get('optimal_clusters', 0)
            silhouette_score = clusters.get('silhouette_score', 0)
            cluster_sizes = clusters.get('cluster_sizes', {})
            
            interpretation = {
                'cluster_interpretation': '',
                'data_pattern_insights': '',
                'business_implications': '',
                'recommended_actions': []
            }
            
            # 聚类数解读
            if optimal_clusters == 2:
                interpretation['cluster_interpretation'] = "数据呈现明显的二元分布模式"
                interpretation['data_pattern_insights'] = "可能存在两种不同的数据状态或条件"
            elif optimal_clusters == 3:
                interpretation['cluster_interpretation'] = "数据呈现三元分布模式"
                interpretation['data_pattern_insights'] = "可能存在三种不同的数据状态或条件"
            else:
                interpretation['cluster_interpretation'] = f"数据呈现 {optimal_clusters} 元分布模式"
                interpretation['data_pattern_insights'] = f"数据具有 {optimal_clusters} 种不同的状态或条件"
            
            # 轮廓系数解读
            if silhouette_score > 0.7:
                interpretation['business_implications'] = "聚类质量很高，数据模式清晰，模型可以很好地区分不同类别"
            elif silhouette_score > 0.5:
                interpretation['business_implications'] = "聚类质量良好，数据模式相对清晰"
            elif silhouette_score > 0.3:
                interpretation['business_implications'] = "聚类质量一般，数据模式有一定重叠"
            else:
                interpretation['business_implications'] = "聚类质量较低，数据模式重叠严重，可能需要更多特征或更好的预处理"
            
            # 聚类大小分析
            if cluster_sizes:
                cluster_balance = max(cluster_sizes.values()) / min(cluster_sizes.values())
                if cluster_balance > 5:
                    interpretation['recommended_actions'].append("聚类大小不平衡，建议检查数据分布是否合理")
                else:
                    interpretation['recommended_actions'].append("聚类大小相对平衡，数据分布合理")
            
            interpretation['recommended_actions'].extend([
                "基于聚类结果优化特征工程",
                "考虑为不同聚类建立专门的预测模型",
                "监控聚类稳定性"
            ])
            
            return interpretation
            
        except Exception as e:
            return {'error': f'聚类分析解读失败: {e}'}
    
    def _interpret_dimensions(self, dimensions: Dict) -> Dict:
        """解读降维分析结果"""
        try:
            n_components = dimensions.get('n_components', 0)
            cumulative_variance = dimensions.get('cumulative_variance', [])
            feature_importance = dimensions.get('feature_importance', {})
            
            interpretation = {
                'dimension_interpretation': '',
                'feature_importance_insights': '',
                'business_implications': '',
                'recommended_actions': []
            }
            
            # 主成分数解读
            if n_components == 2:
                interpretation['dimension_interpretation'] = "数据可以用2个主成分很好地表示"
            elif n_components == 3:
                interpretation['dimension_interpretation'] = "数据可以用3个主成分很好地表示"
            else:
                interpretation['dimension_interpretation'] = f"数据需要 {n_components} 个主成分来表示"
            
            # 方差解释率解读
            if cumulative_variance:
                total_variance = cumulative_variance[-1] if cumulative_variance else 0
                if total_variance > 0.9:
                    interpretation['business_implications'] = f"降维效果很好，{n_components}个主成分解释了{total_variance:.1%}的方差，信息损失很小"
                elif total_variance > 0.8:
                    interpretation['business_implications'] = f"降维效果良好，{n_components}个主成分解释了{total_variance:.1%}的方差，信息损失较小"
                elif total_variance > 0.7:
                    interpretation['business_implications'] = f"降维效果一般，{n_components}个主成分解释了{total_variance:.1%}的方差，有一定信息损失"
                else:
                    interpretation['business_implications'] = f"降维效果较差，{n_components}个主成分只解释了{total_variance:.1%}的方差，信息损失较大"
            
            # 特征重要性解读
            if feature_importance:
                top_features = sorted(feature_importance.items(), 
                                   key=lambda x: max(abs(imp) for imp in x[1]), reverse=True)[:3]
                interpretation['feature_importance_insights'] = f"最重要的特征包括: {', '.join([f[0] for f in top_features])}"
            
            interpretation['recommended_actions'] = [
                "基于主成分分析结果优化特征选择",
                "考虑使用降维后的特征进行模型训练",
                "监控特征重要性的变化"
            ]
            
            return interpretation
            
        except Exception as e:
            return {'error': f'降维分析解读失败: {e}'}
    
    def _interpret_temporal_patterns(self, temporal: Dict) -> Dict:
        """解读时间模式分析结果"""
        try:
            time_columns = temporal.get('time_columns_found', [])
            patterns = temporal.get('patterns', {})
            
            interpretation = {
                'temporal_interpretation': '',
                'pattern_insights': '',
                'business_implications': '',
                'recommended_actions': []
            }
            
            # 时间列解读
            if len(time_columns) >= 3:
                interpretation['temporal_interpretation'] = "数据具有完整的时间维度信息"
            elif len(time_columns) >= 2:
                interpretation['temporal_interpretation'] = "数据具有基本的时间维度信息"
            else:
                interpretation['temporal_interpretation'] = "数据时间维度信息有限"
            
            # 模式洞察
            pattern_insights = []
            if 'yearly' in patterns:
                pattern_insights.append("存在年度变化模式")
            if 'monthly' in patterns:
                pattern_insights.append("存在月度季节性模式")
            if 'daily' in patterns:
                pattern_insights.append("存在日变化模式")
            
            interpretation['pattern_insights'] = "，".join(pattern_insights) if pattern_insights else "未发现明显的时间模式"
            
            # 业务影响
            if len(pattern_insights) >= 2:
                interpretation['business_implications'] = "数据具有丰富的时间模式，适合建立时间序列预测模型"
            elif len(pattern_insights) == 1:
                interpretation['business_implications'] = "数据具有基本的时间模式，可以建立简单的时间序列模型"
            else:
                interpretation['business_implications'] = "数据时间模式不明显，可能需要更多时间特征或不同的分析方法"
            
            interpretation['recommended_actions'] = [
                "基于时间模式优化模型架构",
                "考虑添加时间特征（如季节、星期等）",
                "监控时间模式的稳定性"
            ]
            
            return interpretation
            
        except Exception as e:
            return {'error': f'时间模式解读失败: {e}'}
    
    def _interpret_risk_mechanisms(self, risk_mechanisms: Dict) -> Dict:
        """解读风险机制识别结果"""
        try:
            interpretation = {
                'overall_risk_assessment': '',
                'risk_details': {},
                'business_implications': '',
                'recommended_actions': []
            }
            
            # 整体风险评估
            risk_levels = []
            if 'data_quality' in risk_mechanisms:
                data_quality = risk_mechanisms['data_quality']
                missing_rate = data_quality.get('overall_missing_rate', 0)
                risk_level = data_quality.get('risk_level', 'unknown')
                
                if missing_rate > 0.5:
                    risk_levels.append("high")
                elif missing_rate > 0.2:
                    risk_levels.append("medium")
                else:
                    risk_levels.append("low")
                
                interpretation['risk_details']['data_quality'] = {
                    'risk_level': risk_level,
                    'description': '数据质量风险',
                    'implications': '可能影响模型训练和预测准确性'
                }
            
            if 'extreme_values' in risk_mechanisms:
                extreme_risk = risk_mechanisms['extreme_values']
                risk_level = extreme_risk.get('risk_level', 'unknown')
                risk_levels.append(risk_level)
                interpretation['risk_details']['extreme_values'] = {
                    'risk_level': risk_level,
                    'description': '极端值风险',
                    'implications': '可能影响模型对异常情况的预测能力'
                }
            
            if 'temporal_continuity' in risk_mechanisms:
                temporal_risk = risk_mechanisms['temporal_continuity']
                risk_level = temporal_risk.get('risk_level', 'unknown')
                risk_levels.append(risk_level)
                interpretation['risk_details']['temporal_continuity'] = {
                    'risk_level': risk_level,
                    'description': '时间连续性风险',
                    'implications': '可能影响时间序列模型的训练效果'
                }
            
            # 确定整体风险等级
            if 'high' in risk_levels:
                overall_risk = 'high'
            elif 'medium' in risk_levels:
                overall_risk = 'medium'
            else:
                overall_risk = 'low'
            
            interpretation['overall_risk_assessment'] = overall_risk
            
            # 业务影响
            if overall_risk == 'high':
                interpretation['business_implications'] = "整体风险较高，建议立即关注数据质量问题"
            elif overall_risk == 'medium':
                interpretation['business_implications'] = "整体风险中等，需要持续监控和改进"
            else:
                interpretation['business_implications'] = "整体风险较低，系统运行稳定"
            
            # 建议行动
            if overall_risk == 'high':
                interpretation['recommended_actions'] = [
                    "立即检查数据源和数据收集流程",
                    "验证传感器和设备状态",
                    "暂停使用有问题的数据进行模型训练"
                ]
            elif overall_risk == 'medium':
                interpretation['recommended_actions'] = [
                    "监控数据质量趋势",
                    "检查数据预处理步骤"
                ]
                interpretation['recommended_actions'].extend([
                    "优化数据收集流程",
                    "建立定期质量评估机制"
                ])
            else:
                interpretation['recommended_actions'] = [
                    "继续监控数据质量",
                    "记录最佳实践"
                ]
                interpretation['recommended_actions'].extend([
                    "定期进行质量评估",
                    "分享成功经验"
                ])
            
            return interpretation
            
        except Exception as e:
            return {'error': f'风险机制解读失败: {e}'}
    
    def _interpret_important_factors(self, important_factors: Dict) -> Dict:
        """解读重要影响因素发现结果"""
        try:
            new_discoveries = important_factors.get('new_discoveries', [])
            feature_importance = important_factors.get('feature_importance', {})
            interaction_effects = important_factors.get('interaction_effects', [])
            
            interpretation = {
                'factor_importance_insights': '',
                'interaction_insights': '',
                'business_implications': '',
                'recommended_actions': []
            }
            
            # 特征重要性洞察
            if feature_importance:
                top_features = sorted(feature_importance.items(), 
                                   key=lambda x: x[1].get('importance_score', 0), reverse=True)[:3]
                top_feature_names = [f[0] for f in top_features]
                interpretation['factor_importance_insights'] = f"最重要的影响因素包括: {', '.join(top_feature_names)}"
            
            # 交互效应洞察
            if interaction_effects:
                strong_interactions = [eff for eff in interaction_effects if abs(eff.get('interaction_correlation', 0)) > 0.5]
                if strong_interactions:
                    interpretation['interaction_insights'] = f"发现 {len(strong_interactions)} 个强交互效应，表明特征间存在复杂的非线性关系"
                else:
                    interpretation['interaction_insights'] = "交互效应相对较弱，特征间关系相对简单"
            else:
                interpretation['interaction_insights'] = "未发现显著的交互效应"
            
            # 业务影响
            if len(new_discoveries) >= 3:
                interpretation['business_implications'] = "发现了多个重要影响因素，为模型优化提供了重要信息"
            elif len(new_discoveries) >= 1:
                interpretation['business_implications'] = "发现了一些重要影响因素，有助于模型改进"
            else:
                interpretation['business_implications'] = "影响因素发现有限，可能需要更多数据或不同的分析方法"
            
            # 建议行动
            interpretation['recommended_actions'] = [
                "基于重要影响因素优化特征选择",
                "考虑在模型中添加交互效应项",
                "监控重要影响因素的变化"
            ]
            
            return interpretation
            
        except Exception as e:
            return {'error': f'重要影响因素解读失败: {e}'}
    
    def _interpret_correlation_network(self, correlation_network: Dict) -> Dict:
        """解读相关性网络分析结果"""
        try:
            network_stats = correlation_network.get('network_statistics', {})
            central_features = correlation_network.get('central_features', [])
            feature_clusters = correlation_network.get('feature_clusters', [])
            
            interpretation = {
                'network_structure_insights': '',
                'centrality_insights': '',
                'clustering_insights': '',
                'business_implications': '',
                'recommended_actions': []
            }
            
            # 网络结构洞察
            if network_stats:
                total_features = network_stats.get('total_features', 0)
                total_connections = network_stats.get('total_connections', 0)
                network_density = network_stats.get('network_density', 0)
                
                if network_density > 0.5:
                    interpretation['network_structure_insights'] = f"特征网络密度较高 ({network_density:.2f})，表明特征间关系复杂"
                elif network_density > 0.3:
                    interpretation['network_structure_insights'] = f"特征网络密度中等 ({network_density:.2f})，特征间有一定关联"
                else:
                    interpretation['network_structure_insights'] = f"特征网络密度较低 ({network_density:.2f})，特征间相对独立"
            
            # 中心性洞察
            if central_features:
                top_central = central_features[0] if central_features else {}
                if top_central:
                    interpretation['centrality_insights'] = f"网络中心特征: {top_central.get('feature', 'unknown')}，中心性得分: {top_central.get('centrality_score', 0):.3f}"
            
            # 聚类洞察
            if feature_clusters:
                interpretation['clustering_insights'] = f"发现 {len(feature_clusters)} 个特征聚类，表明存在特征组"
            
            # 业务影响
            if network_stats and network_stats.get('strong_connections', 0) > 0:
                interpretation['business_implications'] = "特征间存在强相关性，需要考虑特征冗余和多重共线性问题"
            else:
                interpretation['business_implications'] = "特征间相关性适中，特征选择相对合理"
            
            # 建议行动
            interpretation['recommended_actions'] = [
                "基于网络分析结果优化特征选择",
                "考虑去除高度相关的冗余特征",
                "监控特征相关性的变化"
            ]
            
            return interpretation
            
        except Exception as e:
            return {'error': f'相关性网络解读失败: {e}'}
    
    def _interpret_swe_cold_factors(self, swe_cold_factors: Dict) -> Dict:
        """解读SWE冷门因素发现结果"""
        try:
            potential_discoveries = swe_cold_factors.get('potential_discoveries', [])
            candidate_cold_factors = swe_cold_factors.get('candidate_cold_factors', {})
            
            interpretation = {
                'cold_factors_insights': '',
                'discovery_insights': '',
                'business_implications': '',
                'recommended_actions': []
            }
            
            # 冷门因素洞察
            if candidate_cold_factors:
                interpretation['cold_factors_insights'] = f"分析了 {len(candidate_cold_factors)} 个冷门因素类别，包括土壤湿度、空间变异性、森林覆盖影响等"
            
            # 发现洞察
            if potential_discoveries:
                hidden_effects = [d for d in potential_discoveries if d.get('type') == 'hidden_effect']
                nonlinear_interactions = [d for d in potential_discoveries if d.get('type') == 'nonlinear_interaction']
                
                insights = []
                if hidden_effects:
                    insights.append(f"发现 {len(hidden_effects)} 个隐藏效应")
                if nonlinear_interactions:
                    insights.append(f"发现 {len(nonlinear_interactions)} 个非线性交互")
                
                interpretation['discovery_insights'] = "，".join(insights) if insights else "发现了一些潜在的重要冷门因素"
            else:
                interpretation['discovery_insights'] = "未发现显著的冷门因素"
            
            # 业务影响
            if len(potential_discoveries) >= 2:
                interpretation['business_implications'] = "发现了多个潜在的重要冷门因素，为SWE预测模型优化提供了新思路"
            elif len(potential_discoveries) == 1:
                interpretation['business_implications'] = "发现了一些潜在的重要冷门因素，值得进一步研究"
            else:
                interpretation['business_implications'] = "冷门因素发现有限，可能需要更多数据或不同的分析方法"
            
            # 建议行动
            interpretation['recommended_actions'] = [
                "进一步研究发现的冷门因素",
                "考虑在SWE预测模型中集成这些因素",
                "建立长期监控机制跟踪冷门因素的变化"
            ]
            
            return interpretation
            
        except Exception as e:
            return {'error': f'SWE冷门因素解读失败: {e}'}
    
    def _interpret_data_quality(self, insights: Dict) -> Dict:
        """解读数据质量洞察"""
        try:
            interpretation = {
                'overall_quality_assessment': '',
                'quality_issues': [],
                'quality_strengths': [],
                'recommended_actions': []
            }
            
            # 整体质量评估
            if 'risk_mechanisms' in insights and 'data_quality' in insights['risk_mechanisms']:
                data_quality = insights['risk_mechanisms']['data_quality']
                missing_rate = data_quality.get('overall_missing_rate', 0)
                risk_level = data_quality.get('risk_level', 'unknown')
                
                if missing_rate > 0.5:
                    interpretation['overall_quality_assessment'] = "数据质量较差，缺失率过高"
                    interpretation['quality_issues'].append(f"总体缺失率: {missing_rate:.1%}")
                elif missing_rate > 0.2:
                    interpretation['overall_quality_assessment'] = "数据质量中等，存在一定缺失"
                    interpretation['quality_issues'].append(f"总体缺失率: {missing_rate:.1%}")
                else:
                    interpretation['overall_quality_assessment'] = "数据质量良好，缺失率较低"
                    interpretation['quality_strengths'].append(f"总体缺失率: {missing_rate:.1%}")
                
                if risk_level == 'high':
                    interpretation['quality_issues'].append("数据质量风险等级: 高")
                elif risk_level == 'medium':
                    interpretation['quality_issues'].append("数据质量风险等级: 中等")
                else:
                    interpretation['quality_strengths'].append("数据质量风险等级: 低")
            
            # 异常检测质量
            if 'anomalies' in insights and 'anomaly_rate' in insights['anomalies']:
                anomaly_rate = insights['anomalies']['anomaly_rate']
                if anomaly_rate > 0.1:
                    interpretation['quality_issues'].append(f"异常数据比例较高: {anomaly_rate:.1%}")
                else:
                    interpretation['quality_strengths'].append(f"异常数据比例正常: {anomaly_rate:.1%}")
            
            # 建议行动
            if interpretation['quality_issues']:
                interpretation['recommended_actions'] = [
                    "立即检查数据源和数据收集流程",
                    "优化数据预处理步骤",
                    "建立数据质量监控机制"
                ]
            else:
                interpretation['recommended_actions'] = [
                    "继续监控数据质量",
                    "定期进行质量评估",
                    "保持现有数据管理流程"
                ]
            
            return interpretation
            
        except Exception as e:
            return {'error': f'数据质量解读失败: {e}'}
    
    def _generate_actionable_recommendations(self, interpretation: Dict) -> Dict:
        """生成可操作建议"""
        try:
            recommendations = {
                'immediate_actions': [],
                'short_term_actions': [],
                'long_term_actions': [],
                'priority_levels': {}
            }
            
            # 基于风险等级确定行动优先级
            overall_risk = interpretation.get('risk_assessment', {}).get('overall_risk_assessment', 'unknown')
            
            if overall_risk == 'high':
                recommendations['immediate_actions'] = [
                    "立即检查数据源和数据收集流程",
                    "验证传感器和设备状态",
                    "暂停使用有问题的数据进行模型训练"
                ]
                recommendations['short_term_actions'] = [
                    "建立数据质量监控机制",
                    "优化数据预处理流程",
                    "培训数据管理人员"
                ]
                recommendations['long_term_actions'] = [
                    "建立数据质量标准和流程",
                    "实施自动化数据质量检查",
                    "建立数据质量报告机制"
                ]
            elif overall_risk == 'medium':
                recommendations['immediate_actions'] = [
                    "监控数据质量趋势",
                    "检查数据预处理步骤"
                ]
                recommendations['short_term_actions'] = [
                    "优化数据收集流程",
                    "建立定期质量评估机制"
                ]
                recommendations['long_term_actions'] = [
                    "持续改进数据质量",
                    "建立数据质量文化"
                ]
            else:
                recommendations['immediate_actions'] = [
                    "继续监控数据质量",
                    "记录最佳实践"
                ]
                recommendations['short_term_actions'] = [
                    "定期进行质量评估",
                    "分享成功经验"
                ]
                recommendations['long_term_actions'] = [
                    "持续优化数据流程",
                    "建立数据质量标杆"
                ]
            
            # 设置优先级
            recommendations['priority_levels'] = {
                'immediate': '高优先级 - 24小时内执行',
                'short_term': '中优先级 - 1周内执行',
                'long_term': '低优先级 - 1个月内执行'
            }
            
            return recommendations
            
        except Exception as e:
            return {'error': f'生成可操作建议失败: {e}'}
    
    def save_insights(self, output_dir: str = "insights") -> str:
        """保存洞察结果"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            insights_file = os.path.join(output_dir, f"insights_discovery_{timestamp}.json")
            
            with open(insights_file, 'w', encoding='utf-8') as f:
                json.dump(self.insights, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ 洞察结果已保存: {insights_file}")
            return insights_file
            
        except Exception as e:
            logger.error(f"❌ 保存洞察结果失败: {e}")
            return ""

def main():
    """主函数"""
    try:
        logger.info("🚀 启动无监督探索模块...")
        
        # 加载Environment Canada数据
        data_path = "data/real/environment_canada/environment_canada_merged.csv"
        
        if not os.path.exists(data_path):
            logger.error(f"❌ 数据文件不存在: {data_path}")
            return
        
        # 读取数据
        data = pd.read_csv(data_path)
        logger.info(f"📊 加载数据: {data.shape}")
        
        # 估算土壤湿度
        if 'Total Precip (mm)' in data.columns and 'Temp (°C)' in data.columns:
            # 简单的土壤湿度估算
            data['estimated_soil_moisture'] = (
                0.3 +  # 基础湿度
                0.1 * np.log1p(data['Total Precip (mm)'].fillna(0)) +  # 降水影响
                0.05 * (1 - (data['Temp (°C)'].fillna(0) + 20) / 60)  # 温度影响
            )
            data['estimated_soil_moisture'] = np.clip(data['estimated_soil_moisture'], 0.1, 0.9)
        
        # 创建探索模块
        explorer = InsightDiscoveryModule()
        
        # 发现模式
        insights = explorer.discover_patterns(data)
        
        if 'status' not in insights:
            # 保存洞察结果
            insights_file = explorer.save_insights()
            
            # 解读洞察结果
            interpretation = explorer.interpret_insights(insights)
            
            logger.info("🎉 无监督探索完成！")
            logger.info(f"📊 发现 {interpretation['executive_summary']['total_discoveries']} 类洞察")
            logger.info(f"⚠️ 风险评估: {interpretation['risk_assessment']['overall_risk_assessment']}")
            
            # 显示关键发现
            if 'executive_summary' in interpretation and 'key_message' in interpretation['executive_summary']:
                logger.info(f"🔍 {interpretation['executive_summary']['key_message']}")
            
            # 显示建议
            if 'actionable_recommendations' in interpretation and 'immediate_actions' in interpretation['actionable_recommendations']:
                for rec in interpretation['actionable_recommendations']['immediate_actions']:
                    logger.info(f"💡 {rec}")
            
            return interpretation
        else:
            logger.error(f"❌ 探索失败: {insights}")
            return None
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return None

if __name__ == "__main__":
    main()
