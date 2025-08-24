#!/usr/bin/env python3
"""
高级洪水预警模块
集成RNN神经网络和聚类分析功能
基于GitHub优秀项目改进
"""

import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedFloodWarningSystem:
    """高级洪水预警系统"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.cluster_model = None
        self.feature_names = []
        self.model_path = "models/advanced_flood_warning_model_improved.pkl"
        self.scaler_path = "models/advanced_flood_warning_scaler_improved.pkl"
        self.cluster_path = "models/advanced_flood_cluster_model_improved.pkl"
        
        # 预警级别配置
        self.alert_levels = {
            'CRITICAL': {'threshold': 0.9, 'color': '#e74c3c', 'action': '立即疏散'},
            'HIGH': {'threshold': 0.7, 'color': '#f39c12', 'action': '准备疏散'},
            'MEDIUM': {'threshold': 0.5, 'color': '#f1c40f', 'action': '加强监测'},
            'LOW': {'threshold': 0.3, 'color': '#27ae60', 'action': '正常监测'},
            'NORMAL': {'threshold': 0.0, 'color': '#95a5a6', 'action': '常规监测'}
        }
        
        self.load_models()
    
    def load_models(self):
        """加载所有模型"""
        try:
            # 加载主模型
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info("✅ 高级洪水预警模型加载成功")
            else:
                logger.warning("⚠️ 高级洪水预警模型文件不存在，将使用基础模型")
            
            # 加载标准化器
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
                logger.info("✅ 高级洪水预警标准化器加载成功")
            else:
                self.scaler = StandardScaler()
                logger.info("✅ 创建新的标准化器")
            
            # 加载聚类模型
            if os.path.exists(self.cluster_path):
                self.cluster_model = joblib.load(self.cluster_path)
                logger.info("✅ 高级洪水预警聚类模型加载成功")
            else:
                self.cluster_model = KMeans(n_clusters=5, random_state=42)
                logger.info("✅ 创建新的聚类模型")
            
            # 设置特征名称 - 与改进模型训练时保持一致
            self.feature_names = [
                'Month', '05OC001_x', '05OC001_y', '05OC011_y', '05OC012_y',
                'DayOfYear', 'WeekOfYear', 'day_of_year_sin', 'day_of_year_cos',
                'month_sin', 'month_cos', 'temp_anomaly', 'flow_change',
                'flow_volatility', 'flow_peak', 'flow_corr_2_3', 'flow_corr_2_4',
                'flow_corr_2_5', 'flow_corr_3_5', 'flow_corr_4_5'
            ]
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            return False
    
    def prepare_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """准备高级特征"""
        try:
            # 确保Date/Time列是datetime类型
            if 'Date/Time' in data.columns:
                data['Date/Time'] = pd.to_datetime(data['Date/Time'])
            
            # 基础特征工程
            features = []
            
            # 积雪相关特征
            if 'Snow on Grnd (cm)' in data.columns:
                features.append('Snow on Grnd (cm)')
                data['snow_change'] = data['Snow on Grnd (cm)'].diff()
                features.append('snow_change')
                
                # 积雪趋势特征
                data['snow_trend'] = data['Snow on Grnd (cm)'].rolling(7).mean()
                features.append('snow_trend')
            
            # 温度相关特征
            if 'Max Temp (°C)' in data.columns:
                features.append('Max Temp (°C)')
                features.append('Min Temp (°C)')
                features.append('Mean Temp (°C)')
                
                # 温度异常和趋势
                data['temp_anomaly'] = data['Mean Temp (°C)'] - data['Mean Temp (°C)'].rolling(30).mean()
                data['temp_trend'] = data['Mean Temp (°C)'].rolling(7).mean()
                features.extend(['temp_anomaly', 'temp_trend'])
            
            # 降水相关特征
            if 'Total Rain (mm)' in data.columns:
                features.append('Total Rain (mm)')
                data['rain_cumulative'] = data['Total Rain (mm)'].rolling(7).sum()
                data['rain_intensity'] = data['Total Rain (mm)'].rolling(3).max()
                features.extend(['rain_cumulative', 'rain_intensity'])
            
            # 径流相关特征
            flow_columns = [col for col in data.columns if col.startswith('05OC')]
            if flow_columns:
                main_flow_column = flow_columns[0]
                features.append(main_flow_column)
                
                # 径流变化和趋势
                data['flow_change'] = data[main_flow_column].pct_change()
                data['flow_anomaly'] = data[main_flow_column] / data[main_flow_column].rolling(30).mean()
                data['flow_trend'] = data[main_flow_column].rolling(7).mean()
                features.extend(['flow_change', 'flow_anomaly', 'flow_trend'])
            
            # 季节性特征
            if 'Month' not in data.columns:
                data['Month'] = data['Date/Time'].dt.month
            
            data['season'] = data['Month'].map({
                12: 'winter', 1: 'winter', 2: 'winter',
                3: 'spring', 4: 'spring', 5: 'spring',
                6: 'summer', 7: 'summer', 8: 'summer',
                9: 'fall', 10: 'fall', 11: 'fall'
            })
            
            season_encoding = pd.get_dummies(data['season'], prefix='season')
            data = pd.concat([data, season_encoding], axis=1)
            
            # 确保所有需要的季节列都存在
            for season in ['season_fall', 'season_winter']:
                if season not in data.columns:
                    data[season] = 0
            
            features.extend(['season_fall', 'season_winter'])
            
            # 时间特征
            if 'DayOfYear' not in data.columns:
                data['DayOfYear'] = data['Date/Time'].dt.dayofyear
            
            data['day_of_year_sin'] = np.sin(2 * np.pi * data['DayOfYear'] / 365)
            data['day_of_year_cos'] = np.cos(2 * np.pi * data['DayOfYear'] / 365)
            features.extend(['day_of_year_sin', 'day_of_year_cos'])
            
            # 聚类分析
            cluster_features = self._perform_clustering(data[features].fillna(0))
            data['cluster_label'] = cluster_features['cluster_label']
            data['risk_trend'] = cluster_features['risk_trend']
            features.extend(['cluster_label', 'risk_trend'])
            
            # 选择特征 - 只选择训练时使用的特征
            available_features = [f for f in self.feature_names if f in data.columns]
            missing_features = [f for f in self.feature_names if f not in data.columns]
            
            if missing_features:
                logger.warning(f"缺少特征: {missing_features}")
            
            # 创建缺失特征，用0填充
            for feature in missing_features:
                data[feature] = 0
            
            # 选择特征
            feature_data = data[self.feature_names].fillna(0)
            
            # 移除无穷值
            feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
            feature_data = feature_data.fillna(0)
            
            logger.info(f"特征矩阵形状: {feature_data.shape}")
            logger.info(f"特征列: {list(feature_data.columns)}")
            
            return feature_data
            
        except Exception as e:
            logger.error(f"高级特征准备失败: {e}")
            raise
    
    def _perform_clustering(self, data: pd.DataFrame) -> Dict:
        """执行聚类分析"""
        try:
            # 标准化数据用于聚类
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # 执行聚类
            cluster_labels = self.cluster_model.fit_predict(scaled_data)
            
            # 计算聚类质量
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            
            # 计算每个聚类的风险趋势
            risk_trends = []
            for i in range(len(cluster_labels)):
                cluster_id = cluster_labels[i]
                cluster_data = scaled_data[cluster_labels == cluster_id]
                
                if len(cluster_data) > 1:
                    # 计算聚类内的风险趋势（基于特征值的方差）
                    risk_trend = np.std(cluster_data).mean()
                else:
                    risk_trend = 0
                
                risk_trends.append(risk_trend)
            
            return {
                'cluster_label': cluster_labels,
                'risk_trend': risk_trends,
                'silhouette_score': silhouette_avg,
                'n_clusters': len(set(cluster_labels))
            }
            
        except Exception as e:
            logger.error(f"聚类分析失败: {e}")
            # 返回默认值
            return {
                'cluster_label': np.zeros(len(data)),
                'risk_trend': np.zeros(len(data)),
                'silhouette_score': 0,
                'n_clusters': 1
            }
    
    def predict_advanced_flood_risk(self, features_data: pd.DataFrame) -> Dict:
        """预测高级洪水风险"""
        try:
            if self.model is None:
                raise ValueError("模型未加载")
            
            # 数据标准化
            features_scaled = self.scaler.fit_transform(features_data)
            
            # 预测
            risk_prediction = self.model.predict(features_scaled)
            risk_probability = self.model.predict_proba(features_scaled)[:, 1]
            
            # 基于概率值重新计算风险等级（而不是使用分类结果）
            risk_level_based_on_prob = (risk_probability >= 0.5).astype(int)
            
            # 计算高级风险指标
            advanced_metrics = self._calculate_advanced_metrics(features_data, risk_probability)
            
            return {
                'risk_level': risk_level_based_on_prob,  # 使用基于概率的风险等级
                'risk_probability': risk_probability,
                'advanced_metrics': advanced_metrics,
                'alert_levels': self._determine_alert_levels(risk_probability),
                'cluster_analysis': self._analyze_clusters(features_data),
                'trend_analysis': self._analyze_trends(features_data, risk_probability)
            }
            
        except Exception as e:
            logger.error(f"高级风险预测失败: {e}")
            raise
    
    def _calculate_advanced_metrics(self, features_data: pd.DataFrame, risk_probability: np.ndarray) -> Dict:
        """计算高级风险指标"""
        try:
            metrics = {}
            
            # 风险稳定性指标
            risk_std = np.std(risk_probability)
            risk_cv = risk_std / np.mean(risk_probability) if np.mean(risk_probability) > 0 else 0
            
            metrics['risk_stability'] = {
                'std': float(risk_std),
                'coefficient_of_variation': float(risk_cv),
                'stability_level': 'HIGH' if risk_cv < 0.3 else 'MEDIUM' if risk_cv < 0.6 else 'LOW'
            }
            
            # 特征重要性分析
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(features_data.columns, self.model.feature_importances_))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                metrics['top_features'] = top_features
            
            # 时间序列分析
            if 'Date/Time' in features_data.columns:
                time_series_metrics = self._analyze_time_series(features_data, risk_probability)
                metrics['time_series'] = time_series_metrics
            
            return metrics
            
        except Exception as e:
            logger.error(f"高级指标计算失败: {e}")
            return {}
    
    def _determine_alert_levels(self, risk_probability: np.ndarray) -> List[Dict]:
        """确定预警级别"""
        try:
            alerts = []
            
            for i, prob in enumerate(risk_probability):
                # 确定预警级别
                alert_level = 'NORMAL'
                for level, config in self.alert_levels.items():
                    if prob >= config['threshold']:
                        alert_level = level
                        break
                
                # 计算预警强度
                alert_intensity = min(prob / config['threshold'], 2.0) if config['threshold'] > 0 else 1.0
                
                alerts.append({
                    'index': i,
                    'probability': float(prob),
                    'alert_level': alert_level,
                    'color': self.alert_levels[alert_level]['color'],
                    'action': self.alert_levels[alert_level]['action'],
                    'intensity': float(alert_intensity),
                    'timestamp': datetime.now() + timedelta(days=i) if i < 7 else None
                })
            
            return alerts
            
        except Exception as e:
            logger.error(f"预警级别确定失败: {e}")
            return []
    
    def _analyze_clusters(self, features_data: pd.DataFrame) -> Dict:
        """分析聚类结果"""
        try:
            if 'cluster_label' not in features_data.columns:
                return {}
            
            cluster_analysis = {}
            unique_clusters = features_data['cluster_label'].unique()
            
            for cluster_id in unique_clusters:
                cluster_data = features_data[features_data['cluster_label'] == cluster_id]
                cluster_size = len(cluster_data)
                
                # 分析聚类特征
                cluster_analysis[f'cluster_{int(cluster_id)}'] = {
                    'size': int(cluster_size),
                    'percentage': float(cluster_size / len(features_data) * 100),
                    'characteristics': self._analyze_cluster_characteristics(cluster_data)
                }
            
            return cluster_analysis
            
        except Exception as e:
            logger.error(f"聚类分析失败: {e}")
            return {}
    
    def _analyze_cluster_characteristics(self, cluster_data: pd.DataFrame) -> Dict:
        """分析聚类特征"""
        try:
            characteristics = {}
            
            # 数值特征统计
            numeric_columns = cluster_data.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col not in ['cluster_label', 'risk_trend']:
                    characteristics[col] = {
                        'mean': float(cluster_data[col].mean()),
                        'std': float(cluster_data[col].std()),
                        'min': float(cluster_data[col].min()),
                        'max': float(cluster_data[col].max())
                    }
            
            return characteristics
            
        except Exception as e:
            logger.error(f"聚类特征分析失败: {e}")
            return {}
    
    def _analyze_trends(self, features_data: pd.DataFrame, risk_probability: np.ndarray) -> Dict:
        """分析趋势"""
        try:
            trends = {}
            
            # 风险趋势
            if len(risk_probability) > 1:
                risk_trend = np.polyfit(range(len(risk_probability)), risk_probability, 1)[0]
                trends['risk_trend'] = {
                    'slope': float(risk_trend),
                    'direction': 'INCREASING' if risk_trend > 0.01 else 'DECREASING' if risk_trend < -0.01 else 'STABLE',
                    'magnitude': abs(float(risk_trend))
                }
            
            # 关键特征趋势
            key_features = ['Snow on Grnd (cm)', 'Mean Temp (°C)', 'Total Rain (mm)']
            for feature in key_features:
                if feature in features_data.columns:
                    feature_data = features_data[feature].values
                    if len(feature_data) > 1:
                        feature_trend = np.polyfit(range(len(feature_data)), feature_data, 1)[0]
                        trends[feature] = {
                            'slope': float(feature_trend),
                            'direction': 'INCREASING' if feature_trend > 0 else 'DECREASING' if feature_trend < 0 else 'STABLE'
                        }
            
            return trends
            
        except Exception as e:
            logger.error(f"趋势分析失败: {e}")
            return {}
    
    def _analyze_time_series(self, features_data: pd.DataFrame, risk_probability: np.ndarray) -> Dict:
        """分析时间序列特征"""
        try:
            time_series = {}
            
            # 季节性分析
            if 'Month' in features_data.columns:
                monthly_risk = {}
                for month in range(1, 13):
                    month_data = risk_probability[features_data['Month'] == month]
                    if len(month_data) > 0:
                        monthly_risk[month] = {
                            'mean_risk': float(np.mean(month_data)),
                            'max_risk': float(np.max(month_data)),
                            'count': int(len(month_data))
                        }
                time_series['seasonal_pattern'] = monthly_risk
            
            # 周期性分析
            if len(risk_probability) > 7:
                # 计算7天移动平均
                weekly_avg = pd.Series(risk_probability).rolling(7).mean().dropna()
                if len(weekly_avg) > 1:
                    weekly_trend = np.polyfit(range(len(weekly_avg)), weekly_avg, 1)[0]
                    time_series['weekly_trend'] = {
                        'slope': float(weekly_trend),
                        'direction': 'INCREASING' if weekly_trend > 0 else 'DECREASING' if weekly_trend < 0 else 'STABLE'
                    }
            
            return time_series
            
        except Exception as e:
            logger.error(f"时间序列分析失败: {e}")
            return {}
    
    def generate_comprehensive_report(self, features_data: pd.DataFrame, prediction_result: Dict) -> Dict:
        """生成综合报告"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_samples': len(features_data),
                    'high_risk_count': int(sum(1 for x in prediction_result['risk_level'] if x == 1)),
                    'average_risk': float(np.mean(prediction_result['risk_probability'])),
                    'risk_stability': prediction_result['advanced_metrics'].get('risk_stability', {}).get('stability_level', 'UNKNOWN')
                },
                'alert_summary': {
                    'critical_alerts': len([a for a in prediction_result['alert_levels'] if a['alert_level'] == 'CRITICAL']),
                    'high_alerts': len([a for a in prediction_result['alert_levels'] if a['alert_level'] == 'HIGH']),
                    'medium_alerts': len([a for a in prediction_result['alert_levels'] if a['alert_level'] == 'MEDIUM']),
                    'low_alerts': len([a for a in prediction_result['alert_levels'] if a['alert_level'] == 'LOW'])
                },
                'cluster_summary': {
                    'n_clusters': len(prediction_result['cluster_analysis']),
                    'largest_cluster': max([c['size'] for c in prediction_result['cluster_analysis'].values()]) if prediction_result['cluster_analysis'] else 0
                },
                'trend_summary': {
                    'risk_direction': prediction_result['trend_analysis'].get('risk_trend', {}).get('direction', 'UNKNOWN'),
                    'key_features_trends': {k: v['direction'] for k, v in prediction_result['trend_analysis'].items() if k != 'risk_trend'}
                },
                'recommendations': self._generate_recommendations(prediction_result)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"综合报告生成失败: {e}")
            return {}
    
    def _generate_recommendations(self, prediction_result: Dict) -> List[str]:
        """生成建议"""
        try:
            recommendations = []
            
            # 基于风险水平的建议
            high_risk_count = len([a for a in prediction_result['alert_levels'] if a['alert_level'] in ['CRITICAL', 'HIGH']])
            if high_risk_count > 0:
                recommendations.append(f"发现{high_risk_count}个高风险区域，建议立即启动应急响应预案")
            
            # 基于趋势的建议
            risk_trend = prediction_result['trend_analysis'].get('risk_trend', {})
            if risk_trend.get('direction') == 'INCREASING':
                recommendations.append("风险呈上升趋势，建议加强监测频率和预警级别")
            
            # 基于稳定性的建议
            stability = prediction_result['advanced_metrics'].get('risk_stability', {}).get('stability_level')
            if stability == 'LOW':
                recommendations.append("风险波动较大，建议优化模型参数和特征选择")
            
            # 基于聚类的建议
            if len(prediction_result['cluster_analysis']) > 3:
                recommendations.append("检测到多个风险模式，建议针对不同模式制定差异化应对策略")
            
            if not recommendations:
                recommendations.append("当前风险状况稳定，建议保持常规监测频率")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"建议生成失败: {e}")
            return ["建议生成失败，请联系技术支持"]

# 创建全局实例
advanced_flood_system = AdvancedFloodWarningSystem()
