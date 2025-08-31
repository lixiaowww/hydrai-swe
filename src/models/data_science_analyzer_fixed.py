"""
简化版数据科学分析器 - 修复终端报错
只保留核心功能，移除复杂的缩进问题
"""
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import logging

class DataScienceAnalyzer:
    def __init__(self):
        self.data = None
        self.scaler = StandardScaler()
        self.analysis_results = {}
    
    def advanced_anomaly_detection(self, column='snow_water_equivalent_mm'):
        """简化的异常检测"""
        print(f"\n🚨 执行高级异常检测: {column}")
        print("=" * 60)
        
        if self.data is None or column not in self.data.columns:
            return {}
        
        series = self.data[column].dropna()
        if len(series) == 0:
            return {}
        
        # Z-score异常检测
        z_scores = np.abs(stats.zscore(series))
        z_anomalies = (z_scores > 2.0).astype(float)
        
        # Isolation Forest
        X = series.values.reshape(-1, 1)
        iso_forest = IsolationForest(contamination=0.02, random_state=42)
        iso_predictions = iso_forest.fit_predict(X)
        iso_anomalies = (iso_predictions == -1).astype(float)
        
        # 集成结果
        ensemble_anomalies = ((z_anomalies + iso_anomalies) >= 1).astype(float)
        
        results = {
            'statistical': {
                'z_score_anomalies': z_anomalies.tolist()
            },
            'machine_learning': {
                'isolation_forest_anomalies': iso_anomalies.tolist()
            },
            'ensemble': {
                'ensemble_anomalies': ensemble_anomalies.tolist()
            },
            'interpretation': f"检测到 {int(ensemble_anomalies.sum())} 个异常点，占总数据的 {ensemble_anomalies.sum()/len(ensemble_anomalies)*100:.1f}%"
        }
        
        print("✅ Advanced anomaly detection completed")
        return results
    
    def clustering_analysis(self):
        """简化的聚类分析"""
        print("\n🔍 执行聚类分析")
        print("=" * 60)
        
        if self.data is None:
            return {}
        
        # 选择数值列
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return {}
        
        data_subset = self.data[numeric_cols].dropna()
        if len(data_subset) == 0:
            return {}
        
        # 标准化数据
        X_scaled = self.scaler.fit_transform(data_subset)
        
        # K-means聚类
        print("🎯 K-means聚类...")
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # DBSCAN聚类
        print("🌐 DBSCAN聚类...")
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        results = {
            'kmeans': {
                'labels': kmeans_labels.tolist(),
                'n_clusters': len(set(kmeans_labels))
            },
            'dbscan': {
                'labels': dbscan_labels.tolist(),
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            },
            'interpretation': f"K-means识别了 {len(set(kmeans_labels))} 个聚类，DBSCAN识别了 {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)} 个聚类"
        }
        
        print("✅ Clustering analysis completed")
        return results
    
    def statistical_hypothesis_testing(self, column='snow_water_equivalent_mm'):
        """简化的统计假设检验"""
        print(f"\n📊 执行统计假设检验: {column}")
        print("=" * 60)
        
        if self.data is None or column not in self.data.columns:
            return {}
        
        series = self.data[column].dropna()
        if len(series) == 0:
            return {}
        
        # 正态性检验
        print("📊 正态性检验...")
        shapiro_stat, shapiro_p = stats.shapiro(series[:5000] if len(series) > 5000 else series)
        
        # 平稳性检验（简化版）
        print("📊 平稳性检验...")
        # 使用ADF检验的简化版本
        try:
            from statsmodels.tsa.stattools import adfuller
            adf_stat, adf_p, _, _, _, _ = adfuller(series)
            stationarity_test = {'adf_statistic': adf_stat, 'p_value': adf_p}
        except:
            stationarity_test = {'adf_statistic': 0, 'p_value': 1.0}
        
        results = {
            'normality': {
                'shapiro_statistic': float(shapiro_stat),
                'shapiro_p_value': float(shapiro_p)
            },
            'stationarity': stationarity_test,
            'interpretation': f"数据{'符合' if shapiro_p > 0.05 else '不符合'}正态分布 (p={shapiro_p:.4f})，{'是' if stationarity_test['p_value'] < 0.05 else '不是'}平稳序列"
        }
        
        print("✅ Statistical hypothesis testing completed")
        return results
    
    def advanced_time_series_decomposition(self, column='snow_water_equivalent_mm'):
        """简化的时间序列分解"""
        print(f"\n🔍 执行高级时间序列分解: {column}")
        print("=" * 60)
        
        if self.data is None or column not in self.data.columns:
            return {}
        
        series = self.data[column].dropna()
        if len(series) == 0:
            return {}
        
        # 简单的移动平均分解
        print("📊 执行简化分解...")
        window = min(365, len(series) // 4)
        
        trend = series.rolling(window=window, center=True).mean()
        seasonal = series - trend
        seasonal = seasonal.rolling(window=7, center=True).mean()  # 周期性
        residual = series - trend - seasonal
        
        # 填充缺失值
        trend = trend.fillna(method='bfill').fillna(method='ffill')
        seasonal = seasonal.fillna(0)
        residual = residual.fillna(0)
        
        results = {
            'stl_decomposition': {
                'trend': {
                    'index': [t.isoformat() for t in series.index],
                    'values': trend.tolist()
                },
                'seasonal': {
                    'index': [t.isoformat() for t in series.index],
                    'values': seasonal.tolist()
                },
                'resid': {
                    'index': [t.isoformat() for t in series.index],
                    'values': residual.tolist()
                }
            },
            'interpretation': f"时间序列分解完成：趋势范围 {trend.min():.1f} 到 {trend.max():.1f}，季节性范围 {seasonal.min():.1f} 到 {seasonal.max():.1f}"
        }
        
        print("✅ Advanced time series decomposition completed")
        return results
    
    def discover_cold_factors(self, target_column='snow_water_equivalent_mm', top_k=10):
        """简化的因子发现"""
        print(f"\n📊 执行因子发现: {target_column}")
        
        if self.data is None or target_column not in self.data.columns:
            return {}
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        factors = []
        
        for col in numeric_cols:
            if col == target_column:
                continue
            
            try:
                correlation = stats.spearmanr(self.data[col].dropna(), 
                                           self.data[target_column].dropna())[0]
                if not np.isnan(correlation):
                    factors.append({
                        'factor': col,
                        'correlation': abs(correlation),
                        'score': abs(correlation)
                    })
            except:
                continue
        
        # 排序并取前k个
        factors.sort(key=lambda x: x['score'], reverse=True)
        factors = factors[:top_k]
        
        results = {
            'high_predictive': factors,
            'interpretation': f"发现 {len(factors)} 个重要因子，最强相关因子是 {factors[0]['factor'] if factors else 'None'}"
        }
        
        print("✅ Factor discovery completed")
        return results


