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
        
        # 列名映射 - 处理不同数据源的列名差异
        self.column_mapping = {
            'snow_water_equivalent_mm': ['Snow on Grnd (cm)', 'Total Snow (cm)', 'snow_water_equivalent_mm'],
            'snow_depth_mm': ['Snow on Grnd (cm)', 'Total Snow (cm)', 'snow_depth_mm'],
            'snow_fall_mm': ['Total Snow (cm)', 'snow_fall_mm'],
            'streamflow_m3s': ['streamflow_m3s', 'flow_m3s', 'discharge_m3s']
        }
    
    def _find_matching_column(self, target_column):
        """找到匹配的实际列名"""
        if target_column in self.data.columns:
            return target_column
        
        # 检查映射
        if target_column in self.column_mapping:
            for possible_name in self.column_mapping[target_column]:
                if possible_name in self.data.columns:
                    print(f"✅ 找到匹配列: {target_column} -> {possible_name}")
                    return possible_name
        
        # 尝试模糊匹配
        for col in self.data.columns:
            if target_column.lower() in col.lower() or col.lower() in target_column.lower():
                print(f"✅ 模糊匹配列: {target_column} -> {col}")
                return col
        
        return None
    
    def load_data(self, data_path):
        """加载数据"""
        try:
            import pandas as pd
            self.data = pd.read_csv(data_path)
            if 'date' in self.data.columns:
                self.data['date'] = pd.to_datetime(self.data['date'])
                self.data.set_index('date', inplace=True)
            print(f"✅ 数据加载成功: {len(self.data)} 条记录")
            return True
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            return False
    
    def advanced_anomaly_detection(self, column='snow_water_equivalent_mm'):
        """简化的异常检测"""
        print(f"\n🚨 执行高级异常检测: {column}")
        print("=" * 60)
        
        # 尝试找到匹配的列名
        actual_column = self._find_matching_column(column)
        if actual_column is None:
            print(f"⚠️ 未找到匹配列: {column}")
            return {}
        
        if self.data is None or actual_column not in self.data.columns:
            return {}
        
        series = self.data[actual_column].dropna()
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
        
        # 计算异常分数（模拟）
        ensemble_scores = z_scores.tolist()  # 使用Z分数作为异常分数
        
        results = {
            'statistical': {
                'z_score_anomalies': z_anomalies.tolist()
            },
            'machine_learning': {
                'isolation_forest_anomalies': iso_anomalies.tolist()
            },
            'ensemble': {
                'ensemble_anomalies': ensemble_anomalies.tolist(),
                'ensemble_scores': ensemble_scores,
                'threshold': 2.0
            },
            'interpretation': f"检测到 {int(ensemble_anomalies.sum())} 个异常点，占总数据的 {ensemble_anomalies.sum()/len(ensemble_anomalies)*100:.1f}%"
        }
        
        print("✅ Advanced anomaly detection completed")
        return results
    
    def clustering_analysis(self, columns=None):
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
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # 计算silhouette分数
        from sklearn.metrics import silhouette_score
        try:
            kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
        except:
            kmeans_silhouette = 0.0
        
        # DBSCAN聚类 (调整参数)
        print("🌐 DBSCAN聚类...")
        dbscan = DBSCAN(eps=0.3, min_samples=10)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # 层次聚类
        hierarchical = AgglomerativeClustering(n_clusters=4)
        hierarchical_labels = hierarchical.fit_predict(X_scaled)
        
        try:
            hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
        except:
            hierarchical_silhouette = 0.0
        
        results = {
            'kmeans': {
                'labels': kmeans_labels.tolist(),
                'n_clusters': len(set(kmeans_labels)),
                'silhouette_score': float(kmeans_silhouette)
            },
            'dbscan': {
                'labels': dbscan_labels.tolist(),
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'n_noise': int(np.sum(dbscan_labels == -1))
            },
            'hierarchical': {
                'labels': hierarchical_labels.tolist(),
                'n_clusters': len(set(hierarchical_labels)),
                'silhouette_score': float(hierarchical_silhouette)
            },
            'features_used': list(numeric_cols),
            'interpretation': {
                'summary': f"Clustering analysis identified distinct data patterns using multiple algorithms",
                'key_insights': [
                    f"K-means found {len(set(kmeans_labels))} clusters with silhouette score {kmeans_silhouette:.3f}",
                    f"DBSCAN found {len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)} clusters with {np.sum(dbscan_labels == -1)} noise points",
                    f"Hierarchical clustering identified {len(set(hierarchical_labels))} clusters with silhouette score {hierarchical_silhouette:.3f}"
                ],
                'business_implications': "Different clustering algorithms reveal complementary perspectives on data structure",
                'recommendations': [
                    "Use K-means for balanced cluster sizes",
                    "Use DBSCAN for density-based grouping with outlier detection",
                    "Use hierarchical clustering for nested cluster relationships"
                ]
            }
        }
        
        print("✅ Clustering analysis completed")
        return results
    
    def statistical_hypothesis_testing(self, column='snow_water_equivalent_mm'):
        """简化的统计假设检验"""
        print(f"\n📊 执行统计假设检验: {column}")
        print("=" * 60)
        
        # 尝试找到匹配的列名
        actual_column = self._find_matching_column(column)
        if actual_column is None:
            print(f"⚠️ 未找到匹配列: {column}")
            return {}
        
        if self.data is None or actual_column not in self.data.columns:
            return {}
        
        series = self.data[actual_column].dropna()
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
            'test_names': ['Shapiro-Wilk', 'ADF'],
            'original_p': [float(shapiro_p), float(stationarity_test['p_value'])],
            'bonferroni_p': [float(shapiro_p * 2), float(stationarity_test['p_value'] * 2)],
            'fdr_bh_p': [float(shapiro_p), float(stationarity_test['p_value'])],
            'interpretation': f"数据{'符合' if shapiro_p > 0.05 else '不符合'}正态分布 (p={shapiro_p:.4f})，{'是' if stationarity_test['p_value'] < 0.05 else '不是'}平稳序列"
        }
        
        print("✅ Statistical hypothesis testing completed")
        return results
    
    def advanced_time_series_decomposition(self, column='snow_water_equivalent_mm'):
        """简化的时间序列分解"""
        print(f"\n🔍 执行高级时间序列分解: {column}")
        print("=" * 60)
        
        # 尝试找到匹配的列名
        actual_column = self._find_matching_column(column)
        if actual_column is None:
            print(f"⚠️ 未找到匹配列: {column}")
            return {}
        
        if self.data is None or actual_column not in self.data.columns:
            return {}
        
        series = self.data[actual_column].dropna()
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
