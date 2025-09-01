#!/usr/bin/env python3
"""
基于GitHub发现的SWE分析系统
整合季节性分析、异常检测、相关性分析等功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from scipy.signal import periodogram
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SWEAnalysisSystem:
    """SWE综合分析系统"""
    
    def __init__(self, data_path=None):
        """
        初始化SWE分析系统
        
        Args:
            data_path (str): 数据文件路径
        """
        self.data = None
        self.analysis_results = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """加载SWE数据"""
        print(f"📊 加载SWE数据: {data_path}")
        
        try:
            # 首先尝试加载数据
            if not os.path.exists(data_path):
                # 尝试备用数据路径
                backup_paths = [
                    "data/processed/eccc_manitoba_snow_processed.csv",
                    "data/raw/eccc_recent/eccc_recent_combined.csv"
                ]
                
                data_loaded = False
                for backup_path in backup_paths:
                    if os.path.exists(backup_path):
                        print(f"📂 使用备用数据: {backup_path}")
                        data_path = backup_path
                        data_loaded = True
                        break
                
                if not data_loaded:
                    raise FileNotFoundError(f"数据文件不存在: {data_path}")
            
            self.data = pd.read_csv(data_path)
            
            # 处理日期列
            date_col = None
            for col_name in ['date', 'Date/Time', 'Date']:
                if col_name in self.data.columns:
                    date_col = col_name
                    break
            
            if date_col is None:
                raise ValueError("未找到日期列")
            
            self.data[date_col] = pd.to_datetime(self.data[date_col], errors='coerce')
            self.data.set_index(date_col, inplace=True)
            
            # 创建或找到snow_water_equivalent_mm列
            if 'snow_water_equivalent_mm' not in self.data.columns:
                # 尝试不同的列名映射
                swe_candidates = [
                    'Snow on Grnd (mm)',  # 已经是mm
                    'Snow on Grnd (cm)',  # 需要转换cm->mm  
                    'Total Snow (mm)',
                    'Total Snow (cm)'
                ]
                
                swe_created = False
                for candidate in swe_candidates:
                    if candidate in self.data.columns:
                        if 'cm' in candidate:
                            # 转换cm到mm
                            self.data['snow_water_equivalent_mm'] = self.data[candidate] * 10.0
                        else:
                            # 已经是mm
                            self.data['snow_water_equivalent_mm'] = self.data[candidate]
                        swe_created = True
                        print(f"✅ 从 {candidate} 创建 snow_water_equivalent_mm 列")
                        break
                
                if not swe_created:
                    # 严格遵循规则：不使用模拟数据
                    print("⚠️ 未找到合适的雪水当量数据，设置为N/A")
                    self.data['snow_water_equivalent_mm'] = 'N/A'
            
            # 数据清理和验证
            self.data['snow_water_equivalent_mm'] = pd.to_numeric(
                self.data['snow_water_equivalent_mm'], errors='coerce'
            )
            
            # 移除空值行
            original_len = len(self.data)
            self.data = self.data.dropna(subset=['snow_water_equivalent_mm'])
            
            if len(self.data) == 0:
                raise ValueError("处理后数据为空")
            
            print(f"✅ 数据加载成功: {len(self.data)} 条记录 (原始: {original_len} 条)")
            print(f"📅 时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
            print(f"📊 SWE数据范围: {self.data['snow_water_equivalent_mm'].min():.1f} - {self.data['snow_water_equivalent_mm'].max():.1f} mm")
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            self.data = None
    
    def seasonal_analysis(self, column='snow_water_equivalent_mm'):
        """
        季节性分析 - 基于CIROH-Snow和Seasonal-Snowfall-Climatology模块
        
        Args:
            column (str): 要分析的列名
            
        Returns:
            dict: 季节性分析结果
        """
        print(f"\n🌍 执行季节性分析: {column}")
        print("=" * 50)
        
        if self.data is None:
            print("❌ 数据未加载")
            return {}
        
        if column not in self.data.columns:
            print(f"❌ 列 {column} 不存在")
            return {}
        
        series = self.data[column].dropna()
        
        if len(series) == 0:
            print("❌ 数据为空，无法进行分析")
            return {}
        
        try:
            # 1. 年度周期分析
            annual_cycle = self._analyze_annual_cycle(series)
            
            # 2. 月度统计
            monthly_stats = self._analyze_monthly_patterns(series)
            
            # 3. 季节性分解
            seasonal_decomposition = self._perform_seasonal_decomposition(series)
            
            # 4. 频谱分析
            frequency_analysis = self._analyze_frequency_domain(series)
            
            results = {
                'annual_cycle': annual_cycle,
                'monthly_patterns': monthly_stats,
                'seasonal_decomposition': seasonal_decomposition,
                'frequency_analysis': frequency_analysis
            }
            
            self.analysis_results['seasonal_analysis'] = results
            print("✅ 季节性分析完成")
            return results
            
        except Exception as e:
            print(f"❌ 季节性分析失败: {e}")
            return {}
    
    def _analyze_annual_cycle(self, series):
        """分析年度周期"""
        print("📅 分析年度周期...")
        
        # 按年分组
        annual_data = series.groupby(series.index.year).agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).dropna()
        
        # 计算年际变化
        years = np.array(annual_data.index)
        means = annual_data['mean'].values
        
        if len(means) > 1:
            # 线性趋势
            slope, intercept, r_value, p_value, std_err = stats.linregress(years, means)
            
            annual_trend = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_per_decade': slope * 10
            }
        else:
            annual_trend = {}
        
        return {
            'annual_statistics': annual_data,
            'trend_analysis': annual_trend
        }
    
    def _analyze_monthly_patterns(self, series):
        """分析月度模式"""
        print("📊 分析月度模式...")
        
        # 按月份分组
        monthly_data = series.groupby(series.index.month).agg([
            'mean', 'std', 'min', 'max', 'count'
        ])
        
        # 计算季节性指数
        overall_mean = series.mean()
        seasonal_indices = monthly_data['mean'] / overall_mean
        
        return {
            'monthly_statistics': monthly_data,
            'seasonal_indices': seasonal_indices,
            'overall_mean': overall_mean
        }
    
    def _perform_seasonal_decomposition(self, series):
        """执行季节性分解"""
        print("🔍 执行季节性分解...")
        
        # 确保数据是等间隔的
        series_resampled = series.resample('D').mean().fillna(method='ffill')
        
        # 计算移动平均
        window = 365  # 一年窗口
        trend = series_resampled.rolling(window=window, center=True).mean()
        
        # 计算季节性成分
        seasonal = series_resampled - trend
        
        # 计算残差
        residual = series_resampled - trend - seasonal
        
        return {
            'original': series_resampled,
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual
        }
    
    def _analyze_frequency_domain(self, series):
        """频域分析"""
        print("📡 执行频域分析...")
        
        # 确保数据是等间隔的
        series_resampled = series.resample('D').mean().fillna(method='ffill')
        
        # 计算功率谱密度
        frequencies, power = periodogram(series_resampled.dropna(), fs=1.0)
        
        # 找到主要频率
        main_freq_idx = np.argmax(power)
        main_frequency = frequencies[main_freq_idx]
        main_period = 1.0 / main_frequency if main_frequency > 0 else np.inf
        
        return {
            'frequencies': frequencies,
            'power': power,
            'main_frequency': main_frequency,
            'main_period': main_period
        }
    
    def anomaly_detection(self, column='snow_water_equivalent_mm'):
        """
        异常检测 - 基于KathiravanNatarajan/SnowDepth_AnomalyDetection模块
        
        Args:
            column (str): 要检测的列名
            
        Returns:
            dict: 异常检测结果
        """
        print(f"\n🚨 执行异常检测: {column}")
        print("=" * 50)
        
        if column not in self.data.columns:
            print(f"❌ 列 {column} 不存在")
            return {}
        
        series = self.data[column].dropna()
        
        # 1. 统计方法异常检测
        statistical_anomalies = self._statistical_anomaly_detection(series)
        
        # 2. 机器学习异常检测
        ml_anomalies = self._machine_learning_anomaly_detection(series)
        
        # 3. 时间序列异常检测
        timeseries_anomalies = self._timeseries_anomaly_detection(series)
        
        # 4. 综合异常评分
        combined_anomalies = self._combine_anomaly_scores(
            statistical_anomalies, ml_anomalies, timeseries_anomalies
        )
        
        results = {
            'statistical': statistical_anomalies,
            'machine_learning': ml_anomalies,
            'timeseries': timeseries_anomalies,
            'combined': combined_anomalies
        }
        
        self.analysis_results['anomaly_detection'] = results
        return results
    
    def _statistical_anomaly_detection(self, series):
        """统计方法异常检测"""
        print("📊 统计方法异常检测...")
        
        # Z-score方法
        z_scores = np.abs(stats.zscore(series))
        z_anomalies = z_scores > 3
        
        # IQR方法
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        iqr_anomalies = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
        
        # 移动窗口方法
        rolling_mean = series.rolling(window=30, center=True).mean()
        rolling_std = series.rolling(window=30, center=True).std()
        rolling_anomalies = np.abs(series - rolling_mean) > 3 * rolling_std
        
        return {
            'z_score_anomalies': z_anomalies,
            'iqr_anomalies': iqr_anomalies,
            'rolling_anomalies': rolling_anomalies,
            'z_scores': z_scores
        }
    
    def _machine_learning_anomaly_detection(self, series):
        """机器学习异常检测"""
        print("🤖 机器学习异常检测...")
        
        # 准备数据
        X = series.values.reshape(-1, 1)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=0.1)
        iso_predictions = iso_forest.fit_predict(X_scaled)
        iso_anomalies = iso_predictions == -1
        
        return {
            'isolation_forest_anomalies': iso_anomalies,
            'isolation_forest_scores': iso_forest.decision_function(X_scaled)
        }
    
    def _timeseries_anomaly_detection(self, series):
        """时间序列异常检测"""
        print("⏰ 时间序列异常检测...")
        
        # 基于趋势的异常检测
        rolling_mean = series.rolling(window=30, center=True).mean()
        trend_anomalies = np.abs(series - rolling_mean) > 2 * series.std()
        
        # 基于季节性的异常检测
        monthly_means = series.groupby(series.index.month).mean()
        seasonal_anomalies = np.abs(series - monthly_means[series.index.month].values) > 2 * series.std()
        
        return {
            'trend_anomalies': trend_anomalies,
            'seasonal_anomalies': seasonal_anomalies
        }
    
    def _combine_anomaly_scores(self, statistical, ml, timeseries):
        """综合异常评分"""
        print("🔗 综合异常评分...")
        
        # 计算综合异常概率
        combined_score = np.zeros(len(self.data))
        
        # 统计方法权重
        if 'z_score_anomalies' in statistical:
            combined_score += statistical['z_score_anomalies'].astype(int) * 0.3
        if 'iqr_anomalies' in statistical:
            combined_score += statistical['iqr_anomalies'].astype(int) * 0.3
        
        # 机器学习权重
        if 'isolation_forest_anomalies' in ml:
            combined_score += ml['isolation_forest_anomalies'].astype(int) * 0.4
        
        # 归一化到0-1
        combined_score = combined_score / combined_score.max() if combined_score.max() > 0 else combined_score
        
        # 确定异常阈值
        threshold = 0.5
        combined_anomalies = combined_score > threshold
        
        return {
            'combined_score': combined_score,
            'combined_anomalies': combined_anomalies,
            'threshold': threshold
        }
    
    def correlation_analysis(self, target_column='snow_water_equivalent_mm'):
        """
        相关性分析 - 基于Bike-sharing-system-Analysis模块
        
        Args:
            target_column (str): 目标变量列名
            
        Returns:
            dict: 相关性分析结果
        """
        print(f"\n🔗 执行相关性分析: {target_column}")
        print("=" * 50)
        
        if target_column not in self.data.columns:
            print(f"❌ 列 {target_column} 不存在")
            return {}
        
        # 选择数值列进行相关性分析
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if target_column not in numeric_columns:
            numeric_columns.append(target_column)
        
        correlation_data = self.data[numeric_columns].dropna()
        
        # 1. 皮尔逊相关系数
        pearson_corr = correlation_data.corr(method='pearson')
        
        # 2. 斯皮尔曼相关系数
        spearman_corr = correlation_data.corr(method='spearman')
        
        # 3. 与目标变量的相关性
        target_correlations = {}
        for col in numeric_columns:
            if col != target_column:
                # 皮尔逊相关
                pearson_r, pearson_p = stats.pearsonr(
                    correlation_data[target_column], correlation_data[col]
                )
                
                # 斯皮尔曼相关
                spearman_r, spearman_p = stats.spearmanr(
                    correlation_data[target_column], correlation_data[col]
                )
                
                target_correlations[col] = {
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'spearman_r': spearman_r,
                    'spearman_p': spearman_p
                }
        
        # 4. 滚动相关性分析
        rolling_corr = self._calculate_rolling_correlations(correlation_data, target_column)
        
        results = {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr,
            'target_correlations': target_correlations,
            'rolling_correlations': rolling_corr
        }
        
        self.analysis_results['correlation_analysis'] = results
        return results
    
    def _calculate_rolling_correlations(self, data, target_column, window=365):
        """计算滚动相关性"""
        print("📈 计算滚动相关性...")
        
        rolling_corrs = {}
        for col in data.columns:
            if col != target_column:
                rolling_corr = data[target_column].rolling(window=window).corr(data[col])
                rolling_corrs[col] = rolling_corr
        
        return rolling_corrs
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n📋 生成综合分析报告")
        print("=" * 60)
        
        if not self.analysis_results:
            print("❌ 没有分析结果，请先运行分析")
            return
        
        # 季节性分析总结
        if 'seasonal_analysis' in self.analysis_results:
            seasonal = self.analysis_results['seasonal_analysis']
            print("\n🌍 季节性分析总结:")
            
            if 'annual_cycle' in seasonal:
                trend = seasonal['annual_cycle'].get('trend_analysis', {})
                if trend:
                    print(f"  年际趋势: {trend.get('trend_per_decade', 0):.4f} 每10年")
                    print(f"  趋势显著性: {'是' if trend.get('p_value', 1) < 0.05 else '否'}")
            
            if 'monthly_patterns' in seasonal:
                monthly = seasonal['monthly_patterns']
                print(f"  季节性指数范围: {monthly.get('seasonal_indices', pd.Series()).min():.2f} - {monthly.get('seasonal_indices', pd.Series()).max():.2f}")
        
        # 异常检测总结
        if 'anomaly_detection' in self.analysis_results:
            anomaly = self.analysis_results['anomaly_detection']
            print("\n🚨 异常检测总结:")
            
            if 'combined' in anomaly:
                combined = anomaly['combined']
                anomaly_count = combined.get('combined_anomalies', pd.Series()).sum()
                total_count = len(combined.get('combined_anomalies', pd.Series()))
                anomaly_rate = anomaly_count / total_count if total_count > 0 else 0
                print(f"  检测到异常: {anomaly_count}/{total_count} ({anomaly_rate:.2%})")
        
        # 相关性分析总结
        if 'correlation_analysis' in self.analysis_results:
            correlation = self.analysis_results['correlation_analysis']
            print("\n🔗 相关性分析总结:")
            
            if 'target_correlations' in correlation:
                target_corr = correlation['target_correlations']
                strong_correlations = []
                for col, corr_data in target_corr.items():
                    if abs(corr_data.get('pearson_r', 0)) > 0.7:
                        strong_correlations.append((col, corr_data['pearson_r']))
                
                if strong_correlations:
                    print(f"  强相关变量: {len(strong_correlations)} 个")
                    for col, r in strong_correlations[:3]:  # 显示前3个
                        print(f"    {col}: r = {r:.3f}")
                else:
                    print("  无强相关变量")
        
        print("\n" + "=" * 60)
    
    def plot_analysis_results(self, save_path=None):
        """绘制分析结果图表"""
        if not self.analysis_results:
            print("❌ 没有分析结果，请先运行分析")
            return
        
        print("📊 绘制分析结果图表...")
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SWE综合分析结果', fontsize=16)
        
        # 1. 季节性分析
        if 'seasonal_analysis' in self.analysis_results:
            seasonal = self.analysis_results['seasonal_analysis']
            
            if 'monthly_patterns' in seasonal:
                monthly = seasonal['monthly_patterns']
                seasonal_indices = monthly.get('seasonal_indices', pd.Series())
                if not seasonal_indices.empty:
                    axes[0, 0].plot(seasonal_indices.index, seasonal_indices.values, 'o-')
                    axes[0, 0].axhline(y=1, color='r', linestyle='--', alpha=0.5)
                    axes[0, 0].set_title('月度季节性指数')
                    axes[0, 0].set_xlabel('月份')
                    axes[0, 0].set_ylabel('季节性指数')
                    axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 异常检测
        if 'anomaly_detection' in self.analysis_results:
            anomaly = self.analysis_results['anomaly_detection']
            
            if 'combined' in anomaly:
                combined = anomaly['combined']
                combined_score = combined.get('combined_score', [])
                if len(combined_score) > 0:
                    axes[0, 1].plot(combined_score, alpha=0.7)
                    axes[0, 1].axhline(y=combined.get('threshold', 0.5), color='r', linestyle='--')
                    axes[0, 1].set_title('综合异常评分')
                    axes[0, 1].set_ylabel('异常评分')
                    axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 相关性热图
        if 'correlation_analysis' in self.analysis_results:
            correlation = self.analysis_results['correlation_analysis']
            
            if 'pearson_correlation' in correlation:
                corr_matrix = correlation['pearson_correlation']
                if not corr_matrix.empty:
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                               ax=axes[1, 0], cbar_kws={'label': '相关系数'})
                    axes[1, 0].set_title('皮尔逊相关性矩阵')
        
        # 4. 时间序列分解
        if 'seasonal_analysis' in self.analysis_results:
            seasonal = self.analysis_results['seasonal_analysis']
            
            if 'seasonal_decomposition' in seasonal:
                decomp = seasonal['seasonal_decomposition']
                if 'trend' in decomp and 'seasonal' in decomp:
                    # 选择最近的数据点进行可视化
                    n_points = min(1000, len(decomp['trend']))
                    x = range(n_points)
                    
                    axes[1, 1].plot(x, decomp['trend'].iloc[-n_points:], label='趋势', alpha=0.7)
                    axes[1, 1].plot(x, decomp['seasonal'].iloc[-n_points:], label='季节性', alpha=0.7)
                    axes[1, 1].set_title('时间序列分解')
                    axes[1, 1].set_xlabel('时间步')
                    axes[1, 1].set_ylabel('值')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 图表保存到: {save_path}")
        
        plt.show()

def main():
    """主函数 - 示例用法"""
    print("🚀 SWE综合分析系统")
    print("=" * 50)
    
    # 创建分析系统
    analyzer = SWEAnalysisSystem()
    
    # 加载数据
    data_path = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
    analyzer.load_data(data_path)
    
    if analyzer.data is not None:
        # 执行综合分析
        print("\n🔍 开始综合分析...")
        
        # 1. 季节性分析
        seasonal_results = analyzer.seasonal_analysis('snow_water_equivalent_mm')
        
        # 2. 异常检测
        anomaly_results = analyzer.anomaly_detection('snow_water_equivalent_mm')
        
        # 3. 相关性分析
        correlation_results = analyzer.correlation_analysis('snow_water_equivalent_mm')
        
        # 4. 生成报告
        analyzer.generate_comprehensive_report()
        
        # 5. 绘制结果
        analyzer.plot_analysis_results()
        
        print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
