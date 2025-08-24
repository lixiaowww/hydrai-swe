#!/usr/bin/env python3
"""
气候变化影响SWE分析模块
实现Mann-Kendall趋势检验、Theil-Sen斜率估计、30年基准期异常计算等功能
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import linregress
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ClimateChangeAnalyzer:
    """气候变化影响SWE分析器"""
    
    def __init__(self, data_path=None):
        """
        初始化分析器
        
        Args:
            data_path (str): 数据文件路径
        """
        self.data = None
        self.baseline_period = (1991, 2020)  # 30年基准期
        self.analysis_results = {}
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """加载数据"""
        print(f"📊 加载数据: {data_path}")
        
        try:
            self.data = pd.read_csv(data_path, parse_dates=['date'])
            self.data.set_index('date', inplace=True)
            
            # 确保有SWE列
            if 'snow_water_equivalent_mm' not in self.data.columns:
                if 'Snow on Grnd (mm)' in self.data.columns:
                    self.data['snow_water_equivalent_mm'] = self.data['Snow on Grnd (mm)']
                else:
                    print("⚠️ 警告: 未找到SWE数据列")
            
            print(f"✅ 数据加载成功: {len(self.data)} 条记录")
            print(f"📅 时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
    
    def check_data_homogeneity(self, column='snow_water_equivalent_mm'):
        """
        检查数据同质化
        检测观测方法变化、站点迁移等导致的非均质性
        
        Args:
            column (str): 要检查的列名
            
        Returns:
            dict: 同质化检测结果
        """
        print(f"🔍 检查数据同质化: {column}")
        
        if column not in self.data.columns:
            print(f"❌ 列 {column} 不存在")
            return {}
        
        series = self.data[column].dropna()
        
        # 1. 双累积曲线分析
        cumulative = series.cumsum()
        time_index = np.arange(len(cumulative))
        
        # 分段线性回归检测断点
        breakpoints = self._detect_breakpoints(cumulative, time_index)
        
        # 2. 标准差变化检测
        rolling_std = series.rolling(window=365, min_periods=30).std()
        std_changes = self._detect_std_changes(rolling_std)
        
        # 3. 均值偏移检测
        mean_shifts = self._detect_mean_shifts(series)
        
        results = {
            'breakpoints': breakpoints,
            'std_changes': std_changes,
            'mean_shifts': mean_shifts,
            'homogeneity_score': self._calculate_homogeneity_score(breakpoints, std_changes, mean_shifts)
        }
        
        print(f"📊 同质化检测结果:")
        print(f"  断点数量: {len(breakpoints)}")
        print(f"  标准差变化: {len(std_changes)}")
        print(f"  均值偏移: {len(mean_shifts)}")
        print(f"  同质化评分: {results['homogeneity_score']:.2f}/100")
        
        return results
    
    def _detect_breakpoints(self, cumulative, time_index, min_segment=30):
        """检测断点"""
        breakpoints = []
        
        if len(cumulative) < min_segment * 2:
            return breakpoints
        
        # 使用分段线性回归检测断点
        for i in range(min_segment, len(cumulative) - min_segment):
            # 分段1
            x1 = time_index[:i]
            y1 = cumulative[:i]
            slope1, _, r1, _, _ = linregress(x1, y1)
            
            # 分段2
            x2 = time_index[i:]
            y2 = cumulative[i:]
            slope2, _, r2, _, _ = linregress(x2, y2)
            
            # 如果斜率差异显著，认为是断点
            if abs(slope1 - slope2) > 0.1 and r1 > 0.8 and r2 > 0.8:
                breakpoints.append(i)
        
        return breakpoints
    
    def _detect_std_changes(self, rolling_std, threshold=2.0):
        """检测标准差变化"""
        mean_std = rolling_std.mean()
        std_changes = []
        
        for i, std_val in enumerate(rolling_std):
            if pd.notna(std_val) and abs(std_val - mean_std) > threshold * rolling_std.std():
                std_changes.append(i)
        
        return std_changes
    
    def _detect_mean_shifts(self, series, window=365, threshold=2.0):
        """检测均值偏移"""
        rolling_mean = series.rolling(window=window, min_periods=30).mean()
        overall_mean = series.mean()
        mean_shifts = []
        
        for i, mean_val in enumerate(rolling_mean):
            if pd.notna(mean_val) and abs(mean_val - overall_mean) > threshold * series.std():
                mean_shifts.append(i)
        
        return mean_shifts
    
    def _calculate_homogeneity_score(self, breakpoints, std_changes, mean_shifts):
        """计算同质化评分 (0-100)"""
        # 基础分100分，每检测到一个问题扣分
        score = 100
        
        # 断点扣分
        score -= len(breakpoints) * 10
        
        # 标准差变化扣分
        score -= len(std_changes) * 5
        
        # 均值偏移扣分
        score -= len(mean_shifts) * 5
        
        return max(0, score)
    
    def calculate_baseline_anomalies(self, column='snow_water_equivalent_mm'):
        """
        计算30年基准期异常
        
        Args:
            column (str): 要分析的列名
            
        Returns:
            pd.Series: 异常值序列
        """
        print(f"📊 计算30年基准期异常: {column}")
        
        if column not in self.data.columns:
            print(f"❌ 列 {column} 不存在")
            return None
        
        # 筛选基准期数据 (1991-2020)
        baseline_mask = (self.data.index.year >= self.baseline_period[0]) & \
                       (self.data.index.year <= self.baseline_period[1])
        
        baseline_data = self.data.loc[baseline_mask, column]
        
        if len(baseline_data) == 0:
            print(f"⚠️ 基准期 {self.baseline_period[0]}-{self.baseline_period[1]} 无数据")
            return None
        
        # 计算基准期统计量
        baseline_mean = baseline_data.mean()
        baseline_std = baseline_data.std()
        
        print(f"📈 基准期统计量:")
        print(f"  均值: {baseline_mean:.2f}")
        print(f"  标准差: {baseline_std:.2f}")
        print(f"  样本数: {len(baseline_data)}")
        
        # 计算异常值 (标准化异常)
        anomalies = (self.data[column] - baseline_mean) / baseline_std
        
        # 添加基准期信息
        self.analysis_results['baseline_stats'] = {
            'mean': baseline_mean,
            'std': baseline_std,
            'period': self.baseline_period,
            'sample_size': len(baseline_data)
        }
        
        return anomalies
    
    def mann_kendall_test(self, series, alpha=0.05):
        """
        Mann-Kendall趋势检验
        
        Args:
            series (pd.Series): 时间序列数据
            alpha (float): 显著性水平
            
        Returns:
            dict: 检验结果
        """
        print("🔍 执行Mann-Kendall趋势检验...")
        
        # 移除缺失值
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            print("⚠️ 数据点太少，无法进行趋势检验")
            return {}
        
        # 计算Mann-Kendall统计量
        n = len(clean_series)
        s = 0
        
        for i in range(n-1):
            for j in range(i+1, n):
                if clean_series.iloc[j] > clean_series.iloc[i]:
                    s += 1
                elif clean_series.iloc[j] < clean_series.iloc[i]:
                    s -= 1
        
        # 计算方差
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # 计算Z统计量
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # 计算p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        # 判断趋势
        if p_value < alpha:
            if z > 0:
                trend = "上升"
            else:
                trend = "下降"
            significant = True
        else:
            trend = "无显著趋势"
            significant = False
        
        results = {
            's_statistic': s,
            'z_statistic': z,
            'p_value': p_value,
            'significant': significant,
            'trend': trend,
            'alpha': alpha,
            'sample_size': n
        }
        
        print(f"📊 Mann-Kendall检验结果:")
        print(f"  S统计量: {s}")
        print(f"  Z统计量: {z:.4f}")
        print(f"  P值: {p_value:.4f}")
        print(f"  显著性: {'是' if significant else '否'}")
        print(f"  趋势: {trend}")
        
        return results
    
    def theil_sen_slope(self, series):
        """
        Theil-Sen稳健斜率估计
        
        Args:
            series (pd.Series): 时间序列数据
            
        Returns:
            dict: 斜率估计结果
        """
        print("📈 计算Theil-Sen稳健斜率...")
        
        # 移除缺失值
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            print("⚠️ 数据点太少，无法计算斜率")
            return {}
        
        # 创建时间索引
        time_index = np.arange(len(clean_series))
        
        # 计算所有点对的斜率
        slopes = []
        for i in range(len(clean_series)):
            for j in range(i+1, len(clean_series)):
                if time_index[j] != time_index[i]:
                    slope = (clean_series.iloc[j] - clean_series.iloc[i]) / (time_index[j] - time_index[i])
                    slopes.append(slope)
        
        if not slopes:
            print("⚠️ 无法计算斜率")
            return {}
        
        # 计算中位数斜率
        median_slope = np.median(slopes)
        
        # 计算置信区间 (使用百分位数)
        slopes_sorted = np.sort(slopes)
        n = len(slopes_sorted)
        
        # 95%置信区间
        lower_idx = int(0.025 * n)
        upper_idx = int(0.975 * n)
        
        ci_lower = slopes_sorted[lower_idx]
        ci_upper = slopes_sorted[upper_idx]
        
        results = {
            'median_slope': median_slope,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': 0.95,
            'sample_size': len(clean_series),
            'slope_pairs': len(slopes)
        }
        
        print(f"📊 Theil-Sen斜率估计结果:")
        print(f"  中位数斜率: {median_slope:.6f}")
        print(f"  95%置信区间: [{ci_lower:.6f}, {ci_upper:.6f}]")
        print(f"  样本数: {len(clean_series)}")
        
        return results
    
    def analyze_climate_change_impacts(self, column='snow_water_equivalent_mm'):
        """
        综合分析气候变化对SWE的影响
        
        Args:
            column (str): 要分析的列名
            
        Returns:
            dict: 综合分析结果
        """
        print(f"\n🌍 综合分析气候变化对SWE的影响: {column}")
        print("=" * 60)
        
        if column not in self.data.columns:
            print(f"❌ 列 {column} 不存在")
            return {}
        
        # 1. 数据同质化检查
        homogeneity_results = self.check_data_homogeneity(column)
        
        # 2. 计算基准期异常
        anomalies = self.calculate_baseline_anomalies(column)
        
        # 3. Mann-Kendall趋势检验
        mk_results = self.mann_kendall_test(self.data[column])
        
        # 4. Theil-Sen斜率估计
        ts_results = self.theil_sen_slope(self.data[column])
        
        # 5. 年际变化分析
        annual_stats = self._analyze_annual_variations(column)
        
        # 综合结果
        comprehensive_results = {
            'homogeneity': homogeneity_results,
            'baseline_anomalies': anomalies,
            'mann_kendall': mk_results,
            'theil_sen': ts_results,
            'annual_variations': annual_stats,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        self.analysis_results = comprehensive_results
        
        # 生成分析报告
        self._generate_analysis_report(comprehensive_results)
        
        return comprehensive_results
    
    def _analyze_annual_variations(self, column):
        """分析年际变化"""
        print("📊 分析年际变化...")
        
        # 按年分组计算统计量
        annual_data = self.data[column].groupby(self.data.index.year).agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).dropna()
        
        # 计算年际变化率
        annual_means = annual_data['mean']
        if len(annual_means) > 1:
            # 线性趋势
            years = np.array(annual_means.index)
            means = annual_means.values
            
            slope, intercept, r_value, p_value, std_err = linregress(years, means)
            
            annual_trend = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'trend_per_decade': slope * 10  # 每10年的变化
            }
        else:
            annual_trend = {}
        
        results = {
            'annual_statistics': annual_data,
            'trend_analysis': annual_trend
        }
        
        return results
    
    def _generate_analysis_report(self, results):
        """生成分析报告"""
        print("\n📋 气候变化影响SWE分析报告")
        print("=" * 60)
        
        # 同质化评分
        homogeneity_score = results['homogeneity'].get('homogeneity_score', 0)
        print(f"📊 数据质量评估:")
        print(f"  同质化评分: {homogeneity_score}/100")
        if homogeneity_score >= 80:
            print("  ✅ 数据质量良好，适合气候变化分析")
        elif homogeneity_score >= 60:
            print("  ⚠️ 数据质量一般，建议谨慎使用")
        else:
            print("  ❌ 数据质量较差，不建议用于气候变化分析")
        
        # 趋势分析
        mk_results = results['mann_kendall']
        if mk_results:
            print(f"\n📈 趋势分析结果:")
            print(f"  Mann-Kendall趋势: {mk_results.get('trend', '未知')}")
            print(f"  显著性: {'是' if mk_results.get('significant', False) else '否'}")
            print(f"  P值: {mk_results.get('p_value', 0):.4f}")
        
        # 斜率估计
        ts_results = results['theil_sen']
        if ts_results:
            print(f"\n📊 变化率估计:")
            print(f"  中位数斜率: {ts_results.get('median_slope', 0):.6f}")
            print(f"  95%置信区间: [{ts_results.get('ci_lower', 0):.6f}, {ts_results.get('ci_upper', 0):.6f}]")
        
        # 年际变化
        annual_trend = results['annual_variations'].get('trend_analysis', {})
        if annual_trend:
            trend_per_decade = annual_trend.get('trend_per_decade', 0)
            print(f"\n🌡️ 年际变化趋势:")
            print(f"  每10年变化: {trend_per_decade:.4f}")
            print(f"  决定系数: {annual_trend.get('r_squared', 0):.4f}")
        
        print("\n" + "=" * 60)
    
    def plot_analysis_results(self, save_path=None):
        """绘制分析结果图表"""
        if not self.analysis_results:
            print("❌ 没有分析结果，请先运行分析")
            return
        
        print("📊 绘制分析结果图表...")
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('气候变化影响SWE分析结果', fontsize=16)
        
        # 1. 原始时间序列
        if 'baseline_anomalies' in self.analysis_results:
            anomalies = self.analysis_results['baseline_anomalies']
            if anomalies is not None:
                axes[0, 0].plot(anomalies.index, anomalies.values, alpha=0.7)
                axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
                axes[0, 0].set_title('SWE基准期异常')
                axes[0, 0].set_ylabel('标准化异常')
                axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 年际变化
        if 'annual_variations' in self.analysis_results:
            annual_stats = self.analysis_results['annual_variations'].get('annual_statistics', pd.DataFrame())
            if not annual_stats.empty:
                annual_means = annual_stats['mean']
                axes[0, 1].plot(annual_means.index, annual_means.values, 'o-')
                axes[0, 1].set_title('年际SWE均值变化')
                axes[0, 1].set_ylabel('SWE (mm)')
                axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 趋势线
        if 'theil_sen' in self.analysis_results:
            ts_results = self.analysis_results['theil_sen']
            if ts_results:
                # 这里可以添加趋势线绘制
                axes[1, 0].set_title('趋势分析')
                axes[1, 0].text(0.1, 0.5, f"斜率: {ts_results.get('median_slope', 0):.6f}", 
                               transform=axes[1, 0].transAxes, fontsize=12)
        
        # 4. 同质化检测
        if 'homogeneity' in self.analysis_results:
            homogeneity_score = self.analysis_results['homogeneity'].get('homogeneity_score', 0)
            axes[1, 1].bar(['同质化评分'], [homogeneity_score], color='skyblue')
            axes[1, 1].set_ylim(0, 100)
            axes[1, 1].set_title('数据同质化评分')
            axes[1, 1].set_ylabel('评分')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 图表保存到: {save_path}")
        
        plt.show()

def main():
    """主函数 - 示例用法"""
    print("🚀 气候变化影响SWE分析模块")
    print("=" * 50)
    
    # 创建分析器
    analyzer = ClimateChangeAnalyzer()
    
    # 加载数据
    data_path = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
    analyzer.load_data(data_path)
    
    if analyzer.data is not None:
        # 执行综合分析
        results = analyzer.analyze_climate_change_impacts('snow_water_equivalent_mm')
        
        # 绘制结果
        analyzer.plot_analysis_results()
        
        print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
