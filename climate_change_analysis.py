#!/usr/bin/env python3
"""
气候变化影响SWE分析模块
实现Mann-Kendall趋势检验、Theil-Sen斜率估计等功能
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

class ClimateChangeAnalyzer:
    """气候变化影响SWE分析器"""
    
    def __init__(self):
        self.baseline_period = (1991, 2020)  # 30年基准期
    
    def mann_kendall_test(self, series):
        """Mann-Kendall趋势检验"""
        print("🔍 执行Mann-Kendall趋势检验...")
        
        clean_series = series.dropna()
        if len(clean_series) < 10:
            return {}
        
        n = len(clean_series)
        s = 0
        
        for i in range(n-1):
            for j in range(i+1, n):
                if clean_series.iloc[j] > clean_series.iloc[i]:
                    s += 1
                elif clean_series.iloc[j] < clean_series.iloc[i]:
                    s -= 1
        
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        if p_value < 0.05:
            trend = "上升" if z > 0 else "下降"
            significant = True
        else:
            trend = "无显著趋势"
            significant = False
        
        results = {
            's_statistic': s,
            'z_statistic': z,
            'p_value': p_value,
            'significant': significant,
            'trend': trend
        }
        
        print(f"📊 趋势: {trend}, 显著性: {'是' if significant else '否'}")
        return results
    
    def theil_sen_slope(self, series):
        """Theil-Sen稳健斜率估计"""
        print("📈 计算Theil-Sen斜率...")
        
        clean_series = series.dropna()
        if len(clean_series) < 10:
            return {}
        
        time_index = np.arange(len(clean_series))
        slopes = []
        
        for i in range(len(clean_series)):
            for j in range(i+1, len(clean_series)):
                if time_index[j] != time_index[i]:
                    slope = (clean_series.iloc[j] - clean_series.iloc[i]) / (time_index[j] - time_index[i])
                    slopes.append(slope)
        
        if not slopes:
            return {}
        
        median_slope = np.median(slopes)
        slopes_sorted = np.sort(slopes)
        n = len(slopes_sorted)
        
        lower_idx = int(0.025 * n)
        upper_idx = int(0.975 * n)
        
        results = {
            'median_slope': median_slope,
            'ci_lower': slopes_sorted[lower_idx],
            'ci_upper': slopes_sorted[upper_idx]
        }
        
        print(f"📊 斜率: {median_slope:.6f}")
        return results
    
    def calculate_baseline_anomalies(self, data, column):
        """计算30年基准期异常"""
        print("📊 计算基准期异常...")
        
        baseline_mask = (data.index.year >= self.baseline_period[0]) & \
                       (data.index.year <= self.baseline_period[1])
        
        baseline_data = data.loc[baseline_mask, column]
        baseline_mean = baseline_data.mean()
        baseline_std = baseline_data.std()
        
        anomalies = (data[column] - baseline_mean) / baseline_std
        
        print(f"📈 基准期均值: {baseline_mean:.2f}, 标准差: {baseline_std:.2f}")
        return anomalies

def main():
    """主函数"""
    print("🚀 气候变化影响SWE分析")
    
    # 加载数据
    data_path = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
    data = pd.read_csv(data_path, parse_dates=['date'])
    data.set_index('date', inplace=True)
    
    # 创建分析器
    analyzer = ClimateChangeAnalyzer()
    
    # 分析SWE变化
    if 'snow_water_equivalent_mm' in data.columns:
        swe_series = data['snow_water_equivalent_mm']
        
        # Mann-Kendall检验
        mk_results = analyzer.mann_kendall_test(swe_series)
        
        # Theil-Sen斜率
        ts_results = analyzer.theil_sen_slope(swe_series)
        
        # 基准期异常
        anomalies = analyzer.calculate_baseline_anomalies(data, 'snow_water_equivalent_mm')
        
        print("\n✅ 分析完成!")

if __name__ == "__main__":
    main()
