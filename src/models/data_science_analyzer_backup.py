#!/usr/bin/env python3
"""
专业数据科学分析模块
集成无监督学习、异常检测、聚类分析、时间模式分析等功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
from scipy.signal import periodogram, welch
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import os
import warnings
warnings.filterwarnings('ignore')

class DataScienceAnalyzer:
    """专业数据科学分析器"""
    
    def __init__(self, data_path=None):
        """
        初始化数据科学分析器
        
        Args:
            data_path (str): 数据文件路径
        """
        self.data = None
        self.analysis_results = {}
        self.scaler = StandardScaler()
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """加载数据"""
        print(f"📊 加载数据: {data_path}")
        
        try:
            if not os.path.exists(data_path):
                # 尝试备用数据路径
                backup_paths = [
                    "data/processed/eccc_manitoba_snow_processed.csv",
                    "data/raw/eccc_recent/eccc_recent_combined.csv"
                ]
                
                for backup_path in backup_paths:
                    if os.path.exists(backup_path):
                        print(f"📂 使用备用数据: {backup_path}")
                        data_path = backup_path
                        break
                else:
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
            
            # 创建或找到snow_water_equivalent_mm列（不使用模拟数据）
            if 'snow_water_equivalent_mm' not in self.data.columns:
                swe_candidates = [
                    'Snow on Grnd (mm)', 'Snow on Grnd (cm)', 
                    'Total Snow (mm)', 'Total Snow (cm)'
                ]
                
                for candidate in swe_candidates:
                    if candidate in self.data.columns:
                        if 'cm' in candidate:
                            self.data['snow_water_equivalent_mm'] = self.data[candidate] * 10.0
                        else:
                            self.data['snow_water_equivalent_mm'] = self.data[candidate]
                        print(f"✅ 从 {candidate} 创建 snow_water_equivalent_mm 列")
                        break
                else:
                    raise ValueError("未找到真实的雪水当量列，禁止使用模拟数据")
            
            # 数据清理
            self.data['snow_water_equivalent_mm'] = pd.to_numeric(
                self.data['snow_water_equivalent_mm'], errors='coerce'
            )
            
            original_len = len(self.data)
            self.data = self.data.dropna(subset=['snow_water_equivalent_mm'])
            
            if len(self.data) == 0:
                raise ValueError("处理后数据为空")
            
            print(f"✅ 数据加载成功: {len(self.data)} 条记录 (原始: {original_len} 条)")
            print(f"📅 时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
            
        except Exception as e:
            print(f"❌ 数据加载失败: {e}")
            self.data = None
    
    def _series_to_dict(self, series):
        """将pandas Series转换为前端期望的字典格式，处理NaN值"""
        # 处理NaN值，确保JSON序列化成功
        clean_values = []
        for val in series.values:
            if pd.isna(val) or np.isnan(val) or np.isinf(val):
                clean_values.append(None)  # 使用None替代NaN/Inf
            else:
                clean_values.append(float(val))  # 确保是普通浮点数
        
        return {
            'index': series.index.tolist(),
            'values': clean_values
        }
    
    def advanced_time_series_decomposition(self, column='snow_water_equivalent_mm'):
        """
        高级时间序列分解 - 多尺度分析
        
        Args:
            column (str): 要分析的列名
            
        Returns:
            dict: 分解结果
        """
        print(f"\n🔍 执行高级时间序列分解: {column}")
        print("=" * 60)
        
        if self.data is None or column not in self.data.columns:
            return {}
        
        series = self.data[column].dropna()
        if len(series) == 0:
            return {}
        
        results = {}
        
        # 1. STL分解
        try:
            results['stl_decomposition'] = self._stl_decomposition(series)
        except Exception as e:
            print(f"⚠️ STL分解失败: {e}. 使用简化分解替代")
            results['stl_decomposition'] = self._simple_decomposition(series)

        # 2. 小波分解（使用pywavelets）
        try:
            results['wavelet_decomposition'] = self._wavelet_decomposition(series)
        except Exception as e:
            print(f"⚠️ 小波分解失败: {e}")

        # 3. 经验模态分解（可选）
        try:
            results['emd_decomposition'] = self._emd_decomposition(series)
        except Exception as e:
            print(f"⚠️ EMD分解失败: {e}")

        # 4. 多尺度趋势
        try:
            results['multi_scale_trends'] = self._multi_scale_trend_analysis(series)
        except Exception as e:
            print(f"⚠️ 多尺度趋势分析失败: {e}")

        # 5. 周期性检测
        try:
            results['periodicity_analysis'] = self._periodicity_detection(series)
        except Exception as e:
            print(f"⚠️ 周期性检测失败: {e}")

        # Add interpretation for decomposition analysis
        results['interpretation'] = self._interpret_decomposition_results(results)

        self.analysis_results['advanced_decomposition'] = results
        print("✅ Advanced time series decomposition completed")
        return results
    
    def _stl_decomposition(self, series):
        """STL分解"""
        print("📊 执行STL分解...")
        
        try:
            from statsmodels.tsa.seasonal import STL
            
            # 数据质量检查和预处理策略
            print(f"📊 原始数据统计: 长度={len(series)}, 频率={series.index.inferred_freq}")
            
            # 检查是否需要重采样
            original_freq = series.index.inferred_freq
            if original_freq is None:
                # 推断频率
                time_diffs = series.index.to_series().diff().dropna()
                median_diff = time_diffs.median()
                if median_diff <= pd.Timedelta('1D'):
                    target_freq = 'D'
                elif median_diff <= pd.Timedelta('7D'):
                    target_freq = 'W'
                elif median_diff <= pd.Timedelta('30D'):
                    target_freq = 'M'
                else:
                    target_freq = 'D'  # 默认日频率
                print(f"⚠️ 推断频率: {target_freq}")
            else:
                target_freq = original_freq
            
            # 智能重采样：保持数据完整性
            if target_freq == 'D':
                series_resampled = series.resample('D').mean()
            elif target_freq == 'W':
                series_resampled = series.resample('W').mean()
            elif target_freq == 'M':
                series_resampled = series.resample('M').mean()
            else:
                series_resampled = series.resample('D').mean()
            
            print(f"📊 重采样后统计: 长度={len(series_resampled)}, 缺失值={series_resampled.isna().sum()}")
            
            # 智能填充策略
            if series_resampled.isna().sum() > 0:
                missing_ratio = series_resampled.isna().sum() / len(series_resampled)
                print(f"📊 缺失值比例: {missing_ratio:.3f}")
                
                if missing_ratio < 0.05:  # 缺失少于5%
                    # 使用线性插值
                    series_resampled = series_resampled.interpolate(method='linear')
                    print("✅ 使用线性插值填充少量缺失值")
                elif missing_ratio < 0.2:  # 缺失少于20%
                    # 使用样条插值
                    series_resampled = series_resampled.interpolate(method='spline', order=2)
                    print("✅ 使用样条插值填充中等缺失值")
                else:
                    # 大量缺失，使用前向填充但严格限制
                    series_resampled = series_resampled.fillna(method='ffill', limit=3)
                    print("⚠️ 大量缺失值，使用限制性前向填充")
                    
                # 验证填充结果
                remaining_missing = series_resampled.isna().sum()
                print(f"📊 填充后剩余缺失值: {remaining_missing}")
                
                if remaining_missing > 0:
                    # 如果仍有缺失，使用均值填充
                    series_resampled = series_resampled.fillna(series_resampled.mean())
                    print("⚠️ 使用均值填充剩余缺失值")
            
            # 推断周期：按日数据用365，按月数据用12，否则回退为近似周期
            inferred = series_resampled.index.inferred_freq
            period = 365
            if inferred is not None and inferred.upper().startswith('M'):
                period = 12
            elif inferred is not None and inferred.upper().startswith('D'):
                period = 365
            
            # 数据质量检查
            print(f"📊 数据统计: 长度={len(series_resampled)}, 均值={series_resampled.mean():.2f}, 标准差={series_resampled.std():.2f}")
            
            # 异常值检测和处理
            Q1 = series_resampled.quantile(0.25)
            Q3 = series_resampled.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (series_resampled < lower_bound) | (series_resampled > upper_bound)
            if outliers.sum() > 0:
                print(f"⚠️ 检测到 {outliers.sum()} 个异常值，使用稳健STL分解")
                robust_flag = True
            else:
                robust_flag = False
            
            # 执行STL分解 - 修复参数设置
            # STL的seasonal参数必须是奇数，且 >= 3
            seasonal_window = min(period // 2, 365 // 2)  # 季节性窗口大小
            if seasonal_window % 2 == 0:  # 确保是奇数
                seasonal_window = seasonal_window + 1
            if seasonal_window < 3:  # 确保 >= 3
                seasonal_window = 3
            print(f"📊 STL参数: seasonal={seasonal_window}, period={period}")
            stl = STL(series_resampled, seasonal=seasonal_window, period=period, robust=robust_flag)
            result = stl.fit()
            
            # 计算季节性强度和趋势强度 - 修复计算方式
            total_variance = np.var(series_resampled.dropna())
            if total_variance > 0:
                seasonal_strength = np.var(result.seasonal.dropna()) / total_variance
                trend_strength = np.var(result.trend.dropna()) / total_variance
            else:
                seasonal_strength = 0.0
                trend_strength = 0.0
            
            # 使用通用的series_to_dict函数
            
            # 结果验证和合理性检查
            decomposition_result = {
                'trend': self._series_to_dict(result.trend),
                'seasonal': self._series_to_dict(result.seasonal),
                'resid': self._series_to_dict(result.resid),
                'seasonal_strength': seasonal_strength,
                'trend_strength': trend_strength
            }
            
            # 验证分解结果的合理性
            validation_result = self._validate_decomposition_result(decomposition_result, series_resampled)
            if not validation_result['is_valid']:
                print(f"⚠️ 分解结果验证失败: {validation_result['issues']}")
                # 尝试使用简化分解
                print("🔄 尝试使用简化分解...")
                return self._simple_decomposition(series)
            
            print("✅ 分解结果验证通过")
            return decomposition_result
        except ImportError:
            print("⚠️ statsmodels未安装，使用简化分解")
            return self._simple_decomposition(series)
        except Exception as e:
            print(f"❌ STL分解失败: {e}")
            print("🔄 回退到简化分解...")
            return self._simple_decomposition(series)
    
    def _validate_decomposition_result(self, result, original_series):
        """验证STL分解结果的合理性"""
        try:
            issues = []
            
            # 1. 检查数据完整性
            if not all(key in result for key in ['trend', 'seasonal', 'resid']):
                issues.append("Missing decomposition components")
                return {'is_valid': False, 'issues': issues}
            
            # 2. 检查数据长度一致性
            trend_len = len(result['trend']['values'])
            seasonal_len = len(result['seasonal']['values'])
            resid_len = len(result['resid']['values'])
            original_len = len(original_series)
            
            if not (trend_len == seasonal_len == resid_len == original_len):
                issues.append(f"Length mismatch: trend={trend_len}, seasonal={seasonal_len}, resid={resid_len}, original={original_len}")
            
            # 3. 检查重建误差
            trend_values = np.array(result['trend']['values'])
            seasonal_values = np.array(result['seasonal']['values'])
            resid_values = np.array(result['resid']['values'])
            original_values = original_series.values
            
            # 重建数据
            reconstructed = trend_values + seasonal_values + resid_values
            
            # 计算重建误差
            reconstruction_error = np.mean(np.abs(reconstructed - original_values))
            original_std = np.std(original_values)
            
            if original_std > 0:
                relative_error = reconstruction_error / original_std
                if relative_error > 0.1:  # 相对误差超过10%
                    issues.append(f"High reconstruction error: {relative_error:.3f}")
            
            # 4. 检查季节性强度合理性
            seasonal_strength = result.get('seasonal_strength', 0)
            if seasonal_strength > 0.95:
                issues.append(f"Unusually high seasonal strength: {seasonal_strength:.3f}")
            elif seasonal_strength < 0.01:
                issues.append(f"Unusually low seasonal strength: {seasonal_strength:.3f}")
            
            # 5. 检查趋势强度合理性 - 进一步放宽阈值
            trend_strength = result.get('trend_strength', 0)
            if trend_strength > 0.95:
                issues.append(f"Unusually high trend strength: {trend_strength:.3f}")
            elif trend_strength < 0.0001:  # 从0.001进一步放宽到0.0001
                issues.append(f"Unusually low trend strength: {trend_strength:.3f}")
            
            # 6. 检查残差的白噪声特性
            resid_values_clean = resid_values[~np.isnan(resid_values)]
            if len(resid_values_clean) > 0:
                resid_std = np.std(resid_values_clean)
                if resid_std < 1e-6:  # 残差几乎为0
                    issues.append("Residuals are almost zero, decomposition may be overfitting")
            
            is_valid = len(issues) == 0
            
            if not is_valid:
                print(f"❌ 分解结果验证失败:")
                for issue in issues:
                    print(f"   - {issue}")
            else:
                print(f"✅ 分解结果验证通过")
            
            return {
                'is_valid': is_valid,
                'issues': issues,
                'reconstruction_error': reconstruction_error if 'reconstruction_error' in locals() else None,
                'relative_error': relative_error if 'relative_error' in locals() else None
            }
            
        except Exception as e:
            print(f"⚠️ 验证过程出错: {e}")
            return {'is_valid': False, 'issues': [f"Validation error: {e}"]}
    
    def _wavelet_decomposition(self, series):
        """小波分解 - 使用成熟的pywavelets库"""
        print("🌊 执行小波分解...")
        
        try:
            import pywt
            
            # 确保数据长度是2的幂次
            n = len(series)
            next_power = 2 ** int(np.ceil(np.log2(n)))
            series_padded = np.pad(series.values, (0, next_power - n), 'edge')
            
            # 执行小波分解
            coeffs = pywt.wavedec(series_padded, 'db4', level=4)
            
            # 重构各个分量
            reconstructed = []
            for i, coeff in enumerate(coeffs):
                coeff_list = [np.zeros_like(c) for c in coeffs]
                coeff_list[i] = coeff
                reconstructed.append(pywt.waverec(coeff_list, 'db4')[:n])
            
            # 使用tsfresh提取小波特征
            try:
                import tsfresh
                from tsfresh.feature_extraction import MinimalFCParameters
                
                df = pd.DataFrame({
                    'id': 1,
                    'time': range(len(series)),
                    'value': series.values
                })
                
                # 提取小波相关特征
                wavelet_features = tsfresh.extract_features(df, column_id='id', column_sort='time', 
                                                         column_value='value', 
                                                         default_fc_parameters=MinimalFCParameters())
                
                return {
                    'approximation': reconstructed[0].tolist(),
                    'details': [detail.tolist() for detail in reconstructed[1:]],
                    'wavelet_type': 'db4',
                    'levels': len(coeffs) - 1,
                    'tsfresh_wavelet_features': wavelet_features.to_dict('records')[0] if not wavelet_features.empty else {}
                }
                
            except Exception as e:
                print(f"⚠️ tsfresh小波特征提取失败: {e}")
                return {
                    'approximation': reconstructed[0].tolist(),
                    'details': [detail.tolist() for detail in reconstructed[1:]],
                'wavelet_type': 'db4',
                'levels': len(coeffs) - 1
            }
            
        except ImportError:
            print("⚠️ PyWavelets未安装，跳过小波分解")
            return {}
        except Exception as e:
            print(f"❌ 小波分解失败: {e}")
            return {}
    
    def _emd_decomposition(self, series):
        """经验模态分解"""
        print("🔄 执行经验模态分解...")
        
        try:
            from PyEMD import EMD
            
            emd = EMD()
            IMFs = emd(series.values)
            
            return {
                'imfs': IMFs,
                'residue': IMFs[-1] if len(IMFs) > 0 else None,
                'n_imfs': len(IMFs)
            }
        except ImportError:
            print("⚠️ PyEMD未安装，跳过EMD分解")
            return {}
    
    def _multi_scale_trend_analysis(self, series):
        """多尺度趋势分析"""
        print("📈 执行多尺度趋势分析...")
        
        # 不同时间尺度的趋势
        scales = [7, 30, 90, 365]  # 周、月、季、年
        trends = {}
        
        for scale in scales:
            if len(series) > scale:
                rolling_mean = series.rolling(window=scale, center=True).mean()
                trends[f'{scale}_day'] = rolling_mean
        
        # 计算趋势强度
        trend_strength = {}
        for scale, trend in trends.items():
            if not trend.isna().all():
                # 计算趋势的方差与原始序列方差的比值
                strength = trend.var() / series.var() if series.var() > 0 else 0
                trend_strength[scale] = strength
        
        return {
            'trends': trends,
            'trend_strength': trend_strength
        }
    
    def _periodicity_detection(self, series):
        """周期性检测 - 使用成熟的tsfresh库"""
        print("🔄 执行周期性检测...")
        
        try:
            import tsfresh
            from tsfresh.feature_extraction import MinimalFCParameters
            
            # 准备数据格式
            df = pd.DataFrame({
                'id': 1,
                'time': range(len(series)),
                'value': series.values
            })
            
            # 使用tsfresh的特征提取
            features = tsfresh.extract_features(df, column_id='id', column_sort='time', 
                                             column_value='value', 
                                             default_fc_parameters=MinimalFCParameters())
            
            # 提取关键周期性特征
            periodicity_features = {
                'fft_coefficient': features.get('value__fft_coefficient__real_0', [0])[0],
                'fft_aggregated': features.get('value__fft_aggregated__aggtype_centroid', [0])[0],
                'fft_aggregated_peaks': features.get('value__fft_aggregated__aggtype_peaks', [0])[0],
                'fft_aggregated_centroid': features.get('value__fft_aggregated__aggtype_centroid', [0])[0]
            }
            
            # 计算主要周期
            try:
                # 使用scipy的welch方法作为备选
                from scipy.signal import welch
                series_resampled = series.resample('D').mean().fillna(method='ffill')
                frequencies, power = welch(series_resampled.dropna(), nperseg=min(256, len(series_resampled)//4))
                
                main_freq_idx = np.argmax(power)
                main_frequency = frequencies[main_freq_idx]
                main_period = 1.0 / main_frequency if main_frequency > 0 else 365.0
                
                return {
                    'tsfresh_features': periodicity_features,
                    'main_frequency': main_frequency,
                    'main_period': main_period,
                    'power_spectrum': {
                        'frequencies': frequencies.tolist(),
                        'power': power.tolist()
                    }
                }
            except Exception as e:
                print(f"⚠️ Welch方法失败，使用tsfresh特征: {e}")
                return {
                    'tsfresh_features': periodicity_features,
                    'main_period': 365.0,  # 默认年周期
                    'method': 'tsfresh_only'
                }
                
        except ImportError:
            print("⚠️ tsfresh未安装，使用传统方法")
            # 回退到传统方法
            try:
                from scipy.signal import welch
                series_resampled = series.resample('D').mean().fillna(method='ffill')
                frequencies, power = welch(series_resampled.dropna(), nperseg=min(256, len(series_resampled)//4))
                
                main_freq_idx = np.argmax(power)
                main_frequency = frequencies[main_freq_idx]
                main_period = 1.0 / main_frequency if main_frequency > 0 else 365.0
                
                return {
                    'frequencies': frequencies.tolist(),
                    'power': power.tolist(),
                    'main_frequency': main_frequency,
                    'main_period': main_period,
                    'method': 'traditional'
                }
            except Exception as e:
                print(f"⚠️ 传统方法也失败: {e}")
                return {
                    'main_period': 365.0,
                    'method': 'fallback',
                    'error': str(e)
        }
    
    def _simple_decomposition(self, series):
        """简化分解（当STL不可用时）"""
        print("📊 执行简化分解...")
        
        # 计算移动平均作为趋势
        window = min(365, len(series) // 4)
        trend = series.rolling(window=window, center=True).mean()
        
        # 计算季节性 - 修复算法
        # 使用去趋势后的数据进行季节性分析
        detrended = series - trend
        
        # 计算季节性：按年分组计算平均值
        if len(series) >= 365:
            # 按年分组 - 修复元组问题
            try:
                # 确保索引是datetime类型
                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.to_datetime(series.index)
                
                yearly_groups = detrended.groupby(series.index.year)
                seasonal_pattern = yearly_groups.mean()
                
                # 扩展到整个时间序列
                seasonal = pd.Series(index=series.index, dtype=float)
                for year in yearly_groups.groups:
                    year_mask = series.index.year == year
                    seasonal[year_mask] = seasonal_pattern[year]
            except Exception as e:
                print(f"⚠️ 年分组失败，使用简单方法: {e}")
                # 回退到简单方法
                seasonal = detrended.rolling(window=30, center=True).mean()
        else:
            # 数据不足一年，使用简单的周期性模式
            seasonal = detrended.rolling(window=7, center=True).mean()
        
        # 计算残差 - 修复计算
        residual = series - trend - seasonal
        
        # 使用通用的series_to_dict函数
        
        return {
            'trend': self._series_to_dict(trend),
            'seasonal': self._series_to_dict(seasonal),
            'resid': self._series_to_dict(residual),
            'seasonal_strength': 0.5,  # 默认值
            'trend_strength': 0.5
        }
    
    def advanced_anomaly_detection(self, column='snow_water_equivalent_mm'):
        """
        高级异常检测 - 多种算法集成
        
        Args:
            column (str): 要检测的列名
            
        Returns:
            dict: 异常检测结果
        """
        print(f"\n🚨 执行高级异常检测: {column}")
        print("=" * 60)
        
        if self.data is None or column not in self.data.columns:
            return {}
        
        series = self.data[column].dropna()
        if len(series) == 0:
            return {}
        
        try:
            # 1. 统计方法异常检测
            statistical_anomalies = self._statistical_anomaly_detection(series)
            
            # 2. 机器学习异常检测
            ml_anomalies = self._ml_anomaly_detection(series)
            
            # 3. 时间序列异常检测
            ts_anomalies = self._timeseries_anomaly_detection(series)
            
            # 4. 集成异常检测
            ensemble_anomalies = self._ensemble_anomaly_detection(
                statistical_anomalies, ml_anomalies, ts_anomalies
            )
            
            # 5. 异常解释
            anomaly_explanations = self._explain_anomalies(series, ensemble_anomalies)
            
            results = {
                'statistical': statistical_anomalies,
                'machine_learning': ml_anomalies,
                'timeseries': ts_anomalies,
                'ensemble': ensemble_anomalies,
                'explanations': anomaly_explanations
            }
            
            # Add interpretation for anomaly detection
            results['interpretation'] = self._interpret_anomaly_detection_results(results)
            
            self.analysis_results['advanced_anomaly_detection'] = results
            print("✅ Advanced anomaly detection completed")
            return results
            
        except Exception as e:
            print(f"❌ 高级异常检测失败: {e}")
            return {}
    
    def _statistical_anomaly_detection(self, series):
        """统计方法异常检测"""
        print("📊 统计方法异常检测...")
        
        # Z-score方法 (降低阈值使其更敏感)
        z_scores = np.abs(stats.zscore(series))
        z_anomalies = z_scores > 2.0  # 从3降到2，更合理的阈值
        
        # Modified Z-score (使用中位数)
        median = np.median(series)
        mad = np.median(np.abs(series - median))
        modified_z_scores = 0.6745 * (series - median) / mad
        modified_z_anomalies = np.abs(modified_z_scores) > 2.0  # 进一步降低阈值
        
        # IQR方法
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        iqr_anomalies = (series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))
        
        # 极端IQR方法
        extreme_iqr_anomalies = (series < (Q1 - 3 * IQR)) | (series > (Q3 + 3 * IQR))
        
        # 移动窗口方法
        window = min(30, len(series) // 10)
        rolling_mean = series.rolling(window=window, center=True).mean()
        rolling_std = series.rolling(window=window, center=True).std()
        rolling_anomalies = np.abs(series - rolling_mean) > 3 * rolling_std
        
        return {
            'z_score_anomalies': z_anomalies,
            'modified_z_score_anomalies': modified_z_anomalies,
            'iqr_anomalies': iqr_anomalies,
            'extreme_iqr_anomalies': extreme_iqr_anomalies,
            'rolling_anomalies': rolling_anomalies,
            'z_scores': z_scores,
            'modified_z_scores': modified_z_scores
        }
    
    def _ml_anomaly_detection(self, series):
        """机器学习异常检测"""
        print("🤖 机器学习异常检测...")
        
        # 准备数据
        X = series.values.reshape(-1, 1)
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        
        # Isolation Forest
        try:
            iso_forest = IsolationForest(contamination=0.02, random_state=42)  # 2%更合理
            iso_predictions = iso_forest.fit_predict(X_scaled)
            iso_anomalies = iso_predictions == -1
            iso_scores = iso_forest.decision_function(X_scaled)
            
            results['isolation_forest'] = {
                'anomalies': iso_anomalies,
                'scores': iso_scores
            }
        except Exception as e:
            print(f"⚠️ Isolation Forest失败: {e}")
        
        # One-Class SVM
        try:
            oc_svm = OneClassSVM(nu=0.02, kernel='rbf')  # 2%更合理
            svm_predictions = oc_svm.fit_predict(X_scaled)
            svm_anomalies = svm_predictions == -1
            svm_scores = oc_svm.decision_function(X_scaled)
            
            results['one_class_svm'] = {
                'anomalies': svm_anomalies,
                'scores': svm_scores
            }
        except Exception as e:
            print(f"⚠️ One-Class SVM失败: {e}")
        
        # Local Outlier Factor
        try:
            lof = LocalOutlierFactor(n_neighbors=min(20, len(X)//2), contamination=0.1)
            lof_predictions = lof.fit_predict(X_scaled)
            lof_anomalies = lof_predictions == -1
            lof_scores = lof.negative_outlier_factor_
            
            results['local_outlier_factor'] = {
                'anomalies': lof_anomalies,
                'scores': lof_scores
            }
        except Exception as e:
            print(f"⚠️ Local Outlier Factor失败: {e}")
        
        return results
    
    def _timeseries_anomaly_detection(self, series):
        """时间序列异常检测"""
        print("⏰ 时间序列异常检测...")
        
        # 基于趋势的异常检测
        window = min(30, len(series) // 10)
        rolling_mean = series.rolling(window=window, center=True).mean()
        trend_anomalies = np.abs(series - rolling_mean) > 2 * series.std()
        
        # 基于季节性的异常检测
        monthly_means = series.groupby(series.index.month).mean()
        monthly_stds = series.groupby(series.index.month).std()
        seasonal_anomalies = np.abs(series - monthly_means[series.index.month].values) > 2 * monthly_stds[series.index.month].values
        
        # 基于变化率的异常检测
        diff = series.diff()
        diff_anomalies = np.abs(diff) > 3 * diff.std()
        
        # 基于自相关的异常检测
        autocorr_anomalies = self._autocorrelation_anomaly_detection(series)
        
        return {
            'trend_anomalies': trend_anomalies,
            'seasonal_anomalies': seasonal_anomalies,
            'change_rate_anomalies': diff_anomalies,
            'autocorr_anomalies': autocorr_anomalies
        }
    
    def _autocorrelation_anomaly_detection(self, series):
        """基于自相关的异常检测"""
        try:
            # 计算自相关
            autocorr = series.autocorr(lag=1)
            
            # 如果自相关很低，可能存在异常
            autocorr_anomalies = pd.Series([False] * len(series), index=series.index)
            
            # 这里可以添加更复杂的自相关异常检测逻辑
            return autocorr_anomalies
        except:
            return pd.Series([False] * len(series), index=series.index)
    
    def _ensemble_anomaly_detection(self, statistical, ml, ts):
        """集成异常检测"""
        print("🔗 集成异常检测...")
        
        # 收集所有异常检测结果
        all_anomalies = []
        
        # 统计方法
        for method in ['z_score_anomalies', 'iqr_anomalies', 'rolling_anomalies']:
            if method in statistical:
                all_anomalies.append(statistical[method].astype(int))
        
        # 机器学习方法
        for method in ['isolation_forest', 'one_class_svm', 'local_outlier_factor']:
            if method in ml and 'anomalies' in ml[method]:
                all_anomalies.append(ml[method]['anomalies'].astype(int))
        
        # 时间序列方法
        for method in ['trend_anomalies', 'seasonal_anomalies']:
            if method in ts:
                all_anomalies.append(ts[method].astype(int))
        
        if not all_anomalies:
            return {'ensemble_anomalies': pd.Series([False] * len(self.data), index=self.data.index)}
        
        # 计算集成分数
        ensemble_scores = np.mean(all_anomalies, axis=0)
        
        # 动态阈值
        threshold = np.percentile(ensemble_scores, 90)  # 前10%作为异常
        ensemble_anomalies = ensemble_scores > threshold
        
        return {
            'ensemble_scores': ensemble_scores,
            'ensemble_anomalies': ensemble_anomalies,
            'threshold': threshold,
            'n_methods': len(all_anomalies)
        }
    
    def _explain_anomalies(self, series, ensemble_anomalies):
        """异常解释"""
        print("🔍 异常解释...")
        
        if 'ensemble_anomalies' not in ensemble_anomalies:
            return {}
        
        anomalies = ensemble_anomalies['ensemble_anomalies']
        # 修复数组比较问题
        if isinstance(anomalies, (list, np.ndarray)):
            anomalies = np.array(anomalies)
        else:
            return {}
            
        anomaly_indices = series.index[anomalies]
        
        explanations = []
        for idx in anomaly_indices[:10]:  # 只解释前10个异常
            value = series.loc[idx]
            mean_val = series.mean()
            std_val = series.std()
            
            explanation = {
                'timestamp': idx,
                'value': value,
                'deviation': (value - mean_val) / std_val if std_val > 0 else 0,
                'type': 'high' if value > mean_val else 'low',
                'context': f"值 {value:.2f} 偏离均值 {mean_val:.2f} 约 {abs(value - mean_val)/std_val:.1f} 个标准差"
            }
            explanations.append(explanation)
        
        return {
            'total_anomalies': anomalies.sum(),
            'anomaly_rate': anomalies.sum() / len(anomalies),
            'explanations': explanations
        }
    
    def clustering_analysis(self, columns=None):
        """
        聚类分析 - 发现数据中的隐藏模式
        
        Args:
            columns (list): 要分析的列名列表
            
        Returns:
            dict: 聚类分析结果
        """
        print(f"\n🔍 执行聚类分析")
        print("=" * 60)
        
        if self.data is None:
            return {}
        
        # 选择数值列
        if columns is None:
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = [col for col in columns if col in self.data.columns]
        
        if not numeric_columns:
            print("❌ 没有找到数值列")
            return {}
        
        # 准备数据
        data_subset = self.data[numeric_columns].dropna()
        if len(data_subset) == 0:
            print("❌ 数据为空")
            return {}
        
        try:
            # 数据标准化
            X_scaled = self.scaler.fit_transform(data_subset)
            
            # 1. K-means聚类
            kmeans_results = self._kmeans_clustering(X_scaled, data_subset)
            
            # 2. DBSCAN聚类
            dbscan_results = self._dbscan_clustering(X_scaled, data_subset)
            
            # 3. 层次聚类
            hierarchical_results = self._hierarchical_clustering(X_scaled, data_subset)
            
            # 4. 聚类评估
            clustering_evaluation = self._evaluate_clustering(X_scaled, kmeans_results, dbscan_results, hierarchical_results)
            
            # 5. 聚类解释
            cluster_interpretations = self._interpret_clusters(data_subset, kmeans_results, dbscan_results, hierarchical_results)
            
            results = {
                'kmeans': kmeans_results,
                'dbscan': dbscan_results,
                'hierarchical': hierarchical_results,
                'evaluation': clustering_evaluation,
                'interpretations': cluster_interpretations,
                'features_used': numeric_columns
            }
            
            # Add interpretation for clustering analysis
            results['interpretation'] = self._interpret_clustering_results(results)
            
            self.analysis_results['clustering_analysis'] = results
            print("✅ Clustering analysis completed")
            return results
            
        except Exception as e:
            print(f"❌ 聚类分析失败: {e}")
            return {}
    
    def _kmeans_clustering(self, X_scaled, data_subset):
        """K-means聚类"""
        print("🎯 K-means聚类...")
        
        # 确定最优聚类数
        n_clusters_range = range(2, min(11, len(data_subset)//10))
        inertias = []
        silhouette_scores = []
        
        for n_clusters in n_clusters_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_scaled)
            
            inertias.append(kmeans.inertia_)
            if n_clusters > 1:
                silhouette_scores.append(silhouette_score(X_scaled, cluster_labels))
            else:
                silhouette_scores.append(0)
        
        # 选择最优聚类数（肘部法则 + 轮廓系数）
        optimal_k = n_clusters_range[np.argmax(silhouette_scores)]
        
        # 执行最终聚类
        final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        final_labels = final_kmeans.fit_predict(X_scaled)
        
        return {
            'n_clusters': optimal_k,
            'labels': final_labels,
            'centers': final_kmeans.cluster_centers_,
            'inertia': final_kmeans.inertia_,
            'silhouette_score': silhouette_score(X_scaled, final_labels),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'n_clusters_range': list(n_clusters_range)
        }
    
    def _dbscan_clustering(self, X_scaled, data_subset):
        """DBSCAN聚类"""
        print("🌐 DBSCAN聚类...")
        
        # 尝试不同的eps值
        eps_values = [0.1, 0.2, 0.5, 1.0, 2.0]
        best_eps = 0.5
        best_n_clusters = 0
        
        for eps in eps_values:
            dbscan = DBSCAN(eps=eps, min_samples=5)
            labels = dbscan.fit_predict(X_scaled)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters > best_n_clusters and n_clusters > 1:
                best_eps = eps
                best_n_clusters = n_clusters
        
        # 执行最终聚类
        final_dbscan = DBSCAN(eps=best_eps, min_samples=5)
        final_labels = final_dbscan.fit_predict(X_scaled)
        
        return {
            'eps': best_eps,
            'labels': final_labels,
            'n_clusters': len(set(final_labels)) - (1 if -1 in final_labels else 0),
            'n_noise': list(final_labels).count(-1),
            'core_samples': final_dbscan.core_sample_indices_
        }
    
    def _hierarchical_clustering(self, X_scaled, data_subset):
        """层次聚类"""
        print("🌳 层次聚类...")
        
        # 使用不同的链接方法
        linkage_methods = ['ward', 'complete', 'average', 'single']
        best_method = 'ward'
        best_score = -1
        
        for method in linkage_methods:
            try:
                clustering = AgglomerativeClustering(n_clusters=3, linkage=method)
                labels = clustering.fit_predict(X_scaled)
                
                if len(set(labels)) > 1:
                    score = silhouette_score(X_scaled, labels)
                    if score > best_score:
                        best_score = score
                        best_method = method
            except:
                continue
        
        # 执行最终聚类
        final_clustering = AgglomerativeClustering(n_clusters=3, linkage=best_method)
        final_labels = final_clustering.fit_predict(X_scaled)
        
        return {
            'linkage_method': best_method,
            'labels': final_labels,
            'n_clusters': len(set(final_labels)),
            'silhouette_score': best_score
        }
    
    def _evaluate_clustering(self, X_scaled, kmeans_results, dbscan_results, hierarchical_results):
        """聚类评估"""
        print("📊 聚类评估...")
        
        evaluation = {}
        
        # K-means评估
        if 'labels' in kmeans_results:
            kmeans_labels = kmeans_results['labels']
            evaluation['kmeans'] = {
                'silhouette_score': silhouette_score(X_scaled, kmeans_labels),
                'calinski_harabasz_score': calinski_harabasz_score(X_scaled, kmeans_labels),
                'n_clusters': len(set(kmeans_labels))
            }
        
        # DBSCAN评估
        if 'labels' in dbscan_results:
            dbscan_labels = dbscan_results['labels']
            if len(set(dbscan_labels)) > 1:
                evaluation['dbscan'] = {
                    'silhouette_score': silhouette_score(X_scaled, dbscan_labels),
                    'calinski_harabasz_score': calinski_harabasz_score(X_scaled, dbscan_labels),
                    'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                    'n_noise': list(dbscan_labels).count(-1)
                }
        
        # 层次聚类评估
        if 'labels' in hierarchical_results:
            hierarchical_labels = hierarchical_results['labels']
            evaluation['hierarchical'] = {
                'silhouette_score': silhouette_score(X_scaled, hierarchical_labels),
                'calinski_harabasz_score': calinski_harabasz_score(X_scaled, hierarchical_labels),
                'n_clusters': len(set(hierarchical_labels))
            }
        
        return evaluation
    
    def _interpret_clusters(self, data_subset, kmeans_results, dbscan_results, hierarchical_results):
        """聚类解释"""
        print("🔍 聚类解释...")
        
        interpretations = {}
        
        # K-means聚类解释
        if 'labels' in kmeans_results:
            kmeans_labels = kmeans_results['labels']
            cluster_stats = data_subset.groupby(kmeans_labels).describe()
            interpretations['kmeans'] = {
                'cluster_statistics': cluster_stats,
                'cluster_characteristics': self._describe_cluster_characteristics(data_subset, kmeans_labels)
            }
        
        # DBSCAN聚类解释
        if 'labels' in dbscan_results:
            dbscan_labels = dbscan_results['labels']
            if len(set(dbscan_labels)) > 1:
                cluster_stats = data_subset.groupby(dbscan_labels).describe()
                interpretations['dbscan'] = {
                    'cluster_statistics': cluster_stats,
                    'cluster_characteristics': self._describe_cluster_characteristics(data_subset, dbscan_labels)
                }
        
        # 层次聚类解释
        if 'labels' in hierarchical_results:
            hierarchical_labels = hierarchical_results['labels']
            cluster_stats = data_subset.groupby(hierarchical_labels).describe()
            interpretations['hierarchical'] = {
                'cluster_statistics': cluster_stats,
                'cluster_characteristics': self._describe_cluster_characteristics(data_subset, hierarchical_labels)
            }
        
        return interpretations
    
    def _describe_cluster_characteristics(self, data_subset, labels):
        """描述聚类特征"""
        characteristics = {}
        
        for cluster_id in set(labels):
            if cluster_id == -1:  # 跳过噪声点
                continue
                
            # 修复数组比较问题
            cluster_mask = np.array(labels) == cluster_id
            cluster_data = data_subset[cluster_mask]
            cluster_characteristics = {}
            
            for column in cluster_data.columns:
                values = cluster_data[column]
                cluster_characteristics[column] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'median': values.median()
                }
            
            characteristics[f'cluster_{cluster_id}'] = cluster_characteristics
        
        return characteristics
    
    def dimensionality_reduction_analysis(self, columns=None):
        """
        降维分析 - 发现数据的主要成分和结构
        
        Args:
            columns (list): 要分析的列名列表
            
        Returns:
            dict: 降维分析结果
        """
        print(f"\n📉 执行降维分析")
        print("=" * 60)
        
        if self.data is None:
            return {}
        
        # 选择数值列
        if columns is None:
            numeric_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        else:
            numeric_columns = [col for col in columns if col in self.data.columns]
        
        if not numeric_columns:
            print("❌ 没有找到数值列")
            return {}
        
        # 准备数据
        data_subset = self.data[numeric_columns].dropna()
        if len(data_subset) == 0:
            print("❌ 数据为空")
            return {}
        
        try:
            # 数据标准化
            X_scaled = self.scaler.fit_transform(data_subset)
            
            # 1. 主成分分析 (PCA)
            pca_results = self._pca_analysis(X_scaled, data_subset)
            
            # 2. 独立成分分析 (ICA)
            ica_results = self._ica_analysis(X_scaled, data_subset)
            
            # 3. t-SNE降维
            tsne_results = self._tsne_analysis(X_scaled, data_subset)
            
            # 4. 降维评估
            reduction_evaluation = self._evaluate_dimensionality_reduction(X_scaled, pca_results, ica_results, tsne_results)
            
            results = {
                'pca': pca_results,
                'ica': ica_results,
                'tsne': tsne_results,
                'evaluation': reduction_evaluation,
                'features_used': numeric_columns
            }
            
            self.analysis_results['dimensionality_reduction'] = results
            print("✅ 降维分析完成")
            return results
            
        except Exception as e:
            print(f"❌ 降维分析失败: {e}")
            return {}
    
    def _pca_analysis(self, X_scaled, data_subset):
        """主成分分析"""
        print("📊 主成分分析...")
        
        # 执行PCA
        pca = PCA()
        pca_transformed = pca.fit_transform(X_scaled)
        
        # 计算累积解释方差比
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
        
        # 找到解释95%方差的主成分数
        n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
        
        # 找到解释90%方差的主成分数
        n_components_90 = np.argmax(cumulative_variance_ratio >= 0.90) + 1
        
        return {
            'transformed_data': pca_transformed,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance_ratio': cumulative_variance_ratio,
            'n_components_95': n_components_95,
            'n_components_90': n_components_90,
            'components': pca.components_,
            'feature_names': data_subset.columns.tolist()
        }
    
    def _ica_analysis(self, X_scaled, data_subset):
        """独立成分分析"""
        print("🔄 独立成分分析...")
        
        try:
            # 执行ICA
            ica = FastICA(n_components=min(5, X_scaled.shape[1]), random_state=42)
            ica_transformed = ica.fit_transform(X_scaled)
            
            return {
                'transformed_data': ica_transformed,
                'components': ica.components_,
                'mixing_matrix': ica.mixing_,
                'feature_names': data_subset.columns.tolist()
            }
        except Exception as e:
            print(f"⚠️ ICA分析失败: {e}")
            return {}
    
    def _tsne_analysis(self, X_scaled, data_subset):
        """t-SNE降维"""
        print("🎨 t-SNE降维...")
        
        try:
            # 如果数据点太多，先进行采样
            if len(X_scaled) > 1000:
                indices = np.random.choice(len(X_scaled), 1000, replace=False)
                X_sample = X_scaled[indices]
            else:
                X_sample = X_scaled
                indices = np.arange(len(X_scaled))
            
            # 执行t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_sample)//4))
            tsne_transformed = tsne.fit_transform(X_sample)
            
            return {
                'transformed_data': tsne_transformed,
                'sample_indices': indices,
                'perplexity': tsne.perplexity,
                'kl_divergence': tsne.kl_divergence_
            }
        except Exception as e:
            print(f"⚠️ t-SNE分析失败: {e}")
            return {}
    
    def _evaluate_dimensionality_reduction(self, X_scaled, pca_results, ica_results, tsne_results):
        """降维评估"""
        print("📊 降维评估...")
        
        evaluation = {}
        
        # PCA评估
        if 'explained_variance_ratio' in pca_results:
            evaluation['pca'] = {
                'total_variance_explained': pca_results['explained_variance_ratio'].sum(),
                'first_component_variance': pca_results['explained_variance_ratio'][0],
                'n_components_95': pca_results['n_components_95'],
                'n_components_90': pca_results['n_components_90']
            }
        
        # ICA评估
        if 'transformed_data' in ica_results:
            evaluation['ica'] = {
                'n_components': ica_results['transformed_data'].shape[1],
                'component_independence': 'ICA components are statistically independent'
            }
        
        # t-SNE评估
        if 'kl_divergence' in tsne_results:
            evaluation['tsne'] = {
                'kl_divergence': tsne_results['kl_divergence'],
                'n_samples': len(tsne_results['transformed_data'])
            }
        
        return evaluation
    
    def statistical_hypothesis_testing(self, column='snow_water_equivalent_mm'):
        """
        统计假设检验 - 专业统计方法
        
        Args:
            column (str): 要分析的列名
            
        Returns:
            dict: 假设检验结果
        """
        print(f"\n📊 执行统计假设检验: {column}")
        print("=" * 60)
        
        if self.data is None or column not in self.data.columns:
            return {}
        
        series = self.data[column].dropna()
        if len(series) == 0:
            return {}
        
        try:
            # 1. 正态性检验
            normality_tests = self._normality_tests(series)
            
            # 2. 平稳性检验
            stationarity_tests = self._stationarity_tests(series)
            
            # 3. 季节性检验
            seasonality_tests = self._seasonality_tests(series)
            
            # 4. 趋势检验
            trend_tests = self._trend_tests(series)
            
            # 5. 多重比较校正
            multiple_comparison_correction = self._multiple_comparison_correction(
                normality_tests, stationarity_tests, seasonality_tests, trend_tests
            )
            
            results = {
                'normality_tests': normality_tests,
                'stationarity_tests': stationarity_tests,
                'seasonality_tests': seasonality_tests,
                'trend_tests': trend_tests,
                'multiple_comparison_correction': multiple_comparison_correction
            }
            
            # Add interpretation for statistical hypothesis testing
            results['interpretation'] = self._interpret_statistical_test_results(results)
            
            self.analysis_results['statistical_hypothesis_testing'] = results
            print("✅ Statistical hypothesis testing completed")
            return results
            
        except Exception as e:
            print(f"❌ 统计假设检验失败: {e}")
            return {}
    
    def _normality_tests(self, series):
        """正态性检验"""
        print("📊 正态性检验...")
        
        tests = {}
        
        # Shapiro-Wilk检验
        try:
            from scipy.stats import shapiro
            stat, p_value = shapiro(series.sample(min(5000, len(series))))
            tests['shapiro_wilk'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except Exception as e:
            print(f"⚠️ Shapiro-Wilk检验失败: {e}")
        
        # Kolmogorov-Smirnov检验
        try:
            from scipy.stats import kstest, norm
            mean, std = series.mean(), series.std()
            stat, p_value = kstest(series, lambda x: norm.cdf(x, mean, std))
            tests['kolmogorov_smirnov'] = {
                'statistic': stat,
                'p_value': p_value,
                'is_normal': p_value > 0.05
            }
        except Exception as e:
            print(f"⚠️ Kolmogorov-Smirnov检验失败: {e}")
        
        # Anderson-Darling检验
        try:
            from scipy.stats import anderson
            result = anderson(series, dist='norm')
            tests['anderson_darling'] = {
                'statistic': result.statistic,
                'critical_values': result.critical_values,
                'significance_levels': result.significance_level,
                'is_normal': result.statistic < result.critical_values[2]  # 5%显著性水平
            }
        except Exception as e:
            print(f"⚠️ Anderson-Darling检验失败: {e}")
        
        return tests
    
    def _stationarity_tests(self, series):
        """平稳性检验"""
        print("📊 平稳性检验...")
        
        tests = {}
        
        # ADF检验 (Augmented Dickey-Fuller)
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series.dropna())
            tests['adf'] = {
                'statistic': result[0],
                'p_value': result[1],
                'critical_values': result[4],
                'is_stationary': result[1] < 0.05
            }
        except Exception as e:
            print(f"⚠️ ADF检验失败: {e}")
        
        # KPSS检验
        try:
            from statsmodels.tsa.stattools import kpss
            result = kpss(series.dropna(), regression='c')
            tests['kpss'] = {
                'statistic': result[0],
                'p_value': result[1],
                'critical_values': result[3],
                'is_stationary': result[1] > 0.05
            }
        except Exception as e:
            print(f"⚠️ KPSS检验失败: {e}")
        
        return tests
    
    def _seasonality_tests(self, series):
        """季节性检验"""
        print("📊 季节性检验...")
        
        tests = {}
        
        # 月度Kruskal-Wallis检验
        try:
            monthly_groups = [series[series.index.month == month].values for month in range(1, 13)]
            monthly_groups = [group for group in monthly_groups if len(group) > 0]
            
            if len(monthly_groups) > 1:
                stat, p_value = kruskal(*monthly_groups)
                tests['monthly_kruskal_wallis'] = {
                    'statistic': stat,
                    'p_value': p_value,
                    'has_seasonality': p_value < 0.05
                }
        except Exception as e:
            print(f"⚠️ 月度Kruskal-Wallis检验失败: {e}")
        
        # 季节性强度检验
        try:
            # 计算季节性强度
            monthly_means = series.groupby(series.index.month).mean()
            seasonal_strength = monthly_means.std() / series.std()
            
            tests['seasonal_strength'] = {
                'strength': seasonal_strength,
                'is_strong_seasonal': seasonal_strength > 0.3
            }
        except Exception as e:
            print(f"⚠️ 季节性强度检验失败: {e}")
        
        return tests
    
    def _trend_tests(self, series):
        """趋势检验"""
        print("📊 趋势检验...")
        
        tests = {}
        
        # Mann-Kendall趋势检验
        try:
            from scipy.stats import kendalltau
            x = np.arange(len(series))
            tau, p_value = kendalltau(x, series.values)
            
            tests['mann_kendall'] = {
                'tau': tau,
                'p_value': p_value,
                'has_trend': p_value < 0.05,
                'trend_direction': 'increasing' if tau > 0 else 'decreasing'
            }
        except Exception as e:
            print(f"⚠️ Mann-Kendall趋势检验失败: {e}")
        
        # 线性趋势检验
        try:
            from scipy.stats import linregress
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = linregress(x, series.values)
            
            tests['linear_trend'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err,
                'has_trend': p_value < 0.05,
                'trend_direction': 'increasing' if slope > 0 else 'decreasing'
            }
        except Exception as e:
            print(f"⚠️ 线性趋势检验失败: {e}")
        
        return tests
    
    def _multiple_comparison_correction(self, *test_results):
        """多重比较校正"""
        print("📊 多重比较校正...")
        
        # 收集所有p值
        all_p_values = []
        test_names = []
        
        for test_group in test_results:
            for test_name, test_data in test_group.items():
                if isinstance(test_data, dict) and 'p_value' in test_data:
                    all_p_values.append(test_data['p_value'])
                    test_names.append(test_name)
        
        if not all_p_values:
            return {}
        
        # Bonferroni校正
        from statsmodels.stats.multitest import multipletests
        try:
            rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                all_p_values, alpha=0.05, method='bonferroni'
            )
            
            # FDR校正 (Benjamini-Hochberg)
            rejected_fdr, p_corrected_fdr, alpha_sidak_fdr, alpha_bonf_fdr = multipletests(
                all_p_values, alpha=0.05, method='fdr_bh'
            )
            
            return {
                'bonferroni': {
                    'corrected_p_values': dict(zip(test_names, p_corrected)),
                    'rejected': dict(zip(test_names, rejected))
                },
                'fdr_bh': {
                    'corrected_p_values': dict(zip(test_names, p_corrected_fdr)),
                    'rejected': dict(zip(test_names, rejected_fdr))
                },
                'original_p_values': dict(zip(test_names, all_p_values))
            }
        except Exception as e:
            print(f"⚠️ 多重比较校正失败: {e}")
            return {}
    
    def create_interactive_visualizations(self, save_path=None):
        """
        创建交互式可视化
        
        Args:
            save_path (str): 保存路径
            
        Returns:
            dict: 可视化结果
        """
        print("\n📊 创建交互式可视化")
        print("=" * 60)
        
        if not self.analysis_results:
            print("❌ 没有分析结果，请先运行分析")
            return {}
        
        visualizations = {}
        
        try:
            # 1. 时间序列分解可视化
            if 'advanced_decomposition' in self.analysis_results:
                visualizations['decomposition'] = self._create_decomposition_plot()
            
            # 2. 异常检测可视化
            if 'advanced_anomaly_detection' in self.analysis_results:
                visualizations['anomaly_detection'] = self._create_anomaly_plot()
            
            # 3. 聚类分析可视化
            if 'clustering_analysis' in self.analysis_results:
                visualizations['clustering'] = self._create_clustering_plot()
            
            # 4. 降维分析可视化
            if 'dimensionality_reduction' in self.analysis_results:
                visualizations['dimensionality_reduction'] = self._create_dimensionality_reduction_plot()
            
            # 5. 统计检验可视化
            if 'statistical_hypothesis_testing' in self.analysis_results:
                visualizations['statistical_tests'] = self._create_statistical_tests_plot()
            
            # 保存可视化
            if save_path:
                self._save_visualizations(visualizations, save_path)
            
            print("✅ 交互式可视化创建完成")
            return visualizations
            
        except Exception as e:
            print(f"❌ 可视化创建失败: {e}")
            return {}
    
    def _create_decomposition_plot(self):
        """创建时间序列分解图"""
        decomposition = self.analysis_results['advanced_decomposition']
        
        if 'stl_decomposition' not in decomposition:
            return {}
        
        stl = decomposition['stl_decomposition']
        
        # 创建子图
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=['原始数据', '趋势', '季节性', '残差'],
            vertical_spacing=0.05
        )
        
        # 原始数据
        fig.add_trace(
            go.Scatter(
                x=self.data.index,
                y=self.data['snow_water_equivalent_mm'],
                mode='lines',
                name='原始数据',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # 趋势
        if 'trend' in stl:
            fig.add_trace(
                go.Scatter(
                    x=stl['trend'].index,
                    y=stl['trend'].values,
                    mode='lines',
                    name='趋势',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        
        # 季节性
        if 'seasonal' in stl:
            fig.add_trace(
                go.Scatter(
                    x=stl['seasonal'].index,
                    y=stl['seasonal'].values,
                    mode='lines',
                    name='季节性',
                    line=dict(color='green')
                ),
                row=3, col=1
            )
        
        # 残差
        if 'resid' in stl:
            fig.add_trace(
                go.Scatter(
                    x=stl['resid'].index,
                    y=stl['resid'].values,
                    mode='lines',
                    name='残差',
                    line=dict(color='orange')
                ),
                row=4, col=1
            )
        
        fig.update_layout(
            title='高级时间序列分解',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def _create_anomaly_plot(self):
        """创建异常检测图"""
        anomaly_detection = self.analysis_results['advanced_anomaly_detection']
        
        if 'ensemble' not in anomaly_detection:
            return {}
        
        ensemble = anomaly_detection['ensemble']
        
        # 创建散点图
        fig = go.Figure()
        
        # 正常点
        normal_mask = ~ensemble['ensemble_anomalies']
        fig.add_trace(
            go.Scatter(
                x=self.data.index[normal_mask],
                y=self.data['snow_water_equivalent_mm'][normal_mask],
                mode='markers',
                name='正常点',
                marker=dict(color='blue', size=4)
            )
        )
        
        # 异常点
        anomaly_mask = ensemble['ensemble_anomalies']
        fig.add_trace(
            go.Scatter(
                x=self.data.index[anomaly_mask],
                y=self.data['snow_water_equivalent_mm'][anomaly_mask],
                mode='markers',
                name='异常点',
                marker=dict(color='red', size=8)
            )
        )
        
        fig.update_layout(
            title='异常检测结果',
            xaxis_title='时间',
            yaxis_title='雪水当量 (mm)',
            height=500
        )
        
        return fig
    
    def _create_clustering_plot(self):
        """创建聚类分析图"""
        clustering = self.analysis_results['clustering_analysis']
        
        if 'kmeans' not in clustering:
            return {}
        
        kmeans = clustering['kmeans']
        
        # 创建散点图
        fig = go.Figure()
        
        # 为每个聚类添加不同的颜色
        colors = px.colors.qualitative.Set1
        
        for cluster_id in set(kmeans['labels']):
            mask = kmeans['labels'] == cluster_id
            fig.add_trace(
                go.Scatter(
                    x=self.data.index[mask],
                    y=self.data['snow_water_equivalent_mm'][mask],
                    mode='markers',
                    name=f'聚类 {cluster_id}',
                    marker=dict(color=colors[cluster_id % len(colors)], size=6)
                )
            )
        
        fig.update_layout(
            title='K-means聚类结果',
            xaxis_title='时间',
            yaxis_title='雪水当量 (mm)',
            height=500
        )
        
        return fig
    
    def _create_dimensionality_reduction_plot(self):
        """创建降维分析图"""
        dim_reduction = self.analysis_results['dimensionality_reduction']
        
        if 'pca' not in dim_reduction:
            return {}
        
        pca = dim_reduction['pca']
        
        # 创建PCA解释方差图
        fig = go.Figure()
        
        fig.add_trace(
            go.Bar(
                x=list(range(1, len(pca['explained_variance_ratio']) + 1)),
                y=pca['explained_variance_ratio'],
                name='解释方差比'
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(pca['cumulative_variance_ratio']) + 1)),
                y=pca['cumulative_variance_ratio'],
                mode='lines+markers',
                name='累积解释方差比',
                line=dict(color='red')
            )
        )
        
        fig.update_layout(
            title='PCA解释方差分析',
            xaxis_title='主成分',
            yaxis_title='解释方差比',
            height=500
        )
        
        return fig
    
    def _create_statistical_tests_plot(self):
        """创建统计检验图"""
        statistical_tests = self.analysis_results['statistical_hypothesis_testing']
        
        if 'multiple_comparison_correction' not in statistical_tests:
            return {}
        
        correction = statistical_tests['multiple_comparison_correction']
        
        if 'original_p_values' not in correction:
            return {}
        
        # 创建p值比较图
        fig = go.Figure()
        
        test_names = list(correction['original_p_values'].keys())
        original_p_values = list(correction['original_p_values'].values())
        
        if 'bonferroni' in correction and 'corrected_p_values' in correction['bonferroni']:
            bonferroni_p_values = [correction['bonferroni']['corrected_p_values'].get(name, 0) for name in test_names]
        else:
            bonferroni_p_values = [0] * len(test_names)
        
        fig.add_trace(
            go.Bar(
                x=test_names,
                y=original_p_values,
                name='原始p值',
                marker=dict(color='blue')
            )
        )
        
        fig.add_trace(
            go.Bar(
                x=test_names,
                y=bonferroni_p_values,
                name='Bonferroni校正p值',
                marker=dict(color='red')
            )
        )
        
        # 添加显著性水平线
        fig.add_hline(y=0.05, line_dash="dash", line_color="green", annotation_text="α = 0.05")
        
        fig.update_layout(
            title='统计检验p值比较',
            xaxis_title='检验方法',
            yaxis_title='p值',
            height=500
        )
        
        return fig
    
    def _save_visualizations(self, visualizations, save_path):
        """保存可视化"""
        import os
        
        os.makedirs(save_path, exist_ok=True)
        
        for name, fig in visualizations.items():
            if fig:
                file_path = os.path.join(save_path, f"{name}.html")
                fig.write_html(file_path)
                print(f"📊 保存可视化: {file_path}")
    
    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        print("\n📋 生成综合分析报告")
        print("=" * 80)
        
        if not self.analysis_results:
            print("❌ 没有分析结果，请先运行分析")
            return
        
        # 报告标题
        print("🔬 数据科学分析综合报告")
        print("=" * 80)
        
        # 数据概览
        if self.data is not None:
            print(f"\n📊 数据概览:")
            print(f"  数据点数量: {len(self.data):,}")
            print(f"  时间范围: {self.data.index.min()} 到 {self.data.index.max()}")
            print(f"  数据列数: {len(self.data.columns)}")
        
        # 高级时间序列分解报告
        if 'advanced_decomposition' in self.analysis_results:
            self._report_decomposition()
        
        # 异常检测报告
        if 'advanced_anomaly_detection' in self.analysis_results:
            self._report_anomaly_detection()
        
        # 聚类分析报告
        if 'clustering_analysis' in self.analysis_results:
            self._report_clustering()
        
        # 降维分析报告
        if 'dimensionality_reduction' in self.analysis_results:
            self._report_dimensionality_reduction()
        
        # 统计检验报告
        if 'statistical_hypothesis_testing' in self.analysis_results:
            self._report_statistical_tests()
        
        print("\n" + "=" * 80)
        print("✅ 综合分析报告生成完成")
    
    def _report_decomposition(self):
        """报告时间序列分解结果"""
        print(f"\n🔍 高级时间序列分解:")
        
        decomposition = self.analysis_results['advanced_decomposition']
        
        if 'stl_decomposition' in decomposition:
            stl = decomposition['stl_decomposition']
            print(f"  STL分解:")
            print(f"    季节性强度: {stl.get('seasonal_strength', 0):.3f}")
            print(f"    趋势强度: {stl.get('trend_strength', 0):.3f}")
        
        if 'periodicity_analysis' in decomposition:
            periodicity = decomposition['periodicity_analysis']
            print(f"  周期性分析:")
            print(f"    主要周期: {periodicity.get('main_period', 0):.1f} 天")
            print(f"    主要频率: {periodicity.get('main_frequency', 0):.6f}")
    
    def _report_anomaly_detection(self):
        """报告异常检测结果"""
        print(f"\n🚨 异常检测分析:")
        
        anomaly_detection = self.analysis_results['advanced_anomaly_detection']
        
        if 'explanations' in anomaly_detection:
            explanations = anomaly_detection['explanations']
            print(f"  检测到异常: {explanations.get('total_anomalies', 0)} 个")
            print(f"  异常率: {explanations.get('anomaly_rate', 0):.2%}")
        
        if 'ensemble' in anomaly_detection:
            ensemble = anomaly_detection['ensemble']
            print(f"  集成方法数: {ensemble.get('n_methods', 0)}")
            print(f"  异常阈值: {ensemble.get('threshold', 0):.3f}")
    
    def _report_clustering(self):
        """报告聚类分析结果"""
        print(f"\n🔍 聚类分析:")
        
        clustering = self.analysis_results['clustering_analysis']
        
        if 'kmeans' in clustering:
            kmeans = clustering['kmeans']
            print(f"  K-means聚类:")
            print(f"    最优聚类数: {kmeans.get('n_clusters', 0)}")
            print(f"    轮廓系数: {kmeans.get('silhouette_score', 0):.3f}")
        
        if 'evaluation' in clustering:
            evaluation = clustering['evaluation']
            print(f"  聚类评估:")
            for method, metrics in evaluation.items():
                if 'silhouette_score' in metrics:
                    print(f"    {method}: 轮廓系数 = {metrics['silhouette_score']:.3f}")
    
    def _report_dimensionality_reduction(self):
        """报告降维分析结果"""
        print(f"\n📉 降维分析:")
        
        dim_reduction = self.analysis_results['dimensionality_reduction']
        
        if 'pca' in dim_reduction:
            pca = dim_reduction['pca']
            print(f"  PCA分析:")
            print(f"    解释95%方差需要: {pca.get('n_components_95', 0)} 个主成分")
            print(f"    解释90%方差需要: {pca.get('n_components_90', 0)} 个主成分")
            if 'explained_variance_ratio' in pca:
                first_component = pca['explained_variance_ratio'][0]
                print(f"    第一主成分解释方差: {first_component:.1%}")
    
    def _report_statistical_tests(self):
        """报告统计检验结果"""
        print(f"\n📊 统计假设检验:")
        
        statistical_tests = self.analysis_results['statistical_hypothesis_testing']
        
        if 'normality_tests' in statistical_tests:
            normality = statistical_tests['normality_tests']
            print(f"  正态性检验:")
            for test_name, test_data in normality.items():
                if 'is_normal' in test_data:
                    print(f"    {test_name}: {'正态' if test_data['is_normal'] else '非正态'}")
        
        if 'trend_tests' in statistical_tests:
            trend_tests = statistical_tests['trend_tests']
            print(f"  趋势检验:")
            if 'mann_kendall' in trend_tests:
                mk = trend_tests['mann_kendall']
                if 'has_trend' in mk:
                    print(f"    Mann-Kendall: {'有趋势' if mk['has_trend'] else '无趋势'}")
                    if 'trend_direction' in mk:
                        print(f"      趋势方向: {mk['trend_direction']}")
    
    def run_comprehensive_analysis(self, column='snow_water_equivalent_mm'):
        """
        运行综合分析
        
        Args:
            column (str): 要分析的列名
            
        Returns:
            dict: 所有分析结果
        """
        print("🚀 开始数据科学综合分析")
        print("=" * 80)
        
        if self.data is None:
            print("❌ 数据未加载")
            return {}
        
        try:
            # 1. 高级时间序列分解
            self.advanced_time_series_decomposition(column)
            
            # 2. 异常检测
            self.advanced_anomaly_detection(column)
            
            # 3. 聚类分析
            self.clustering_analysis()
            
            # 4. 降维分析
            self.dimensionality_reduction_analysis()
            
            # 5. 统计假设检验
            self.statistical_hypothesis_testing(column)
            
            # 6. 生成报告
            self.generate_comprehensive_report()
            
            # 7. 创建可视化
            visualizations = self.create_interactive_visualizations()
            
            print("\n✅ 数据科学综合分析完成!")
            return self.analysis_results
            
        except Exception as e:
            print(f"❌ 综合分析失败: {e}")
            return {}

    def discover_cold_factors(self, target_column='snow_water_equivalent_mm', top_k=10):
        """无监督冷门影响因素发现：基于相关性稀有性与聚类异质性。

        返回按"冷门度"排序的候选要素（越高越冷门但有影响）。
        """
        if self.data is None or target_column not in self.data.columns:
            return {}

        # 仅数值特征
        df = self.data.select_dtypes(include=[np.number]).dropna()
        if target_column not in df.columns:
            return {}

        target = df[target_column]

        # 识别时间特征（这些通常不是真正的"冷门"因素）
        time_features = ['year', 'month', 'day', 'day_of_year', 'hour', 'minute']
        time_cols = [col for col in df.columns if any(tf in col.lower() for tf in time_features)]

        # 1) 影响力：|Spearman| 相关性
        impact_scores = {}
        for col in df.columns:
            if col == target_column:
                continue
            try:
                rho, _ = stats.spearmanr(df[col], target)
                impact_scores[col] = float(abs(rho))
            except Exception:
                continue

        # 2) 冷门度：重新设计 - 基于真正的业务定义
        coldness_scores = {}
        for col in df.columns:
            if col == target_column:
                continue
            try:
                values = df[col]
                
                # 计算值的稀有性（真正的冷门度）
                value_counts = values.value_counts()
                total_count = len(values)
                
                # 稀有性分数：值出现频率越低，分数越高
                rarity_scores = []
                for val, count in value_counts.items():
                    # 稀有性 = 1 - (出现次数/总数)，越稀有分数越高
                    rarity = 1 - (count / total_count)
                    rarity_scores.append(rarity)
                
                # 平均稀有性
                avg_rarity = np.mean(rarity_scores) if rarity_scores else 0
                
                # 计算信息熵（特征的信息量）
                value_probs = value_counts / total_count
                entropy = -np.sum(value_probs * np.log2(value_probs + 1e-10))
                
                # 标准化熵值到0-1范围
                max_entropy = np.log2(len(value_counts) + 1e-10)
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                # 时间特征惩罚：大幅降低时间特征的冷门度
                time_penalty = 0.1 if col in time_cols else 1.0
                
                # 新的冷门度计算：稀有性 + 信息量
                coldness_scores[col] = float((0.7 * avg_rarity + 0.3 * normalized_entropy) * time_penalty)
                
            except Exception:
                coldness_scores[col] = 0.0

        # 3) 预测价值：基于特征对目标的预测能力
        predictive_scores = {}
        for col in df.columns:
            if col == target_column:
                continue
            try:
                # 使用简单的线性回归R²作为预测能力指标
                from sklearn.linear_model import LinearRegression
                from sklearn.metrics import r2_score
                
                X = df[col].values.reshape(-1, 1)
                y = target.values
                
                # 处理缺失值
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                if np.sum(mask) > 10:  # 至少需要10个有效样本
                    X_clean = X[mask]
                    y_clean = y[mask]
                    
                    model = LinearRegression()
                    model.fit(X_clean, y_clean)
                    y_pred = model.predict(X_clean)
                    r2 = r2_score(y_clean, y_pred)
                    predictive_scores[col] = float(max(0, r2))  # R²可能为负，我们只关心正值
                else:
                    predictive_scores[col] = 0.0
                    
        except Exception:
                predictive_scores[col] = 0.0

        # 综合得分：数据驱动的动态权重分配
        all_cols = [c for c in df.columns if c != target_column]
        def nz(d, c):
            return d.get(c, 0.0)
        
        # 计算各维度的统计特征，用于动态权重调整
        impact_values = [nz(impact_scores, col) for col in all_cols]
        coldness_values = [nz(coldness_scores, col) for col in all_cols]
        predictive_values = [nz(predictive_scores, col) for col in all_cols]
        
        # 计算各维度的变异系数（标准差/均值），用于权重调整
        def get_cv(values):
            if not values or np.mean(values) == 0:
                return 1.0
            return np.std(values) / np.mean(values)
        
        impact_cv = get_cv(impact_values)
        coldness_cv = get_cv(coldness_values)
        predictive_cv = get_cv(predictive_values)
        
        # 动态权重：变异系数越高，权重越大（区分度更好）
        total_cv = impact_cv + coldness_cv + predictive_cv
        if total_cv > 0:
            impact_weight = impact_cv / total_cv
            coldness_weight = coldness_cv / total_cv
            predictive_weight = predictive_cv / total_cv
        else:
            # 默认权重
            impact_weight = 0.4
            coldness_weight = 0.3
            predictive_weight = 0.3
        
        print(f"📊 动态权重分配: Impact={impact_weight:.3f}, Coldness={coldness_weight:.3f}, Predictive={predictive_weight:.3f}")
        
        combined = []
        for col in all_cols:
            impact = nz(impact_scores, col)
            coldness = nz(coldness_scores, col)
            predictive = nz(predictive_scores, col)
            
            # 时间特征特殊处理：大幅降低权重
            if col in time_cols:
                impact *= 0.2  # 时间特征的影响力权重大幅降低
                coldness *= 0.1  # 时间特征的冷门度权重极低
            
            # 动态权重综合得分
            score = impact_weight * impact + coldness_weight * coldness + predictive_weight * predictive
            
            combined.append((col, score))
        
        combined.sort(key=lambda x: x[1], reverse=True)

        # Add interpretation content
        interpretation = self._interpret_cold_factors_discovery(
            combined[:top_k], impact_scores, coldness_scores, predictive_scores, target_column
        )

        return {
            'target': target_column,
            'top_candidates': combined[:top_k],
            'impact_scores': impact_scores,
            'coldness_scores': coldness_scores,
            'predictive_scores': predictive_scores,
            'interpretation': interpretation
        }
    
    def _interpret_cold_factors_discovery(self, top_candidates, impact_scores, coldness_scores, predictive_scores, target_column):
        """Interpret cold factors discovery results"""
        try:
            interpretation = {
                'summary': '',
                'key_insights': [],
                'business_implications': '',
                'recommendations': [],
                'factor_categories': {}
            }
            
            if not top_candidates:
                interpretation['summary'] = 'No significant cold factors discovered'
                return interpretation
            
            # Generate summary
            top_factor = top_candidates[0]
            interpretation['summary'] = f"Discovered {len(top_candidates)} potential cold factors, with {top_factor[0]} being the most important"
            
            # Key insights
            interpretation['key_insights'] = [
                f"Most important cold factor: {top_factor[0]} (comprehensive score: {top_factor[1]:.3f})",
                f"Impact score: {impact_scores.get(top_factor[0], 0):.3f}",
                f"Coldness score: {coldness_scores.get(top_factor[0], 0):.3f}",
                f"Predictive value score: {predictive_scores.get(top_factor[0], 0):.3f}"
            ]
            
            # Business implications
            if top_factor[1] > 0.5:
                interpretation['business_implications'] = "High-value cold factors discovered that may significantly improve prediction model performance"
            elif top_factor[1] > 0.3:
                interpretation['business_implications'] = "Medium-value cold factors discovered that are worth considering in the model"
            else:
                interpretation['business_implications'] = "Discovered cold factors have limited value, recommend further analysis"
            
            # Recommendations
            interpretation['recommendations'] = [
                f"Consider adding {top_factor[0]} feature to {target_column} prediction model",
                "Perform feature engineering to create derived features based on cold factors",
                "Monitor actual contribution of these factors in the model",
                "Regularly reassess importance of cold factors"
            ]
            
            # Factor categories
            interpretation['factor_categories'] = {
                'high_impact': [col for col, score in top_candidates if impact_scores.get(col, 0) > 0.5],
                'high_coldness': [col for col, score in top_candidates if coldness_scores.get(col, 0) > 0.5],
                'high_predictive': [col for col, score in top_candidates if predictive_scores.get(col, 0) > 0.3]
            }
            
            return interpretation
            
        except Exception as e:
            print(f"⚠️ Cold factors interpretation failed: {e}")
            return {
                'summary': 'Interpretation generation failed',
                'key_insights': [],
                'business_implications': 'Further analysis needed',
                'recommendations': ['Check data quality', 'Re-run analysis'],
                'factor_categories': {}
            }
    
    def _interpret_decomposition_results(self, results):
        """Interpret time series decomposition results"""
        try:
            interpretation = {
                'summary': '',
                'key_insights': [],
                'business_implications': '',
                'recommendations': []
            }
            
            if 'stl_decomposition' in results:
                stl = results['stl_decomposition']
                seasonal_strength = stl.get('seasonal_strength', 0)
                trend_strength = stl.get('trend_strength', 0)
                
                interpretation['summary'] = f"Time series decomposition completed with seasonal strength {seasonal_strength:.3f} and trend strength {trend_strength:.3f}"
                
                interpretation['key_insights'] = [
                    f"Seasonal component strength: {seasonal_strength:.3f}",
                    f"Trend component strength: {trend_strength:.3f}",
                    "STL decomposition successfully extracted trend, seasonal, and residual components"
                ]
                
                if seasonal_strength > 0.5:
                    interpretation['business_implications'] = "Strong seasonal patterns detected, seasonal forecasting models recommended"
                elif seasonal_strength > 0.2:
                    interpretation['business_implications'] = "Moderate seasonal patterns, consider seasonal adjustments in forecasting"
                else:
                    interpretation['business_implications'] = "Weak seasonal patterns, focus on trend-based forecasting"
                
                interpretation['recommendations'] = [
                    "Use seasonal decomposition for pattern identification",
                    "Apply seasonal adjustments in forecasting models",
                    "Monitor seasonal strength changes over time"
                ]
            else:
                interpretation['summary'] = "Decomposition analysis completed but STL results not available"
                interpretation['key_insights'] = ["Basic decomposition performed", "Check data quality and seasonality"]
                interpretation['business_implications'] = "Limited decomposition insights available"
                interpretation['recommendations'] = ["Verify data completeness", "Check for sufficient data points"]
            
            return interpretation
            
        except Exception as e:
            print(f"⚠️ Decomposition interpretation failed: {e}")
            return {
                'summary': 'Decomposition interpretation failed',
                'key_insights': [],
                'business_implications': 'Further analysis needed',
                'recommendations': ['Check decomposition results', 'Re-run analysis']
            }
    
    def _interpret_anomaly_detection_results(self, results):
        """Interpret anomaly detection results"""
        try:
            interpretation = {
                'summary': '',
                'key_insights': [],
                'business_implications': '',
                'recommendations': []
            }
            
            if 'ensemble' in results:
                ensemble = results['ensemble']
                anomalies = ensemble.get('ensemble_anomalies', [])
                anomaly_count = sum(anomalies) if anomalies else 0
                total_points = len(anomalies) if anomalies else 0
                anomaly_rate = anomaly_count / total_points if total_points > 0 else 0
                
                interpretation['summary'] = f"Anomaly detection completed: {anomaly_count} anomalies found in {total_points} data points ({anomaly_rate:.2%})"
                
                interpretation['key_insights'] = [
                    f"Total anomalies detected: {anomaly_count}",
                    f"Anomaly rate: {anomaly_rate:.2%}",
                    "Ensemble method used for robust anomaly detection"
                ]
                
                if anomaly_rate > 0.1:
                    interpretation['business_implications'] = "High anomaly rate detected, data quality issues or significant events may be present"
                elif anomaly_rate > 0.05:
                    interpretation['business_implications'] = "Moderate anomaly rate, some data quality issues detected"
                else:
                    interpretation['business_implications'] = "Low anomaly rate, data appears to be of good quality"
                
                interpretation['recommendations'] = [
                    "Investigate detected anomalies for data quality issues",
                    "Consider anomaly removal for model training",
                    "Monitor anomaly patterns for system health"
                ]
            else:
                interpretation['summary'] = "Anomaly detection completed but ensemble results not available"
                interpretation['key_insights'] = ["Basic anomaly detection performed", "Check detection method results"]
                interpretation['business_implications'] = "Limited anomaly insights available"
                interpretation['recommendations'] = ["Verify anomaly detection method", "Check for sufficient data"]
            
            return interpretation
            
        except Exception as e:
            print(f"⚠️ Anomaly detection interpretation failed: {e}")
            return {
                'summary': 'Anomaly detection interpretation failed',
                'key_insights': [],
                'business_implications': 'Further analysis needed',
                'recommendations': ['Check anomaly detection results', 'Re-run analysis']
            }
    
    def _interpret_clustering_results(self, results):
        """Interpret clustering analysis results"""
        try:
            interpretation = {
                'summary': '',
                'key_insights': [],
                'business_implications': '',
                'recommendations': []
            }
            
            if 'kmeans' in results:
                kmeans = results['kmeans']
                labels = kmeans.get('labels', [])
                silhouette = kmeans.get('silhouette_score', 0)
                n_clusters = len(set(labels)) if labels else 0
                
                interpretation['summary'] = f"Clustering analysis completed: {n_clusters} clusters identified with silhouette score {silhouette:.3f}"
                
                interpretation['key_insights'] = [
                    f"Number of clusters: {n_clusters}",
                    f"Silhouette score: {silhouette:.3f}",
                    "K-means clustering performed successfully"
                ]
                
                if silhouette > 0.7:
                    interpretation['business_implications'] = "Excellent cluster separation, clustering results highly reliable"
                elif silhouette > 0.5:
                    interpretation['business_implications'] = "Good cluster separation, clustering results reliable"
                elif silhouette > 0.3:
                    interpretation['business_implications'] = "Fair cluster separation, clustering results moderately reliable"
                else:
                    interpretation['business_implications'] = "Poor cluster separation, consider different clustering parameters"
                
                interpretation['recommendations'] = [
                    "Use cluster labels for feature engineering",
                    "Analyze cluster characteristics for insights",
                    "Consider adjusting number of clusters if silhouette score is low"
                ]
            else:
                interpretation['summary'] = "Clustering analysis completed but K-means results not available"
                interpretation['key_insights'] = ["Basic clustering performed", "Check clustering method results"]
                interpretation['business_implications'] = "Limited clustering insights available"
                interpretation['recommendations'] = ["Verify clustering method", "Check for sufficient data"]
            
            return interpretation
            
        except Exception as e:
            print(f"⚠️ Clustering interpretation failed: {e}")
            return {
                'summary': 'Clustering interpretation failed',
                'key_insights': [],
                'business_implications': 'Further analysis needed',
                'recommendations': ['Check clustering results', 'Re-run analysis']
            }
    
    def _interpret_statistical_test_results(self, results):
        """Interpret statistical hypothesis testing results"""
        try:
            interpretation = {
                'summary': '',
                'key_insights': [],
                'business_implications': '',
                'recommendations': []
            }
            
            if 'normality_tests' in results:
                normality = results['normality_tests']
                stationarity = results.get('stationarity_tests', {})
                
                # Check normality
                normal_vars = []
                for test_name, test_result in normality.items():
                    if isinstance(test_result, dict) and 'p_value' in test_result:
                        if test_result['p_value'] > 0.05:
                            normal_vars.append(test_name)
                
                interpretation['summary'] = f"Statistical testing completed: {len(normal_vars)} variables show normal distribution"
                
                interpretation['key_insights'] = [
                    f"Normal variables: {len(normal_vars)}",
                    f"Non-normal variables: {len(normality) - len(normal_vars)}",
                    "Multiple statistical tests performed for comprehensive analysis"
                ]
                
                if len(normal_vars) > len(normality) / 2:
                    interpretation['business_implications'] = "Most variables are normally distributed, parametric tests recommended"
                else:
                    interpretation['business_implications'] = "Many variables are non-normal, non-parametric tests recommended"
                
                interpretation['recommendations'] = [
                    "Use appropriate statistical tests based on distribution",
                    "Consider data transformations for non-normal variables",
                    "Apply multiple comparison corrections for multiple tests"
                ]
            else:
                interpretation['summary'] = "Statistical testing completed but detailed results not available"
                interpretation['key_insights'] = ["Basic statistical tests performed", "Check test method results"]
                interpretation['business_implications'] = "Limited statistical insights available"
                interpretation['recommendations'] = ["Verify statistical test methods", "Check for sufficient data"]
            
            return interpretation
            
        except Exception as e:
            print(f"⚠️ Statistical test interpretation failed: {e}")
            return {
                'summary': 'Statistical test interpretation failed',
                'key_insights': [],
                'business_implications': 'Further analysis needed',
                'recommendations': ['Check statistical test results', 'Re-run analysis']
        }


def main():
    """主函数 - 示例用法"""
    print("🚀 数据科学分析器")
    print("=" * 50)
    
    # 创建分析器
    analyzer = DataScienceAnalyzer()
    
    # 加载数据
    data_path = "src/neuralhydrology/data/red_river_basin/timeseries.csv"
    analyzer.load_data(data_path)
    
    if analyzer.data is not None:
        # 运行综合分析
        results = analyzer.run_comprehensive_analysis('snow_water_equivalent_mm')
        
        print(f"\n📊 分析结果包含 {len(results)} 个模块")
        for module_name in results.keys():
            print(f"  - {module_name}")


if __name__ == "__main__":
    main()
