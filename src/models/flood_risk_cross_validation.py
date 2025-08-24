#!/usr/bin/env python3
"""
洪水预警系统历史数据交叉验证测试
使用时间序列交叉验证评估模型性能
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 导入洪水风险评估模型
from .flood_risk_assessment import FloodRiskAssessment

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FloodRiskCrossValidator:
    """洪水风险交叉验证器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化交叉验证器"""
        self.risk_assessor = FloodRiskAssessment(config_path)
        self.validation_results = []
        self.performance_metrics = {}
        
        logger.info("洪水风险交叉验证器初始化完成")
    
    def load_historical_data(self, data_path: str) -> pd.DataFrame:
        """加载历史数据"""
        logger.info(f"加载历史数据: {data_path}")
        
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError("不支持的数据格式，请使用CSV或JSON")
            
            # 数据预处理
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            # 检查必需列
            required_columns = ['date', 'station_id', 'flow_value']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"缺少必需列: {missing_columns}")
            
            logger.info(f"历史数据加载成功: {len(df)} 条记录")
            logger.info(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
            logger.info(f"站点数量: {df['station_id'].nunique()}")
            
            return df
            
        except Exception as e:
            logger.error(f"加载历史数据失败: {e}")
            raise
    
    def generate_forecast_scenarios(self, 
                                  historical_flows: List[float], 
                                  forecast_horizon: int = 7) -> List[List[float]]:
        """生成预测场景"""
        scenarios = []
        
        for i in range(len(historical_flows) - forecast_horizon):
            # 使用历史数据作为预测
            scenario = historical_flows[i:i + forecast_horizon]
            scenarios.append(scenario)
        
        return scenarios
    
    def time_series_cross_validation(self, 
                                   data: pd.DataFrame,
                                   station_id: str,
                                   validation_windows: int = 5,
                                   forecast_horizon: int = 7,
                                   min_training_size: int = 30) -> Dict:
        """时间序列交叉验证"""
        logger.info(f"开始站点 {station_id} 的时间序列交叉验证...")
        
        # 筛选站点数据
        station_data = data[data['station_id'] == station_id].copy()
        if len(station_data) < min_training_size + forecast_horizon:
            logger.warning(f"站点 {station_id} 数据不足，跳过验证")
            return {"error": "数据不足"}
        
        # 按时间排序
        station_data = station_data.sort_values('date').reset_index(drop=True)
        flows = station_data['flow_value'].values
        dates = station_data['date'].values
        
        # 计算验证窗口大小
        total_size = len(flows)
        window_size = (total_size - min_training_size) // validation_windows
        
        validation_results = []
        
        for window in range(validation_windows):
            # 计算训练和验证的起始位置
            train_start = window * window_size
            train_end = min_training_size + window * window_size
            val_start = train_end
            val_end = min(val_start + forecast_horizon, total_size)
            
            if val_end <= val_start:
                break
            
            # 训练数据
            train_flows = flows[train_start:train_end]
            train_dates = dates[train_start:train_end]
            
            # 验证数据
            val_flows = flows[val_start:val_end]
            val_dates = dates[val_start:val_end]
            
            # 生成预测场景
            forecast_scenarios = self.generate_forecast_scenarios(
                train_flows, forecast_horizon
            )
            
            # 执行风险评估
            try:
                # 使用最后一个训练数据作为当前流量
                current_flow = train_flows[-1]
                
                # 选择最佳预测场景（基于历史模式）
                best_scenario = self._select_best_scenario(train_flows, forecast_scenarios)
                
                # 执行风险评估
                assessment = self.risk_assessor.assess_risk(
                    station_id=station_id,
                    current_flow=current_flow,
                    forecast_flows=best_scenario,
                    forecast_hours=list(range(6, 6 + len(best_scenario) * 6, 6))
                )
                
                # 计算预测误差
                if len(best_scenario) == len(val_flows):
                    mse = mean_squared_error(val_flows, best_scenario)
                    mae = mean_absolute_error(val_flows, best_scenario)
                    r2 = r2_score(val_flows, best_scenario)
                else:
                    mse = mae = r2 = np.nan
                
                # 记录验证结果
                window_result = {
                    "window": window,
                    "train_start": train_dates[0],
                    "train_end": train_dates[-1],
                    "val_start": val_dates[0],
                    "val_end": val_dates[-1],
                    "current_flow": current_flow,
                    "predicted_flows": best_scenario,
                    "actual_flows": val_flows.tolist(),
                    "risk_assessment": assessment,
                    "prediction_metrics": {
                        "mse": mse,
                        "mae": mae,
                        "r2": r2
                    }
                }
                
                validation_results.append(window_result)
                
                logger.info(f"  窗口 {window}: 训练 {len(train_flows)} 天, 验证 {len(val_flows)} 天")
                
            except Exception as e:
                logger.error(f"窗口 {window} 验证失败: {e}")
                continue
        
        # 计算整体性能指标
        overall_metrics = self._calculate_overall_metrics(validation_results)
        
        return {
            "station_id": station_id,
            "validation_windows": len(validation_results),
            "forecast_horizon": forecast_horizon,
            "results": validation_results,
            "overall_metrics": overall_metrics
        }
    
    def _select_best_scenario(self, 
                             historical_flows: np.ndarray, 
                             scenarios: List[List[float]]) -> List[float]:
        """选择最佳预测场景"""
        if not scenarios:
            return []
        
        # 计算历史流量变化模式
        historical_changes = np.diff(historical_flows)
        historical_pattern = np.mean(historical_changes)
        historical_volatility = np.std(historical_changes)
        
        # 计算每个场景的相似度分数
        scenario_scores = []
        
        for scenario in scenarios:
            if len(scenario) < 2:
                scenario_scores.append(0)
                continue
            
            # 计算场景变化模式
            scenario_changes = np.diff(scenario)
            scenario_pattern = np.mean(scenario_changes)
            scenario_volatility = np.std(scenario_changes)
            
            # 计算相似度分数（基于变化模式和波动性）
            pattern_similarity = 1 / (1 + abs(scenario_pattern - historical_pattern))
            volatility_similarity = 1 / (1 + abs(scenario_volatility - historical_volatility))
            
            # 综合分数
            score = (pattern_similarity + volatility_similarity) / 2
            scenario_scores.append(score)
        
        # 选择最高分数的场景
        best_index = np.argmax(scenario_scores)
        return scenarios[best_index]
    
    def _calculate_overall_metrics(self, validation_results: List[Dict]) -> Dict:
        """计算整体性能指标"""
        if not validation_results:
            return {}
        
        # 收集所有预测指标
        mse_values = []
        mae_values = []
        r2_values = []
        risk_scores = []
        risk_levels = []
        
        for result in validation_results:
            if "prediction_metrics" in result:
                metrics = result["prediction_metrics"]
                if not np.isnan(metrics["mse"]):
                    mse_values.append(metrics["mse"])
                    mae_values.append(metrics["mae"])
                    r2_values.append(metrics["r2"])
            
            if "risk_assessment" in result:
                risk_scores.append(result["risk_assessment"]["risk_score"])
                risk_levels.append(result["risk_assessment"]["risk_level"])
        
        # 计算统计指标
        overall_metrics = {
            "prediction_performance": {
                "mse_mean": np.mean(mse_values) if mse_values else np.nan,
                "mse_std": np.std(mse_values) if mse_values else np.nan,
                "mae_mean": np.mean(mae_values) if mae_values else np.nan,
                "mae_std": np.std(mae_values) if mae_values else np.nan,
                "r2_mean": np.mean(r2_values) if r2_values else np.nan,
                "r2_std": np.std(r2_values) if r2_values else np.nan
            },
            "risk_assessment_performance": {
                "risk_score_mean": np.mean(risk_scores) if risk_scores else np.nan,
                "risk_score_std": np.std(risk_scores) if risk_scores else np.nan,
                "risk_level_distribution": pd.Series(risk_levels).value_counts().to_dict() if risk_levels else {}
            },
            "validation_summary": {
                "total_windows": len(validation_results),
                "successful_windows": len([r for r in validation_results if "error" not in r]),
                "failed_windows": len([r for r in validation_results if "error" in r])
            }
        }
        
        return overall_metrics
    
    def run_cross_validation(self, 
                           data_path: str,
                           stations: Optional[List[str]] = None,
                           validation_windows: int = 5,
                           forecast_horizon: int = 7) -> Dict:
        """运行完整的交叉验证"""
        logger.info("开始洪水风险交叉验证...")
        
        # 加载历史数据
        data = self.load_historical_data(data_path)
        
        # 确定要验证的站点
        if stations is None:
            stations = data['station_id'].unique().tolist()
        
        # 执行交叉验证
        all_results = {}
        
        for station_id in stations:
            logger.info(f"验证站点: {station_id}")
            
            try:
                station_result = self.time_series_cross_validation(
                    data=data,
                    station_id=station_id,
                    validation_windows=validation_windows,
                    forecast_horizon=forecast_horizon
                )
                
                all_results[station_id] = station_result
                
            except Exception as e:
                logger.error(f"站点 {station_id} 验证失败: {e}")
                all_results[station_id] = {"error": str(e)}
        
        # 计算整体性能
        overall_performance = self._calculate_cross_station_metrics(all_results)
        
        # 保存验证结果
        self.validation_results = all_results
        self.performance_metrics = overall_performance
        
        # 生成验证报告
        validation_report = {
            "validation_time": datetime.now().isoformat(),
            "data_source": data_path,
            "validation_parameters": {
                "validation_windows": validation_windows,
                "forecast_horizon": forecast_horizon,
                "stations": stations
            },
            "station_results": all_results,
            "overall_performance": overall_performance
        }
        
        logger.info("交叉验证完成")
        return validation_report
    
    def _calculate_cross_station_metrics(self, station_results: Dict) -> Dict:
        """计算跨站点性能指标"""
        successful_stations = [s for s, r in station_results.items() if "error" not in r]
        
        if not successful_stations:
            return {"error": "所有站点验证都失败"}
        
        # 收集所有站点的性能指标
        all_mse = []
        all_mae = []
        all_r2 = []
        all_risk_scores = []
        
        for station_id in successful_stations:
            result = station_results[station_id]
            if "overall_metrics" in result:
                metrics = result["overall_metrics"]
                
                if "prediction_performance" in metrics:
                    pred_metrics = metrics["prediction_performance"]
                    if not np.isnan(pred_metrics["mse_mean"]):
                        all_mse.append(pred_metrics["mse_mean"])
                        all_mae.append(pred_metrics["mae_mean"])
                        all_r2.append(pred_metrics["r2_mean"])
                
                if "risk_assessment_performance" in metrics:
                    risk_metrics = metrics["risk_assessment_performance"]
                    if not np.isnan(risk_metrics["risk_score_mean"]):
                        all_risk_scores.append(risk_metrics["risk_score_mean"])
        
        # 计算跨站点统计
        cross_station_metrics = {
            "prediction_performance": {
                "mse_mean": np.mean(all_mse) if all_mse else np.nan,
                "mse_std": np.std(all_mse) if all_mse else np.nan,
                "mae_mean": np.mean(all_mae) if all_mae else np.nan,
                "mae_std": np.std(all_mae) if all_mae else np.nan,
                "r2_mean": np.mean(all_r2) if all_r2 else np.nan,
                "r2_std": np.std(all_r2) if all_r2 else np.nan
            },
            "risk_assessment_performance": {
                "risk_score_mean": np.mean(all_risk_scores) if all_risk_scores else np.nan,
                "risk_score_std": np.std(all_risk_scores) if all_risk_scores else np.nan
            },
            "validation_summary": {
                "total_stations": len(station_results),
                "successful_stations": len(successful_stations),
                "failed_stations": len(station_results) - len(successful_stations),
                "success_rate": len(successful_stations) / len(station_results)
            }
        }
        
        return cross_station_metrics
    
    def generate_validation_report(self, output_path: str):
        """生成验证报告"""
        if not self.validation_results:
            logger.warning("没有验证结果，无法生成报告")
            return
        
        # 创建输出目录
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 生成报告内容
        report = {
            "validation_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_stations": len(self.validation_results),
                "overall_performance": self.performance_metrics
            },
            "detailed_results": self.validation_results
        }
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"验证报告已保存到: {output_file}")
    
    def plot_validation_results(self, output_dir: str = "data/processed/validation_plots"):
        """绘制验证结果图表"""
        if not self.validation_results:
            logger.warning("没有验证结果，无法生成图表")
            return
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 设置图表样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. 预测性能对比图
        self._plot_prediction_performance(output_path)
        
        # 2. 风险评估性能图
        self._plot_risk_assessment_performance(output_path)
        
        # 3. 时间序列验证图
        self._plot_time_series_validation(output_path)
        
        logger.info(f"验证结果图表已保存到: {output_path}")
    
    def _plot_prediction_performance(self, output_path: Path):
        """绘制预测性能图表"""
        successful_stations = [s for s, r in self.validation_results.items() if "error" not in r]
        
        if not successful_stations:
            return
        
        # 收集性能指标
        station_names = []
        mse_values = []
        mae_values = []
        r2_values = []
        
        for station_id in successful_stations:
            result = self.validation_results[station_id]
            if "overall_metrics" in result:
                metrics = result["overall_metrics"]["prediction_performance"]
                station_names.append(station_id)
                mse_values.append(metrics["mse_mean"])
                mae_values.append(metrics["mae_mean"])
                r2_values.append(metrics["r2_mean"])
        
        # 创建子图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MSE对比
        axes[0].bar(station_names, mse_values, color='skyblue')
        axes[0].set_title('Mean Squared Error (MSE)')
        axes[0].set_ylabel('MSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        # MAE对比
        axes[1].bar(station_names, mae_values, color='lightcoral')
        axes[1].set_title('Mean Absolute Error (MAE)')
        axes[1].set_ylabel('MAE')
        axes[1].tick_params(axis='x', rotation=45)
        
        # R²对比
        axes[2].bar(station_names, r2_values, color='lightgreen')
        axes[2].set_title('R² Score')
        axes[2].set_ylabel('R²')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_path / 'prediction_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_risk_assessment_performance(self, output_path: Path):
        """绘制风险评估性能图表"""
        successful_stations = [s for s, r in self.validation_results.items() if "error" not in r]
        
        if not successful_stations:
            return
        
        # 收集风险评分
        station_names = []
        risk_scores = []
        
        for station_id in successful_stations:
            result = self.validation_results[station_id]
            if "overall_metrics" in result:
                metrics = result["overall_metrics"]["risk_assessment_performance"]
                station_names.append(station_id)
                risk_scores.append(metrics["risk_score_mean"])
        
        # 创建图表
        plt.figure(figsize=(10, 6))
        bars = plt.bar(station_names, risk_scores, color='gold')
        plt.title('Average Risk Score by Station')
        plt.xlabel('Station ID')
        plt.ylabel('Risk Score')
        plt.xticks(rotation=45)
        
        # 添加数值标签
        for bar, score in zip(bars, risk_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{score:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(output_path / 'risk_assessment_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_time_series_validation(self, output_path: Path):
        """绘制时间序列验证图表"""
        successful_stations = [s for s, r in self.validation_results.items() if "error" not in r]
        
        if not successful_stations:
            return
        
        # 选择第一个成功站点进行详细展示
        station_id = successful_stations[0]
        result = self.validation_results[station_id]
        
        if "results" not in result:
            return
        
        # 创建时间序列图
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # 第一个子图：流量预测对比
        for i, window_result in enumerate(result["results"][:3]):  # 只显示前3个窗口
            if "predicted_flows" in window_result and "actual_flows" in window_result:
                predicted = window_result["predicted_flows"]
                actual = window_result["actual_flows"]
                dates = pd.date_range(
                    start=window_result["val_start"], 
                    periods=len(actual), 
                    freq='D'
                )
                
                axes[0].plot(dates, predicted, 'o-', label=f'Window {i+1} (Predicted)', alpha=0.7)
                axes[0].plot(dates, actual, 's-', label=f'Window {i+1} (Actual)', alpha=0.7)
        
        axes[0].set_title(f'Flow Prediction vs Actual - Station {station_id}')
        axes[0].set_ylabel('Flow (m³/s)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 第二个子图：风险评估时间序列
        risk_scores = []
        dates = []
        
        for window_result in result["results"]:
            if "risk_assessment" in window_result:
                risk_scores.append(window_result["risk_assessment"]["risk_score"])
                dates.append(window_result["val_start"])
        
        if risk_scores:
            axes[1].plot(dates, risk_scores, 'o-', color='red', linewidth=2, markersize=8)
            axes[1].set_title(f'Risk Score Time Series - Station {station_id}')
            axes[1].set_ylabel('Risk Score')
            axes[1].set_xlabel('Date')
            axes[1].grid(True, alpha=0.3)
            
            # 添加风险等级阈值线
            axes[1].axhline(y=20, color='green', linestyle='--', alpha=0.7, label='Low Risk')
            axes[1].axhline(y=40, color='orange', linestyle='--', alpha=0.7, label='Medium Risk')
            axes[1].axhline(y=60, color='red', linestyle='--', alpha=0.7, label='High Risk')
            axes[1].axhline(y=80, color='darkred', linestyle='--', alpha=0.7, label='Extreme Risk')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'time_series_validation.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """测试交叉验证功能"""
    print("🧪 测试洪水风险交叉验证...")
    
    # 创建验证器实例
    validator = FloodRiskCrossValidator()
    
    # 检查是否有历史数据
    data_paths = [
        "data/raw/manitoba_streamflow_processed.csv",
        "data/raw/manitoba_streamflow_sample.csv",
        "data/processed/hydat_streamflow_processed.csv"
    ]
    
    available_data = None
    for path in data_paths:
        if Path(path).exists():
            available_data = path
            break
    
    if not available_data:
        print("❌ 未找到历史数据，请先运行HYDAT数据下载")
        return
    
    print(f"📊 使用历史数据: {available_data}")
    
    # 运行交叉验证
    try:
        validation_report = validator.run_cross_validation(
            data_path=available_data,
            validation_windows=3,  # 减少窗口数以加快测试
            forecast_horizon=7
        )
        
        print("✅ 交叉验证完成")
        print(f"   验证站点数: {len(validation_report['station_results'])}")
        
        if "overall_performance" in validation_report:
            perf = validation_report["overall_performance"]
            if "validation_summary" in perf:
                summary = perf["validation_summary"]
                print(f"   成功率: {summary.get('success_rate', 0):.1%}")
        
        # 生成报告和图表
        validator.generate_validation_report("data/processed/flood_risk_validation_report.json")
        validator.plot_validation_results()
        
        print("📊 验证报告和图表已生成")
        
    except Exception as e:
        print(f"❌ 交叉验证失败: {e}")

if __name__ == "__main__":
    main()
