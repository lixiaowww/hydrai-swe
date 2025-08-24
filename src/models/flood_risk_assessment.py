#!/usr/bin/env python3
"""
洪水预警风险评估模型
基于径流预测进行洪水风险分级和预警
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FloodRiskAssessment:
    """洪水风险评估模型"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化洪水风险评估模型"""
        self.config = self._load_config(config_path)
        self.risk_thresholds = self.config['risk_thresholds']
        self.station_configs = self.config['station_configs']
        
        logger.info("洪水风险评估模型初始化完成")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置文件"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        # 默认配置
        return {
            "risk_thresholds": {
                "low": {"flow_ratio": 0.5, "duration_hours": 24, "color": "#10B981"},
                "medium": {"flow_ratio": 1.0, "duration_hours": 12, "color": "#F59E0B"},
                "high": {"flow_ratio": 1.5, "duration_hours": 6, "color": "#EF4444"},
                "extreme": {"flow_ratio": 2.0, "duration_hours": 3, "color": "#7C2D12"}
            },
            "station_configs": {
                # 红河流域站点 (东部)
                "05OC001": {  # Red River at Emerson
                    "name": "Red River at Emerson",
                    "normal_flow": 100.0,  # m³/s
                    "flood_stage": 200.0,  # m³/s
                    "critical_stage": 400.0,  # m³/s
                    "basin_area": 116000,  # km²
                    "response_time_hours": 6,
                    "region": "east"
                },
                "05OC011": {  # Red River at Winnipeg
                    "name": "Red River at Winnipeg",
                    "normal_flow": 150.0,  # m³/s
                    "flood_stage": 300.0,  # m³/s
                    "critical_stage": 600.0,  # m³/s
                    "basin_area": 116000,  # km²
                    "response_time_hours": 12,
                    "region": "east"
                },
                "05OC012": {  # Red River at Lockport
                    "name": "Red River at Lockport",
                    "normal_flow": 200.0,  # m³/s
                    "flood_stage": 400.0,  # m³/s
                    "critical_stage": 800.0,  # m³/s
                    "basin_area": 116000,  # km²
                    "response_time_hours": 18,
                    "region": "east"
                },
                # 南部区域站点
                "5010140": {  # BALDUR
                    "name": "Baldur",
                    "normal_flow": 25.0,  # m³/s
                    "flood_stage": 50.0,  # m³/s
                    "critical_stage": 100.0,  # m³/s
                    "basin_area": 25000,  # km²
                    "response_time_hours": 8,
                    "region": "south",
                    "coordinates": [-99.29, 49.28]
                },
                # 西部区域站点
                "5010481": {  # BRANDON A
                    "name": "Brandon A",
                    "normal_flow": 35.0,  # m³/s
                    "flood_stage": 70.0,  # m³/s
                    "critical_stage": 140.0,  # m³/s
                    "basin_area": 35000,  # km²
                    "response_time_hours": 10,
                    "region": "west",
                    "coordinates": [-99.95, 49.91]
                }
            },
            "warning_levels": {
                "watch": {"description": "洪水观察", "action": "密切监控水位变化"},
                "warning": {"description": "洪水预警", "action": "准备防洪措施"},
                "critical": {"description": "洪水危险", "action": "立即采取防洪行动"},
                "emergency": {"description": "洪水紧急", "action": "紧急疏散"}
            }
        }
    
    def assess_risk(self, 
                    station_id: str, 
                    current_flow: float, 
                    forecast_flows: List[float],
                    forecast_hours: List[int]) -> Dict:
        """
        评估洪水风险
        
        Args:
            station_id: 水文站点ID
            current_flow: 当前流量 (m³/s)
            forecast_flows: 预测流量列表 (m³/s)
            forecast_hours: 预测时间列表 (小时)
        
        Returns:
            风险评估结果字典
        """
        logger.info(f"开始评估站点 {station_id} 的洪水风险...")
        
        if station_id not in self.station_configs:
            raise ValueError(f"未知的站点ID: {station_id}")
        
        station_config = self.station_configs[station_id]
        
        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(
            current_flow, forecast_flows, forecast_hours, station_config
        )
        
        # 确定风险等级
        risk_level = self._determine_risk_level(risk_metrics, station_config)
        
        # 生成预警信息
        warning_info = self._generate_warning_info(risk_level, risk_metrics, station_config)
        
        # 计算风险概率
        risk_probability = self._calculate_risk_probability(risk_metrics, station_config)
        
        # 生成风险评估报告
        assessment_result = {
            "station_id": station_id,
            "station_name": station_config["name"],
            "assessment_time": datetime.now().isoformat(),
            "current_flow": current_flow,
            "current_flow_ratio": risk_metrics["current_flow_ratio"],
            "risk_level": risk_level,
            "risk_score": risk_metrics["risk_score"],
            "risk_probability": risk_probability,
            "warning_level": warning_info["warning_level"],
            "warning_description": warning_info["description"],
            "recommended_action": warning_info["action"],
            "risk_metrics": risk_metrics,
            "forecast_summary": {
                "peak_flow": max(forecast_flows),
                "peak_flow_time": forecast_hours[forecast_flows.index(max(forecast_flows))],
                "duration_above_threshold": risk_metrics["duration_above_threshold"]
            }
        }
        
        logger.info(f"站点 {station_id} 风险评估完成: {risk_level}")
        return assessment_result
    
    def _calculate_risk_metrics(self, 
                               current_flow: float, 
                               forecast_flows: List[float],
                               forecast_hours: List[int],
                               station_config: Dict) -> Dict:
        """计算风险指标"""
        
        normal_flow = station_config["normal_flow"]
        flood_stage = station_config["flood_stage"]
        critical_stage = station_config["critical_stage"]
        
        # 当前流量比率
        current_flow_ratio = current_flow / normal_flow
        
        # 预测流量比率
        forecast_ratios = [flow / normal_flow for flow in forecast_flows]
        max_forecast_ratio = max(forecast_ratios)
        
        # 超过洪水阈值的持续时间
        duration_above_threshold = 0
        for i, flow in enumerate(forecast_flows):
            if flow > flood_stage:
                if i < len(forecast_hours) - 1:
                    duration_above_threshold += forecast_hours[i+1] - forecast_hours[i]
                else:
                    duration_above_threshold += 1  # 假设最后时段为1小时
        
        # 风险评分 (0-100)
        risk_score = 0
        
        # 基于当前流量
        if current_flow > critical_stage:
            risk_score += 40
        elif current_flow > flood_stage:
            risk_score += 25
        elif current_flow > normal_flow * 1.5:
            risk_score += 15
        
        # 基于预测流量
        if max_forecast_ratio >= 2.0: # Changed from max_forecast_flows to max_forecast_ratio
            risk_score += 35
        elif max_forecast_ratio >= 1.5:
            risk_score += 20
        elif max_forecast_ratio >= 1.2:
            risk_score += 10
        
        # 基于持续时间
        if duration_above_threshold > 24:
            risk_score += 15
        elif duration_above_threshold > 12:
            risk_score += 10
        elif duration_above_threshold > 6:
            risk_score += 5
        
        # 限制风险评分在0-100范围内
        risk_score = min(100, max(0, risk_score))
        
        # Enhanced metrics based on project reports (60% completion)
        trend_score = self._calculate_trend_score(forecast_flows)
        acceleration = self._calculate_acceleration(forecast_flows, forecast_hours)
        
        return {
            "current_flow_ratio": current_flow_ratio,
            "max_forecast_ratio": max_forecast_ratio,
            "duration_above_threshold": duration_above_threshold,
            "trend_score": trend_score,
            "acceleration": acceleration,
            "risk_score": risk_score,
            "peak_flow": max(forecast_flows),
            "peak_flow_ratio": max_forecast_ratio
        }
    
    def _determine_risk_level(self, risk_metrics: Dict, station_config: Dict) -> str:
        """确定风险等级"""
        risk_score = risk_metrics["risk_score"]
        current_flow_ratio = risk_metrics["current_flow_ratio"]
        peak_flow_ratio = risk_metrics["peak_flow_ratio"]
        
        # 基于风险评分和流量比率确定风险等级
        if risk_score >= 80 or peak_flow_ratio >= 3.0:
            return "extreme"
        elif risk_score >= 60 or peak_flow_ratio >= 2.0:
            return "high"
        elif risk_score >= 40 or peak_flow_ratio >= 1.5:
            return "medium"
        elif risk_score >= 20 or peak_flow_ratio >= 1.2:
            return "low"
        else:
            return "minimal"
    
    def _generate_warning_info(self, risk_level: str, risk_metrics: Dict, station_config: Dict) -> Dict:
        """生成预警信息"""
        
        warning_mapping = {
            "extreme": {
                "warning_level": "emergency",
                "description": "洪水紧急状态 - 极高风险",
                "action": "立即启动紧急防洪预案，准备疏散"
            },
            "high": {
                "warning_level": "critical",
                "description": "洪水危险状态 - 高风险",
                "action": "启动防洪预案，加强监测，准备应急响应"
            },
            "medium": {
                "warning_level": "warning",
                "description": "洪水预警状态 - 中等风险",
                "action": "密切监控水位变化，准备防洪措施"
            },
            "low": {
                "warning_level": "watch",
                "description": "洪水观察状态 - 低风险",
                "action": "继续监控，注意水位变化"
            },
            "minimal": {
                "warning_level": "normal",
                "description": "正常状态 - 无风险",
                "action": "继续常规监测"
            }
        }
        
        return warning_mapping.get(risk_level, warning_mapping["minimal"])
    
    def _calculate_risk_probability(self, risk_metrics: Dict, station_config: Dict) -> float:
        """计算风险概率 (0-1)"""
        
        # 基于历史数据和当前指标计算风险概率
        risk_score = risk_metrics["risk_score"]
        
        # 简单的线性映射 (可以根据历史数据调整)
        if risk_score >= 80:
            probability = 0.9 + (risk_score - 80) * 0.005  # 90-100%
        elif risk_score >= 60:
            probability = 0.7 + (risk_score - 60) * 0.01   # 70-90%
        elif risk_score >= 40:
            probability = 0.4 + (risk_score - 40) * 0.015  # 40-70%
        elif risk_score >= 20:
            probability = 0.2 + (risk_score - 20) * 0.01   # 20-40%
        else:
            probability = risk_score * 0.01  # 0-20%
        
        return min(1.0, max(0.0, probability))
    
    def _calculate_trend_score(self, forecast_flows: List[float]) -> float:
        """计算流量趋势评分"""
        if len(forecast_flows) < 2:
            return 0.0
        
        # 计算相邻时间点的变化率
        changes = []
        for i in range(1, len(forecast_flows)):
            change = (forecast_flows[i] - forecast_flows[i-1]) / forecast_flows[i-1] * 100
            changes.append(change)
        
        # 平均变化率
        avg_change = sum(changes) / len(changes)
        
        # 趋势评分 (-100 到 +100)
        return max(-100, min(100, avg_change))
    
    def _calculate_acceleration(self, forecast_flows: List[float], forecast_hours: List[int]) -> float:
        """计算流量加速度 (m³/s²)"""
        if len(forecast_flows) < 3 or len(forecast_hours) < 3:
            return 0.0
        
        # 计算速度 (流量变化率)
        velocities = []
        for i in range(1, len(forecast_flows)):
            dt = forecast_hours[i] - forecast_hours[i-1]  # 时间间隔 (小时)
            if dt > 0:
                velocity = (forecast_flows[i] - forecast_flows[i-1]) / dt
                velocities.append(velocity)
        
        if len(velocities) < 2:
            return 0.0
        
        # 计算加速度 (速度变化率)
        accelerations = []
        for i in range(1, len(velocities)):
            dt = forecast_hours[i+1] - forecast_hours[i]
            if dt > 0:
                acceleration = (velocities[i] - velocities[i-1]) / dt
                accelerations.append(acceleration)
        
        return sum(accelerations) / len(accelerations) if accelerations else 0.0
    
    def assess_multiple_stations(self, 
                                station_data: Dict[str, Dict]) -> Dict[str, Dict]:
        """评估多个站点的洪水风险"""
        
        logger.info(f"开始评估 {len(station_data)} 个站点的洪水风险...")
        
        results = {}
        overall_risk = "minimal"
        max_risk_score = 0
        
        for station_id, data in station_data.items():
            try:
                assessment = self.assess_risk(
                    station_id=station_id,
                    current_flow=data["current_flow"],
                    forecast_flows=data["forecast_flows"],
                    forecast_hours=data["forecast_hours"]
                )
                
                results[station_id] = assessment
                
                # 更新整体风险等级
                if assessment["risk_score"] > max_risk_score:
                    max_risk_score = assessment["risk_score"]
                    overall_risk = assessment["risk_level"]
                    
            except Exception as e:
                logger.error(f"评估站点 {station_id} 时出错: {e}")
                results[station_id] = {
                    "error": str(e),
                    "station_id": station_id
                }
        
        # 生成综合风险评估
        summary = {
            "assessment_time": datetime.now().isoformat(),
            "total_stations": len(station_data),
            "overall_risk_level": overall_risk,
            "max_risk_score": max_risk_score,
            "station_results": results,
            "recommendations": self._generate_overall_recommendations(results)
        }
        
        logger.info(f"多站点风险评估完成，整体风险等级: {overall_risk}")
        return summary
    
    def _generate_overall_recommendations(self, station_results: Dict[str, Dict]) -> List[str]:
        """生成整体建议"""
        recommendations = []
        
        # 统计各风险等级的站点数量
        risk_counts = {"extreme": 0, "high": 0, "medium": 0, "low": 0, "minimal": 0}
        
        for result in station_results.values():
            if "risk_level" in result:
                risk_counts[result["risk_level"]] += 1
        
        # 基于风险分布生成建议
        if risk_counts["extreme"] > 0:
            recommendations.append("立即启动区域紧急防洪预案")
            recommendations.append("准备疏散高风险区域居民")
        
        if risk_counts["high"] > 0:
            recommendations.append("加强高风险站点的监测频率")
            recommendations.append("启动防洪物资调配")
        
        if risk_counts["medium"] > 0:
            recommendations.append("密切监控中等风险站点")
            recommendations.append("准备防洪设备和人员")
        
        if risk_counts["low"] > 0:
            recommendations.append("继续常规监测，注意风险变化")
        
        if not recommendations:
            recommendations.append("当前无特殊防洪要求，继续常规监测")
        
        return recommendations
    
    def save_assessment_report(self, assessment_result: Dict, output_path: str):
        """保存风险评估报告"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(assessment_result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"风险评估报告已保存到: {output_file}")

def main():
    """测试洪水风险评估模型"""
    
    # 创建评估模型实例
    risk_assessor = FloodRiskAssessment()
    
    # 测试数据
    test_data = {
        "05OC001": {  # Red River at Emerson
            "current_flow": 150.0,  # m³/s
            "forecast_flows": [160.0, 180.0, 220.0, 280.0, 320.0, 280.0, 220.0],
            "forecast_hours": [6, 12, 18, 24, 30, 36, 42]
        }
    }
    
    # 执行风险评估
    try:
        assessment = risk_assessor.assess_multiple_stations(test_data)
        print("✅ 洪水风险评估测试成功")
        print(f"整体风险等级: {assessment['overall_risk_level']}")
        
        # 保存报告
        risk_assessor.save_assessment_report(
            assessment, 
            "data/processed/flood_risk_assessment_report.json"
        )
        
    except Exception as e:
        print(f"❌ 洪水风险评估测试失败: {e}")

if __name__ == "__main__":
    main()
