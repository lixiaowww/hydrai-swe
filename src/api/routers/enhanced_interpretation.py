#!/usr/bin/env python3
"""
Enhanced Interpretation Service for HydrAI-SWE
集成水文知识库的专业解读服务

This module provides professional hydrological interpretation based on:
- Comprehensive knowledge base
- Regional context (Manitoba/Red River Basin)
- Climate change considerations
- Technical expertise
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException, Body
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import json

# Import knowledge base
try:
    from src.knowledge.hydrology_knowledge_base import HydrologyKnowledgeBase
except ImportError:
    # Fallback if knowledge base not available
    HydrologyKnowledgeBase = None

router = APIRouter(prefix="/api/v1/interpretation", tags=["enhanced_interpretation"])

class EnhancedInterpretationService:
    """增强解读服务"""
    
    def __init__(self):
        self.knowledge_base = HydrologyKnowledgeBase() if HydrologyKnowledgeBase else None
        if not self.knowledge_base:
            print("⚠️ Warning: Hydrology knowledge base not available, using basic interpretation")
    
    def analyze_trend_significance(self, trend_data: List[float], 
                                 timestamps: List[str]) -> Dict[str, Any]:
        """分析趋势的统计显著性"""
        if len(trend_data) < 10:
            return {"significant": False, "reason": "Insufficient data points for trend analysis"}
        
        # Calculate trend using linear regression
        x = np.arange(len(trend_data))
        y = np.array(trend_data)
        
        # Simple linear regression
        slope, intercept = np.polyfit(x, y, 1)
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate trend magnitude
        total_change = (y_pred[-1] - y_pred[0]) / y_pred[0] * 100 if y_pred[0] != 0 else 0
        
        # Determine significance (R² > 0.5 and sufficient change)
        significant = r_squared > 0.5 and abs(total_change) > 5
        
        return {
            "significant": significant,
            "slope": float(slope),
            "r_squared": float(r_squared),
            "total_change_percent": float(total_change),
            "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "stable",
            "confidence": "high" if r_squared > 0.7 else "moderate" if r_squared > 0.5 else "low"
        }
    
    def detect_seasonal_patterns(self, data: List[float], 
                               timestamps: List[str]) -> Dict[str, Any]:
        """检测季节性模式"""
        if len(data) < 365:  # Need at least one year of data
            return {"pattern": "insufficient_data", "description": "Need at least one year of data for seasonal analysis"}
        
        # Convert to pandas for easier analysis
        df = pd.DataFrame({
            'value': data,
            'date': pd.to_datetime(timestamps)
        })
        df['month'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        
        # Monthly averages
        monthly_avg = df.groupby('month')['value'].mean()
        
        # Find peak month
        peak_month = monthly_avg.idxmax()
        peak_value = monthly_avg.max()
        
        # Determine seasonal pattern
        if peak_month in [12, 1, 2]:  # Winter peak
            pattern = "winter_peak"
            description = "Peak SWE occurs during winter months, typical of continental climates"
        elif peak_month in [3, 4]:  # Spring peak
            pattern = "spring_peak"
            description = "Peak SWE occurs during spring, indicating late accumulation or delayed melt"
        elif peak_month in [5, 6]:  # Late spring peak
            pattern = "late_spring_peak"
            description = "Peak SWE occurs in late spring, suggesting prolonged winter conditions"
        else:
            pattern = "atypical"
            description = "Atypical seasonal pattern requiring further investigation"
        
        return {
            "pattern": pattern,
            "peak_month": int(peak_month),
            "peak_value": float(peak_value),
            "description": description,
            "monthly_averages": monthly_avg.to_dict()
        }
    
    def calculate_anomaly_score(self, data: List[float]) -> float:
        """计算异常分数"""
        if len(data) < 10:
            return 0.0
        
        # Calculate z-score
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return 0.0
        
        # Use the most recent value for anomaly detection
        latest_value = data[-1]
        z_score = (latest_value - mean_val) / std_val
        
        return float(z_score)
    
    def assess_data_quality(self, data: List[float], 
                           timestamps: List[str]) -> Dict[str, Any]:
        """评估数据质量"""
        if not data:
            return {"quality": "poor", "score": 0.0, "issues": ["No data available"]}
        
        issues = []
        score = 100.0
        
        # Check for missing values
        missing_count = sum(1 for x in data if pd.isna(x) or x is None)
        if missing_count > 0:
            missing_rate = missing_count / len(data)
            issues.append(f"Missing data: {missing_rate:.1%}")
            score -= missing_rate * 30
        
        # Check for extreme outliers
        if len(data) > 10:
            mean_val = np.mean([x for x in data if not pd.isna(x) and x is not None])
            std_val = np.std([x for x in data if not pd.isna(x) and x is not None])
            
            if std_val > 0:
                outliers = [x for x in data if not pd.isna(x) and x is not None and abs(x - mean_val) > 3 * std_val]
                if outliers:
                    issues.append(f"Extreme outliers detected: {len(outliers)} values")
                    score -= len(outliers) / len(data) * 20
        
        # Check temporal consistency
        if len(timestamps) > 1:
            try:
                dates = pd.to_datetime(timestamps)
                time_diffs = dates.diff().dropna()
                if len(time_diffs) > 0:
                    irregular_intervals = time_diffs[time_diffs != time_diffs.iloc[0]]
                    if len(irregular_intervals) > 0:
                        issues.append("Irregular time intervals detected")
                        score -= 10
            except:
                issues.append("Timestamp parsing issues")
                score -= 15
        
        # Determine quality level
        if score >= 80:
            quality = "excellent"
        elif score >= 60:
            quality = "good"
        elif score >= 40:
            quality = "fair"
        else:
            quality = "poor"
        
        return {
            "quality": quality,
            "score": max(0.0, score),
            "issues": issues,
            "data_points": len(data),
            "missing_rate": missing_count / len(data) if data else 0.0
        }
    
    def generate_professional_interpretation(self, 
                                          trend_analysis: Dict[str, Any],
                                          seasonal_patterns: Dict[str, Any],
                                          anomaly_score: float,
                                          data_quality: Dict[str, Any]) -> Dict[str, Any]:
        """生成专业解读"""
        
        if not self.knowledge_base:
            return self._generate_basic_interpretation(trend_analysis, seasonal_patterns, anomaly_score, data_quality)
        
        # Use knowledge base for professional interpretation
        interpretation = self.knowledge_base.get_swe_interpretation(
            trend_direction=trend_analysis.get("trend_direction", "stable"),
            trend_magnitude=trend_analysis.get("total_change_percent", 0.0),
            seasonal_pattern=seasonal_patterns.get("pattern", "normal"),
            anomaly_score=anomaly_score,
            forecast_confidence=data_quality.get("score", 0.0) / 100.0
        )
        
        # Add technical details
        interpretation["technical_analysis"] = {
            "trend_significance": trend_analysis.get("significant", False),
            "r_squared": trend_analysis.get("r_squared", 0.0),
            "data_quality_score": data_quality.get("score", 0.0),
            "anomaly_z_score": anomaly_score
        }
        
        return interpretation
    
    def _generate_basic_interpretation(self, 
                                     trend_analysis: Dict[str, Any],
                                     seasonal_patterns: Dict[str, Any],
                                     anomaly_score: float,
                                     data_quality: Dict[str, Any]) -> Dict[str, Any]:
        """生成基础解读（当知识库不可用时）"""
        
        trend_desc = f"SWE shows a {trend_analysis.get('trend_direction', 'stable')} trend "
        if trend_analysis.get("significant", False):
            trend_desc += f"with {abs(trend_analysis.get('total_change_percent', 0)):.1f}% change "
            trend_desc += f"(R² = {trend_analysis.get('r_squared', 0):.3f})"
        else:
            trend_desc += "but the trend is not statistically significant"
        
        seasonal_desc = seasonal_patterns.get("description", "Seasonal pattern analysis requires more data")
        
        anomaly_desc = f"Current SWE values are "
        if abs(anomaly_score) > 2.0:
            anomaly_desc += "extremely anomalous"
        elif abs(anomaly_score) > 1.0:
            anomaly_desc += "moderately anomalous"
        else:
            anomaly_desc += "within normal range"
        
        quality_desc = f"Data quality is {data_quality.get('quality', 'unknown')} "
        quality_desc += f"(score: {data_quality.get('score', 0):.1f}/100)"
        
        return {
            "summary": {
                "trend": trend_desc,
                "seasonal": seasonal_desc,
                "anomaly": anomaly_desc,
                "quality": quality_desc
            },
            "technical_details": {
                "trend_analysis": trend_analysis,
                "seasonal_patterns": seasonal_patterns,
                "anomaly_score": anomaly_score,
                "data_quality": data_quality
            },
            "recommendations": [
                "Continue monitoring SWE patterns",
                "Validate data quality regularly",
                "Consider additional data sources if available"
            ]
        }

# Initialize service
interpretation_service = EnhancedInterpretationService()

@router.post("/swe-comprehensive")
async def comprehensive_swe_interpretation(
    data: Dict[str, Any] = Body(..., description="SWE data for interpretation")
):
    """
    综合SWE解读服务
    
    输入数据应包含：
    - values: SWE数值列表
    - timestamps: 对应的时间戳列表
    - region: 地理区域（可选）
    """
    
    try:
        # Extract data
        values = data.get("values", [])
        timestamps = data.get("timestamps", [])
        region = data.get("region", "manitoba")
        
        if not values or not timestamps:
            raise HTTPException(status_code=400, detail="Values and timestamps are required")
        
        if len(values) != len(timestamps):
            raise HTTPException(status_code=400, detail="Values and timestamps must have same length")
        
        # Perform comprehensive analysis
        trend_analysis = interpretation_service.analyze_trend_significance(values, timestamps)
        seasonal_patterns = interpretation_service.detect_seasonal_patterns(values, timestamps)
        anomaly_score = interpretation_service.calculate_anomaly_score(values)
        data_quality = interpretation_service.assess_data_quality(values, timestamps)
        
        # Generate professional interpretation
        interpretation = interpretation_service.generate_professional_interpretation(
            trend_analysis, seasonal_patterns, anomaly_score, data_quality
        )
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "region": region,
            "data_summary": {
                "data_points": len(values),
                "time_range": f"{timestamps[0]} to {timestamps[-1]}",
                "value_range": f"{min(values):.2f} to {max(values):.2f}"
            },
            "interpretation": interpretation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interpretation failed: {str(e)}")

@router.get("/knowledge-base/glossary")
async def get_technical_glossary():
    """获取技术术语词汇表"""
    try:
        if interpretation_service.knowledge_base:
            glossary = interpretation_service.knowledge_base.get_technical_glossary()
        else:
            glossary = {
                "SWE": "Snow Water Equivalent - depth of water from melted snow",
                "Degree-day": "Temperature-based index for snowmelt modeling",
                "Infiltration": "Process of water entering soil from surface",
                "Runoff": "Water flowing over land surface to streams"
            }
        
        return {
            "status": "success",
            "glossary": glossary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve glossary: {str(e)}")

@router.get("/knowledge-base/climate-context")
async def get_climate_context():
    """获取气候变化背景信息"""
    try:
        if interpretation_service.knowledge_base:
            climate_info = interpretation_service.knowledge_base.get_climate_context()
        else:
            climate_info = {
                "global_context": "Climate change information requires knowledge base access",
                "manitoba_specific": "Regional climate data not available"
            }
        
        return {
            "status": "success",
            "climate_context": climate_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve climate context: {str(e)}")

@router.get("/knowledge-base/regional-info")
async def get_regional_information(region: str = "manitoba"):
    """获取区域特定信息"""
    try:
        if interpretation_service.knowledge_base:
            regional_info = interpretation_service.knowledge_base.regional_context
        else:
            regional_info = {
                "manitoba_hydrology": "Regional information requires knowledge base access",
                "red_river_basin": "Basin-specific data not available"
            }
        
        return {
            "status": "success",
            "region": region,
            "regional_information": regional_info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve regional information: {str(e)}")

@router.post("/quick-assessment")
async def quick_swe_assessment(
    data: Dict[str, Any] = Body(..., description="Quick SWE assessment data")
):
    """
    快速SWE评估服务
    
    适用于快速分析，返回关键指标和解读
    """
    
    try:
        values = data.get("values", [])
        if not values:
            raise HTTPException(status_code=400, detail="Values are required")
        
        # Quick calculations
        current_value = values[-1] if values else 0
        mean_value = np.mean(values) if values else 0
        std_value = np.std(values) if values else 0
        
        # Simple anomaly detection
        if std_value > 0:
            anomaly_score = (current_value - mean_value) / std_value
        else:
            anomaly_score = 0
        
        # Quick assessment
        if abs(anomaly_score) > 2.0:
            assessment = "Extremely anomalous conditions detected"
            risk_level = "high"
        elif abs(anomaly_score) > 1.0:
            assessment = "Moderately anomalous conditions"
            risk_level = "moderate"
        else:
            assessment = "Conditions within normal range"
            risk_level = "low"
        
        return {
            "status": "success",
            "assessment": assessment,
            "risk_level": risk_level,
            "metrics": {
                "current_value": float(current_value),
                "mean_value": float(mean_value),
                "anomaly_score": float(anomaly_score),
                "data_points": len(values)
            },
            "recommendations": [
                "Continue monitoring if risk level is moderate or high",
                "Validate data quality",
                "Consider detailed analysis if anomalies persist"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quick assessment failed: {str(e)}")

if __name__ == "__main__":
    # Test the service
    print("🌊 Enhanced Interpretation Service Test")
    print("=" * 40)
    
    # Test data
    test_values = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    test_timestamps = [f"2024-01-{i:02d}" for i in range(1, 11)]
    
    # Test analysis
    trend = interpretation_service.analyze_trend_significance(test_values, test_timestamps)
    seasonal = interpretation_service.detect_seasonal_patterns(test_values, test_timestamps)
    anomaly = interpretation_service.calculate_anomaly_score(test_values)
    quality = interpretation_service.assess_data_quality(test_values, test_timestamps)
    
    print(f"Trend Analysis: {trend}")
    print(f"Seasonal Patterns: {seasonal}")
    print(f"Anomaly Score: {anomaly}")
    print(f"Data Quality: {quality}")
    
    print("✅ Service test completed!")
