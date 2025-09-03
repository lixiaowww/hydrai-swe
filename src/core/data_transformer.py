#!/usr/bin/env python3
"""
数据转换器
统一不同数据源的格式，为各个模块提供标准化的数据接口
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DataTransformer:
    """数据转换器基类"""
    
    def __init__(self):
        self.standard_columns = {
            "datetime": "Date/Time",
            "precipitation": "precipitation_mm", 
            "temperature": "temperature_c",
            "humidity": "humidity_percent",
            "pressure": "pressure_hpa",
            "wind_speed": "wind_speed_kmh",
            "wind_direction": "wind_direction_deg"
        }
    
    def transform(self, data: pd.DataFrame, source: str) -> pd.DataFrame:
        """
        转换数据格式
        
        Args:
            data: 原始数据
            source: 数据源类型
            
        Returns:
            DataFrame: 转换后的数据
        """
        if source == "swe":
            return self._transform_swe_data(data)
        elif source == "flood":
            return self._transform_flood_data(data)
        elif source == "hydrology":
            return self._transform_hydrology_data(data)
        elif source == "weather":
            return self._transform_weather_data(data)
        elif source == "agriculture":
            return self._transform_agriculture_data(data)
        else:
            logger.warning(f"未知数据源类型: {source}")
            return data
    
    def _transform_swe_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换SWE数据格式"""
        transformed = data.copy()
        
        # 标准化列名
        column_mapping = {
            "snow_water_equivalent_mm": "swe_mm",
            "snow_depth_cm": "snow_depth_cm",
            "temperature": "temperature_c",
            "precipitation": "precipitation_mm"
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in transformed.columns:
                transformed = transformed.rename(columns={old_col: new_col})
        
        # 确保有日期列
        if "Date/Time" not in transformed.columns and "date" in transformed.columns:
            transformed["Date/Time"] = pd.to_datetime(transformed["date"])
        
        return transformed
    
    def _transform_flood_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换洪水数据格式"""
        transformed = data.copy()
        
        # 标准化列名
        column_mapping = {
            "streamflow_m3s": "flow_m3s",
            "water_level_m": "level_m",
            "risk_level": "risk_level",
            "precipitation_mm": "precip_mm"
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in transformed.columns:
                transformed = transformed.rename(columns={old_col: new_col})
        
        # 确保有日期列
        if "Date/Time" not in transformed.columns and "date" in transformed.columns:
            transformed["Date/Time"] = pd.to_datetime(transformed["date"])
        
        return transformed
    
    def _transform_hydrology_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换水文数据格式"""
        transformed = data.copy()
        
        # 标准化列名
        column_mapping = {
            "streamflow": "flow_m3s",
            "precipitation": "precip_mm",
            "evapotranspiration": "et_mm",
            "runoff": "runoff_mm"
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in transformed.columns:
                transformed = transformed.rename(columns={old_col: new_col})
        
        return transformed
    
    def _transform_weather_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换天气数据格式"""
        transformed = data.copy()
        
        # 标准化列名
        column_mapping = {
            "Mean Temp (°C)": "temperature_c",
            "Total Precip (mm)": "precipitation_mm",
            "Spd of Max Gust (km/h)": "wind_speed_kmh"
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in transformed.columns:
                transformed = transformed.rename(columns={old_col: new_col})
        
        return transformed
    
    def _transform_agriculture_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """转换农业数据格式"""
        transformed = data.copy()
        
        # 标准化列名
        column_mapping = {
            "soil_moisture": "soil_moisture_percent",
            "crop_yield": "yield_kg_ha",
            "precipitation": "precip_mm"
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in transformed.columns:
                transformed = transformed.rename(columns={old_col: new_col})
        
        return transformed

class SWEDataTransformer(DataTransformer):
    """SWE数据专用转换器"""
    
    def transform_for_analysis(self, data: pd.DataFrame) -> pd.DataFrame:
        """为SWE分析转换数据"""
        transformed = self._transform_swe_data(data)
        
        # 添加分析所需的衍生列
        if "Date/Time" in transformed.columns:
            # 确保日期列是datetime类型
            try:
                if not pd.api.types.is_datetime64_any_dtype(transformed["Date/Time"]):
                    transformed["Date/Time"] = pd.to_datetime(transformed["Date/Time"], errors='coerce')
                
                # 添加时间衍生列
                transformed["year"] = transformed["Date/Time"].dt.year
                transformed["month"] = transformed["Date/Time"].dt.month
                transformed["day"] = transformed["Date/Time"].dt.day
                transformed["day_of_year"] = transformed["Date/Time"].dt.dayofyear
                
                logger.info("✅ 日期列转换成功")
            except Exception as e:
                logger.warning(f"⚠️ 日期列转换失败: {e}")
        
        # 确保数值列的类型
        numeric_columns = ["swe_mm", "snow_depth_cm", "temperature_c", "precipitation_mm"]
        for col in numeric_columns:
            if col in transformed.columns:
                try:
                    transformed[col] = pd.to_numeric(transformed[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"⚠️ 数值列{col}转换失败: {e}")
        
        return transformed

class FloodDataTransformer(DataTransformer):
    """洪水数据专用转换器"""
    
    def transform_for_prediction(self, data: pd.DataFrame) -> pd.DataFrame:
        """为洪水预测转换数据"""
        transformed = self._transform_flood_data(data)
        
        # 添加预测所需的特征
        if "Date/Time" in transformed.columns:
            transformed["hour"] = transformed["Date/Time"].dt.hour
            transformed["day_of_year"] = transformed["Date/Time"].dt.dayofyear
        
        # 计算移动平均
        if "flow_m3s" in transformed.columns:
            transformed["flow_ma_3"] = transformed["flow_m3s"].rolling(window=3).mean()
            transformed["flow_ma_7"] = transformed["flow_m3s"].rolling(window=7).mean()
        
        return transformed

# 全局转换器实例
data_transformer = DataTransformer()
swe_transformer = SWEDataTransformer()
flood_transformer = FloodDataTransformer()

def get_transformer(source: str) -> DataTransformer:
    """获取指定数据源的转换器"""
    if source == "swe":
        return swe_transformer
    elif source == "flood":
        return flood_transformer
    else:
        return data_transformer
