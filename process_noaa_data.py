#!/usr/bin/env python3
"""
处理NOAA数据
将下载的NOAA数据转换为适合模型训练的格式
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NOAADataProcessor:
    """NOAA数据处理器"""
    
    def __init__(self):
        """初始化"""
        self.processed_data = None
        
    def process_daily_summaries(self, file_path: str) -> Optional[pd.DataFrame]:
        """处理NOAA每日摘要数据"""
        try:
            logger.info(f"🔧 处理NOAA每日摘要数据: {file_path}")
            
            # 读取数据
            df = pd.read_csv(file_path)
            logger.info(f"📊 原始数据: {df.shape}")
            
            # 数据清理
            df_cleaned = self._clean_daily_data(df)
            
            # 特征工程
            df_engineered = self._engineer_daily_features(df_cleaned)
            
            # 估算土壤湿度
            df_final = self._estimate_soil_moisture_daily(df_engineered)
            
            logger.info(f"✅ 每日摘要数据处理完成: {df_final.shape}")
            return df_final
            
        except Exception as e:
            logger.error(f"❌ 处理每日摘要数据失败: {e}")
            return None
    
    def process_hourly_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """处理NOAA小时数据"""
        try:
            logger.info(f"🔧 处理NOAA小时数据: {file_path}")
            
            # 读取数据
            df = pd.read_csv(file_path)
            logger.info(f"📊 原始数据: {df.shape}")
            
            # 数据清理
            df_cleaned = self._clean_hourly_data(df)
            
            # 特征工程
            df_engineered = self._engineer_hourly_features(df_cleaned)
            
            # 估算土壤湿度
            df_final = self._estimate_soil_moisture_hourly(df_engineered)
            
            logger.info(f"✅ 小时数据处理完成: {df_final.shape}")
            return df_final
            
        except Exception as e:
            logger.error(f"❌ 处理小时数据失败: {e}")
            return None
    
    def _clean_daily_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理每日摘要数据"""
        try:
            # 处理缺失值
            df = df.replace('999.9', np.nan)
            df = df.replace('99.99', np.nan)
            
            # 转换数值列
            numeric_columns = ['TEMP', 'DEWP', 'SLP', 'STP', 'VISIB', 'WDSP', 'MXSPD', 'GUST', 'MAX', 'MIN', 'PRCP', 'SNDP']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 处理日期
            df['DATE'] = pd.to_datetime(df['DATE'])
            df['year'] = df['DATE'].dt.year
            df['month'] = df['DATE'].dt.month
            df['day'] = df['DATE'].dt.day
            df['day_of_year'] = df['DATE'].dt.dayofyear
            df['day_of_week'] = df['DATE'].dt.dayofweek
            
            # 移除完全缺失的行
            df = df.dropna(subset=['TEMP', 'PRCP'], how='all')
            
            logger.info(f"✅ 每日数据清理完成: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ 每日数据清理失败: {e}")
            return df
    
    def _clean_hourly_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """清理小时数据"""
        try:
            # 处理缺失值
            df = df.replace('99999,9,9,9', np.nan)
            df = df.replace('999999,9,9,9', np.nan)
            
            # 处理日期时间
            df['DATE'] = pd.to_datetime(df['DATE'])
            df['year'] = df['DATE'].dt.year
            df['month'] = df['DATE'].dt.month
            df['day'] = df['DATE'].dt.day
            df['hour'] = df['DATE'].dt.hour
            df['day_of_year'] = df['DATE'].dt.dayofyear
            df['day_of_week'] = df['DATE'].dt.dayofweek
            
            # 处理风向风速数据 (格式: "318,1,N,0061,1")
            df['wind_direction'] = df['WND'].str.extract(r'(\d+),')[0].astype(float)
            df['wind_speed'] = df['WND'].str.extract(r',(\d+),')[0].astype(float)
            
            # 处理温度数据 (格式: "-0070,1")
            df['temperature'] = df['TMP'].str.extract(r'([+-]?\d+),')[0].astype(float) / 10
            
            # 处理露点数据
            df['dewpoint'] = df['DEW'].str.extract(r'([+-]?\d+),')[0].astype(float) / 10
            
            # 处理气压数据 (格式: "10208,1")
            df['pressure'] = df['SLP'].str.extract(r'(\d+),')[0].astype(float) / 10
            
            # 移除完全缺失的行
            df = df.dropna(subset=['temperature'], how='all')
            
            logger.info(f"✅ 小时数据清理完成: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ 小时数据清理失败: {e}")
            return df
    
    def _engineer_daily_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """每日数据特征工程"""
        try:
            # 基础特征
            features = df[['year', 'month', 'day', 'day_of_year', 'day_of_week']].copy()
            
            # 数值特征
            if 'TEMP' in df.columns:
                features['temperature'] = df['TEMP']
                features['temp_squared'] = df['TEMP'] ** 2
            
            if 'MAX' in df.columns:
                features['max_temp'] = df['MAX']
            
            if 'MIN' in df.columns:
                features['min_temp'] = df['MIN']
                features['temp_range'] = df['MAX'] - df['MIN']
            
            if 'PRCP' in df.columns:
                features['precipitation'] = df['PRCP']
                features['precip_log'] = np.log1p(df['PRCP'].fillna(0))
            
            if 'SNDP' in df.columns:
                features['snow_depth'] = df['SNDP']
            
            if 'WDSP' in df.columns:
                features['wind_speed'] = df['WDSP']
            
            if 'SLP' in df.columns:
                features['pressure'] = df['SLP']
            
            # 季节性特征
            features['is_winter'] = features['month'].isin([12, 1, 2]).astype(int)
            features['is_spring'] = features['month'].isin([3, 4, 5]).astype(int)
            features['is_summer'] = features['month'].isin([6, 7, 8]).astype(int)
            features['is_fall'] = features['month'].isin([9, 10, 11]).astype(int)
            
            # 周期性特征
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
            
            # 移除包含NaN的行
            features = features.dropna()
            
            logger.info(f"✅ 每日特征工程完成: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"❌ 每日特征工程失败: {e}")
            return pd.DataFrame()
    
    def _engineer_hourly_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """小时数据特征工程"""
        try:
            # 基础特征
            features = df[['year', 'month', 'day', 'hour', 'day_of_year', 'day_of_week']].copy()
            
            # 数值特征
            if 'temperature' in df.columns:
                features['temperature'] = df['temperature']
                features['temp_squared'] = df['temperature'] ** 2
            
            if 'dewpoint' in df.columns:
                features['dewpoint'] = df['dewpoint']
                features['humidity'] = self._calculate_humidity(df['temperature'], df['dewpoint'])
            
            if 'wind_direction' in df.columns:
                features['wind_direction'] = df['wind_direction']
                features['wind_direction_sin'] = np.sin(np.radians(df['wind_direction']))
                features['wind_direction_cos'] = np.cos(np.radians(df['wind_direction']))
            
            if 'wind_speed' in df.columns:
                features['wind_speed'] = df['wind_speed']
                features['wind_speed_squared'] = df['wind_speed'] ** 2
            
            if 'pressure' in df.columns:
                features['pressure'] = df['pressure']
            
            # 时间特征
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            
            # 季节性特征
            features['is_winter'] = features['month'].isin([12, 1, 2]).astype(int)
            features['is_spring'] = features['month'].isin([3, 4, 5]).astype(int)
            features['is_summer'] = features['month'].isin([6, 7, 8]).astype(int)
            features['is_fall'] = features['month'].isin([9, 10, 11]).astype(int)
            
            # 周期性特征
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
            
            # 移除包含NaN的行
            features = features.dropna()
            
            logger.info(f"✅ 小时特征工程完成: {features.shape}")
            return features
            
        except Exception as e:
            logger.error(f"❌ 小时特征工程失败: {e}")
            return pd.DataFrame()
    
    def _calculate_humidity(self, temperature: pd.Series, dewpoint: pd.Series) -> pd.Series:
        """计算相对湿度"""
        try:
            # 使用Magnus公式计算相对湿度
            # 饱和水汽压
            es_t = 6.112 * np.exp((17.67 * temperature) / (temperature + 243.5))
            es_d = 6.112 * np.exp((17.67 * dewpoint) / (dewpoint + 243.5))
            
            # 相对湿度
            humidity = (es_d / es_t) * 100
            return np.clip(humidity, 0, 100)
            
        except Exception as e:
            logger.warning(f"⚠️ 湿度计算失败: {e}")
            return pd.Series([50] * len(temperature), index=temperature.index)
    
    def _estimate_soil_moisture_daily(self, df: pd.DataFrame) -> pd.DataFrame:
        """估算每日土壤湿度"""
        try:
            # 基础土壤湿度
            base_moisture = 0.3
            
            # 温度影响
            if 'temperature' in df.columns:
                temp_factor = 1 - (df['temperature'] + 20) / 60
                temp_factor = np.clip(temp_factor, 0, 1)
            else:
                temp_factor = 0.5
            
            # 降水影响
            if 'precipitation' in df.columns:
                precip_factor = np.log1p(df['precipitation'].fillna(0)) / 20
                precip_factor = np.clip(precip_factor, 0, 0.3)
            else:
                precip_factor = 0
            
            # 季节性影响
            seasonal_factor = np.where(
                df['month'].isin([12, 1, 2]), 0.1,  # 冬季
                np.where(
                    df['month'].isin([3, 4, 5]), 0.2,  # 春季
                    np.where(
                        df['month'].isin([6, 7, 8]), 0.0,  # 夏季
                        0.1  # 秋季
                    )
                )
            )
            
            # 计算估算土壤湿度
            estimated_moisture = (
                base_moisture * 0.4 +
                temp_factor * 0.3 +
                precip_factor * 0.2 +
                seasonal_factor * 0.1
            )
            
            # 限制在合理范围内
            estimated_moisture = np.clip(estimated_moisture, 0.1, 0.9)
            df['estimated_soil_moisture'] = estimated_moisture
            
            logger.info("✅ 每日土壤湿度估算完成")
            return df
            
        except Exception as e:
            logger.error(f"❌ 每日土壤湿度估算失败: {e}")
            return df
    
    def _estimate_soil_moisture_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """估算小时土壤湿度"""
        try:
            # 基础土壤湿度
            base_moisture = 0.3
            
            # 温度影响
            if 'temperature' in df.columns:
                temp_factor = 1 - (df['temperature'] + 20) / 60
                temp_factor = np.clip(temp_factor, 0, 1)
            else:
                temp_factor = 0.5
            
            # 湿度影响
            if 'humidity' in df.columns:
                humidity_factor = df['humidity'] / 100
            else:
                humidity_factor = 0.5
            
            # 季节性影响
            seasonal_factor = np.where(
                df['month'].isin([12, 1, 2]), 0.1,  # 冬季
                np.where(
                    df['month'].isin([3, 4, 5]), 0.2,  # 春季
                    np.where(
                        df['month'].isin([6, 7, 8]), 0.0,  # 夏季
                        0.1  # 秋季
                    )
                )
            )
            
            # 计算估算土壤湿度
            estimated_moisture = (
                base_moisture * 0.4 +
                temp_factor * 0.3 +
                humidity_factor * 0.2 +
                seasonal_factor * 0.1
            )
            
            # 限制在合理范围内
            estimated_moisture = np.clip(estimated_moisture, 0.1, 0.9)
            df['estimated_soil_moisture'] = estimated_moisture
            
            logger.info("✅ 小时土壤湿度估算完成")
            return df
            
        except Exception as e:
            logger.error(f"❌ 小时土壤湿度估算失败: {e}")
            return df
    
    def save_processed_data(self, df: pd.DataFrame, output_dir: str, filename: str) -> str:
        """保存处理后的数据"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            filepath = os.path.join(output_dir, filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"✅ 处理后数据已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ 保存处理后数据失败: {e}")
            return ""

def main():
    """主函数"""
    try:
        logger.info("🚀 启动NOAA数据处理...")
        
        # 创建处理器
        processor = NOAADataProcessor()
        
        # 处理每日摘要数据
        daily_file = "data/real/noaa_daily/noaa_daily_2024_sample_20250821_191859.csv"
        if os.path.exists(daily_file):
            logger.info("🔧 处理每日摘要数据...")
            daily_processed = processor.process_daily_summaries(daily_file)
            
            if daily_processed is not None and not daily_processed.empty:
                # 保存处理后的每日数据
                daily_output = processor.save_processed_data(
                    daily_processed,
                    "data/processed/noaa_daily",
                    f"noaa_daily_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                logger.info(f"📊 每日数据处理完成: {daily_processed.shape}")
            else:
                logger.warning("⚠️ 每日数据处理失败")
        else:
            logger.warning("⚠️ 每日摘要数据文件不存在")
        
        # 处理小时数据
        hourly_file = "data/real/noaa_hourly/noaa_hourly_2024_sample_20250821_191901.csv"
        if os.path.exists(hourly_file):
            logger.info("🔧 处理小时数据...")
            hourly_processed = processor.process_hourly_data(hourly_file)
            
            if hourly_processed is not None and not hourly_processed.empty:
                # 保存处理后的小时数据
                hourly_output = processor.save_processed_data(
                    hourly_processed,
                    "data/processed/noaa_hourly",
                    f"noaa_hourly_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                )
                logger.info(f"📊 小时数据处理完成: {hourly_processed.shape}")
            else:
                logger.warning("⚠️ 小时数据处理失败")
        else:
            logger.warning("⚠️ 小时数据文件不存在")
        
        logger.info("🎉 NOAA数据处理完成！")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return False

if __name__ == "__main__":
    main()
