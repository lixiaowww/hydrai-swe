#!/usr/bin/env python3
"""
修复曼省数据泄露问题
移除所有包含目标变量信息的特征，重新进行特征工程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ManitobaDataLeakageFixer:
    """曼省数据泄露修复器"""
    
    def __init__(self):
        """初始化"""
        logger.info("🔧 曼省数据泄露修复器初始化完成")
    
    def fix_data_leakage(self, input_file: str) -> pd.DataFrame:
        """修复数据泄露问题"""
        try:
            logger.info(f"🔧 开始修复数据泄露问题: {input_file}")
            
            # 读取原始数据
            data = pd.read_csv(input_file)
            logger.info(f"📊 原始数据: {data.shape}")
            
            # 检查数据泄露
            leakage_issues = self._check_data_leakage(data)
            if leakage_issues:
                logger.warning(f"⚠️ 发现 {len(leakage_issues)} 个数据泄露问题:")
                for issue in leakage_issues:
                    logger.warning(f"  - {issue}")
            
            # 修复数据泄露
            fixed_data = self._remove_leakage_features(data)
            
            # 重新进行特征工程（无泄露）
            engineered_data = self._engineer_features_no_leakage(fixed_data)
            
            logger.info(f"✅ 数据泄露修复完成: {engineered_data.shape}")
            return engineered_data
            
        except Exception as e:
            logger.error(f"❌ 修复数据泄露失败: {e}")
            return pd.DataFrame()
    
    def _check_data_leakage(self, data: pd.DataFrame) -> List[str]:
        """检查数据泄露问题"""
        issues = []
        
        # 检查目标变量是否在特征中
        target_col = 'estimated_soil_moisture'
        if target_col in data.columns:
            issues.append(f"目标变量 '{target_col}' 直接出现在特征中")
        
        # 检查是否有其他泄露问题
        suspicious_cols = ['soil_moisture', 'moisture', 'soil']
        for col in data.columns:
            if any(susp in col.lower() for susp in suspicious_cols):
                if col != target_col:
                    issues.append(f"可疑列 '{col}' 可能包含目标变量信息")
        
        return issues
    
    def _remove_leakage_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """移除泄露特征"""
        try:
            # 移除目标变量
            target_col = 'estimated_soil_moisture'
            if target_col in data.columns:
                data = data.drop(columns=[target_col])
                logger.info(f"✅ 已移除目标变量: {target_col}")
            
            # 保留基础特征
            safe_columns = [
                'date', 'year', 'month', 'day', 'day_of_year',
                'temperature', 'precipitation', 'crop_growth_status',
                'region', 'climate_zone'
            ]
            
            # 只保留安全的列
            safe_data = data[safe_columns].copy()
            logger.info(f"✅ 保留安全特征: {list(safe_columns)}")
            
            return safe_data
            
        except Exception as e:
            logger.error(f"❌ 移除泄露特征失败: {e}")
            return data
    
    def _engineer_features_no_leakage(self, data: pd.DataFrame) -> pd.DataFrame:
        """无泄露的特征工程"""
        try:
            features = data.copy()
            
            # 时间特征
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
            
            # 季节性特征
            features['is_winter'] = (features['month'].isin([12, 1, 2])).astype(int)
            features['is_spring'] = (features['month'].isin([3, 4, 5])).astype(int)
            features['is_summer'] = (features['month'].isin([6, 7, 8])).astype(int)
            features['is_fall'] = (features['month'].isin([9, 10, 11])).astype(int)
            
            # 数值特征变换
            features['temp_squared'] = features['temperature'] ** 2
            features['temp_cubed'] = features['temperature'] ** 3
            features['precip_log'] = np.log1p(features['precipitation'])
            features['precip_squared'] = features['precipitation'] ** 2
            
            # 曼省特有特征
            features['growing_season'] = (features['month'].isin([5, 6, 7, 8, 9])).astype(int)
            features['freezing_season'] = (features['month'].isin([11, 12, 1, 2, 3])).astype(int)
            
            # 移除不需要的列
            features = features.drop(['date', 'region', 'climate_zone'], axis=1)
            
            logger.info(f"✅ 无泄露特征工程完成: {features.shape[1]} 个特征")
            return features
            
        except Exception as e:
            logger.error(f"❌ 无泄露特征工程失败: {e}")
            return data
    
    def save_fixed_data(self, data: pd.DataFrame, output_dir: str) -> str:
        """保存修复后的数据"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"manitoba_fixed_no_leakage_{timestamp}.csv"
            filepath = os.path.join(output_dir, filename)
            
            data.to_csv(filepath, index=False)
            
            logger.info(f"✅ 修复后数据已保存: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ 保存修复后数据失败: {e}")
            return ""

def main():
    """主函数"""
    try:
        logger.info("🚀 启动曼省数据泄露修复...")
        
        # 创建修复器
        fixer = ManitobaDataLeakageFixer()
        
        # 修复数据泄露
        input_file = "data/real/manitoba/agriculture/manitoba_agriculture_20250821_193550.csv"
        if os.path.exists(input_file):
            fixed_data = fixer.fix_data_leakage(input_file)
            
            if not fixed_data.empty:
                # 保存修复后的数据
                output_dir = "data/real/manitoba/fixed"
                output_file = fixer.save_fixed_data(fixed_data, output_dir)
                
                if output_file:
                    logger.info("🎉 曼省数据泄露修复完成！")
                    logger.info(f"📊 修复后数据: {fixed_data.shape}")
                    logger.info(f"💾 保存位置: {output_file}")
                    
                    # 显示特征列表
                    logger.info(f"🔍 修复后特征: {list(fixed_data.columns)}")
                    
                    return True
                else:
                    logger.error("❌ 保存修复后数据失败")
                    return False
            else:
                logger.error("❌ 数据泄露修复失败")
                return False
        else:
            logger.error(f"❌ 输入文件不存在: {input_file}")
            return False
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return False

if __name__ == "__main__":
    main()
