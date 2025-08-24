#!/usr/bin/env python3
"""
修复真实数据质量问题
处理Environment Canada数据中的缺失值和异常值
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import logging
from datetime import datetime
import json

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_real_data_quality():
    """修复真实数据质量"""
    try:
        logger.info("🔧 开始修复真实数据质量...")
        
        # 加载原始数据
        data_path = "data/real/environment_canada/environment_canada_merged.csv"
        data = pd.read_csv(data_path)
        
        logger.info(f"📊 原始数据: {data.shape}")
        logger.info(f"📋 列名: {list(data.columns)}")
        
        # 检查缺失值
        missing_info = data.isnull().sum()
        logger.info(f"📊 缺失值统计:")
        for col, missing_count in missing_info.items():
            if missing_count > 0:
                missing_rate = missing_count / len(data) * 100
                logger.info(f"  {col}: {missing_count} ({missing_rate:.1f}%)")
        
        # 步骤1: 清理数据
        logger.info("🔧 步骤1: 清理数据...")
        
        # 移除完全缺失的行
        data_cleaned = data.dropna(subset=['Max Temp (°C)', 'Min Temp (°C)', 'Total Precip (mm)'], how='all')
        logger.info(f"✅ 移除完全缺失行后: {data_cleaned.shape}")
        
        # 步骤2: 处理时间列
        logger.info("🔧 步骤2: 处理时间列...")
        
        if 'Date/Time' in data_cleaned.columns:
            data_cleaned['Date/Time'] = pd.to_datetime(data_cleaned['Date/Time'], errors='coerce')
            data_cleaned = data_cleaned.dropna(subset=['Date/Time'])
            
            # 提取时间特征
            data_cleaned['year'] = data_cleaned['Date/Time'].dt.year
            data_cleaned['month'] = data_cleaned['Date/Time'].dt.month
            data_cleaned['day'] = data_cleaned['Date/Time'].dt.day
            data_cleaned['hour'] = data_cleaned['Date/Time'].dt.hour
            data_cleaned['day_of_week'] = data_cleaned['Date/Time'].dt.dayofweek
            
            logger.info(f"✅ 时间列处理完成: {data_cleaned.shape}")
        
        # 步骤3: 处理数值列
        logger.info("🔧 步骤3: 处理数值列...")
        
        # 选择关键数值列
        key_numeric_cols = [
            'Max Temp (°C)', 'Min Temp (°C)', 'Mean Temp (°C)',
            'Heat Deg Days (°C)', 'Cool Deg Days (°C)',
            'Total Rain (mm)', 'Total Snow (cm)', 'Total Precip (mm)',
            'Snow on Grnd (cm)'
        ]
        
        # 只保留存在的列
        available_numeric_cols = [col for col in key_numeric_cols if col in data_cleaned.columns]
        logger.info(f"📋 可用数值列: {available_numeric_cols}")
        
        # 处理缺失值
        for col in available_numeric_cols:
            if data_cleaned[col].isnull().sum() > 0:
                # 使用前向填充和后向填充
                data_cleaned[col] = data_cleaned[col].ffill().bfill()
                
                # 如果仍有缺失值，使用列均值填充
                if data_cleaned[col].isnull().sum() > 0:
                    col_mean = data_cleaned[col].mean()
                    if pd.notna(col_mean):
                        data_cleaned[col] = data_cleaned[col].fillna(col_mean)
                        logger.info(f"  {col}: 使用均值 {col_mean:.2f} 填充")
        
        # 步骤4: 异常值处理
        logger.info("🔧 步骤4: 异常值处理...")
        
        for col in available_numeric_cols:
            if col in data_cleaned.columns:
                # 计算IQR
                Q1 = data_cleaned[col].quantile(0.25)
                Q3 = data_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # 定义异常值边界
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # 统计异常值
                outliers = ((data_cleaned[col] < lower_bound) | (data_cleaned[col] > upper_bound)).sum()
                if outliers > 0:
                    logger.info(f"  {col}: 发现 {outliers} 个异常值")
                    
                    # 将异常值限制在边界内
                    data_cleaned[col] = np.clip(data_cleaned[col], lower_bound, upper_bound)
        
        # 步骤5: 估算土壤湿度
        logger.info("🔧 步骤5: 估算土壤湿度...")
        
        # 基于温度和降水估算土壤湿度
        base_moisture = 0.3
        
        # 温度影响
        if 'Mean Temp (°C)' in data_cleaned.columns:
            temp_factor = 1 - (data_cleaned['Mean Temp (°C)'] + 20) / 60
            temp_factor = np.clip(temp_factor, 0, 1)
        else:
            temp_factor = 0.5
        
        # 降水影响
        if 'Total Precip (mm)' in data_cleaned.columns:
            precip_factor = np.log1p(data_cleaned['Total Precip (mm)'].fillna(0)) / 10
            precip_factor = np.clip(precip_factor, 0, 0.3)
        else:
            precip_factor = 0
        
        # 季节性影响
        if 'month' in data_cleaned.columns:
            seasonal_factor = np.where(
                data_cleaned['month'].isin([12, 1, 2]), 0.1,  # 冬季
                np.where(
                    data_cleaned['month'].isin([3, 4, 5]), 0.2,  # 春季
                    np.where(
                        data_cleaned['month'].isin([6, 7, 8]), 0.0,  # 夏季
                        0.1  # 秋季
                    )
                )
            )
        else:
            seasonal_factor = 0
        
        # 计算估算土壤湿度
        estimated_moisture = (
            base_moisture * 0.4 +
            temp_factor * 0.3 +
            precip_factor * 0.2 +
            seasonal_factor * 0.1
        )
        
        # 限制在合理范围内
        estimated_moisture = np.clip(estimated_moisture, 0.1, 0.9)
        data_cleaned['estimated_soil_moisture'] = estimated_moisture
        
        logger.info("✅ 土壤湿度估算完成")
        
        # 步骤6: 最终数据清理
        logger.info("🔧 步骤6: 最终数据清理...")
        
        # 移除包含NaN的行
        data_final = data_cleaned.dropna()
        logger.info(f"✅ 最终清理后: {data_final.shape}")
        
        # 检查数据质量
        logger.info("📊 最终数据质量检查:")
        logger.info(f"  总记录数: {len(data_final)}")
        logger.info(f"  特征数: {len(data_final.columns)}")
        logger.info(f"  缺失值: {data_final.isnull().sum().sum()}")
        
        # 保存修复后的数据
        output_dir = "data/processed/real_data_fixed"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"real_data_fixed_{timestamp}.csv")
        data_final.to_csv(output_file, index=False)
        
        # 保存数据摘要
        summary = {
            'timestamp': datetime.now().isoformat(),
            'original_shape': data.shape,
            'final_shape': data_final.shape,
            'available_features': list(data_final.columns),
            'data_quality': {
                'missing_values': int(data_final.isnull().sum().sum()),
                'outliers_handled': True,
                'soil_moisture_estimated': True
            }
        }
        
        summary_file = os.path.join(output_dir, f"data_summary_{timestamp}.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ 修复后的数据已保存: {output_file}")
        logger.info(f"✅ 数据摘要已保存: {summary_file}")
        
        return {
            'status': 'success',
            'output_file': output_file,
            'summary': summary
        }
        
    except Exception as e:
        logger.error(f"❌ 数据质量修复失败: {e}")
        return {'status': 'error', 'error': str(e)}

def main():
    """主函数"""
    try:
        logger.info("🚀 启动真实数据质量修复...")
        
        result = fix_real_data_quality()
        
        if result['status'] == 'success':
            logger.info("🎉 数据质量修复成功！")
            logger.info(f"📁 输出文件: {result['output_file']}")
            
            # 显示摘要
            summary = result['summary']
            logger.info(f"📊 数据修复摘要:")
            logger.info(f"  原始数据: {summary['original_shape']}")
            logger.info(f"  修复后: {summary['final_shape']}")
            logger.info(f"  可用特征: {len(summary['available_features'])}")
            
            return result
        else:
            logger.error(f"❌ 修复失败: {result}")
            return None
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return None

if __name__ == "__main__":
    main()
