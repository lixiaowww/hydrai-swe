#!/usr/bin/env python3
"""
修复 Environment Canada 数据质量问题
解决时间参数错误导致的高缺失率问题
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import requests
import time
from typing import Dict, List, Optional

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnvironmentCanadaDataFixer:
    """修复 Environment Canada 数据质量问题的工具"""
    
    def __init__(self):
        # 红河流域相关的Environment Canada站点
        self.stations = {
            'winnipeg_airport': {
                'station_id': '27174',
                'name': 'Winnipeg Richardson International Airport',
                'province': 'MB',
                'lat': 49.9100,
                'lon': -97.2394
            },
            'morris': {
                'station_id': '3025', 
                'name': 'Morris',
                'province': 'MB',
                'lat': 49.3558,
                'lon': -97.3642
            },
            'emerson': {
                'station_id': '3017',
                'name': 'Emerson',
                'province': 'MB', 
                'lat': 49.0042,
                'lon': -97.2189
            }
        }
        
        # Environment Canada 数据URL模板
        self.base_url = "https://climate.weather.gc.ca/climate_data/bulk_data_e.html"
        
        logger.info("✅ Environment Canada数据修复工具初始化完成")
        logger.info(f"📊 可用站点: {len(self.stations)} 个")
    
    def download_corrected_data(self, output_dir: str = "data/real/environment_canada_fixed") -> Dict:
        """下载修正后的历史数据"""
        try:
            logger.info("🚀 开始下载修正后的Environment Canada历史数据")
            
            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)
            
            # 使用正确的历史日期范围
            # Environment Canada通常有1-2个月的数据延迟
            end_date = datetime.now() - timedelta(days=90)  # 3个月前
            start_date = datetime(2023, 1, 1)  # 从2023年开始
            
            logger.info(f"📅 下载范围: {start_date.strftime('%Y-%m')} 到 {end_date.strftime('%Y-%m')}")
            
            all_downloads = []
            successful_downloads = 0
            
            # 遍历每个站点
            for station_key, station_info in self.stations.items():
                logger.info(f"📍 处理站点: {station_info['name']}")
                
                station_downloads = []
                current_date = start_date
                
                # 按月下载数据
                while current_date <= end_date:
                    year = current_date.year
                    month = current_date.month
                    
                    # 下载月度数据
                    result = self._download_monthly_data(station_key, year, month)
                    
                    if result['status'] == 'success':
                        # 验证下载的数据质量
                        if self._validate_data_quality(result['data']):
                            # 保存数据文件
                            filename = f"{station_key}_{year}_{month:02d}.csv"
                            filepath = os.path.join(output_dir, filename)
                            
                            with open(filepath, 'w', encoding='utf-8') as f:
                                f.write(result['data'])
                            
                            result['local_file'] = filepath
                            successful_downloads += 1
                            
                            logger.info(f"💾 保存: {filename}")
                        else:
                            logger.warning(f"⚠️ {station_key} {year}-{month:02d} 数据质量验证失败")
                            result['status'] = 'quality_failed'
                    elif result['status'] == 'no_data':
                        logger.info(f"ℹ️ {station_key} {year}-{month:02d} 无数据")
                    else:
                        logger.warning(f"⚠️ {station_key} {year}-{month:02d} 下载失败: {result.get('error', 'Unknown error')}")
                    
                    station_downloads.append(result)
                    all_downloads.append(result)
                    
                    # 移动到下个月
                    if month == 12:
                        current_date = current_date.replace(year=year+1, month=1)
                    else:
                        current_date = current_date.replace(month=month+1)
                    
                    # 避免请求过快
                    time.sleep(2)
                
                logger.info(f"✅ 完成站点 {station_info['name']}: {len([d for d in station_downloads if d['status'] == 'success'])} 个月的数据")
            
            # 合并所有数据
            merged_file = self._merge_fixed_data(output_dir)
            
            summary = {
                'status': 'success',
                'total_downloads': len(all_downloads),
                'successful_downloads': successful_downloads,
                'stations': len(self.stations),
                'output_dir': output_dir,
                'merged_file': merged_file,
                'downloads': all_downloads
            }
            
            logger.info(f"🎉 修正后的Environment Canada数据下载完成!")
            logger.info(f"📊 成功下载: {successful_downloads}/{len(all_downloads)} 个文件")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ 下载修正后的数据失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _download_monthly_data(self, station_key: str, year: int, month: int) -> Dict:
        """下载特定站点的月度数据"""
        try:
            station = self.stations.get(station_key)
            if not station:
                raise ValueError(f"未知站点: {station_key}")
            
            logger.info(f"📥 下载 {station['name']} 数据: {year}-{month:02d}")
            
            # 构建下载URL
            params = {
                'format': 'csv',
                'stationID': station['station_id'],
                'Year': year,
                'Month': month,
                'Day': '1',
                'timeframe': 2,  # 2=daily, 1=hourly
                'submit': 'Download Data'
            }
            
            # 发送请求
            response = requests.get(self.base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                # 检查是否是有效的CSV数据
                content = response.text
                if 'Date/Time' in content and len(content.strip().split('\n')) > 3:
                    logger.info(f"✅ 成功下载 {station['name']} {year}-{month:02d} 数据")
                    
                    return {
                        'status': 'success',
                        'station_key': station_key,
                        'station_info': station,
                        'year': year,
                        'month': month,
                        'data': content,
                        'url': response.url
                    }
                else:
                    logger.warning(f"⚠️ {station['name']} {year}-{month:02d} 数据内容无效")
                    return {
                        'status': 'no_data',
                        'station_key': station_key,
                        'year': year,
                        'month': month,
                        'message': '数据内容无效'
                    }
            else:
                logger.error(f"❌ 下载失败: HTTP {response.status_code}")
                return {
                    'status': 'error',
                    'station_key': station_key,
                    'error': f'HTTP {response.status_code}'
                }
                
        except Exception as e:
            logger.error(f"❌ 下载 {station_key} 数据失败: {e}")
            return {
                'status': 'error',
                'station_key': station_key,
                'error': str(e)
            }
    
    def _validate_data_quality(self, data_content: str) -> bool:
        """验证数据质量"""
        try:
            lines = data_content.strip().split('\n')
            if len(lines) < 4:  # 至少需要标题行和3行数据
                return False
            
            # 检查数据行
            data_lines = [line for line in lines[1:] if line.strip() and ',' in line]
            if len(data_lines) < 3:
                return False
            
            # 检查是否有足够的数值列
            first_data_line = data_lines[0]
            values = first_data_line.split(',')
            
            # 计算有效数值列
            numeric_values = 0
            for value in values:
                value = value.strip().strip('"')
                if value and value not in ['', 'M', 'E', 'NA', 'N/A', 'null']:
                    try:
                        float(value)
                        numeric_values += 1
                    except ValueError:
                        continue
            
            # 至少需要5个有效数值列
            is_valid = numeric_values >= 5
            
            if not is_valid:
                logger.debug(f"数据质量验证失败: 只有 {numeric_values} 个有效数值列")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"数据质量验证过程出错: {e}")
            return False
    
    def _merge_fixed_data(self, data_dir: str) -> Optional[str]:
        """合并修正后的数据"""
        try:
            logger.info("🔗 合并修正后的数据...")
            
            all_dataframes = []
            csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            
            for csv_file in csv_files:
                try:
                    filepath = os.path.join(data_dir, csv_file)
                    
                    # 读取CSV文件
                    df = pd.read_csv(filepath, encoding='utf-8', low_memory=False)
                    
                    # 添加站点信息
                    station_key = csv_file.split('_')[0]
                    if station_key in self.stations:
                        station_info = self.stations[station_key]
                        df['station_key'] = station_key
                        df['station_name'] = station_info['name']
                        df['station_lat'] = station_info['lat']
                        df['station_lon'] = station_info['lon']
                    
                    all_dataframes.append(df)
                    logger.info(f"📄 处理: {csv_file} ({len(df)} 行)")
                    
                except Exception as e:
                    logger.warning(f"⚠️ 处理文件 {csv_file} 失败: {e}")
                    continue
            
            if all_dataframes:
                # 合并所有数据
                merged_df = pd.concat(all_dataframes, ignore_index=True)
                
                # 保存合并后的数据
                merged_file = os.path.join(data_dir, 'environment_canada_fixed.csv')
                merged_df.to_csv(merged_file, index=False, encoding='utf-8')
                
                logger.info(f"✅ 数据合并完成: {merged_file}")
                logger.info(f"📊 合并后数据: {len(merged_df)} 行, {len(merged_df.columns)} 列")
                
                return merged_file
            else:
                logger.warning("⚠️ 没有有效数据可合并")
                return None
                
        except Exception as e:
            logger.error(f"❌ 合并数据失败: {e}")
            return None
    
    def analyze_fixed_data(self, data_dir: str) -> Dict:
        """分析修正后的数据质量"""
        try:
            logger.info("🔍 分析修正后的数据质量...")
            
            merged_file = os.path.join(data_dir, 'environment_canada_fixed.csv')
            
            if not os.path.exists(merged_file):
                return {'status': 'no_merged_file', 'message': '未找到合并数据文件'}
            
            # 读取合并数据
            df = pd.read_csv(merged_file, low_memory=False)
            
            # 数据质量检查
            validation_result = {
                'status': 'success',
                'total_records': len(df),
                'date_range': {
                    'start': df['Year'].min() if 'Year' in df.columns else 'Unknown',
                    'end': df['Year'].max() if 'Year' in df.columns else 'Unknown'
                },
                'stations': df['station_key'].nunique() if 'station_key' in df.columns else 0,
                'variables': list(df.columns),
                'missing_data': {},
                'data_quality': 'government_official_fixed'
            }
            
            # 检查缺失数据
            for col in df.columns:
                missing_count = df[col].isna().sum()
                if missing_count > 0:
                    validation_result['missing_data'][col] = {
                        'count': int(missing_count),
                        'percentage': float(missing_count / len(df) * 100)
                    }
            
            # 计算总体缺失率
            total_missing = sum([info['count'] for info in validation_result['missing_data'].values()])
            overall_missing_rate = total_missing / (len(df) * len(df.columns)) * 100
            validation_result['overall_missing_rate'] = overall_missing_rate
            
            # 检查关键变量
            key_variables = ['Temp (°C)', 'Rel Hum (%)', 'Wind Spd (km/h)', 'Total Precip (mm)']
            available_key_vars = [var for var in key_variables if var in df.columns]
            
            validation_result['key_variables_available'] = available_key_vars
            validation_result['data_completeness'] = len(available_key_vars) / len(key_variables)
            
            logger.info(f"✅ 数据质量分析完成:")
            logger.info(f"  📊 总记录: {validation_result['total_records']}")
            logger.info(f"  📍 站点数: {validation_result['stations']}")
            logger.info(f"  📈 关键变量: {len(available_key_vars)}/{len(key_variables)}")
            logger.info(f"  📉 总体缺失率: {overall_missing_rate:.1f}%")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ 数据质量分析失败: {e}")
            return {'status': 'error', 'error': str(e)}

def main():
    """主函数"""
    print("🔧 Environment Canada 数据质量修复工具")
    print("=" * 60)
    
    try:
        # 创建修复工具
        fixer = EnvironmentCanadaDataFixer()
        
        # 下载修正后的数据
        print("\n🚀 开始下载修正后的历史数据...")
        result = fixer.download_corrected_data()
        
        if result['status'] == 'success':
            print(f"✅ 下载成功!")
            print(f"📊 成功下载: {result['successful_downloads']}/{result['total_downloads']} 个文件")
            print(f"📁 保存位置: {result['output_dir']}")
            
            if result['merged_file']:
                print(f"🔗 合并文件: {result['merged_file']}")
                
                # 分析数据质量
                print("\n🔍 分析修正后的数据质量...")
                validation = fixer.analyze_fixed_data(result['output_dir'])
                
                if validation['status'] == 'success':
                    print(f"✅ 数据质量分析完成!")
                    print(f"📊 总记录: {validation['total_records']}")
                    print(f"📍 站点数: {validation['stations']}")
                    print(f"📈 数据完整性: {validation['data_completeness']:.1%}")
                    print(f"📉 总体缺失率: {validation['overall_missing_rate']:.1f}%")
                    print(f"🏷️ 数据质量: {validation['data_quality']}")
                    
                    # 比较修复前后的质量
                    print(f"\n📊 修复效果对比:")
                    print(f"  修复前缺失率: 64.7%")
                    print(f"  修复后缺失率: {validation['overall_missing_rate']:.1f}%")
                    
                    if validation['overall_missing_rate'] < 64.7:
                        improvement = 64.7 - validation['overall_missing_rate']
                        print(f"  ✅ 改善效果: {improvement:.1f}%")
                    else:
                        print(f"  ⚠️ 需要进一步优化")
                        
                else:
                    print(f"❌ 数据质量分析失败: {validation.get('error', 'Unknown error')}")
        else:
            print(f"❌ 下载失败: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")

if __name__ == "__main__":
    main()
