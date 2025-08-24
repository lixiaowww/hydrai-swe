#!/usr/bin/env python3
"""
下载真实数据
从搜索到的可用数据源下载实际数据
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import requests
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
from typing import Dict, List, Optional
import time
import zipfile
import io
import re

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataDownloader:
    """真实数据下载器"""
    
    def __init__(self):
        """初始化"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def download_noaa_daily_summaries(self) -> Optional[str]:
        """下载NOAA每日摘要数据"""
        try:
            logger.info("📥 下载NOAA每日摘要数据...")
            
            # 创建下载目录
            download_dir = "data/real/noaa_daily"
            os.makedirs(download_dir, exist_ok=True)
            
            # 尝试下载最近的每日摘要数据
            base_url = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"
            
            # 获取可用年份列表
            response = self.session.get(base_url, timeout=10)
            if response.status_code != 200:
                logger.warning("⚠️ 无法访问NOAA基础URL")
                return None
            
            # 尝试下载2024年的数据
            year = 2024
            year_url = f"{base_url}{year}/"
            
            response = self.session.get(year_url, timeout=10)
            if response.status_code == 200:
                # 尝试下载一个具体文件
                sample_file = f"{year_url}01001099999.csv"
                response = self.session.get(sample_file, timeout=30)
                
                if response.status_code == 200:
                    # 保存数据
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"noaa_daily_{year}_sample_{timestamp}.csv"
                    filepath = os.path.join(download_dir, filename)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    logger.info(f"✅ NOAA每日摘要数据已保存: {filepath}")
                    return filepath
                else:
                    logger.warning(f"⚠️ 无法下载NOAA {year}年数据: HTTP {response.status_code}")
            else:
                logger.warning(f"⚠️ 无法访问NOAA {year}年目录: HTTP {response.status_code}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 下载NOAA每日摘要数据失败: {e}")
            return None
    
    def download_noaa_hourly_data(self) -> Optional[str]:
        """下载NOAA小时数据"""
        try:
            logger.info("📥 下载NOAA小时数据...")
            
            # 创建下载目录
            download_dir = "data/real/noaa_hourly"
            os.makedirs(download_dir, exist_ok=True)
            
            # 尝试下载最近的小时数据
            base_url = "https://www.ncei.noaa.gov/data/global-hourly/access/"
            
            # 尝试下载2024年的数据
            year = 2024
            year_url = f"{base_url}{year}/"
            
            response = self.session.get(year_url, timeout=10)
            if response.status_code == 200:
                # 尝试下载一个具体文件
                sample_file = f"{year_url}01001099999.csv"
                response = self.session.get(sample_file, timeout=30)
                
                if response.status_code == 200:
                    # 保存数据
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"noaa_hourly_{year}_sample_{timestamp}.csv"
                    filepath = os.path.join(download_dir, filename)
                    
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    
                    logger.info(f"✅ NOAA小时数据已保存: {filepath}")
                    return filepath
                else:
                    logger.warning(f"⚠️ 无法下载NOAA {year}年小时数据: HTTP {response.status_code}")
            else:
                logger.warning(f"⚠️ 无法访问NOAA {year}年小时目录: HTTP {response.status_code}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 下载NOAA小时数据失败: {e}")
            return None
    
    def download_eobs_daily_data(self) -> Optional[str]:
        """下载EOBS每日数据"""
        try:
            logger.info("📥 下载EOBS每日数据...")
            
            # 创建下载目录
            download_dir = "data/real/eobs_daily"
            os.makedirs(download_dir, exist_ok=True)
            
            # EOBS数据下载页面
            eobs_url = "https://www.ecad.eu/download/ensembles/download.php"
            
            response = self.session.get(eobs_url, timeout=10)
            if response.status_code == 200:
                # 尝试下载一个具体的数据文件
                # EOBS通常提供温度、降水等数据
                sample_url = "https://www.ecad.eu/download/ensembles/data/Grid_0.1deg_reg_ensemble/tg_0.1deg_reg_2024.01.nc"
                
                response = self.session.get(sample_url, timeout=30)
                if response.status_code == 200:
                    # 保存数据
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"eobs_daily_sample_{timestamp}.nc"
                    filepath = os.path.join(download_dir, filename)
                    
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    
                    logger.info(f"✅ EOBS每日数据已保存: {filepath}")
                    return filepath
                else:
                    logger.warning(f"⚠️ 无法下载EOBS数据文件: HTTP {response.status_code}")
            else:
                logger.warning(f"⚠️ 无法访问EOBS下载页面: HTTP {response.status_code}")
            
            return None
            
        except Exception as e:
            logger.error(f"❌ 下载EOBS每日数据失败: {e}")
            return None
    
    def download_openmeteo_data(self) -> Optional[str]:
        """下载Open-Meteo数据"""
        try:
            logger.info("📥 下载Open-Meteo数据...")
            
            # 创建下载目录
            download_dir = "data/real/openmeteo"
            os.makedirs(download_dir, exist_ok=True)
            
            # Open-Meteo API (免费，无需API密钥)
            # 获取加拿大几个城市的天气数据
            cities = [
                {'name': 'Winnipeg', 'lat': 49.8951, 'lon': -97.1384},
                {'name': 'Toronto', 'lat': 43.6532, 'lon': -79.3832},
                {'name': 'Vancouver', 'lat': 49.2827, 'lon': -123.1207}
            ]
            
            all_data = []
            
            for city in cities:
                try:
                    # 获取历史天气数据
                    url = f"https://archive-api.open-meteo.com/v1/archive"
                    params = {
                        'latitude': city['lat'],
                        'longitude': city['lon'],
                        'start_date': '2024-01-01',
                        'end_date': '2024-12-31',
                        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,soil_moisture_0_to_7cm',
                        'timezone': 'America/Winnipeg'
                    }
                    
                    response = self.session.get(url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'daily' in data:
                            # 转换为DataFrame
                            df = pd.DataFrame(data['daily'])
                            df['city'] = city['name']
                            df['latitude'] = city['lat']
                            df['longitude'] = city['lon']
                            all_data.append(df)
                            
                            logger.info(f"✅ 成功获取 {city['name']} 数据: {len(df)} 条记录")
                        else:
                            logger.warning(f"⚠️ {city['name']} 数据格式不正确")
                    else:
                        logger.warning(f"⚠️ 无法获取 {city['name']} 数据: HTTP {response.status_code}")
                    
                    time.sleep(1)  # 避免请求过快
                    
                except Exception as e:
                    logger.warning(f"⚠️ 获取 {city['name']} 数据失败: {e}")
            
            if all_data:
                # 合并所有城市数据
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # 保存数据
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"openmeteo_canada_{timestamp}.csv"
                filepath = os.path.join(download_dir, filename)
                
                combined_df.to_csv(filepath, index=False)
                
                logger.info(f"✅ Open-Meteo数据已保存: {filepath}")
                logger.info(f"📊 总记录数: {len(combined_df)}")
                logger.info(f"🏙️ 城市数: {len(cities)}")
                
                return filepath
            else:
                logger.warning("⚠️ 未获取到任何Open-Meteo数据")
                return None
            
        except Exception as e:
            logger.error(f"❌ 下载Open-Meteo数据失败: {e}")
            return None
    
    def download_visualcrossing_data(self) -> Optional[str]:
        """下载Visual Crossing数据"""
        try:
            logger.info("📥 下载Visual Crossing数据...")
            
            # 创建下载目录
            download_dir = "data/real/visualcrossing"
            os.makedirs(download_dir, exist_ok=True)
            
            # Visual Crossing提供免费的历史天气数据
            # 获取加拿大几个城市的数据
            cities = [
                {'name': 'Edmonton', 'lat': 53.5461, 'lon': -113.4938},
                {'name': 'Calgary', 'lat': 51.0447, 'lon': -114.0719},
                {'name': 'Montreal', 'lat': 45.5017, 'lon': -73.5673}
            ]
            
            all_data = []
            
            for city in cities:
                try:
                    # 使用免费的历史天气API
                    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city['lat']},{city['lon']}/2024-01-01/2024-12-31"
                    params = {
                        'unitGroup': 'metric',
                        'include': 'days',
                        'key': 'demo',  # 使用演示密钥
                        'contentType': 'json'
                    }
                    
                    response = self.session.get(url, params=params, timeout=30)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'days' in data:
                            # 转换为DataFrame
                            df = pd.DataFrame(data['days'])
                            df['city'] = city['name']
                            df['latitude'] = city['lat']
                            df['longitude'] = city['lon']
                            all_data.append(df)
                            
                            logger.info(f"✅ 成功获取 {city['name']} 数据: {len(df)} 条记录")
                        else:
                            logger.warning(f"⚠️ {city['name']} 数据格式不正确")
                    else:
                        logger.warning(f"⚠️ 无法获取 {city['name']} 数据: HTTP {response.status_code}")
                    
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"⚠️ 获取 {city['name']} 数据失败: {e}")
            
            if all_data:
                # 合并所有城市数据
                combined_df = pd.concat(all_data, ignore_index=True)
                
                # 保存数据
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"visualcrossing_canada_{timestamp}.csv"
                filepath = os.path.join(download_dir, filename)
                
                combined_df.to_csv(filepath, index=False)
                
                logger.info(f"✅ Visual Crossing数据已保存: {filepath}")
                logger.info(f"📊 总记录数: {len(combined_df)}")
                logger.info(f"🏙️ 城市数: {len(cities)}")
                
                return filepath
            else:
                logger.warning("⚠️ 未获取到任何Visual Crossing数据")
                return None
            
        except Exception as e:
            logger.error(f"❌ 下载Visual Crossing数据失败: {e}")
            return None
    
    def generate_download_report(self, download_results: Dict) -> Dict:
        """生成下载报告"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_sources': len(download_results),
                    'successful_downloads': 0,
                    'failed_downloads': 0,
                    'total_records': 0
                },
                'download_details': download_results,
                'recommendations': []
            }
            
            # 统计结果
            for source, result in download_results.items():
                if result and os.path.exists(result):
                    report['summary']['successful_downloads'] += 1
                    
                    # 尝试统计记录数
                    try:
                        if result.endswith('.csv'):
                            df = pd.read_csv(result)
                            report['summary']['total_records'] += len(df)
                    except:
                        pass
                else:
                    report['summary']['failed_downloads'] += 1
            
            # 生成建议
            if report['summary']['successful_downloads'] > 0:
                report['recommendations'].append(f"成功下载 {report['summary']['successful_downloads']} 个数据源")
                report['recommendations'].append(f"总记录数: {report['summary']['total_records']}")
                report['recommendations'].append("建议使用这些真实数据重新训练模型")
            else:
                report['recommendations'].append("下载失败，需要检查网络连接和API访问")
                report['recommendations'].append("考虑使用其他数据源或API密钥")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ 生成下载报告失败: {e}")
            return {'status': 'error', 'error': str(e)}

def main():
    """主函数"""
    try:
        logger.info("🚀 启动真实数据下载...")
        
        # 创建下载器
        downloader = RealDataDownloader()
        
        # 下载各种数据源
        download_results = {}
        
        # 1. NOAA每日摘要数据
        logger.info("📥 1/5 下载NOAA每日摘要数据...")
        noaa_daily = downloader.download_noaa_daily_summaries()
        download_results['noaa_daily'] = noaa_daily
        
        # 2. NOAA小时数据
        logger.info("📥 2/5 下载NOAA小时数据...")
        noaa_hourly = downloader.download_noaa_hourly_data()
        download_results['noaa_hourly'] = noaa_hourly
        
        # 3. EOBS每日数据
        logger.info("📥 3/5 下载EOBS每日数据...")
        eobs_daily = downloader.download_eobs_daily_data()
        download_results['eobs_daily'] = eobs_daily
        
        # 4. Open-Meteo数据
        logger.info("📥 4/5 下载Open-Meteo数据...")
        openmeteo = downloader.download_openmeteo_data()
        download_results['openmeteo'] = openmeteo
        
        # 5. Visual Crossing数据
        logger.info("📥 5/5 下载Visual Crossing数据...")
        visualcrossing = downloader.download_visualcrossing_data()
        download_results['visualcrossing'] = visualcrossing
        
        # 生成下载报告
        logger.info("📊 生成下载报告...")
        report = downloader.generate_download_report(download_results)
        
        # 保存报告
        output_dir = "data/download_reports"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f"real_data_download_report_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ 下载报告已保存: {report_file}")
        
        # 显示结果摘要
        logger.info("🎉 真实数据下载完成！")
        logger.info(f"📊 下载摘要:")
        logger.info(f"  总数据源: {report['summary']['total_sources']}")
        logger.info(f"  成功下载: {report['summary']['successful_downloads']}")
        logger.info(f"  下载失败: {report['summary']['failed_downloads']}")
        logger.info(f"  总记录数: {report['summary']['total_records']}")
        
        # 显示建议
        for i, rec in enumerate(report['recommendations'], 1):
            logger.info(f"💡 建议 {i}: {rec}")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return None

if __name__ == "__main__":
    main()
