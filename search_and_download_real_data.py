#!/usr/bin/env python3
"""
搜索和下载真实数据源
替代合成数据，获取高质量真实观测数据
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

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealDataSearcher:
    """真实数据搜索器"""
    
    def __init__(self):
        """初始化"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def search_noaa_climate_data(self) -> Dict:
        """搜索NOAA气候数据"""
        try:
            logger.info("🔍 搜索NOAA气候数据...")
            
            # NOAA气候数据在线搜索
            noaa_urls = {
                'daily_summaries': 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/',
                'hourly_data': 'https://www.ncei.noaa.gov/data/global-hourly/access/',
                'precipitation': 'https://www.ncei.noaa.gov/data/global-precipitation-climatology-centre/access/',
                'soil_moisture': 'https://www.ncei.noaa.gov/data/soil-moisture/access/'
            }
            
            results = {}
            
            for data_type, url in noaa_urls.items():
                try:
                    logger.info(f"📥 检查 {data_type} 数据可用性...")
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        results[data_type] = {
                            'status': 'available',
                            'url': url,
                            'size': len(response.content)
                        }
                        logger.info(f"✅ {data_type} 数据可用")
                    else:
                        results[data_type] = {
                            'status': 'unavailable',
                            'url': url,
                            'error': f"HTTP {response.status_code}"
                        }
                        logger.info(f"❌ {data_type} 数据不可用: HTTP {response.status_code}")
                        
                except Exception as e:
                    results[data_type] = {
                        'status': 'error',
                        'url': url,
                        'error': str(e)
                    }
                    logger.warning(f"⚠️ {data_type} 数据检查失败: {e}")
                
                time.sleep(1)  # 避免请求过快
            
            return results
            
        except Exception as e:
            logger.error(f"❌ NOAA数据搜索失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def search_nasa_earth_data(self) -> Dict:
        """搜索NASA地球数据"""
        try:
            logger.info("🔍 搜索NASA地球数据...")
            
            # NASA Earthdata搜索
            nasa_urls = {
                'smap_soil_moisture': 'https://cmr.earthdata.nasa.gov/search/collections.json?keyword=SMAP&type=dataset',
                'modis_land': 'https://cmr.earthdata.nasa.gov/search/collections.json?keyword=MODIS&type=dataset',
                'grace_water': 'https://cmr.earthdata.nasa.gov/search/collections.json?keyword=GRACE&type=dataset'
            }
            
            results = {}
            
            for data_type, url in nasa_urls.items():
                try:
                    logger.info(f"📥 检查 {data_type} 数据可用性...")
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if 'feed' in data and 'entry' in data['feed']:
                            results[data_type] = {
                                'status': 'available',
                                'url': url,
                                'datasets': len(data['feed']['entry'])
                            }
                            logger.info(f"✅ {data_type} 数据可用: {len(data['feed']['entry'])} 个数据集")
                        else:
                            results[data_type] = {
                                'status': 'available',
                                'url': url,
                                'datasets': 'unknown'
                            }
                            logger.info(f"✅ {data_type} 数据可用")
                    else:
                        results[data_type] = {
                            'status': 'unavailable',
                            'url': url,
                            'error': f"HTTP {response.status_code}"
                        }
                        logger.info(f"❌ {data_type} 数据不可用: HTTP {response.status_code}")
                        
                except Exception as e:
                    results[data_type] = {
                        'status': 'error',
                        'url': url,
                        'error': str(e)
                    }
                    logger.warning(f"⚠️ {data_type} 数据检查失败: {e}")
                
                time.sleep(1)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ NASA数据搜索失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def search_european_weather_data(self) -> Dict:
        """搜索欧洲天气数据"""
        try:
            logger.info("🔍 搜索欧洲天气数据...")
            
            # ECMWF和欧洲天气数据
            european_urls = {
                'era5_reanalysis': 'https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels',
                'era5_land': 'https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land',
                'eobs_daily': 'https://www.ecad.eu/download/ensembles/download.php'
            }
            
            results = {}
            
            for data_type, url in european_urls.items():
                try:
                    logger.info(f"📥 检查 {data_type} 数据可用性...")
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        results[data_type] = {
                            'status': 'available',
                            'url': url,
                            'size': len(response.content)
                        }
                        logger.info(f"✅ {data_type} 数据可用")
                    else:
                        results[data_type] = {
                            'status': 'unavailable',
                            'url': url,
                            'error': f"HTTP {response.status_code}"
                        }
                        logger.info(f"❌ {data_type} 数据不可用: HTTP {response.status_code}")
                        
                except Exception as e:
                    results[data_type] = {
                        'status': 'error',
                        'url': url,
                        'error': str(e)
                    }
                    logger.warning(f"⚠️ {data_type} 数据检查失败: {e}")
                
                time.sleep(1)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 欧洲数据搜索失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def search_canadian_weather_data(self) -> Dict:
        """搜索加拿大天气数据"""
        try:
            logger.info("🔍 搜索加拿大天气数据...")
            
            # Environment Canada和加拿大天气数据
            canadian_urls = {
                'environment_canada': 'https://climate.weather.gc.ca/',
                'hydat_water': 'https://www.canada.ca/en/environment-climate-change/services/water-overview/quantity/monitoring/survey/data-products-services/national-archive-hydat.html',
                'agriculture_data': 'https://agriculture.canada.ca/en/agriculture-and-environment/weather-and-climate'
            }
            
            results = {}
            
            for data_type, url in canadian_urls.items():
                try:
                    logger.info(f"📥 检查 {data_type} 数据可用性...")
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        results[data_type] = {
                            'status': 'available',
                            'url': url,
                            'size': len(response.content)
                        }
                        logger.info(f"✅ {data_type} 数据可用")
                    else:
                        results[data_type] = {
                            'status': 'unavailable',
                            'url': url,
                            'error': f"HTTP {response.status_code}"
                        }
                        logger.info(f"❌ {data_type} 数据不可用: HTTP {response.status_code}")
                        
                except Exception as e:
                    results[data_type] = {
                        'status': 'error',
                        'url': url,
                        'error': str(e)
                    }
                    logger.warning(f"⚠️ {data_type} 数据检查失败: {e}")
                
                time.sleep(1)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 加拿大数据搜索失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def search_open_weather_datasets(self) -> Dict:
        """搜索开放天气数据集"""
        try:
            logger.info("🔍 搜索开放天气数据集...")
            
            # 开放天气数据平台
            open_weather_urls = {
                'openweathermap': 'https://openweathermap.org/api',
                'weatherbit': 'https://www.weatherbit.io/api',
                'visualcrossing': 'https://www.visualcrossing.com/weather-api',
                'openmeteo': 'https://open-meteo.com/en/docs'
            }
            
            results = {}
            
            for data_type, url in open_weather_urls.items():
                try:
                    logger.info(f"📥 检查 {data_type} 数据可用性...")
                    response = self.session.get(url, timeout=10)
                    
                    if response.status_code == 200:
                        results[data_type] = {
                            'status': 'available',
                            'url': url,
                            'size': len(response.content)
                        }
                        logger.info(f"✅ {data_type} 数据可用")
                    else:
                        results[data_type] = {
                            'status': 'unavailable',
                            'url': url,
                            'error': f"HTTP {response.status_code}"
                        }
                        logger.info(f"❌ {data_type} 数据不可用: HTTP {response.status_code}")
                        
                except Exception as e:
                    results[data_type] = {
                        'status': 'error',
                        'url': url,
                        'error': str(e)
                    }
                    logger.warning(f"⚠️ {data_type} 数据检查失败: {e}")
                
                time.sleep(1)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ 开放天气数据搜索失败: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def download_sample_data(self, data_source: str, url: str) -> Optional[str]:
        """下载样本数据"""
        try:
            logger.info(f"📥 尝试下载 {data_source} 样本数据...")
            
            # 创建下载目录
            download_dir = f"data/real_samples/{data_source}"
            os.makedirs(download_dir, exist_ok=True)
            
            # 尝试下载
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                # 保存数据
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{data_source}_sample_{timestamp}.txt"
                filepath = os.path.join(download_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                logger.info(f"✅ 样本数据已保存: {filepath}")
                return filepath
            else:
                logger.warning(f"⚠️ 下载失败: HTTP {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 下载样本数据失败: {e}")
            return None
    
    def generate_search_report(self, all_results: Dict) -> Dict:
        """生成搜索报告"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': {
                    'total_sources': len(all_results),
                    'available_sources': 0,
                    'unavailable_sources': 0,
                    'error_sources': 0
                },
                'recommendations': [],
                'data_sources': all_results
            }
            
            # 统计结果
            for source_type, results in all_results.items():
                for data_type, result in results.items():
                    if result['status'] == 'available':
                        report['summary']['available_sources'] += 1
                    elif result['status'] == 'unavailable':
                        report['summary']['unavailable_sources'] += 1
                    else:
                        report['summary']['error_sources'] += 1
            
            # 生成建议
            if report['summary']['available_sources'] > 0:
                report['recommendations'].append("发现多个可用数据源，建议优先使用")
                report['recommendations'].append("可以组合多个数据源提高数据质量")
            else:
                report['recommendations'].append("未发现可用数据源，需要进一步调查")
                report['recommendations'].append("考虑使用API密钥或注册账户")
            
            return report
            
        except Exception as e:
            logger.error(f"❌ 生成搜索报告失败: {e}")
            return {'status': 'error', 'error': str(e)}

def main():
    """主函数"""
    try:
        logger.info("🚀 启动真实数据源搜索...")
        
        # 创建搜索器
        searcher = RealDataSearcher()
        
        # 搜索各种数据源
        all_results = {}
        
        # 1. NOAA气候数据
        logger.info("🔍 1/5 搜索NOAA气候数据...")
        noaa_results = searcher.search_noaa_climate_data()
        all_results['noaa'] = noaa_results
        
        # 2. NASA地球数据
        logger.info("🔍 2/5 搜索NASA地球数据...")
        nasa_results = searcher.search_nasa_earth_data()
        all_results['nasa'] = nasa_results
        
        # 3. 欧洲天气数据
        logger.info("🔍 3/5 搜索欧洲天气数据...")
        european_results = searcher.search_european_weather_data()
        all_results['european'] = european_results
        
        # 4. 加拿大天气数据
        logger.info("🔍 4/5 搜索加拿大天气数据...")
        canadian_results = searcher.search_canadian_weather_data()
        all_results['canadian'] = canadian_results
        
        # 5. 开放天气数据集
        logger.info("🔍 5/5 搜索开放天气数据集...")
        open_weather_results = searcher.search_open_weather_datasets()
        all_results['open_weather'] = open_weather_results
        
        # 生成搜索报告
        logger.info("📊 生成搜索报告...")
        report = searcher.generate_search_report(all_results)
        
        # 保存报告
        output_dir = "data/search_reports"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = os.path.join(output_dir, f"real_data_search_report_{timestamp}.json")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✅ 搜索报告已保存: {report_file}")
        
        # 显示结果摘要
        logger.info("🎉 真实数据源搜索完成！")
        logger.info(f"📊 搜索摘要:")
        logger.info(f"  总数据源: {report['summary']['total_sources']}")
        logger.info(f"  可用数据源: {report['summary']['available_sources']}")
        logger.info(f"  不可用数据源: {report['summary']['unavailable_sources']}")
        logger.info(f"  错误数据源: {report['summary']['error_sources']}")
        
        # 显示建议
        for i, rec in enumerate(report['recommendations'], 1):
            logger.info(f"💡 建议 {i}: {rec}")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ 主函数执行失败: {e}")
        return None

if __name__ == "__main__":
    main()
