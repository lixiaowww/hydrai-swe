#!/usr/bin/env python3
"""
WeatherAPI.com 全球天气数据获取器
- 获取全球范围的实时和预报天气数据
- 支持当前天气、3日预报、历史数据
- 免费额度: 100万次调用/月
- 集成到HydrAI-SWE数据管道系统

数据来源: WeatherAPI.com
更新频率: 实时(每10分钟)
覆盖区域: 全球

使用示例:
    python3 weatherapi_fetch.py --api-key YOUR_API_KEY --locations "49.9,-97.2;49.91,-99.95"
    
环境变量:
    WEATHER_API_KEY: WeatherAPI.com API密钥
"""

import os
import sys
import json
import csv
import requests
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Manitoba地区预定义位置
MANITOBA_LOCATIONS = {
    'winnipeg': {
        'name': 'Winnipeg',
        'coordinates': (49.9, -97.2394),
        'region': 'Manitoba, Canada'
    },
    'brandon': {
        'name': 'Brandon', 
        'coordinates': (49.91, -99.9519),
        'region': 'Manitoba, Canada'
    },
    'thompson': {
        'name': 'Thompson',
        'coordinates': (55.8011, -97.8642),
        'region': 'Manitoba, Canada'
    },
    'churchill': {
        'name': 'Churchill',
        'coordinates': (58.7684, -94.1647),
        'region': 'Manitoba, Canada'
    },
    'flin_flon': {
        'name': 'Flin Flon',
        'coordinates': (54.7682, -101.8651),
        'region': 'Manitoba, Canada'
    },
    'portage_la_prairie': {
        'name': 'Portage La Prairie',
        'coordinates': (49.9736, -98.2914),
        'region': 'Manitoba, Canada'
    }
}

class WeatherAPIFetcher:
    """WeatherAPI.com天气数据获取器"""
    
    def __init__(self, api_key: str, output_dir: str):
        self.api_key = api_key
        self.output_dir = output_dir
        self.base_url = "https://api.weatherapi.com/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'HydrAI-SWE/1.0 (Climate Research)'
        })
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # API调用限制 (WeatherAPI.com限制更宽松)
        self.call_interval = 0.1  # 秒
        self.last_call_time = 0
    
    def _rate_limit(self):
        """API调用频率限制"""
        current_time = time.time()
        time_since_last_call = current_time - self.last_call_time
        
        if time_since_last_call < self.call_interval:
            sleep_time = self.call_interval - time_since_last_call
            time.sleep(sleep_time)
        
        self.last_call_time = time.time()
    
    def fetch_current_weather(self, lat: float, lon: float, location_name: str = None) -> Optional[Dict]:
        """获取指定坐标的当前天气数据"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/current.json"
            params = {
                'key': self.api_key,
                'q': f"{lat},{lon}",
                'aqi': 'yes'  # 包含空气质量数据
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # 格式化数据
            weather_data = self._format_current_weather(data, lat, lon, location_name)
            
            self.logger.info(f"✅ 成功获取 {location_name or f'({lat}, {lon})'} 当前天气")
            return weather_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ 获取 {location_name} 当前天气失败: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ 处理 {location_name} 天气数据失败: {e}")
            return None
    
    def fetch_3day_forecast(self, lat: float, lon: float, location_name: str = None) -> Optional[List[Dict]]:
        """获取3日天气预报"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/forecast.json"
            params = {
                'key': self.api_key,
                'q': f"{lat},{lon}",
                'days': 3,  # WeatherAPI.com免费版最多3天
                'aqi': 'yes',  # 包含空气质量预报
                'alerts': 'yes'  # 包含天气警告
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # 格式化预报数据
            forecast_data = self._format_forecast_data(data, lat, lon, location_name)
            
            self.logger.info(f"✅ 成功获取 {location_name or f'({lat}, {lon})'} 3日预报")
            return forecast_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ 获取 {location_name} 3日预报失败: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ 处理 {location_name} 预报数据失败: {e}")
            return None
    
    def _format_current_weather(self, data: Dict, lat: float, lon: float, location_name: str) -> Dict:
        """格式化当前天气数据"""
        location = data.get('location', {})
        current = data.get('current', {})
        condition = current.get('condition', {})
        air_quality = current.get('air_quality', {})
        
        return {
            'location_name': location_name or location.get('name', 'Unknown'),
            'coordinates': (lat, lon),
            'country': location.get('country', ''),
            'region': location.get('region', ''),
            'timezone': location.get('tz_id', ''),
            'local_time': location.get('localtime', ''),
            'temperature_c': current.get('temp_c'),
            'feels_like_c': current.get('feelslike_c'),
            'temperature_f': current.get('temp_f'),
            'feels_like_f': current.get('feelslike_f'),
            'pressure_mb': current.get('pressure_mb'),
            'pressure_in': current.get('pressure_in'),
            'humidity_percent': current.get('humidity'),
            'visibility_km': current.get('vis_km'),
            'visibility_miles': current.get('vis_miles'),
            'uv_index': current.get('uv'),
            'wind_speed_kph': current.get('wind_kph'),
            'wind_speed_mph': current.get('wind_mph'),
            'wind_direction_deg': current.get('wind_degree'),
            'wind_direction': current.get('wind_dir'),
            'wind_gust_kph': current.get('gust_kph'),
            'wind_gust_mph': current.get('gust_mph'),
            'cloud_cover_percent': current.get('cloud'),
            'weather_condition': condition.get('text'),
            'weather_icon': condition.get('icon'),
            'weather_code': condition.get('code'),
            'precipitation_mm': current.get('precip_mm'),
            'precipitation_in': current.get('precip_in'),
            'is_day': current.get('is_day') == 1,
            # 空气质量数据
            'air_quality': {
                'co': air_quality.get('co'),
                'no2': air_quality.get('no2'),
                'o3': air_quality.get('o3'),
                'so2': air_quality.get('so2'),
                'pm2_5': air_quality.get('pm2_5'),
                'pm10': air_quality.get('pm10'),
                'us_epa_index': air_quality.get('us-epa-index'),
                'gb_defra_index': air_quality.get('gb-defra-index')
            },
            'last_updated': current.get('last_updated'),
            'fetch_time_utc': datetime.utcnow().isoformat(),
            'data_source': 'WeatherAPI.com',
            'api_version': 'v1'
        }
    
    def _format_forecast_data(self, data: Dict, lat: float, lon: float, location_name: str) -> List[Dict]:
        """格式化预报数据"""
        forecast_list = []
        
        location = data.get('location', {})
        forecast_days = data.get('forecast', {}).get('forecastday', [])
        
        for forecast_day in forecast_days:
            day_data = forecast_day.get('day', {})
            astro_data = forecast_day.get('astro', {})
            condition = day_data.get('condition', {})
            
            # 处理小时预报
            hourly_forecasts = []
            for hour_data in forecast_day.get('hour', []):
                hour_condition = hour_data.get('condition', {})
                hourly_forecasts.append({
                    'time': hour_data.get('time'),
                    'temperature_c': hour_data.get('temp_c'),
                    'feels_like_c': hour_data.get('feelslike_c'),
                    'condition': hour_condition.get('text'),
                    'wind_kph': hour_data.get('wind_kph'),
                    'humidity': hour_data.get('humidity'),
                    'cloud': hour_data.get('cloud'),
                    'precipitation_mm': hour_data.get('precip_mm'),
                    'chance_of_rain': hour_data.get('chance_of_rain'),
                    'chance_of_snow': hour_data.get('chance_of_snow')
                })
            
            forecast_item = {
                'location_name': location_name or location.get('name', 'Unknown'),
                'coordinates': (lat, lon),
                'date': forecast_day.get('date'),
                'temperature_max_c': day_data.get('maxtemp_c'),
                'temperature_min_c': day_data.get('mintemp_c'),
                'temperature_avg_c': day_data.get('avgtemp_c'),
                'max_wind_kph': day_data.get('maxwind_kph'),
                'total_precipitation_mm': day_data.get('totalprecip_mm'),
                'total_snow_cm': day_data.get('totalsnow_cm'),
                'avg_visibility_km': day_data.get('avgvis_km'),
                'avg_humidity': day_data.get('avghumidity'),
                'daily_chance_of_rain': day_data.get('daily_chance_of_rain'),
                'daily_chance_of_snow': day_data.get('daily_chance_of_snow'),
                'condition': condition.get('text'),
                'condition_icon': condition.get('icon'),
                'condition_code': condition.get('code'),
                'uv_index': day_data.get('uv'),
                # 天文数据
                'sunrise': astro_data.get('sunrise'),
                'sunset': astro_data.get('sunset'),
                'moonrise': astro_data.get('moonrise'),
                'moonset': astro_data.get('moonset'),
                'moon_phase': astro_data.get('moon_phase'),
                'moon_illumination': astro_data.get('moon_illumination'),
                # 小时预报
                'hourly_forecast': hourly_forecasts,
                'data_source': 'WeatherAPI.com_Forecast',
                'fetch_time_utc': datetime.utcnow().isoformat()
            }
            
            forecast_list.append(forecast_item)
        
        return forecast_list
    
    def fetch_all_manitoba_weather(self, include_forecast: bool = True) -> Dict:
        """获取所有Manitoba地区的天气数据"""
        all_weather_data = {
            'current_weather': {},
            'forecasts': {},
            'metadata': {
                'fetch_time': datetime.utcnow().isoformat(),
                'locations_count': len(MANITOBA_LOCATIONS),
                'data_source': 'WeatherAPI.com',
                'include_forecast': include_forecast
            }
        }
        
        # 使用线程池并发获取数据
        with ThreadPoolExecutor(max_workers=5) as executor:  # WeatherAPI.com限制更宽松
            futures = {}
            
            # 提交当前天气获取任务
            for location_key, location_info in MANITOBA_LOCATIONS.items():
                lat, lon = location_info['coordinates']
                
                # 当前天气
                future_current = executor.submit(
                    self.fetch_current_weather, 
                    lat, lon, location_info['name']
                )
                futures[f"current_{location_key}"] = future_current
                
                # 3日预报
                if include_forecast:
                    future_forecast = executor.submit(
                        self.fetch_3day_forecast,
                        lat, lon, location_info['name']
                    )
                    futures[f"forecast_{location_key}"] = future_forecast
            
            # 收集结果
            for future_key, future in futures.items():
                try:
                    result = future.result(timeout=30)
                    
                    if future_key.startswith('current_'):
                        location_key = future_key.replace('current_', '')
                        if result:
                            all_weather_data['current_weather'][location_key] = result
                    
                    elif future_key.startswith('forecast_'):
                        location_key = future_key.replace('forecast_', '')
                        if result:
                            all_weather_data['forecasts'][location_key] = result
                            
                except Exception as e:
                    self.logger.error(f"❌ 获取 {future_key} 数据失败: {e}")
        
        return all_weather_data
    
    def save_weather_data(self, weather_data: Dict) -> Tuple[str, str]:
        """保存天气数据到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON格式
        json_filename = f"weatherapi_{timestamp}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(weather_data, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式 (仅当前天气数据)
        csv_filename = f"weatherapi_current_{timestamp}.csv"
        csv_path = os.path.join(self.output_dir, csv_filename)
        
        current_weather = weather_data.get('current_weather', {})
        if current_weather:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # CSV标题行
                headers = [
                    'location_name', 'lat', 'lon', 'temperature_c', 'feels_like_c',
                    'humidity_percent', 'pressure_mb', 'wind_speed_kph', 'wind_direction_deg',
                    'weather_condition', 'cloud_cover_percent', 'uv_index', 'precipitation_mm',
                    'visibility_km', 'last_updated'
                ]
                writer.writerow(headers)
                
                # 数据行
                for location_key, data in current_weather.items():
                    lat, lon = data.get('coordinates', (0, 0))
                    row = [
                        data.get('location_name', ''),
                        lat, lon,
                        data.get('temperature_c', ''),
                        data.get('feels_like_c', ''),
                        data.get('humidity_percent', ''),
                        data.get('pressure_mb', ''),
                        data.get('wind_speed_kph', ''),
                        data.get('wind_direction_deg', ''),
                        data.get('weather_condition', ''),
                        data.get('cloud_cover_percent', ''),
                        data.get('uv_index', ''),
                        data.get('precipitation_mm', ''),
                        data.get('visibility_km', ''),
                        data.get('last_updated', '')
                    ]
                    writer.writerow(row)
        
        return json_path, csv_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='获取WeatherAPI.com天气数据')
    parser.add_argument('--api-key', '-k', 
                      default=os.getenv('WEATHER_API_KEY'),
                      help='WeatherAPI.com API密钥')
    parser.add_argument('--output', '-o', default='data/raw/openweather',
                      help='输出目录路径')
    parser.add_argument('--locations', '-l',
                      help='自定义位置坐标，格式: "lat1,lon1;lat2,lon2"')
    parser.add_argument('--no-forecast', action='store_true',
                      help='不获取预报数据')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='详细日志输出')
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("❌ 错误: 需要WeatherAPI.com API密钥")
        print("   可通过 --api-key 参数或 WEATHER_API_KEY 环境变量提供")
        print("   在 https://www.weatherapi.com/ 获取免费API密钥")
        return 1
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🌍 启动WeatherAPI.com天气数据获取...")
    print(f"💾 输出目录: {args.output}")
    print(f"📊 预报数据: {'否' if args.no_forecast else '是'}")
    
    # 创建获取器
    fetcher = WeatherAPIFetcher(args.api_key, args.output)
    
    # 获取天气数据
    start_time = time.time()
    
    if args.locations:
        # 处理自定义位置
        print("🎯 使用自定义位置")
        # 这里可以扩展处理自定义位置的逻辑
        weather_data = {'current_weather': {}, 'forecasts': {}}
    else:
        # 获取Manitoba地区数据
        print("📍 获取Manitoba地区天气数据...")
        weather_data = fetcher.fetch_all_manitoba_weather(
            include_forecast=not args.no_forecast
        )
    
    fetch_duration = time.time() - start_time
    
    if weather_data and weather_data['current_weather']:
        success_count = len(weather_data['current_weather'])
        print(f"✅ 成功获取 {success_count} 个位置的天气数据")
        
        # 保存数据
        json_path, csv_path = fetcher.save_weather_data(weather_data)
        
        print(f"💾 数据已保存:")
        print(f"   📄 JSON: {json_path}")
        print(f"   📊 CSV:  {csv_path}")
        print(f"⏱️  获取耗时: {fetch_duration:.2f}秒")
        
        # 显示数据摘要
        print(f"\n📊 当前天气摘要:")
        for location_key, data in weather_data['current_weather'].items():
            temp = data.get('temperature_c', 'n/a')
            condition = data.get('weather_condition', 'n/a')
            name = data.get('location_name', location_key)
            print(f"   🌡️  {name}: {temp}°C, {condition}")
        
        if weather_data.get('forecasts'):
            print(f"\n🔮 预报数据已获取 ({len(weather_data['forecasts'])} 个位置)")
        
    else:
        print("❌ 未能获取任何天气数据")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
