#!/usr/bin/env python3
"""
WeatherAPI.com 全球天气数据获取器
- 获取全球范围的实时和预报天气数据
- 支持当前天气、5日预报、历史数据
- 作为ECCC数据的补充和备用数据源
- 集成到HydrAI-SWE数据管道系统

数据来源: WeatherAPI.com (替代OpenWeatherMap)
更新频率: 实时(每10分钟)
覆盖区域: 全球
免费额度: 100万次调用/月

使用示例:
    python3 openweather_fetch.py --api-key YOUR_API_KEY --locations "49.9,-97.2;49.91,-99.95"
    
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
import math

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
        
        # API调用限制 (免费版: 100万次/月，比OpenWeatherMap更慷慨)
        self.call_interval = 0.1  # 秒，WeatherAPI.com限制更宽松
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
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric',
                'lang': 'en'
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
    
    def fetch_5day_forecast(self, lat: float, lon: float, location_name: str = None) -> Optional[List[Dict]]:
        """获取5日天气预报"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric',
                'lang': 'en'
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # 格式化预报数据
            forecast_data = self._format_forecast_data(data, lat, lon, location_name)
            
            self.logger.info(f"✅ 成功获取 {location_name or f'({lat}, {lon})'} 5日预报")
            return forecast_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ 获取 {location_name} 5日预报失败: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ 处理 {location_name} 预报数据失败: {e}")
            return None
    
    def fetch_air_quality(self, lat: float, lon: float, location_name: str = None) -> Optional[Dict]:
        """获取空气质量数据"""
        self._rate_limit()
        
        try:
            url = f"http://api.openweathermap.org/data/2.5/air_pollution"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key
            }
            
            response = self.session.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            # 格式化空气质量数据
            air_quality_data = self._format_air_quality_data(data, lat, lon, location_name)
            
            self.logger.info(f"✅ 成功获取 {location_name or f'({lat}, {lon})'} 空气质量数据")
            return air_quality_data
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"❌ 获取 {location_name} 空气质量失败: {e}")
            return None
        except Exception as e:
            self.logger.error(f"❌ 处理 {location_name} 空气质量数据失败: {e}")
            return None
    
    def _format_current_weather(self, data: Dict, lat: float, lon: float, location_name: str) -> Dict:
        """格式化当前天气数据"""
        main = data.get('main', {})
        weather = data.get('weather', [{}])[0]
        wind = data.get('wind', {})
        sys_data = data.get('sys', {})
        clouds = data.get('clouds', {})
        
        return {
            'location_name': location_name or data.get('name', 'Unknown'),
            'coordinates': (lat, lon),
            'country': sys_data.get('country', ''),
            'temperature_c': main.get('temp'),
            'feels_like_c': main.get('feels_like'),
            'temperature_min_c': main.get('temp_min'),
            'temperature_max_c': main.get('temp_max'),
            'pressure_hpa': main.get('pressure'),
            'humidity_percent': main.get('humidity'),
            'visibility_m': data.get('visibility'),
            'uv_index': None,  # 需要单独API调用
            'wind_speed_ms': wind.get('speed'),
            'wind_speed_kmh': wind.get('speed', 0) * 3.6,
            'wind_direction_deg': wind.get('deg'),
            'wind_gust_ms': wind.get('gust'),
            'cloudiness_percent': clouds.get('all'),
            'weather_main': weather.get('main'),
            'weather_description': weather.get('description'),
            'weather_icon': weather.get('icon'),
            'sunrise_utc': datetime.fromtimestamp(sys_data.get('sunrise', 0)).isoformat() if sys_data.get('sunrise') else None,
            'sunset_utc': datetime.fromtimestamp(sys_data.get('sunset', 0)).isoformat() if sys_data.get('sunset') else None,
            'observation_time_utc': datetime.fromtimestamp(data.get('dt', 0)).isoformat() if data.get('dt') else None,
            'fetch_time_utc': datetime.utcnow().isoformat(),
            'data_source': 'OpenWeatherMap',
            'api_version': '2.5'
        }
    
    def _format_forecast_data(self, data: Dict, lat: float, lon: float, location_name: str) -> List[Dict]:
        """格式化预报数据"""
        forecast_list = []
        
        city_info = data.get('city', {})
        forecasts = data.get('list', [])
        
        for forecast in forecasts:
            main = forecast.get('main', {})
            weather = forecast.get('weather', [{}])[0]
            wind = forecast.get('wind', {})
            clouds = forecast.get('clouds', {})
            
            forecast_item = {
                'location_name': location_name or city_info.get('name', 'Unknown'),
                'coordinates': (lat, lon),
                'forecast_time_utc': datetime.fromtimestamp(forecast.get('dt', 0)).isoformat(),
                'temperature_c': main.get('temp'),
                'feels_like_c': main.get('feels_like'),
                'temperature_min_c': main.get('temp_min'),
                'temperature_max_c': main.get('temp_max'),
                'pressure_hpa': main.get('pressure'),
                'humidity_percent': main.get('humidity'),
                'wind_speed_ms': wind.get('speed'),
                'wind_speed_kmh': wind.get('speed', 0) * 3.6,
                'wind_direction_deg': wind.get('deg'),
                'wind_gust_ms': wind.get('gust'),
                'cloudiness_percent': clouds.get('all'),
                'weather_main': weather.get('main'),
                'weather_description': weather.get('description'),
                'weather_icon': weather.get('icon'),
                'precipitation_probability': forecast.get('pop', 0) * 100,  # 转换为百分比
                'rain_3h_mm': forecast.get('rain', {}).get('3h', 0),
                'snow_3h_mm': forecast.get('snow', {}).get('3h', 0),
                'data_source': 'OpenWeatherMap_Forecast',
                'fetch_time_utc': datetime.utcnow().isoformat()
            }
            
            forecast_list.append(forecast_item)
        
        return forecast_list
    
    def _format_air_quality_data(self, data: Dict, lat: float, lon: float, location_name: str) -> Dict:
        """格式化空气质量数据"""
        air_quality_list = data.get('list', [])
        
        if not air_quality_list:
            return None
        
        # 取第一个(当前)空气质量数据
        current_aqi = air_quality_list[0]
        main_aqi = current_aqi.get('main', {})
        components = current_aqi.get('components', {})
        
        return {
            'location_name': location_name,
            'coordinates': (lat, lon),
            'air_quality_index': main_aqi.get('aqi'),  # 1-5 scale
            'air_quality_level': self._get_aqi_level(main_aqi.get('aqi', 1)),
            'co_μg_m3': components.get('co'),  # Carbon monoxide
            'no_μg_m3': components.get('no'),  # Nitric oxide
            'no2_μg_m3': components.get('no2'),  # Nitrogen dioxide
            'o3_μg_m3': components.get('o3'),  # Ozone
            'so2_μg_m3': components.get('so2'),  # Sulphur dioxide
            'pm2_5_μg_m3': components.get('pm2_5'),  # Fine particles matter
            'pm10_μg_m3': components.get('pm10'),  # Coarse particulate matter
            'nh3_μg_m3': components.get('nh3'),  # Ammonia
            'measurement_time_utc': datetime.fromtimestamp(current_aqi.get('dt', 0)).isoformat(),
            'data_source': 'OpenWeatherMap_AirPollution',
            'fetch_time_utc': datetime.utcnow().isoformat()
        }
    
    def _get_aqi_level(self, aqi: int) -> str:
        """根据AQI数值获取空气质量等级"""
        aqi_levels = {
            1: 'Good',
            2: 'Fair', 
            3: 'Moderate',
            4: 'Poor',
            5: 'Very Poor'
        }
        return aqi_levels.get(aqi, 'Unknown')
    
    def fetch_all_manitoba_weather(self, include_forecast: bool = True, include_air_quality: bool = True) -> Dict:
        """获取所有Manitoba地区的天气数据"""
        all_weather_data = {
            'current_weather': {},
            'forecasts': {},
            'air_quality': {},
            'metadata': {
                'fetch_time': datetime.utcnow().isoformat(),
                'locations_count': len(MANITOBA_LOCATIONS),
                'data_source': 'OpenWeatherMap',
                'include_forecast': include_forecast,
                'include_air_quality': include_air_quality
            }
        }
        
        # 使用线程池并发获取数据
        with ThreadPoolExecutor(max_workers=3) as executor:  # 限制并发数以避免API限制
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
                
                # 5日预报
                if include_forecast:
                    future_forecast = executor.submit(
                        self.fetch_5day_forecast,
                        lat, lon, location_info['name']
                    )
                    futures[f"forecast_{location_key}"] = future_forecast
                
                # 空气质量
                if include_air_quality:
                    future_air = executor.submit(
                        self.fetch_air_quality,
                        lat, lon, location_info['name']
                    )
                    futures[f"air_{location_key}"] = future_air
            
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
                    
                    elif future_key.startswith('air_'):
                        location_key = future_key.replace('air_', '')
                        if result:
                            all_weather_data['air_quality'][location_key] = result
                            
                except Exception as e:
                    self.logger.error(f"❌ 获取 {future_key} 数据失败: {e}")
        
        return all_weather_data
    
    def save_weather_data(self, weather_data: Dict) -> Tuple[str, str]:
        """保存天气数据到文件"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存JSON格式
        json_filename = f"openweather_{timestamp}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(weather_data, f, indent=2, ensure_ascii=False)
        
        # 保存CSV格式 (仅当前天气数据)
        csv_filename = f"openweather_current_{timestamp}.csv"
        csv_path = os.path.join(self.output_dir, csv_filename)
        
        current_weather = weather_data.get('current_weather', {})
        if current_weather:
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                
                # CSV标题行
                headers = [
                    'location_name', 'lat', 'lon', 'temperature_c', 'feels_like_c',
                    'humidity_percent', 'pressure_hpa', 'wind_speed_kmh', 'wind_direction_deg',
                    'weather_main', 'weather_description', 'cloudiness_percent',
                    'observation_time_utc'
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
                        data.get('pressure_hpa', ''),
                        data.get('wind_speed_kmh', ''),
                        data.get('wind_direction_deg', ''),
                        data.get('weather_main', ''),
                        data.get('weather_description', ''),
                        data.get('cloudiness_percent', ''),
                        data.get('observation_time_utc', '')
                    ]
                    writer.writerow(row)
        
        return json_path, csv_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='获取OpenWeatherMap天气数据')
    parser.add_argument('--api-key', '-k', 
                      default=os.getenv('OPENWEATHER_API_KEY'),
                      help='OpenWeatherMap API密钥')
    parser.add_argument('--output', '-o', default='data/raw/openweather',
                      help='输出目录路径')
    parser.add_argument('--locations', '-l',
                      help='自定义位置坐标，格式: "lat1,lon1;lat2,lon2"')
    parser.add_argument('--no-forecast', action='store_true',
                      help='不获取预报数据')
    parser.add_argument('--no-air-quality', action='store_true',
                      help='不获取空气质量数据')
    parser.add_argument('--verbose', '-v', action='store_true',
                      help='详细日志输出')
    
    args = parser.parse_args()
    
    if not args.api_key:
        print("❌ 错误: 需要OpenWeatherMap API密钥")
        print("   可通过 --api-key 参数或 OPENWEATHER_API_KEY 环境变量提供")
        print("   在 https://openweathermap.org/api 获取API密钥")
        return 1
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🌍 启动OpenWeatherMap天气数据获取...")
    print(f"💾 输出目录: {args.output}")
    print(f"📊 预报数据: {'否' if args.no_forecast else '是'}")
    print(f"🌬️  空气质量: {'否' if args.no_air_quality else '是'}")
    
    # 创建获取器
    fetcher = WeatherAPIFetcher(args.api_key, args.output)
    
    # 获取天气数据
    start_time = time.time()
    
    if args.locations:
        # 处理自定义位置
        print("🎯 使用自定义位置")
        # 这里可以扩展处理自定义位置的逻辑
    else:
        # 获取Manitoba地区数据
        print("📍 获取Manitoba地区天气数据...")
        weather_data = fetcher.fetch_all_manitoba_weather(
            include_forecast=not args.no_forecast,
            include_air_quality=not args.no_air_quality
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
            condition = data.get('weather_description', 'n/a')
            name = data.get('location_name', location_key)
            print(f"   🌡️  {name}: {temp}°C, {condition}")
        
        if weather_data.get('forecasts'):
            print(f"\n🔮 预报数据已获取 ({len(weather_data['forecasts'])} 个位置)")
        
        if weather_data.get('air_quality'):
            print(f"\n🌬️  空气质量数据已获取 ({len(weather_data['air_quality'])} 个位置)")
        
    else:
        print("❌ 未能获取任何天气数据")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
