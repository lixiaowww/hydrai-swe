"""
真实天气数据API路由
获取曼省各城市的实时天气数据，替换模拟数据
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import httpx
import asyncio
import logging
from datetime import datetime
import json
import os

# 配置日志
logger = logging.getLogger(__name__)

router = APIRouter()

# 曼省主要城市配置
MANITOBA_CITIES = {
    'Winnipeg': {'lat': 49.8951, 'lon': -97.1384, 'name': '温尼伯', 'displayName': 'Winnipeg'},
    'Brandon': {'lat': 49.8483, 'lon': -99.9530, 'name': '布兰登', 'displayName': 'Brandon'},
    'Thompson': {'lat': 55.7435, 'lon': -97.8551, 'name': '汤普森', 'displayName': 'Thompson'},
    'Steinbach': {'lat': 49.5253, 'lon': -96.6845, 'name': '斯坦巴赫', 'displayName': 'Steinbach'},
    'Portage_la_Prairie': {'lat': 49.9728, 'lon': -98.2926, 'name': '草原港', 'displayName': 'Portage la Prairie'},
    'Selkirk': {'lat': 50.1439, 'lon': -96.8839, 'name': '塞尔扣克', 'displayName': 'Selkirk'},
    'Dauphin': {'lat': 51.1454, 'lon': -100.0506, 'name': '多芬', 'displayName': 'Dauphin'},
    'Flin_Flon': {'lat': 54.7682, 'lon': -101.8647, 'name': '弗林弗伦', 'displayName': 'Flin Flon'}
}

class WeatherData(BaseModel):
    city: str
    temperature: float
    feels_like: float
    humidity: int
    precipitation: float
    precipitation_1h: Optional[float] = None
    precipitation_3h: Optional[float] = None
    wind_speed: float
    wind_direction: int
    pressure: float
    visibility: Optional[float] = None
    weather_main: str
    weather_description: str
    cloud_cover: int
    uv_index: Optional[float] = None
    lat: float
    lon: float
    data_quality: int
    status: str
    last_updated: str

class SystemMetrics(BaseModel):
    active_stations: int
    data_quality_avg: float
    avg_temperature: float
    total_precipitation: float
    last_updated: str
    data_sources: List[str]

# WeatherAPI.com 配置 - 免费服务，每月100万次调用
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # 从环境变量获取
WEATHER_API_BASE_URL = "https://api.weatherapi.com/v1"

# OpenWeatherMap API 配置 (主要数据源)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    # 使用有效的OpenWeatherMap API密钥
    OPENWEATHER_API_KEY = "74e70f7d7b079d2308f516716ffc1b06"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"

# 如果没有配置WeatherAPI key，使用OpenWeatherMap作为主要数据源
if not WEATHER_API_KEY:
    WEATHER_API_KEY = None  # 不使用无效的演示密钥

async def get_weatherapi_data(lat: float, lon: float, city: str) -> Optional[WeatherData]:
    """从WeatherAPI.com获取天气数据"""
    if not WEATHER_API_KEY:
        logger.warning("WeatherAPI.com key not configured")
        return None
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # WeatherAPI.com 当前天气端点
            current_url = f"{WEATHER_API_BASE_URL}/current.json"
            params = {
                "key": WEATHER_API_KEY,
                "q": f"{lat},{lon}",
                "aqi": "no"  # 不需要空气质量数据
            }
            
            response = await client.get(current_url, params=params)
            if response.status_code == 200:
                data = response.json()
                current = data['current']
                location = data['location']
                
                # WeatherAPI返回的降水数据
                precipitation = current.get('precip_mm', 0.0)
                
                return WeatherData(
                    city=city,
                    temperature=round(current['temp_c'], 1),
                    feels_like=round(current['feelslike_c'], 1),
                    humidity=current['humidity'],
                    precipitation=round(precipitation, 1),
                    precipitation_1h=round(precipitation, 1) if precipitation > 0 else None,
                    wind_speed=round(current['wind_kph'], 1),  # 已经是km/h
                    wind_direction=current['wind_degree'],
                    pressure=current['pressure_mb'],
                    visibility=current.get('vis_km', 10.0),
                    weather_main=current['condition']['text'].split()[0],  # 取第一个词
                    weather_description=current['condition']['text'],
                    cloud_cover=current['cloud'],
                    uv_index=current.get('uv'),
                    lat=lat,
                    lon=lon,
                    data_quality=97,  # WeatherAPI.com 质量很高
                    status="Online",
                    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
            elif response.status_code == 401:
                logger.error("WeatherAPI.com: Invalid API key")
            elif response.status_code == 400:
                logger.error(f"WeatherAPI.com: Invalid location {lat},{lon}")
            else:
                logger.error(f"WeatherAPI.com: HTTP {response.status_code}")
                
    except Exception as e:
        logger.error(f"Error fetching WeatherAPI.com data for {city}: {e}")
    
    return None

async def get_openweather_data(lat: float, lon: float, city: str) -> Optional[WeatherData]:
    """从OpenWeatherMap获取天气数据"""
    if not OPENWEATHER_API_KEY:
        logger.warning("OpenWeatherMap API key not configured")
        return None
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 当前天气
            current_url = f"{OPENWEATHER_BASE_URL}/weather"
            params = {
                "lat": lat,
                "lon": lon,
                "appid": OPENWEATHER_API_KEY,
                "units": "metric",
                "lang": "en"
            }
            
            response = await client.get(current_url, params=params)
            if response.status_code == 200:
                data = response.json()
                
                # 处理降水数据
                precipitation = 0.0
                rain_1h = data.get('rain', {}).get('1h', 0.0) if 'rain' in data else 0.0
                snow_1h = data.get('snow', {}).get('1h', 0.0) if 'snow' in data else 0.0
                precipitation = rain_1h + snow_1h  # 雨雪总量
                
                return WeatherData(
                    city=city,
                    temperature=round(data['main']['temp'], 1),
                    feels_like=round(data['main']['feels_like'], 1),
                    humidity=data['main']['humidity'],
                    precipitation=round(precipitation, 1),
                    precipitation_1h=round(precipitation, 1) if precipitation > 0 else None,
                    wind_speed=round(data['wind']['speed'] * 3.6, 1),  # m/s to km/h
                    wind_direction=data['wind'].get('deg', 0),
                    pressure=data['main']['pressure'],
                    visibility=data.get('visibility', 10000) / 1000,  # meters to km
                    weather_main=data['weather'][0]['main'],
                    weather_description=data['weather'][0]['description'].title(),
                    cloud_cover=data['clouds']['all'],
                    lat=lat,
                    lon=lon,
                    data_quality=95,  # OpenWeatherMap 通常质量很高
                    status="Online",
                    last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                )
                
    except Exception as e:
        logger.error(f"Error fetching OpenWeather data for {city}: {e}")
    
    return None

# 删除fallback函数 - 严格遵循规则：不使用模拟数据

@router.get("/weather/cities", response_model=Dict[str, WeatherData])
async def get_cities_weather():
    """获取曼省所有城市的实时天气数据 - 仅返回真实数据"""
    try:
        weather_data = {}
        
        # 并发获取所有城市的天气数据
        tasks = []
        for city_key, city_info in MANITOBA_CITIES.items():
            tasks.append(get_real_weather_only(city_key, city_info))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (city_key, city_info) in enumerate(MANITOBA_CITIES.items()):
            result = results[i]
            if isinstance(result, Exception):
                logger.error(f"Error getting weather for {city_key}: {result}")
                # 返回n/a数据而不是模拟数据
                weather_data[city_key] = get_na_weather_data(city_info['displayName'], city_info['lat'], city_info['lon'])
            else:
                weather_data[city_key] = result
        
        return weather_data
        
    except Exception as e:
        logger.error(f"Error in get_cities_weather: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch weather data")

async def get_real_weather_only(city_key: str, city_info: dict) -> WeatherData:
    """仅获取真实天气数据，失败时抛出异常"""
    city_name = city_info['displayName']
    lat = city_info['lat']
    lon = city_info['lon']
    
    # 优先尝试OpenWeatherMap（更稳定可靠）
    real_data = await get_openweather_data(lat, lon, city_name)
    if real_data:
        return real_data
    
    # 备用：尝试WeatherAPI.com
    real_data = await get_weatherapi_data(lat, lon, city_name)
    if real_data:
        return real_data
    
    # 如果所有真实API都不可用，抛出异常
    raise Exception(f"No real weather data available for {city_name}")

def get_na_weather_data(city: str, lat: float, lon: float) -> WeatherData:
    """返回n/a数据结构"""
    return WeatherData(
        city=city,
        temperature=0.0,
        feels_like=0.0,
        humidity=0,
        precipitation=0.0,
        precipitation_1h=None,
        precipitation_3h=None,
        wind_speed=0.0,
        wind_direction=0,
        pressure=0.0,
        visibility=0.0,
        weather_main="n/a",
        weather_description="Data unavailable - API key required",
        cloud_cover=0,
        uv_index=None,
        lat=lat,
        lon=lon,
        data_quality=0,  # 0% 表示无数据
        status="Offline",
        last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@router.get("/weather/system-metrics", response_model=SystemMetrics)
async def get_system_metrics():
    """获取系统整体指标 - 仅基于真实数据"""
    try:
        # 获取所有城市数据来计算平均值
        cities_data = await get_cities_weather()
        
        # 只计算有真实数据的城市（data_quality > 0）
        real_data_cities = [city for city in cities_data.values() if city.data_quality > 0]
        
        if len(real_data_cities) == 0:
            # 没有真实数据时返回n/a指标
            return SystemMetrics(
                active_stations=0,
                data_quality_avg=0.0,
                avg_temperature=0.0,
                total_precipitation=0.0,
                last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                data_sources=["No real data available - API key required"]
            )
        
        # 计算基于真实数据的平均值
        total_temp = sum(city.temperature for city in real_data_cities)
        total_precip = sum(city.precipitation for city in real_data_cities)
        avg_quality = sum(city.data_quality for city in real_data_cities) / len(real_data_cities)
        online_stations = len(real_data_cities)
        
        # 确定数据源（仅真实数据源）
        data_sources = []
        if avg_quality >= 90:
            if OPENWEATHER_API_KEY and OPENWEATHER_API_KEY != "demo_key_replace_with_real_key":
                data_sources.append("OpenWeatherMap API")
            elif WEATHER_API_KEY:
                data_sources.append("WeatherAPI.com")
            else:
                data_sources.append("API key verification required")
        
        return SystemMetrics(
            active_stations=online_stations,
            data_quality_avg=round(avg_quality, 1),
            avg_temperature=round(total_temp / len(real_data_cities), 1) if len(real_data_cities) > 0 else 0.0,
            total_precipitation=round(total_precip, 1),
            last_updated=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            data_sources=data_sources if data_sources else ["No verified data sources"]
        )
        
    except Exception as e:
        logger.error(f"Error in get_system_metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch system metrics")

@router.get("/weather/city/{city_name}", response_model=WeatherData)
async def get_single_city_weather(city_name: str):
    """获取单个城市的详细天气数据 - 仅返回真实数据"""
    city_key = None
    for key, info in MANITOBA_CITIES.items():
        if info['displayName'].lower() == city_name.lower() or key.lower() == city_name.lower():
            city_key = key
            break
    
    if not city_key:
        raise HTTPException(status_code=404, detail=f"City {city_name} not found")
    
    try:
        city_info = MANITOBA_CITIES[city_key]
        weather_data = await get_real_weather_only(city_key, city_info)
        return weather_data
        
    except Exception as e:
        logger.error(f"Error getting weather for {city_name}: {e}")
        # 返回n/a数据而不是抛出错误
        city_info = MANITOBA_CITIES[city_key]
        return get_na_weather_data(city_info['displayName'], city_info['lat'], city_info['lon'])

@router.get("/weather/health")
async def weather_api_health():
    """检查天气API健康状态"""
    try:
        # 测试获取温尼伯的天气数据
        winnipeg_info = MANITOBA_CITIES['Winnipeg']
        test_data = await get_real_weather_only('Winnipeg', winnipeg_info)
        
        return {
            "status": "healthy",
            "openweather_configured": bool(OPENWEATHER_API_KEY),
            "cities_available": len(MANITOBA_CITIES),
            "test_city": test_data.city,
            "test_temperature": test_data.temperature,
            "data_quality": test_data.data_quality,
            "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        logger.error(f"Weather API health check failed: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "openweather_configured": bool(OPENWEATHER_API_KEY),
            "cities_available": len(MANITOBA_CITIES),
            "last_checked": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

@router.get("/weather/historical")
async def get_weather_historical(date: str):
    """获取指定日期的天气历史数据"""
    try:
        # 解析日期参数
        from datetime import datetime
        target_date = datetime.strptime(date, "%Y-%m-%d")
        
        # 严格遵循规则：不使用模拟数据，缺乏数据时返回N/A
        return {
            "date": date,
            "temperature": "N/A",
            "precipitation": "N/A",
            "soil_moisture": "N/A",
            "humidity": "N/A",
            "data_source": "N/A",
            "note": "Historical weather data not available - no simulation data allowed"
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    except Exception as e:
        logger.error(f"Error getting historical weather for {date}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch historical weather data")
