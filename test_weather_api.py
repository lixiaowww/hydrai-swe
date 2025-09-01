#!/usr/bin/env python3
"""
测试天气API功能的脚本
"""
import asyncio
import httpx
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000"

async def test_weather_apis():
    """测试所有天气API端点"""
    
    async with httpx.AsyncClient() as client:
        
        print("🧪 测试天气API...")
        print("=" * 50)
        
        # 测试API健康状态
        try:
            print("1️⃣ 测试天气API健康状态...")
            response = await client.get(f"{API_BASE_URL}/api/v1/weather/health")
            if response.status_code == 200:
                health_data = response.json()
                print("✅ 天气API健康检查通过")
                print(f"   状态: {health_data['status']}")
                print(f"   OpenWeather配置: {health_data['openweather_configured']}")
                print(f"   可用城市数量: {health_data['cities_available']}")
                if 'test_temperature' in health_data:
                    print(f"   测试温度: {health_data['test_temperature']}°C")
            else:
                print(f"❌ 健康检查失败: HTTP {response.status_code}")
        except Exception as e:
            print(f"❌ 健康检查错误: {e}")
        
        print()
        
        # 测试系统指标API
        try:
            print("2️⃣ 测试系统指标API...")
            response = await client.get(f"{API_BASE_URL}/api/v1/weather/system-metrics")
            if response.status_code == 200:
                metrics_data = response.json()
                print("✅ 系统指标获取成功")
                print(f"   活跃站点: {metrics_data['active_stations']}")
                print(f"   数据质量: {metrics_data['data_quality_avg']:.1f}%")
                print(f"   平均温度: {metrics_data['avg_temperature']}°C")
                print(f"   总降水量: {metrics_data['total_precipitation']} mm")
                print(f"   数据源: {', '.join(metrics_data['data_sources'])}")
                print(f"   更新时间: {metrics_data['last_updated']}")
            else:
                print(f"❌ 系统指标获取失败: HTTP {response.status_code}")
                print(f"   响应: {response.text}")
        except Exception as e:
            print(f"❌ 系统指标获取错误: {e}")
        
        print()
        
        # 测试所有城市天气API
        try:
            print("3️⃣ 测试所有城市天气API...")
            response = await client.get(f"{API_BASE_URL}/api/v1/weather/cities")
            if response.status_code == 200:
                cities_data = response.json()
                print(f"✅ 获取到 {len(cities_data)} 个城市的天气数据")
                
                for city_key, weather_data in cities_data.items():
                    print(f"   📍 {weather_data['city']}:")
                    print(f"      温度: {weather_data['temperature']}°C")
                    print(f"      降水: {weather_data['precipitation']} mm")
                    print(f"      湿度: {weather_data['humidity']}%")
                    print(f"      风速: {weather_data['wind_speed']} km/h")
                    print(f"      天气: {weather_data['weather_description']}")
                    print(f"      数据质量: {weather_data['data_quality']}%")
                    print(f"      状态: {weather_data['status']}")
                    print()
            else:
                print(f"❌ 城市天气获取失败: HTTP {response.status_code}")
                print(f"   响应: {response.text}")
        except Exception as e:
            print(f"❌ 城市天气获取错误: {e}")
        
        print()
        
        # 测试单个城市天气API
        try:
            print("4️⃣ 测试单个城市天气API (温尼伯)...")
            response = await client.get(f"{API_BASE_URL}/api/v1/weather/city/Winnipeg")
            if response.status_code == 200:
                winnipeg_data = response.json()
                print("✅ 温尼伯天气数据获取成功")
                print(f"   城市: {winnipeg_data['city']}")
                print(f"   温度: {winnipeg_data['temperature']}°C (体感: {winnipeg_data['feels_like']}°C)")
                print(f"   天气: {winnipeg_data['weather_main']} - {winnipeg_data['weather_description']}")
                print(f"   湿度: {winnipeg_data['humidity']}%")
                print(f"   气压: {winnipeg_data['pressure']} hPa")
                print(f"   风速: {winnipeg_data['wind_speed']} km/h")
                print(f"   云量: {winnipeg_data['cloud_cover']}%")
                if winnipeg_data['precipitation'] > 0:
                    print(f"   降水: {winnipeg_data['precipitation']} mm")
                print(f"   数据质量: {winnipeg_data['data_quality']}%")
                print(f"   更新时间: {winnipeg_data['last_updated']}")
            else:
                print(f"❌ 温尼伯天气获取失败: HTTP {response.status_code}")
                print(f"   响应: {response.text}")
        except Exception as e:
            print(f"❌ 温尼伯天气获取错误: {e}")
        
        print()
        print("🎯 API测试完成!")
        print(f"🕒 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def main():
    """主函数"""
    print("🌤️  HydrAI-SWE 天气API测试工具")
    print(f"🔗 测试目标: {API_BASE_URL}")
    print()
    
    try:
        asyncio.run(test_weather_apis())
    except KeyboardInterrupt:
        print("\n⚠️  测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")

if __name__ == "__main__":
    main()
