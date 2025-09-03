# 🐍 Python基础学习指南

## 📚 目录
1. [Python环境搭建](#python环境搭建)
2. [基础语法](#基础语法)
3. [数据结构](#数据结构)
4. [函数和模块](#函数和模块)
5. [面向对象编程](#面向对象编程)
6. [文件操作](#文件操作)
7. [异常处理](#异常处理)
8. [实践项目](#实践项目)

---

## Python环境搭建

### 安装Python
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip python3-venv

# macOS
brew install python3

# Windows
# 从 https://python.org 下载安装包
```

### 创建虚拟环境
```bash
# 创建虚拟环境
python3 -m venv hydrai_env

# 激活虚拟环境
# Linux/macOS
source hydrai_env/bin/activate

# Windows
hydrai_env\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 验证安装
```python
# 检查Python版本
python3 --version

# 检查pip版本
pip --version

# 测试Python
python3 -c "print('Hello, HydrAI-SWE!')"
```

---

## 基础语法

### 变量和数据类型
```python
# 基本数据类型
name = "HydrAI-SWE"           # 字符串
version = 1.0                 # 浮点数
year = 2025                   # 整数
is_production = True          # 布尔值

# 类型检查
print(type(name))             # <class 'str'>
print(type(version))          # <class 'float'>
print(type(year))             # <class 'int'>
print(type(is_production))    # <class 'bool'>

# 类型转换
str_version = str(version)    # "1.0"
int_version = int(version)    # 1
float_year = float(year)      # 2025.0
```

### 字符串操作
```python
# 字符串格式化
project_name = "HydrAI-SWE"
version = "1.0"
print(f"项目名称: {project_name}, 版本: {version}")

# 字符串方法
text = "  HydrAI-SWE  "
print(text.strip())           # "HydrAI-SWE"
print(text.upper())           # "  HYDRAI-SWE  "
print(text.lower())           # "  hydrai-swe  "
print(text.replace("SWE", "Snow Water Equivalent"))  # "  HydrAI-Snow Water Equivalent  "
```

### 控制流
```python
# 条件语句
temperature = 25
if temperature > 30:
    print("高温天气")
elif temperature > 20:
    print("温暖天气")
else:
    print("凉爽天气")

# 循环语句
# for循环
for i in range(5):
    print(f"循环次数: {i}")

# while循环
count = 0
while count < 3:
    print(f"计数: {count}")
    count += 1

# 列表推导式
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
```

---

## 数据结构

### 列表 (List)
```python
# 创建列表
swe_data = [10, 15, 20, 25, 30]
temperatures = [5.2, 8.1, 12.3, 15.7, 18.9]

# 列表操作
print(len(swe_data))          # 5
print(swe_data[0])            # 10
print(swe_data[-1])           # 30
print(swe_data[1:3])          # [15, 20]

# 添加元素
swe_data.append(35)           # [10, 15, 20, 25, 30, 35]
swe_data.insert(0, 5)         # [5, 10, 15, 20, 25, 30, 35]

# 删除元素
swe_data.remove(20)           # [5, 10, 15, 25, 30, 35]
popped = swe_data.pop()       # 35, swe_data = [5, 10, 15, 25, 30]

# 列表方法
print(max(swe_data))          # 30
print(min(swe_data))          # 5
print(sum(swe_data))          # 85
print(sorted(swe_data))       # [5, 10, 15, 25, 30]
```

### 字典 (Dictionary)
```python
# 创建字典
weather_data = {
    "temperature": 25.5,
    "humidity": 60,
    "precipitation": 0.0,
    "wind_speed": 10.2
}

# 访问字典
print(weather_data["temperature"])    # 25.5
print(weather_data.get("pressure", "N/A"))  # N/A (默认值)

# 修改字典
weather_data["pressure"] = 1013.25
weather_data.update({"visibility": 10, "cloud_cover": 30})

# 字典方法
print(weather_data.keys())            # dict_keys(['temperature', 'humidity', ...])
print(weather_data.values())          # dict_values([25.5, 60, ...])
print(weather_data.items())           # dict_items([('temperature', 25.5), ...])

# 遍历字典
for key, value in weather_data.items():
    print(f"{key}: {value}")
```

### 元组 (Tuple)
```python
# 创建元组
coordinates = (49.895, -97.239)  # 温尼伯坐标
dimensions = (1920, 1080)        # 屏幕分辨率

# 元组解包
lat, lon = coordinates
width, height = dimensions

# 元组是不可变的
# coordinates[0] = 50.0  # 这会报错

# 元组方法
print(coordinates.count(49.895))  # 1
print(coordinates.index(-97.239)) # 1
```

### 集合 (Set)
```python
# 创建集合
unique_temperatures = {5.2, 8.1, 12.3, 15.7, 18.9, 5.2}  # 自动去重
print(unique_temperatures)  # {5.2, 8.1, 12.3, 15.7, 18.9}

# 集合操作
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

print(set1.union(set2))        # {1, 2, 3, 4, 5, 6, 7, 8}
print(set1.intersection(set2)) # {4, 5}
print(set1.difference(set2))   # {1, 2, 3}
```

---

## 函数和模块

### 函数定义
```python
# 基本函数
def calculate_swe(snow_depth, snow_density):
    """计算雪水当量"""
    return snow_depth * snow_density

# 调用函数
swe = calculate_swe(50, 0.3)  # 50cm雪深，0.3密度
print(f"雪水当量: {swe} cm")

# 带默认参数的函数
def predict_runoff(swe, temperature=0, precipitation=0):
    """预测径流"""
    base_runoff = swe * 0.1
    temp_factor = temperature * 0.05
    precip_factor = precipitation * 0.8
    return base_runoff + temp_factor + precip_factor

# 调用函数
runoff1 = predict_runoff(100)                    # 只传必需参数
runoff2 = predict_runoff(100, temperature=5)     # 传部分可选参数
runoff3 = predict_runoff(100, temperature=5, precipitation=10)  # 传所有参数

# 可变参数
def calculate_average(*numbers):
    """计算平均值"""
    return sum(numbers) / len(numbers)

avg = calculate_average(10, 20, 30, 40, 50)  # 30.0

# 关键字参数
def create_weather_report(**kwargs):
    """创建天气报告"""
    report = "天气报告:\n"
    for key, value in kwargs.items():
        report += f"{key}: {value}\n"
    return report

report = create_weather_report(
    temperature=25,
    humidity=60,
    wind_speed=10
)
print(report)
```

### 模块和包
```python
# 导入标准库
import math
import random
from datetime import datetime

# 使用导入的模块
print(math.pi)                    # 3.141592653589793
print(random.randint(1, 100))     # 随机整数
print(datetime.now())             # 当前时间

# 导入第三方库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建自定义模块
# 在 weather_utils.py 文件中
def celsius_to_fahrenheit(celsius):
    """摄氏度转华氏度"""
    return celsius * 9/5 + 32

def fahrenheit_to_celsius(fahrenheit):
    """华氏度转摄氏度"""
    return (fahrenheit - 32) * 5/9

# 在主文件中导入
from weather_utils import celsius_to_fahrenheit, fahrenheit_to_celsius

temp_c = 25
temp_f = celsius_to_fahrenheit(temp_c)
print(f"{temp_c}°C = {temp_f}°F")
```

---

## 面向对象编程

### 类定义
```python
class WeatherStation:
    """天气监测站类"""
    
    def __init__(self, name, latitude, longitude):
        """初始化监测站"""
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.temperature = 0
        self.humidity = 0
        self.precipitation = 0
    
    def update_weather(self, temp, hum, precip):
        """更新天气数据"""
        self.temperature = temp
        self.humidity = hum
        self.precipitation = precip
    
    def get_weather_summary(self):
        """获取天气摘要"""
        return f"""
        监测站: {self.name}
        位置: ({self.latitude}, {self.longitude})
        温度: {self.temperature}°C
        湿度: {self.humidity}%
        降水: {self.precipitation}mm
        """
    
    def calculate_heat_index(self):
        """计算热指数"""
        if self.temperature < 27:
            return self.temperature
        
        # 简化的热指数计算
        hi = -8.78469475556 + 1.61139411 * self.temperature + \
             2.33854883889 * self.humidity + \
             -0.14611605 * self.temperature * self.humidity
        return round(hi, 1)

# 使用类
station = WeatherStation("温尼伯站", 49.895, -97.239)
station.update_weather(25, 60, 0)
print(station.get_weather_summary())
print(f"热指数: {station.calculate_heat_index()}°C")
```

### 继承
```python
class SWEMonitoringStation(WeatherStation):
    """SWE监测站类，继承自WeatherStation"""
    
    def __init__(self, name, latitude, longitude, elevation):
        """初始化SWE监测站"""
        super().__init__(name, latitude, longitude)
        self.elevation = elevation
        self.swe_data = []
    
    def add_swe_measurement(self, swe_value, date):
        """添加SWE测量数据"""
        self.swe_data.append({
            'date': date,
            'swe': swe_value
        })
    
    def get_average_swe(self):
        """计算平均SWE"""
        if not self.swe_data:
            return 0
        return sum(measurement['swe'] for measurement in self.swe_data) / len(self.swe_data)
    
    def get_max_swe(self):
        """获取最大SWE"""
        if not self.swe_data:
            return 0
        return max(measurement['swe'] for measurement in self.swe_data)

# 使用继承的类
swe_station = SWEMonitoringStation("红河SWE站", 49.895, -97.239, 230)
swe_station.update_weather(20, 65, 5)
swe_station.add_swe_measurement(50, "2025-01-15")
swe_station.add_swe_measurement(55, "2025-01-16")
swe_station.add_swe_measurement(60, "2025-01-17")

print(swe_station.get_weather_summary())
print(f"平均SWE: {swe_station.get_average_swe()}mm")
print(f"最大SWE: {swe_station.get_max_swe()}mm")
```

---

## 文件操作

### 文件读写
```python
# 写入文件
weather_data = {
    "date": "2025-01-15",
    "temperature": 25.5,
    "humidity": 60,
    "precipitation": 0.0
}

# 写入文本文件
with open("weather_data.txt", "w", encoding="utf-8") as f:
    for key, value in weather_data.items():
        f.write(f"{key}: {value}\n")

# 读取文本文件
with open("weather_data.txt", "r", encoding="utf-8") as f:
    content = f.read()
    print(content)

# 按行读取
with open("weather_data.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
    for line in lines:
        print(line.strip())

# CSV文件操作
import csv

# 写入CSV
with open("weather_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["date", "temperature", "humidity", "precipitation"])
    writer.writerow(["2025-01-15", 25.5, 60, 0.0])
    writer.writerow(["2025-01-16", 26.0, 65, 2.5])

# 读取CSV
with open("weather_data.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"日期: {row['date']}, 温度: {row['temperature']}°C")
```

### JSON文件操作
```python
import json

# 写入JSON
weather_data = {
    "station": "温尼伯",
    "coordinates": [49.895, -97.239],
    "measurements": [
        {"date": "2025-01-15", "temperature": 25.5, "humidity": 60},
        {"date": "2025-01-16", "temperature": 26.0, "humidity": 65}
    ]
}

with open("weather_data.json", "w", encoding="utf-8") as f:
    json.dump(weather_data, f, ensure_ascii=False, indent=2)

# 读取JSON
with open("weather_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    print(f"监测站: {data['station']}")
    print(f"坐标: {data['coordinates']}")
    for measurement in data['measurements']:
        print(f"日期: {measurement['date']}, 温度: {measurement['temperature']}°C")
```

---

## 异常处理

### 基本异常处理
```python
# try-except语句
def safe_divide(a, b):
    """安全除法"""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("错误: 除数不能为零")
        return None
    except TypeError:
        print("错误: 参数类型不正确")
        return None

# 测试异常处理
print(safe_divide(10, 2))    # 5.0
print(safe_divide(10, 0))    # 错误: 除数不能为零, None
print(safe_divide(10, "2"))  # 错误: 参数类型不正确, None

# 多个异常处理
def process_weather_data(data):
    """处理天气数据"""
    try:
        temperature = float(data['temperature'])
        humidity = int(data['humidity'])
        precipitation = float(data['precipitation'])
        
        return {
            'temperature': temperature,
            'humidity': humidity,
            'precipitation': precipitation
        }
    except KeyError as e:
        print(f"错误: 缺少键 {e}")
        return None
    except ValueError as e:
        print(f"错误: 数值转换失败 {e}")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None

# 测试数据
test_data = {
    'temperature': '25.5',
    'humidity': '60',
    'precipitation': '0.0'
}

result = process_weather_data(test_data)
if result:
    print(f"处理结果: {result}")
```

### 自定义异常
```python
class WeatherDataError(Exception):
    """天气数据异常"""
    pass

class InvalidTemperatureError(WeatherDataError):
    """无效温度异常"""
    def __init__(self, temperature):
        self.temperature = temperature
        super().__init__(f"无效温度: {temperature}°C")

class InvalidHumidityError(WeatherDataError):
    """无效湿度异常"""
    def __init__(self, humidity):
        self.humidity = humidity
        super().__init__(f"无效湿度: {humidity}%")

def validate_weather_data(temperature, humidity):
    """验证天气数据"""
    if not -50 <= temperature <= 60:
        raise InvalidTemperatureError(temperature)
    
    if not 0 <= humidity <= 100:
        raise InvalidHumidityError(humidity)
    
    return True

# 测试自定义异常
try:
    validate_weather_data(25, 60)  # 正常数据
    print("数据验证通过")
except WeatherDataError as e:
    print(f"数据验证失败: {e}")

try:
    validate_weather_data(100, 60)  # 异常温度
except WeatherDataError as e:
    print(f"数据验证失败: {e}")

try:
    validate_weather_data(25, 150)  # 异常湿度
except WeatherDataError as e:
    print(f"数据验证失败: {e}")
```

---

## 实践项目

### 项目1: 天气数据管理器
```python
import json
from datetime import datetime

class WeatherDataManager:
    """天气数据管理器"""
    
    def __init__(self, filename="weather_data.json"):
        self.filename = filename
        self.data = self.load_data()
    
    def load_data(self):
        """加载数据"""
        try:
            with open(self.filename, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"stations": [], "measurements": []}
    
    def save_data(self):
        """保存数据"""
        with open(self.filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
    
    def add_station(self, name, latitude, longitude):
        """添加监测站"""
        station = {
            "id": len(self.data["stations"]) + 1,
            "name": name,
            "latitude": latitude,
            "longitude": longitude,
            "created_at": datetime.now().isoformat()
        }
        self.data["stations"].append(station)
        self.save_data()
        return station
    
    def add_measurement(self, station_id, temperature, humidity, precipitation):
        """添加测量数据"""
        measurement = {
            "station_id": station_id,
            "temperature": temperature,
            "humidity": humidity,
            "precipitation": precipitation,
            "timestamp": datetime.now().isoformat()
        }
        self.data["measurements"].append(measurement)
        self.save_data()
        return measurement
    
    def get_station_measurements(self, station_id):
        """获取监测站的所有测量数据"""
        return [m for m in self.data["measurements"] if m["station_id"] == station_id]
    
    def get_average_temperature(self, station_id):
        """计算平均温度"""
        measurements = self.get_station_measurements(station_id)
        if not measurements:
            return None
        return sum(m["temperature"] for m in measurements) / len(measurements)

# 使用天气数据管理器
manager = WeatherDataManager()

# 添加监测站
station = manager.add_station("温尼伯站", 49.895, -97.239)
print(f"添加监测站: {station['name']}")

# 添加测量数据
manager.add_measurement(station["id"], 25.5, 60, 0.0)
manager.add_measurement(station["id"], 26.0, 65, 2.5)
manager.add_measurement(station["id"], 24.8, 58, 0.0)

# 获取数据
measurements = manager.get_station_measurements(station["id"])
print(f"测量数据数量: {len(measurements)}")

avg_temp = manager.get_average_temperature(station["id"])
print(f"平均温度: {avg_temp:.1f}°C")
```

### 项目2: SWE计算器
```python
import math

class SWECalculator:
    """雪水当量计算器"""
    
    def __init__(self):
        self.measurements = []
    
    def add_measurement(self, snow_depth, snow_density, date=None):
        """添加测量数据"""
        swe = snow_depth * snow_density
        measurement = {
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "snow_depth": snow_depth,
            "snow_density": snow_density,
            "swe": swe
        }
        self.measurements.append(measurement)
        return measurement
    
    def calculate_total_swe(self):
        """计算总SWE"""
        return sum(m["swe"] for m in self.measurements)
    
    def calculate_average_swe(self):
        """计算平均SWE"""
        if not self.measurements:
            return 0
        return self.calculate_total_swe() / len(self.measurements)
    
    def predict_melt_rate(self, temperature, wind_speed=0):
        """预测融化速率"""
        # 简化的融化速率计算
        base_rate = 0.1  # 基础融化速率
        temp_factor = max(0, temperature - 0) * 0.05  # 温度因子
        wind_factor = wind_speed * 0.01  # 风速因子
        
        return base_rate + temp_factor + wind_factor
    
    def generate_report(self):
        """生成报告"""
        if not self.measurements:
            return "没有测量数据"
        
        total_swe = self.calculate_total_swe()
        avg_swe = self.calculate_average_swe()
        max_swe = max(m["swe"] for m in self.measurements)
        min_swe = min(m["swe"] for m in self.measurements)
        
        report = f"""
        SWE测量报告
        ============
        测量次数: {len(self.measurements)}
        总SWE: {total_swe:.2f} mm
        平均SWE: {avg_swe:.2f} mm
        最大SWE: {max_swe:.2f} mm
        最小SWE: {min_swe:.2f} mm
        
        详细数据:
        """
        
        for i, m in enumerate(self.measurements, 1):
            report += f"{i}. {m['date']}: 雪深{m['snow_depth']}cm, 密度{m['snow_density']}, SWE{m['swe']:.2f}mm\n"
        
        return report

# 使用SWE计算器
calculator = SWECalculator()

# 添加测量数据
calculator.add_measurement(50, 0.3, "2025-01-15")  # 50cm雪深，0.3密度
calculator.add_measurement(55, 0.32, "2025-01-16")
calculator.add_measurement(60, 0.35, "2025-01-17")

# 生成报告
print(calculator.generate_report())

# 预测融化速率
melt_rate = calculator.predict_melt_rate(5, 10)  # 5°C，10km/h风速
print(f"预测融化速率: {melt_rate:.2f} mm/小时")
```

---

## 📚 学习资源

### 在线教程
- **Python官方教程**: https://docs.python.org/3/tutorial/
- **菜鸟教程**: https://www.runoob.com/python3/python3-tutorial.html
- **廖雪峰Python教程**: https://www.liaoxuefeng.com/wiki/1016959663602400

### 实践平台
- **LeetCode**: https://leetcode.cn/
- **HackerRank**: https://www.hackerrank.com/
- **Codecademy**: https://www.codecademy.com/

### 推荐书籍
- 《Python编程：从入门到实践》
- 《流畅的Python》
- 《Python Cookbook》

---

## 🎯 学习检查点

### 第1周目标
- [ ] 能够安装和配置Python环境
- [ ] 掌握基本语法和数据类型
- [ ] 能够编写简单的程序

### 第2周目标
- [ ] 掌握函数和模块的使用
- [ ] 理解面向对象编程基础
- [ ] 能够处理文件和异常

### 第3周目标
- [ ] 能够独立完成小项目
- [ ] 掌握常用库的使用
- [ ] 为学习机器学习做好准备

---

**下一步**: 学习 [机器学习基础指南](MACHINE_LEARNING_BASICS.md)

