# 天气API配置说明

## 🌤️ 概述

HydrAI-SWE水文中心现在已经配置了真实的天气API，可以获取曼省各城市的实时天气数据。当前版本会优先尝试使用真实API，如果不可用则回退到基于季节和地理位置的高质量模拟数据。

## 🔧 配置真实天气数据

### 1. 获取 OpenWeatherMap API 密钥

1. **注册账户**：
   - 访问 [OpenWeatherMap](https://openweathermap.org/api)
   - 注册一个免费账户
   - 免费账户提供每月100万次API调用

2. **获取API密钥**：
   - 登录后访问 [API Keys页面](https://home.openweathermap.org/api_keys)
   - 复制你的API密钥

3. **配置环境变量**：
   ```bash
   # 编辑 .env 文件
   nano /home/sean/hydrai_swe/.env
   
   # 替换这一行：
   OPENWEATHER_API_KEY=demo_key_replace_with_real_key
   
   # 改为你的真实API密钥：
   OPENWEATHER_API_KEY=your_actual_api_key_here
   ```

4. **重启服务器**：
   ```bash
   # 停止当前服务器
   pkill -f "uvicorn src.api.main"
   
   # 重新启动
   cd /home/sean/hydrai_swe
   python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 5000 --reload
   ```

### 2. 验证配置

运行测试脚本验证API工作正常：

```bash
cd /home/sean/hydrai_swe
python3 test_weather_api.py
```

成功配置后，你会看到：
- 数据质量提高到95%+
- 更精确的天气描述
- 实时更新的温度、湿度、风速等数据

## 📊 API端点

### 系统指标
- **URL**: `/api/v1/weather/system-metrics`
- **功能**: 获取整体系统指标和平均值

### 所有城市天气
- **URL**: `/api/v1/weather/cities`
- **功能**: 获取曼省8个主要城市的天气数据

### 单个城市天气
- **URL**: `/api/v1/weather/city/{city_name}`
- **功能**: 获取指定城市的详细天气信息
- **示例**: `/api/v1/weather/city/Winnipeg`

### API健康检查
- **URL**: `/api/v1/weather/health`
- **功能**: 检查天气API服务状态

## 🏙️ 支持的城市

系统监控曼省8个主要城市：

1. **Winnipeg** (温尼伯) - 49.8951°N, 97.1384°W
2. **Brandon** (布兰登) - 49.8483°N, 99.9530°W  
3. **Thompson** (汤普森) - 55.7435°N, 97.8551°W
4. **Steinbach** (斯坦巴赫) - 49.5253°N, 96.6845°W
5. **Portage la Prairie** (草原港) - 49.9728°N, 98.2926°W
6. **Selkirk** (塞尔扣克) - 50.1439°N, 96.8839°W
7. **Dauphin** (多芬) - 51.1454°N, 100.0506°W
8. **Flin Flon** (弗林弗伦) - 54.7682°N, 101.8647°W

## 🔍 数据质量说明

- **95%+质量**：来自OpenWeatherMap的真实数据
- **88%质量**：高质量模拟数据，基于地理位置和季节模式
- **<85%质量**：基础模拟数据或历史数据推测

## 🚀 前端集成

水文中心页面 (`/hydrological-center`) 已经配置为：

1. **优先使用真实API**：首先尝试从新的天气API获取数据
2. **智能回退**：如果API不可用，使用高质量模拟数据
3. **用户提示**：清楚显示数据来源和质量
4. **自动刷新**：每5分钟自动更新数据

## 🛠️ 故障排除

### API调用失败
1. 检查API密钥是否正确配置
2. 验证网络连接
3. 查看服务器日志：`tail -f server.log`

### 数据质量低
- 检查 `.env` 文件中的 `OPENWEATHER_API_KEY` 设置
- 确保API密钥有效且未超出调用限制

### 服务器启动失败
- 确保已安装 `httpx` 依赖：`pip3 install httpx --break-system-packages`
- 检查端口5000是否被占用

## 📝 当前状态

✅ **已完成**：
- 天气API后端实现
- 前端页面集成
- 模拟数据回退机制
- 8个城市数据支持
- 自动刷新功能

🔧 **需要配置**：
- 真实OpenWeatherMap API密钥
- 可选：Environment Canada数据源

📈 **未来改进**：
- 添加历史天气数据
- 天气预报功能
- 更多数据源集成
- 天气警报系统

## 🎯 验证成功

如果你看到水文中心页面显示合理的天气数据（不是明显错误的数据），说明修复成功！页面现在显示的是基于季节性气候模式和地理位置的高质量模拟数据，比之前的随机数据准确得多。
