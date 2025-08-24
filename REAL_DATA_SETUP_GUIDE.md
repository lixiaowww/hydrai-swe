# 真实数据源配置指南

## 🚨 重要原则
**模型训练严禁使用任何模拟/Mock数据！只能使用真实观测数据！**

## 📊 推荐的真实数据源

### 1. NOAA Climate Data Online (免费)
**优先级: 高 ⭐⭐⭐⭐⭐**

- **网站**: https://www.ncdc.noaa.gov/cdo-web/
- **数据质量**: 政府官方，极高可靠性
- **覆盖范围**: 美国及部分国际站点
- **申请步骤**:
  1. 访问: https://www.ncdc.noaa.gov/cdo-web/token
  2. 填写邮箱申请免费API token
  3. 在 `config/credentials.env` 中设置: `NOAA_API_TOKEN=your_token`

### 2. Environment and Climate Change Canada (免费)
**优先级: 高 ⭐⭐⭐⭐⭐**

- **网站**: https://climate.weather.gc.ca/
- **数据质量**: 政府官方，极高可靠性
- **覆盖范围**: 加拿大全境，包括红河流域
- **获取方式**:
  - 历史数据下载: https://climate.weather.gc.ca/climate_data/bulk_data_e.html
  - API访问需要申请: https://api.weather.gc.ca/

### 3. USGS Water Data (免费)
**优先级: 高 ⭐⭐⭐⭐**

- **网站**: https://waterdata.usgs.gov/
- **数据类型**: 河流流量、水位、水质
- **API**: https://waterservices.usgs.gov/
- **无需注册**: 直接API访问

### 4. Agriculture and Agri-Food Canada (免费)
**优先级: 中 ⭐⭐⭐**

- **网站**: https://agriculture.canada.ca/
- **数据类型**: 土壤湿度、农业气象
- **需要申请**: 研究用途可申请数据访问

### 5. Global Runoff Data Centre (免费/收费)
**优先级: 中 ⭐⭐⭐**

- **网站**: https://www.bafg.de/GRDC/
- **数据类型**: 全球河流径流数据
- **申请**: 研究用途免费

## 🔧 立即行动步骤

### Step 1: 申请NOAA API Token
```bash
# 1. 访问 https://www.ncdc.noaa.gov/cdo-web/token
# 2. 填写邮箱申请
# 3. 收到token后，在credentials.env中添加：
echo "NOAA_API_TOKEN=your_actual_token" >> config/credentials.env
```

### Step 2: 配置Environment Canada数据
```bash
# 下载历史数据
wget "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?format=csv&stationID=27174&Year=2024&Month=1&Day=1&timeframe=2"

# 或使用我们的脚本自动下载
python3 src/data/download_environment_canada.py
```

### Step 3: 验证数据获取
```bash
# 运行真实数据收集器
python3 src/data/real_data_collector.py

# 应该看到成功获取真实数据的消息
```

## 📋 数据质量检查清单

在使用任何数据源之前，必须验证：

- [ ] 数据来源是政府机构或权威研究机构
- [ ] 数据标注为"观测数据"而非"模型输出"
- [ ] 有明确的测站位置和设备信息
- [ ] 数据有质量控制标记
- [ ] 时间序列连续且合理
- [ ] 数值在物理合理范围内

## 🚫 严格禁止的数据类型

- ❌ 任何标注为"模拟"、"生成"、"Mock"的数据
- ❌ 数值天气预报模式输出（如GFS、ECMWF模式）
- ❌ 插值或统计生成的数据
- ❌ 人工合成的测试数据
- ❌ 未经验证的第三方数据

## 🎯 红河流域具体站点

### 加拿大侧
- **Winnipeg Airport (YWG)**: Station ID 27174
- **Morris, MB**: Station ID 3025
- **Emerson, MB**: Station ID 3017

### 美国侧  
- **Fargo, ND**: NOAA Station USW00014922
- **Grand Forks, ND**: NOAA Station USW00014918

## ⚡ 紧急方案

如果短期内无法获取足够的真实历史数据：

1. **暂停模型训练** - 等待真实数据
2. **使用现有预训练模型** - 但要明确标注
3. **联系研究机构** - 获取观测数据集
4. **购买商业数据** - 如需要立即开始训练

## 📞 联系方式

### NOAA支持
- Email: ncei.orders@noaa.gov
- 文档: https://www.ncdc.noaa.gov/cdo-web/webservices/v2

### Environment Canada支持  
- Email: ec.dps-client.ec@canada.ca
- 文档: https://api.weather.gc.ca/

## 🔄 定期检查

- 每月检查API配额使用情况
- 验证数据质量和完整性
- 更新数据源配置
- 备份重要数据集

---

**记住: 宁可延迟训练，也不使用虚假数据！数据质量决定模型质量！**
