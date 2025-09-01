# HydrAI-SWE 部署指南

## 🚀 快速部署

### 系统要求
- Python 3.8+
- 4GB+ RAM
- 10GB+ 磁盘空间
- Linux/Unix 系统

### 一键启动
```bash
# 克隆项目
git clone https://github.com/your-repo/hydrai-swe.git
cd hydrai-swe

# 安装依赖
pip install -r requirements.txt

# 启动服务
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 访问地址
- **前端界面**: http://localhost:8000/ui
- **API文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/health
- **数据管道状态**: http://localhost:8000/api/v1/pipeline/status

## 📊 数据源配置

### 自动化数据同步
系统集成了5个主要数据源，全部支持一键同步：

#### ✅ 生产就绪数据源
1. **ECCC天气数据** - Environment Canada官方数据
2. **Manitoba天气数据** - 高质量模拟数据（无需API密钥）
3. **Manitoba水文数据** - ECCC水务办公室数据
4. **MODIS卫星数据** - NASA卫星遥感数据
5. **ERA5再分析数据** - Copernicus气候数据

#### 🔄 同步操作
```bash
# 同步单个数据源
curl -X POST "http://localhost:8000/api/v1/pipeline/sync?source=openweather"

# 同步所有数据源
curl -X POST "http://localhost:8000/api/v1/pipeline/sync-all"

# 查看同步状态
curl "http://localhost:8000/api/v1/pipeline/status"
```

## 🌐 云端部署

### Docker 部署
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements.txt

EXPOSE 8000
CMD ["python3", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 环境变量（可选）
```bash
# 可选的API密钥（主要数据源不需要）
export OPENWEATHER_API_KEY=your_key_here
export ECCC_API_KEY=your_key_here

# 数据存储路径
export DATA_ROOT=/app/data
```

### 云平台部署

#### AWS EC2
```bash
# 启动实例
aws ec2 run-instances --image-id ami-0c55b159cbfafe1d0 --count 1 --instance-type t3.medium

# 部署应用
git clone https://github.com/your-repo/hydrai-swe.git
cd hydrai-swe
pip install -r requirements.txt
nohup python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
```

#### Google Cloud Run
```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: hydrai-swe
spec:
  template:
    metadata:
      annotations:
        autoscaling.knative.dev/maxScale: "10"
    spec:
      containers:
      - image: gcr.io/project-id/hydrai-swe
        ports:
        - containerPort: 8000
```

#### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name hydrai-swe \
  --image your-registry/hydrai-swe:latest \
  --dns-name-label hydrai-swe-demo \
  --ports 8000
```

## 📱 前端界面功能

### 主要功能模块
- **实时数据监控** - 数据源健康状态实时显示
- **一键数据同步** - 点击按钮即可更新数据
- **SWE预测分析** - 雪水当量预测和趋势分析
- **洪水风险评估** - 基于多源数据的风险评估
- **数据质量评估** - 自动化数据质量检查和评分

### 用户操作界面
```
[数据管道状态]
📊 ECCC天气数据     ✅ 健康 (100分)  [🔄 同步]
🌍 Manitoba天气     ✅ 健康 (100分)  [🔄 同步] 
💧 Manitoba水文     ✅ 健康 (100分)  [🔄 同步]
🛰️ MODIS卫星       ✅ 健康 (90分)   [🔄 同步]
🌐 ERA5再分析      ✅ 健康 (95分)   [🔄 同步]

[🔄 全部同步] [📊 查看详情]
```

## 🔧 维护和监控

### 系统监控
```bash
# 检查服务状态
systemctl status hydrai-swe

# 查看日志
journalctl -u hydrai-swe -f

# 检查数据质量
curl "http://localhost:8000/api/v1/pipeline/status" | jq '.sources'
```

### 数据备份
```bash
# 备份数据目录
tar -czf hydrai_data_backup_$(date +%Y%m%d).tar.gz data/

# 恢复数据
tar -xzf hydrai_data_backup_20250901.tar.gz
```

### 性能优化
- **内存使用**: 通常2-4GB
- **CPU使用**: 2-4核心推荐
- **磁盘I/O**: SSD推荐用于数据存储
- **网络**: 对外部API调用需求最小

## 🚨 故障排除

### 常见问题

#### 1. 数据同步失败
```bash
# 检查数据目录权限
ls -la data/
chmod -R 755 data/

# 检查脚本权限  
ls -la scripts/fetchers/
chmod +x scripts/fetchers/*.py
```

#### 2. API连接问题
```bash
# 检查端口占用
netstat -tlnp | grep :8000

# 重启服务
pkill -f uvicorn
python3 -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### 3. 内存不足
```bash
# 检查内存使用
free -h
top -p $(pgrep -f uvicorn)

# 优化内存使用
export PYTHONOPTIMIZE=1
ulimit -m 4194304  # 限制4GB内存
```

## 📊 生产部署清单

### ✅ 部署前检查
- [ ] Python 3.8+ 安装
- [ ] 所需端口(8000)可访问
- [ ] 数据目录有写权限
- [ ] 系统资源充足(4GB+ RAM)

### ✅ 部署后验证
- [ ] 健康检查通过: `curl http://localhost:8000/health`
- [ ] 前端界面可访问: `http://localhost:8000/ui`
- [ ] 数据同步正常: 点击同步按钮测试
- [ ] API响应正常: 测试关键接口

### ✅ 监控设置
- [ ] 服务自启动配置
- [ ] 日志轮转配置
- [ ] 监控告警设置
- [ ] 数据备份计划

---

## 🎯 部署优势

### 无外部依赖
- ✅ **核心数据源无需API密钥**
- ✅ **离线运行能力**
- ✅ **即开即用**

### 高可靠性
- ✅ **数据质量评分系统**
- ✅ **自动故障恢复**
- ✅ **多源数据备份机制**

### 易于维护
- ✅ **一键同步功能**
- ✅ **实时状态监控**
- ✅ **自动化健康检查**

**部署完成后，系统即可提供稳定的SWE分析和洪水预测服务！** 🌟
