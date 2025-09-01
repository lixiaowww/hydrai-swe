#!/usr/bin/env python3
"""
简化的服务器启动脚本用于测试天气API修改
"""
import sys
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# 直接导入天气路由
from api.routers import weather

app = FastAPI(
    title="HydrAI-SWE Test API",
    description="Test server for weather API modifications",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 包含天气API路由
app.include_router(weather.router, prefix="/api/v1", tags=["weather"])

# 配置模板
templates = Jinja2Templates(directory="templates")

# 静态文件
if os.path.exists("templates/ui"):
    app.mount("/static", StaticFiles(directory="templates/ui"), name="ui_static")

@app.get("/")
def read_root():
    return {"message": "Weather API Test Server", "weather_endpoints": [
        "/api/v1/weather/cities",
        "/api/v1/weather/system-metrics", 
        "/api/v1/weather/health"
    ]}

@app.get("/hydrological-center")
def hydrological_center_dashboard(request):
    # 水文中心仪表板
    from fastapi import Request
    return templates.TemplateResponse("ui/flood_warning_dashboard.html", {"request": request})

if __name__ == "__main__":
    print("🚀 Starting HydrAI-SWE Weather API Test Server...")
    print("📊 Visit http://localhost:8001/hydrological-center to test the UI")
    print("🔧 API docs available at http://localhost:8001/docs")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
