"""
HydrAI-SWE API Main Application
基于真实数据，禁止硬编码和模拟数据
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from fastapi import FastAPI, Request, Body, HTTPException
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from datetime import datetime

# 只导入必要的模块
try:
    from src.api.routers import swe
    SWE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SWE router: {e}")
    SWE_AVAILABLE = False

try:
    from src.api.routers import real_data_api
    REAL_DATA_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import real data API router: {e}")
    REAL_DATA_AVAILABLE = False

try:
    from src.api.routers import bayesian_network
    BAYESIAN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import Bayesian network router: {e}")
    BAYESIAN_AVAILABLE = False

try:
    from src.api.routers import advanced_features_api
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import advanced features API router: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

try:
    from src.api.routers import scientific_validation_api
    SCIENTIFIC_VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import scientific validation API router: {e}")
    SCIENTIFIC_VALIDATION_AVAILABLE = False

try:
    from src.api.routers import flood_warning
    FLOOD_AVAILABLE = True
except ImportError:
    FLOOD_AVAILABLE = False

try:
    from src.api.routers import water_resources_management
    WATER_RESOURCES_AVAILABLE = True
except ImportError:
    WATER_RESOURCES_AVAILABLE = False

try:
    from src.api.routers import flood_prediction_api
    FLOOD_PREDICTION_AVAILABLE = True
except ImportError:
    FLOOD_PREDICTION_AVAILABLE = False

try:
    from src.api.routers import water_quality_api
    WATER_QUALITY_AVAILABLE = True
except ImportError:
    WATER_QUALITY_AVAILABLE = False

try:
    from src.api.routers import data_sources_api
    DATA_SOURCES_AVAILABLE = True
except ImportError:
    DATA_SOURCES_AVAILABLE = False

try:
    from src.api.routers import hydrological_analysis_api
    HYDROLOGICAL_ANALYSIS_AVAILABLE = True
except ImportError:
    HYDROLOGICAL_ANALYSIS_AVAILABLE = False

try:
    from src.api.routers import validation_report_api
    VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import validation report API: {e}")
    VALIDATION_AVAILABLE = False

try:
    from src.api.routers import agriculture
    AGRICULTURE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agriculture router: {e}")
    AGRICULTURE_AVAILABLE = False

try:
    from src.api.routers import model_prediction
    MODEL_PREDICTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import model prediction router: {e}")
    MODEL_PREDICTION_AVAILABLE = False

try:
    from src.api.routers import openmeteo
    OPENMETEO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import OpenMeteo router: {e}")
    OPENMETEO_AVAILABLE = False

try:
    from src.api.routers import model_training
    MODEL_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import model training router: {e}")
    MODEL_TRAINING_AVAILABLE = False

# Prediction enhancement removed due to simulated data usage
PREDICTION_ENHANCEMENT_AVAILABLE = False

# 创建FastAPI应用
app = FastAPI(
    title="HydrAI-SWE API",
    description="积雪水当量预测与径流分析 - 基于真实数据",
    version="1.0.0",
)

# 启用gzip压缩
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 启用CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务 - 检查目录是否存在
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Favicon路由
@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

# 模板配置
templates = Jinja2Templates(directory="templates")

# 包含可用的路由器
if SWE_AVAILABLE:
    app.include_router(swe.router, prefix="/api/swe", tags=["swe"])

if FLOOD_AVAILABLE:
    app.include_router(flood_warning.router, prefix="/api/v1/flood", tags=["flood_warning"])

if WATER_RESOURCES_AVAILABLE:
    app.include_router(water_resources_management.router, prefix="/api/water", tags=["water-resources"])

if FLOOD_PREDICTION_AVAILABLE:
    app.include_router(flood_prediction_api.router, tags=["flood-prediction"])

if WATER_QUALITY_AVAILABLE:
    app.include_router(water_quality_api.router, tags=["water-quality"])

## Watchdog: Disabled placeholder data-sources endpoints (no mock/Hardcode policy)
# if DATA_SOURCES_AVAILABLE:
#     app.include_router(data_sources_api.router, prefix="/api", tags=["data-sources"])

## Watchdog: Disabled placeholder hydrological-analysis endpoints (no mock/Hardcode policy)
# if HYDROLOGICAL_ANALYSIS_AVAILABLE:
#     app.include_router(hydrological_analysis_api.router, prefix="/api", tags=["hydrological-analysis"])

if AGRICULTURE_AVAILABLE:
    app.include_router(agriculture.router, prefix="/api/v1/agriculture", tags=["agriculture"])

if MODEL_PREDICTION_AVAILABLE:
    app.include_router(model_prediction.router, prefix="/api/v1/prediction", tags=["model_prediction"])

if OPENMETEO_AVAILABLE:
    app.include_router(openmeteo.router, tags=["openmeteo"])

if MODEL_TRAINING_AVAILABLE:
    app.include_router(model_training.router, prefix="/api/training", tags=["model_training"])

if REAL_DATA_AVAILABLE:
    app.include_router(real_data_api.router, tags=["real-data"])

if BAYESIAN_AVAILABLE:
    app.include_router(bayesian_network.router, tags=["bayesian-network"])

if ADVANCED_FEATURES_AVAILABLE:
    app.include_router(advanced_features_api.router, tags=["advanced-features"])

if SCIENTIFIC_VALIDATION_AVAILABLE:
    app.include_router(scientific_validation_api.router, tags=["scientific-validation"])

if VALIDATION_AVAILABLE:
    app.include_router(validation_report_api.router)

# Prediction enhancement router removed due to simulated data usage

# 科学验证页面路由
@app.get("/knowledge")
async def hydrological_knowledge_base():
    """水文知识库页面"""
    try:
        with open("templates/ui/hydrological_knowledge_base.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Hydrological Knowledge Base</h1><p>Page not found</p>", status_code=404)

# 根路径重定向到首页
@app.get("/")
async def root():
    return RedirectResponse(url="/home")

# 首页
@app.get("/home", response_class=HTMLResponse)
async def home(request: Request):
    """系统首页"""
    with open("templates/ui/home.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

# UI界面
@app.get("/ui", response_class=HTMLResponse)
async def ui(request: Request):
    """Main UI Interface - Enhanced dashboard with detailed analysis"""
    with open("templates/ui/enhanced_dashboard.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

# 下线法语界面（已废弃）
# @app.get("/ui/francais", response_class=HTMLResponse)
# async def ui_francais(request: Request):
#     """法语界面（已下线）"""
#     return templates.TemplateResponse("ui/enhanced_fr.html", {"request": request})

@app.get("/model", response_class=HTMLResponse)
async def model_training(request: Request):
    """模型训练界面"""
    with open("templates/ui/model_training_dashboard.html", "r", encoding="utf-8") as f:
        content = f.read()
    return HTMLResponse(content=content)

# Prediction enhancement page removed due to simulated data usage

@app.get("/analysis", response_class=HTMLResponse)
async def analysis_dashboard():
    """数据分析界面 - 包含交互式图表"""
    with open("templates/ui/analysis_dashboard_simple.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/comprehensive", response_class=HTMLResponse)
async def comprehensive_dashboard():
    """综合数据仪表板 - 展示所有数据源"""
    with open("templates/ui/comprehensive_dashboard.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/hydrological", response_class=HTMLResponse)
async def hydrological_analysis_dashboard():
    """水文分析仪表板 - 专业水文分析和时间序列展示"""
    with open("templates/ui/hydrological_analysis_dashboard.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/validation-report", response_class=HTMLResponse)
async def validation_report_page():
    """Model validation report page."""
    with open("templates/ui/validation_report.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/bayesian-network-dashboard", response_class=HTMLResponse)
async def bayesian_network_dashboard():
    """贝叶斯网络分析仪表板"""
    with open("templates/bayesian_network_dashboard.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.get("/agriculture")
async def agriculture_redirect():
    """农业模块重定向"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/api/v1/agriculture/health")

# 调度器启动和关闭事件
@app.on_event("startup")
async def startup_event():
    """应用启动时启动数据刷新调度器"""
    try:
        # 暂时禁用调度器，避免启动阻塞
        # from src.core.scheduler import start_scheduler
        # start_scheduler()
        print("Data refresh scheduler disabled for now")
    except Exception as e:
        print(f"Warning: Could not start scheduler: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时停止数据刷新调度器"""
    try:
        from src.core.scheduler import stop_scheduler
        stop_scheduler()
        print("Data refresh scheduler stopped")
    except Exception as e:
        print(f"Warning: Could not stop scheduler: {e}")

# 健康检查
@app.get("/health")
async def health_check():
    """系统健康检查"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "modules": {
            "swe": SWE_AVAILABLE,
            "flood": FLOOD_AVAILABLE,
            "agriculture": AGRICULTURE_AVAILABLE,
            "model_training": MODEL_TRAINING_AVAILABLE,
            "prediction_enhancement": PREDICTION_ENHANCEMENT_AVAILABLE,
            "openmeteo": OPENMETEO_AVAILABLE
        },
        "real_data_only": True
    }

# API文档 - 移除循环重定向

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)