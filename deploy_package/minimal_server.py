#!/usr/bin/env python3
"""
最小化服务器，只包含核心功能
"""

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="HydrAI-SWE Minimal")

@app.get("/health")
async def health():
    return {"status": "healthy"}

# 只添加SWE路由
from src.api.routers import swe
app.include_router(swe.router, prefix="/api/swe", tags=["swe"])

# 添加洪水预测API
try:
    from src.api.routers import flood_prediction_api
    app.include_router(flood_prediction_api.router, tags=["flood-prediction"])
    print("✅ Flood prediction API loaded")
except ImportError as e:
    print(f"⚠️  Could not load flood prediction API: {e}")

# 添加水质分析API
try:
    from src.api.routers import water_quality_api
    app.include_router(water_quality_api.router, tags=["water-quality"])
    print("✅ Water quality API loaded")
except ImportError as e:
    print(f"⚠️  Could not load water quality API: {e}")

@app.get("/ui", response_class=HTMLResponse)
async def ui_dashboard():
    with open("templates/ui/enhanced_dashboard.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

if __name__ == "__main__":
    print("Starting minimal server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
