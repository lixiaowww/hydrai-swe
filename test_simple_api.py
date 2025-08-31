#!/usr/bin/env python3
"""
简单的API测试脚本
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# 创建简单的FastAPI应用
app = FastAPI(title="Simple Test API")

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "Simple test API is running"}

@app.get("/api/swe/insight-discovery")
def test_insight_discovery():
    return {
        "success": True,
        "message": "Test endpoint working",
        "data": {"test": "insight discovery"}
    }

@app.get("/api/swe/historical")
def test_historical():
    return {
        "success": True,
        "message": "Test endpoint working",
        "data": {"test": "historical data"}
    }

if __name__ == "__main__":
    print("🚀 启动简单测试API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
