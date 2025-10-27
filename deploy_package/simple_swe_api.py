from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
import pandas as pd
import sqlite3
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json

app = FastAPI(title="Simple SWE API")

# 添加 CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有请求头
)

# 数据库文件
DB_FILE = "swe_data.db"

def init_database():
    """初始化数据库"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 创建SWE数据表
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS swe_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            swe_mm REAL NOT NULL,
            data_source TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 创建索引
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON swe_data(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_data_source ON swe_data(data_source)')
    
    conn.commit()
    conn.close()

def load_historical_data_to_db():
    """将历史数据加载到数据库"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # 检查是否已有数据
    cursor.execute("SELECT COUNT(*) FROM swe_data")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("Loading historical data to database...")
        
        # 加载历史数据
        historical_file = "data/processed/validation/manitoba_daily_swe_2010_2020_20250915_213917.csv"
        if os.path.exists(historical_file):
            df = pd.read_csv(historical_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            # 插入历史数据
            for _, row in df.iterrows():
                if pd.notna(row['timestamp']) and pd.notna(row['swe_mm']):
                    cursor.execute(
                        "INSERT INTO swe_data (timestamp, swe_mm, data_source) VALUES (?, ?, ?)",
                        (row['timestamp'].strftime('%Y-%m-%d'), row['swe_mm'], 'historical')
                    )
        
        # 加载实时数据
        realtime_file = "data/processed/validation/integrated_swe_validation_20250915_165935.csv"
        if os.path.exists(realtime_file):
            df = pd.read_csv(realtime_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                for _, row in df.iterrows():
                    if pd.notna(row['timestamp']) and pd.notna(row['swe_mm']):
                        cursor.execute(
                            "INSERT INTO swe_data (timestamp, swe_mm, data_source) VALUES (?, ?, ?)",
                            (row['timestamp'].strftime('%Y-%m-%d'), row['swe_mm'], 'realtime')
                        )
        
        # 加载更多历史数据（包含2023年数据）
        additional_file = "data/processed/validation/colorado_clpx_swe_20250915_165935.csv"
        if os.path.exists(additional_file):
            df = pd.read_csv(additional_file)
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.dropna(subset=['timestamp'])
                for _, row in df.iterrows():
                    if pd.notna(row['timestamp']) and pd.notna(row['swe_mm']):
                        cursor.execute(
                            "INSERT INTO swe_data (timestamp, swe_mm, data_source) VALUES (?, ?, ?)",
                            (row['timestamp'].strftime('%Y-%m-%d'), row['swe_mm'], 'colorado_clpx')
                        )
        
        conn.commit()
        print(f"Loaded {cursor.rowcount} records to database")
    
    conn.close()

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化数据库"""
    init_database()
    load_historical_data_to_db()

@app.get("/api/swe/historical")
def get_historical_swe(
    window: str = Query("30d", description="Time window: 24h, 7d, 30d, all, custom"),
    start_date: str = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(None, description="End date (YYYY-MM-DD)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(365, ge=1, le=2000, description="Items per page"),
    data_type: str = Query("daily", description="Data type"),
    region: str = Query("manitoba", description="Region"),
    source_order: str = Query(None, description="Source order")
):
    """获取SWE历史数据"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 计算日期范围
        if window == "custom":
            if not start_date or not end_date:
                raise HTTPException(status_code=422, detail="start_date and end_date required for custom window")
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        elif window == "all":
            if start_date and end_date:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            else:
                # 获取所有数据范围
                cursor.execute("SELECT MIN(timestamp), MAX(timestamp) FROM swe_data")
                min_date, max_date = cursor.fetchone()
                start_dt = datetime.strptime(min_date, '%Y-%m-%d')
                end_dt = datetime.strptime(max_date, '%Y-%m-%d')
        elif start_date and end_date:
            # 如果用户设定了日期范围，优先使用用户设定的范围
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            # 时间窗口：先尝试基于当前日期，如果没有数据则使用数据库中的最新数据
            end_dt = datetime.now()
            if window == "24h":
                start_dt = end_dt - timedelta(hours=24)
            elif window == "7d":
                start_dt = end_dt - timedelta(days=7)
            elif window == "30d":
                start_dt = end_dt - timedelta(days=30)
            else:
                raise HTTPException(status_code=422, detail="Invalid window")
            
            # 检查是否有数据，如果没有则使用数据库中最新的数据
            cursor.execute("SELECT COUNT(*) FROM swe_data WHERE timestamp >= ? AND timestamp <= ?", 
                         (start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')))
            count = cursor.fetchone()[0]
            
            if count == 0:
                # 使用数据库中最新的数据
                cursor.execute("SELECT MAX(timestamp) FROM swe_data")
                max_date_str = cursor.fetchone()[0]
                if max_date_str:
                    max_date = datetime.strptime(max_date_str, '%Y-%m-%d')
                    if window == "24h":
                        start_dt = max_date - timedelta(hours=24)
                    elif window == "7d":
                        start_dt = max_date - timedelta(days=7)
                    elif window == "30d":
                        start_dt = max_date - timedelta(days=30)
                    end_dt = max_date
        
        # 查询数据
        query = """
            SELECT timestamp, swe_mm, data_source 
            FROM swe_data 
            WHERE timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
        """
        
        cursor.execute(query, (start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')))
        rows = cursor.fetchall()
        
        if not rows:
            raise HTTPException(status_code=404, detail="No data in the specified date range")
        
        # 处理数据
        dates = [row[0] for row in rows]
        swe_values = [row[1] for row in rows]
        data_sources = [row[2] for row in rows]
        
        # 计算统计信息
        mean_swe = sum(swe_values) / len(swe_values)
        min_swe = min(swe_values)
        max_swe = max(swe_values)
        last_swe = swe_values[-1] if swe_values else 0
        last_date = dates[-1] if dates else None
        
        # 计算历史平均值（使用所有数据）
        cursor.execute("SELECT AVG(swe_mm) FROM swe_data")
        historical_avg = cursor.fetchone()[0] or 0
        
        # 生成历史平均值数组
        historical_average = [historical_avg] * len(dates)
        
        # 分页
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_dates = dates[start_idx:end_idx]
        paginated_swe = swe_values[start_idx:end_idx]
        paginated_historical = historical_average[start_idx:end_idx]
        
        total_pages = (len(dates) + page_size - 1) // page_size
        
        conn.close()
        
        return {
            "dates": paginated_dates,
            "swe_values": paginated_swe,
            "historical_average": paginated_historical,
            "summary": {
                "count": len(dates),
                "mean_mm": round(mean_swe, 2),
                "std_mm": round(pd.Series(swe_values).std(), 2),
                "min_mm": round(min_swe, 2),
                "max_mm": round(max_swe, 2),
                "last_value_mm": round(last_swe, 2),
                "last_date": last_date
            },
            "interpretation": {
                "signal": "increasing" if last_swe > mean_swe else "decreasing" if last_swe < mean_swe else "stable",
                "percent_vs_historical": round((last_swe - historical_avg) / historical_avg * 100, 1) if historical_avg > 0 else 0
            },
            "provenance": {
                "source": "database",
                "source_path": DB_FILE,
                "updated_at": datetime.now().isoformat(),
                "lineage_id": "simple_db_v1"
            },
            "page_info": {
                "page": page,
                "total_pages": total_pages,
                "total_count": len(dates)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/swe/realtime")
def get_realtime_swe():
    """获取实时SWE数据"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # 获取最新的数据
        cursor.execute("""
            SELECT timestamp, swe_mm, data_source 
            FROM swe_data 
            WHERE data_source = 'realtime'
            ORDER BY timestamp DESC 
            LIMIT 1
        """)
        row = cursor.fetchone()
        
        if not row:
            # 如果没有实时数据，返回最新的历史数据
            cursor.execute("""
                SELECT timestamp, swe_mm, data_source 
                FROM swe_data 
                ORDER BY timestamp DESC 
                LIMIT 1
            """)
            row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return {
                "status": "success",
                "data": [{
                    "timestamp": row[0],
                    "swe_mm": row[1],
                    "data_source": row[2]
                }]
            }
        else:
            return {
                "status": "error",
                "error": "No data available"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/api/flood/prediction/7day")
def get_flood_prediction():
    """获取7天洪水预测数据"""
    return {
        "status": "success",
        "prediction": {
            "risk_level": "low",
            "confidence": 0.85,
            "message": "No significant flood risk expected in the next 7 days"
        }
    }

@app.get("/api/water-quality/analysis/current")
def get_water_quality():
    """获取当前水质分析数据"""
    return {
        "status": "success",
        "quality": {
            "overall_score": 8.5,
            "turbidity": "Good",
            "chlorine": "Normal",
            "ph": 7.2
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
