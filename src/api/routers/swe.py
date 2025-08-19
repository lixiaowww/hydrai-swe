from fastapi import APIRouter, Query
from datetime import datetime, timedelta
import os
import pandas as pd
from src.models.predict_service import NeuralHydrologyPredictor
import math

router = APIRouter()

@router.get("/runoff-forecast")
def get_runoff_forecast(
    station_id: str = Query(..., description="The ID of the station to get the forecast for."),
    start_date: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end_date: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    mode: str = Query("nowcast", description="Prediction mode: nowcast or scenario"),
    scenario_year: int | None = Query(None, description="Historical year to use for scenario mode"),
):
    """
    Get the runoff forecast for a given station.
    Modes:
      - nowcast: use trained model if available, otherwise deterministic fallback
      - scenario: use historical snow depth (Manitoba ECCC) with simple melt-based flow estimate
    """
    # Derive date range (default to 7 days if not provided)
    try:
        if start_date:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            start_dt = datetime.today()
        if end_date:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        else:
            end_dt = start_dt + timedelta(days=6)
        if end_dt < start_dt:
            start_dt, end_dt = end_dt, start_dt
    except Exception:
        start_dt = datetime.today()
        end_dt = start_dt + timedelta(days=6)

    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    if mode == "scenario":
        # Map requested dates to scenario_year or use climatology by day-of-year if year not present
        if scenario_year is None:
            scenario_year = start_dt.year - 1

        # Prefer recent combined ECCC (2020-2024) for better coverage
        recent_path = "data/raw/eccc_recent/eccc_recent_combined.csv"
        processed_path = "data/processed/eccc_manitoba_snow_processed.csv"
        src_path = recent_path if os.path.exists(recent_path) else processed_path

        forecasts: list[dict] = []
        try:
            df = pd.read_csv(src_path)
            # Normalize date column
            date_col = 'date' if 'date' in df.columns else ('Date/Time' if 'Date/Time' in df.columns else None)
            if date_col is None:
                raise ValueError("No date column in ECCC data")
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

            if 'Snow on Grnd (cm)' not in df.columns:
                raise ValueError("Missing 'Snow on Grnd (cm)' column in ECCC data")

            # Region filter if lat/lon present (Red River Basin)
            if {'Latitude (y)', 'Longitude (x)'}.issubset(df.columns):
                lon_min, lat_min, lon_max, lat_max = (-97.5, 49.0, -96.5, 50.5)
                df = df[(df['Longitude (x)'] >= lon_min) & (df['Longitude (x)'] <= lon_max) & (df['Latitude (y)'] >= lat_min) & (df['Latitude (y)'] <= lat_max)]

            # Daily mean snow depth (cm)
            daily = (
                df[[date_col, 'Snow on Grnd (cm)']]
                .dropna()
                .groupby(date_col)['Snow on Grnd (cm)']
                .mean()
                .sort_index()
            )

            dates_req = pd.date_range(start=start_dt, end=end_dt, freq='D')

            # If chosen scenario_year exists, use that series; else build climatology by day-of-year
            has_year = any(d.year == scenario_year for d in daily.index)
            if has_year:
                series_year = daily[daily.index.year == scenario_year]
                mapping = { (d.month, d.day): v for d, v in series_year.items() }
                default_cm = float(series_year.mean()) if len(series_year) else 0.0
                snow_mm = [ float(mapping.get((d.month, d.day), default_cm)) * 10.0 for d in dates_req ]
            else:
                # Climatology across all years by month/day
                df_day = daily.reset_index()
                df_day['month'] = df_day[date_col].dt.month
                df_day['day'] = df_day[date_col].dt.day
                clim = df_day.groupby(['month','day'])['Snow on Grnd (cm)'].mean()
                default_cm = float(clim.mean()) if len(clim) else 0.0
                snow_mm = [ float(clim.get((d.month, d.day), default_cm)) * 10.0 for d in dates_req ]

            # Simple melt-to-flow model with smoothing
            base_flow = 950.0
            coef = 0.2
            # sanitize NaNs
            prev = 0.0
            if snow_mm:
                v0 = snow_mm[0]
                prev = 0.0 if (v0 is None or math.isnan(v0) or math.isinf(v0)) else v0
            for i, req_date in enumerate(dates_req):
                v = snow_mm[i] if i < len(snow_mm) else prev
                current = 0.0 if (v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))) else float(v)
                snow_melt = max(prev - current, 0.0)
                flow = base_flow + coef * snow_melt
                # final safety for JSON serializable finite numbers
                flow = 0.0 if (isinstance(flow, float) and (math.isnan(flow) or math.isinf(flow))) else float(flow)
                forecasts.append({
                    "date": req_date.strftime("%Y-%m-%d"),
                    "predicted_flow": float(flow),
                })
                prev = current
        except Exception:
            # Fallback deterministic series
            for i in range((end_dt - start_dt).days + 1):
                forecasts.append({
                    "date": (start_dt + timedelta(days=i)).strftime("%Y-%m-%d"),
                    "predicted_flow": 1000.0,
                })

        return {
            "station_id": station_id,
            "forecasts": forecasts,
        }

    # Default: nowcast using predictor (with internal fallback)
    predictor = NeuralHydrologyPredictor()
    forecasts = predictor.predict_daily(station_id=station_id, start_date=start_str, end_date=end_str)
    return {
        "station_id": station_id,
        "forecasts": forecasts,
    }
