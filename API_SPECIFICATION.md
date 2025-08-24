# HydrAI-SWE Backend API Specification

## Overview
This document defines the complete API specification for the HydrAI-SWE Enhanced System backend. All endpoints are designed to provide real-time snow water equivalent (SWE) and runoff prediction data for water resource management.

## Base URL
```
/api
```

## Authentication
Currently, no authentication is required for the API endpoints. In production, consider implementing API key authentication.

## Common Response Format
All API responses follow this structure:
```json
{
  "status": "success|error",
  "timestamp": "ISO 8601 timestamp",
  "data": {},
  "error": {
    "code": "error_code",
    "message": "human readable error message"
  }
}
```

## Error Codes
- `400` - Bad Request (Invalid parameters)
- `404` - Not Found (Resource not found)
- `500` - Internal Server Error
- `503` - Service Unavailable (Data source unavailable)

---

## Historical Data Endpoints

### GET /api/swe/historical
Retrieves historical snow water equivalent data for specified date range and region.

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| start_date | string | Yes | Start date in YYYY-MM-DD format |
| end_date | string | Yes | End date in YYYY-MM-DD format |
| region | string | Yes | Region code: 'all', 'alberta', 'bc', 'manitoba', 'saskatchewan' |

#### Response
```json
{
  "status": "success",
  "timestamp": "2024-08-20T16:24:23Z",
  "data": {
    "region_name": "All Regions",
    "dates": ["2020-01-01", "2020-02-01", "..."],
    "swe_values": [45.2, 67.8, 89.1, "..."],
    "historical_average": [42.1, 65.2, 85.3, "..."],
    "data_sources": [
      "Environment Canada",
      "Provincial Water Survey Networks"
    ]
  }
}
```

---

## Forecast Endpoints

### GET /api/swe/forecast
Retrieves ML-generated forecast data for snow water equivalent or runoff.

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| days | integer | Yes | Number of forecast days (7, 14, 30) |
| forecast_type | string | Yes | Type: 'swe', 'runoff', 'combined' |
| region | string | No | Region code (default: 'all') |

#### Response
```json
{
  "status": "success",
  "timestamp": "2024-08-20T16:24:23Z",
  "data": {
    "forecast_type": "SWE",
    "region": "all",
    "dates": ["2024-08-21", "2024-08-22", "..."],
    "forecast_values": [195.2, 198.7, 201.3, "..."],
    "upper_confidence": [215.8, 219.6, 222.1, "..."],
    "lower_confidence": [174.6, 177.8, 180.5, "..."],
    "confidence_level": 0.95,
    "y_axis_label": "SWE (mm)",
    "model_info": {
      "name": "HydrAI-SWE v2.1",
      "last_trained": "2024-08-15T10:30:00Z",
      "accuracy_score": 0.87
    }
  }
}
```

### GET /api/swe/runoff-forecast
Retrieves runoff predictions for multiple river basins.

#### Parameters
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| days | integer | No | Number of forecast days (default: 14) |

#### Response
```json
{
  "status": "success",
  "timestamp": "2024-08-20T16:24:23Z",
  "data": {
    "dates": ["2024-08-21", "2024-08-22", "..."],
    "basins": [
      {
        "name": "Bow River",
        "basin_id": "bow_river",
        "predictions": [245.3, 248.1, 251.7, "..."],
        "color": "#e74c3c",
        "current_flow": 245.3,
        "average_flow": 232.1
      },
      {
        "name": "Fraser River",
        "basin_id": "fraser_river",
        "predictions": [1120.5, 1125.2, 1130.8, "..."],
        "color": "#27ae60",
        "current_flow": 1120.5,
        "average_flow": 1089.3
      }
    ]
  }
}
```

---

## Current Data Endpoints

### GET /api/swe/current-season-summary
Retrieves current season summary metrics.

#### Response
```json
{
  "status": "success",
  "timestamp": "2024-08-20T16:24:23Z",
  "data": {
    "total_snow": {
      "value": "285 mm",
      "trend": "increasing",
      "change_24h": "+3.2 mm"
    },
    "vs_historical": {
      "value": "+16.3%",
      "trend": "above-average",
      "percentile": 73
    },
    "peak_date": {
      "value": "Mar 15",
      "trend": "typical",
      "confidence": "high"
    },
    "active_stations": {
      "value": "187",
      "trend": "stable",
      "total_stations": 203
    },
    "season_status": "Above-average snowpack across most regions. Peak accumulation recorded in mid-March.",
    "last_updated": "2024-08-20T15:30:00Z"
  }
}
```

### GET /api/swe/regional-trends
Retrieves current regional trend comparison data.

#### Response
```json
{
  "status": "success",
  "timestamp": "2024-08-20T16:24:23Z",
  "data": {
    "regions": ["Alberta", "BC", "Manitoba", "Saskatchewan"],
    "current_values": [285, 198, 167, 234],
    "historical_averages": [245, 210, 145, 220],
    "percent_of_average": [116, 94, 115, 106],
    "trends": ["increasing", "stable", "increasing", "stable"],
    "last_updated": "2024-08-20T12:00:00Z"
  }
}
```

---

## Analysis Endpoints

### GET /api/swe/basin-analysis
Retrieves detailed basin-by-basin analysis.

#### Response
```json
{
  "status": "success",
  "timestamp": "2024-08-20T16:24:23Z",
  "data": {
    "basins": [
      {
        "name": "Bow River Basin",
        "basin_id": "bow_river",
        "current_swe": "312 mm",
        "historical_avg": "285 mm",
        "percent_of_avg": "109%",
        "trend": "increasing",
        "status": "Above Normal",
        "coordinates": {
          "lat": 51.0447,
          "lng": -114.0719
        },
        "drainage_area_km2": 26200,
        "elevation_range": {
          "min": 1050,
          "max": 3650
        }
      }
    ],
    "summary": {
      "above_normal": 4,
      "normal": 2,
      "below_normal": 0
    }
  }
}
```

### GET /api/swe/flood-risk
Retrieves flood risk assessment data.

#### Response
```json
{
  "status": "success",
  "timestamp": "2024-08-20T16:24:23Z",
  "data": {
    "risk_level": {
      "value": "Moderate",
      "color": "#f39c12",
      "level": 3,
      "max_level": 5
    },
    "peak_risk_period": {
      "value": "May 5-15",
      "start_date": "2024-05-05",
      "end_date": "2024-05-15",
      "days_away": 18
    },
    "regions_at_risk": {
      "value": "3",
      "count": 3,
      "list": ["Bow River", "Red Deer", "North Saskatchewan"],
      "details": [
        {
          "name": "Bow River",
          "risk_level": "Moderate",
          "probability": 0.35
        }
      ]
    },
    "alert_lead_time": {
      "value": "18-21 days",
      "min_days": 18,
      "max_days": 21,
      "confidence": "high"
    },
    "alert_message": "Moderate flood risk for Bow River, Red Deer, and North Saskatchewan basins due to above-average snowpack. Monitor forecasts closely in early May.",
    "recommendations": [
      "Monitor weather forecasts for temperature increases",
      "Check flood preparedness plans for affected areas",
      "Review dam and reservoir management protocols"
    ]
  }
}
```

### GET /api/swe/regional-forecast
Retrieves detailed regional forecast information.

#### Response
```json
{
  "status": "success",
  "timestamp": "2024-08-20T16:24:23Z",
  "data": {
    "regions": [
      {
        "name": "Alberta Rockies",
        "region_id": "ab_rockies",
        "current_swe": "312 mm",
        "forecast_7day": "+15 mm",
        "forecast_change": 15.2,
        "peak_runoff_date": "May 12-18",
        "expected_volume": "High (120%)",
        "volume_percent": 120,
        "risk_level": "Moderate",
        "confidence": 0.82,
        "coordinates": {
          "center_lat": 52.2,
          "center_lng": -117.1
        }
      }
    ],
    "forecast_metadata": {
      "model_run": "2024-08-20T06:00:00Z",
      "next_update": "2024-08-20T18:00:00Z",
      "data_quality": "good"
    }
  }
}
```

---

## Data Quality and Metadata

### GET /api/swe/data-status
Retrieves system status and data quality information.

#### Response
```json
{
  "status": "success",
  "timestamp": "2024-08-20T16:24:23Z",
  "data": {
    "system_status": "operational",
    "data_sources": [
      {
        "name": "Environment Canada",
        "status": "online",
        "last_update": "2024-08-20T15:00:00Z",
        "coverage": "national",
        "station_count": 145
      },
      {
        "name": "Provincial Networks",
        "status": "online",
        "last_update": "2024-08-20T14:30:00Z",
        "coverage": "regional",
        "station_count": 58
      }
    ],
    "model_status": {
      "swe_model": {
        "status": "operational",
        "last_trained": "2024-08-15T10:30:00Z",
        "accuracy": 0.87,
        "version": "2.1.3"
      },
      "runoff_model": {
        "status": "operational",
        "last_trained": "2024-08-14T08:15:00Z",
        "accuracy": 0.84,
        "version": "1.8.2"
      }
    },
    "data_coverage": {
      "alberta": 0.92,
      "bc": 0.88,
      "manitoba": 0.85,
      "saskatchewan": 0.89
    }
  }
}
```

---

## Error Handling Examples

### 400 Bad Request
```json
{
  "status": "error",
  "timestamp": "2024-08-20T16:24:23Z",
  "data": null,
  "error": {
    "code": "INVALID_DATE_RANGE",
    "message": "Start date must be before end date",
    "details": {
      "start_date": "2024-12-01",
      "end_date": "2024-01-01"
    }
  }
}
```

### 503 Service Unavailable
```json
{
  "status": "error",
  "timestamp": "2024-08-20T16:24:23Z",
  "data": null,
  "error": {
    "code": "DATA_SOURCE_UNAVAILABLE",
    "message": "Environment Canada data feed is temporarily unavailable",
    "retry_after": 300
  }
}
```

---

## Implementation Notes

### Database Schema Considerations
- **historical_swe**: Store daily SWE measurements with station_id, date, value, region
- **forecasts**: Store model predictions with model_version, run_time, forecast_date, value
- **stations**: Store metadata about monitoring stations
- **basins**: Store basin boundaries and characteristics

### Caching Strategy
- Cache historical data for 1 hour (updates daily)
- Cache forecast data for 15 minutes (updates every 6 hours)
- Cache current season summary for 30 minutes
- Use Redis or similar for high-performance caching

### Rate Limiting
- Implement rate limiting: 1000 requests per hour per IP
- Use 429 status code when limits exceeded

### Data Validation
- Validate all date parameters (YYYY-MM-DD format)
- Validate region parameters against allowed values
- Validate forecast days parameter (max 90 days)

### Monitoring and Logging
- Log all API requests with response times
- Monitor data source availability
- Alert on model prediction accuracy degradation
- Track API usage statistics

This specification provides a complete foundation for implementing the backend API that supports the HydrAI-SWE Enhanced System frontend.
