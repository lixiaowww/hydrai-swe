import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

# Import only the SWE router for now
try:
    from src.api.routers import swe
    SWE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import SWE router: {e}")
    SWE_AVAILABLE = False

# Try to import other routers
try:
    from src.api.routers import flood_warning
    FLOOD_AVAILABLE = True
except ImportError:
    FLOOD_AVAILABLE = False

# Try to import enhanced interpretation service
try:
    from src.api.routers import enhanced_interpretation
    ENHANCED_INTERPRETATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import enhanced interpretation router: {e}")
    ENHANCED_INTERPRETATION_AVAILABLE = False

try:
    from src.api.routers import advanced_flood_warning
    ADVANCED_FLOOD_AVAILABLE = True
except ImportError:
    ADVANCED_FLOOD_AVAILABLE = False

try:
    from src.api.routers import agriculture
    AGRICULTURE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agriculture router: {e}")
    AGRICULTURE_AVAILABLE = False

try:
    from src.api.routers import pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

try:
    from src.api.routers import data_science
    DATA_SCIENCE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import data science router: {e}")
    DATA_SCIENCE_AVAILABLE = False

app = FastAPI(
    title="HydrAI-SWE API",
    description="API for the HydrAI-SWE project to serve snow water equivalent (SWE), runoff predictions, flood warning services, and historical data cross-validation.",
    version="1.0.0",
)

# Enable gzip compression to speed up payload transfer
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Enable CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Include available routers
if SWE_AVAILABLE:
    app.include_router(swe.router, prefix="/api/swe", tags=["swe"])

if FLOOD_AVAILABLE:
    app.include_router(flood_warning.router, prefix="/api/v1/flood", tags=["flood_warning"])
    app.include_router(flood_warning.router, prefix="/api/manitoba-flood", tags=["manitoba_flood"])

if AGRICULTURE_AVAILABLE:
    app.include_router(agriculture.router, prefix="/api/v1/agriculture", tags=["agriculture"])

# Temporarily disabled advanced flood warning due to data quality issues
# if ADVANCED_FLOOD_AVAILABLE:
#     app.include_router(advanced_flood_warning.router, prefix="/api/v2/flood", tags=["advanced_flood_warning"])
#     app.include_router(advanced_flood_warning.router, prefix="/api/manitoba-flood/advanced", tags=["manitoba_flood_advanced"])

if PIPELINE_AVAILABLE:
    app.include_router(pipeline.router, prefix="/api/v1", tags=["pipeline"])

if DATA_SCIENCE_AVAILABLE:
    app.include_router(data_science.router, prefix="/api/v1", tags=["data_science"])

# Include enhanced interpretation service
if ENHANCED_INTERPRETATION_AVAILABLE:
    app.include_router(enhanced_interpretation.router, tags=["enhanced_interpretation"])

@app.get("/")
def read_root():
    # Redirect root to the main UI homepage
    return RedirectResponse(url="/ui")

@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "HydrAI-SWE API is running"}

templates = Jinja2Templates(directory="templates")

# Mount static files for UI assets
app.mount("/static", StaticFiles(directory="templates/ui"), name="ui_static")

# Serve generated analysis visualizations (HTML files)
if os.path.isdir("analysis_results"):
    app.mount("/analysis_results", StaticFiles(directory="analysis_results"), name="analysis_results")

@app.get('/favicon.ico')
def favicon():
    # Provide a small inline SVG as favicon to avoid 404s
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'>"
        "<rect width='64' height='64' rx='12' fill='#4A90E2'/><path d='M16 40 L32 16 L48 40' stroke='white' stroke-width='6' fill='none'/>"
        "</svg>"
    )
    return HTMLResponse(content=svg, media_type='image/svg+xml')

@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    # Main end-user interface - English enhanced version
    return templates.TemplateResponse("ui/enhanced_en.html", {"request": request})

@app.get("/ui/enhanced_en", response_class=HTMLResponse)
def ui_enhanced_english(request: Request):
    # Enhanced English UI - direct access
    return templates.TemplateResponse("ui/enhanced_en.html", {"request": request})

@app.get("/model", response_class=HTMLResponse)
def model_interface(request: Request):
    # Model training, configuration, and technical settings interface
    return templates.TemplateResponse("ui/model_interface.html", {"request": request})

@app.get("/ui/legacy", response_class=HTMLResponse)
def ui_legacy(request: Request):
    # Legacy UI for backwards compatibility
    return templates.TemplateResponse("index.html", {"request": request})

# Removed Chinese enhanced UI route - deprecated and removed due to data import errors

@app.get("/ui/francais", response_class=HTMLResponse)
def ui_francais(request: Request):
    # French enhanced interface
    return templates.TemplateResponse("ui/enhanced_fr.html", {"request": request})

@app.get("/ui/factors-discovery-echarts", response_class=HTMLResponse)
def ui_factors_discovery_echarts(request: Request):
    # Factors Discovery Module with ECharts - Advanced Data Science Analysis
    return templates.TemplateResponse("ui/enhanced_en_echarts.html", {"request": request})

@app.get("/ui/cree", response_class=HTMLResponse)
def ui_cree(request: Request):
    # Cree language interface
    return templates.TemplateResponse("ui/enhanced_cr.html", {"request": request})

@app.get("/ui/simple_fr", response_class=HTMLResponse)
def ui_simple_french(request: Request):
    # Simple French interface with translated content
    return templates.TemplateResponse("ui/simple_fr.html", {"request": request})

@app.get("/ui/simple_cr", response_class=HTMLResponse)
def ui_simple_cree(request: Request):
    # Simple Cree interface with translated content
    return templates.TemplateResponse("ui/simple_cr.html", {"request": request})

@app.get("/ui/multilingual", response_class=HTMLResponse)
def ui_multilingual(request: Request):
    # True multilingual interface with dynamic language switching
    return templates.TemplateResponse("ui/multilingual_dashboard.html", {"request": request})

@app.get("/ui/vnext", response_class=HTMLResponse)
def ui_vnext(request: Request):
    # Next-generation UI prototype
    return templates.TemplateResponse("ui_vnext.html", {"request": request})

@app.get("/hydrological-center", response_class=HTMLResponse)
def hydrological_center_dashboard(request: Request):
    # Manitoba Real-time Hydrological Data Center with Flood Warning System
    return templates.TemplateResponse("ui/flood_warning_dashboard.html", {"request": request})

@app.get("/ui/flood-warning-demo", response_class=HTMLResponse)
def flood_warning_demo(request: Request):
    # Flood Warning Feature Demo
    return templates.TemplateResponse("ui/flood_warning_demo.html", {"request": request})

@app.get("/agriculture", response_class=HTMLResponse)
def agriculture(request: Request):
    # Agriculture Intelligence Suite - dedicated applications interface
    return templates.TemplateResponse("ui/applications.html", {"request": request})

@app.get("/applications", response_class=HTMLResponse)
def applications(request: Request):
    # Applications interface - alternative route
    return templates.TemplateResponse("ui/applications.html", {"request": request})

@app.get("/applications/data-authenticity", response_class=HTMLResponse)
def applications_data_authenticity_report(request: Request):
    # Data Authenticity Analysis Report for Applications Module
    return templates.TemplateResponse("ui/data_authenticity_report.html", {"request": request})

@app.get("/agriculture/data-authenticity", response_class=HTMLResponse)
def agriculture_data_authenticity_report(request: Request):
    # Data Authenticity Analysis Report for Agriculture Module
    return templates.TemplateResponse("ui/data_authenticity_report.html", {"request": request})

@app.get("/guides", response_class=HTMLResponse)
def user_guide(request: Request):
    # Comprehensive User Guide for HydrAI-SWE System
    return templates.TemplateResponse("ui/user_guide.html", {"request": request})



@app.get("/api-docs", response_class=HTMLResponse)
def api_documentation(request: Request):
    # Custom API Documentation with consistent navigation header
    return templates.TemplateResponse("ui/api_docs.html", {"request": request})

# Removed additional UI variants to avoid accidental use of heavy UIs
