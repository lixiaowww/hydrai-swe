
# Navigation Routes matching production_server.py
@app.get("/home")
async def home_page():
    """Home Page"""
    return FileResponse("templates/ui/home.html")

@app.get("/knowledge")
async def knowledge_page():
    """Knowledge Base Page"""
    return FileResponse("templates/ui/hydrological_knowledge_base.html")

@app.get("/about")
async def about_page():
    """About Page"""
    return FileResponse("templates/ui/about.html")

@app.get("/model")
async def model_page():
    """Model Training Page"""
    return FileResponse("templates/ui/model_training_dashboard.html")

@app.get("/analysis")
async def analysis_page():
    """Data Analysis Page"""
    # Check for best available analysis template
    if os.path.exists("templates/ui/analysis_dashboard_simple.html"):
        return FileResponse("templates/ui/analysis_dashboard_simple.html")
    elif os.path.exists("templates/real_data_analysis_page.html"):
        return FileResponse("templates/real_data_analysis_page.html")
    else:
        return HTMLResponse("<h1>Analysis Dashboard Not Found</h1>")
