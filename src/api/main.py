from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from src.api.routers import swe

app = FastAPI(
    title="HydrAI-SWE API",
    description="API for the HydrAI-SWE project to serve snow water equivalent (SWE) and runoff predictions.",
    version="1.0.0",
)

app.include_router(swe.router, prefix="/api/v1", tags=["swe"])

@app.get("/")
def read_root():
    return {"message": "Welcome to the HydrAI-SWE API"}

templates = Jinja2Templates(directory="templates")

@app.get("/ui", response_class=HTMLResponse)
def ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
