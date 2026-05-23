from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
import os
import sys

from .routes import isa, ai, verify, export

app = FastAPI(title="ISA AI Model API", version="1.0", description="Цифровая модель МСА с ИИ-аппроксимацией")

# CORS для фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(isa.router)
app.include_router(ai.router)
app.include_router(verify.router)
app.include_router(export.router)

# Поиск папки frontend
def get_frontend_path():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS if hasattr(sys, '_MEIPASS') else os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    possible_paths = [
        os.path.join(base_path, "frontend"),
        os.path.join(os.path.dirname(sys.executable), "frontend"),
        os.path.join(os.getcwd(), "frontend"),
    ]
    for path in possible_paths:
        if os.path.exists(path):
            return path
    raise RuntimeError(f"Не найдена папка frontend. Искали в: {possible_paths}")

frontend_path = get_frontend_path()
print(f"Frontend path: {frontend_path}")

app.mount("/static", StaticFiles(directory=os.path.join(frontend_path, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(frontend_path, "templates"))

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})