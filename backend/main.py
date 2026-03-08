"""
FastAPI application entry point.

Start with:
    uvicorn backend.main:app --reload --port 8000

Endpoints:
    GET  /app          → serves frontend/index.html
    GET  /static/*     → serves frontend/ static assets
    WS   /ws/generate  → streaming generation WebSocket
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import HOST, PORT
from backend.model_loader import get_model_and_tokenizer
from backend.websocket_handler import handle_generation

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model once at startup so the first request isn't slow
    get_model_and_tokenizer()
    yield


app = FastAPI(title="LLM Visualizer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# Serve static assets from frontend/
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/app")
async def serve_app():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/")
async def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.websocket("/ws/generate")
async def ws_generate(websocket: WebSocket):
    await handle_generation(websocket)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=HOST, port=PORT, reload=True)
