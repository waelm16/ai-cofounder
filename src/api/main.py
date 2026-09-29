"""FastAPI application entry point.

Creates the app instance, registers CORS middleware, mounts all route modules,
and initialises the database on startup via the lifespan context manager.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import calls, chat, company, knowledge, users, validation
from src.memory.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler — creates DB tables on startup."""
    init_db()
    yield


app = FastAPI(
    title="AI Cofounder",
    description="Multi-user AI startup advisor with RAG over business knowledge and persistent memory",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(users.router)
app.include_router(company.router)
app.include_router(validation.router)
app.include_router(knowledge.router)
app.include_router(calls.router)
app.include_router(chat.router)


@app.get("/health")
def health_check():
    """Lightweight liveness probe for monitoring and load-balancer health checks.

    Returns:
        dict: ``{"status": "ok"}``
    """
    return {"status": "ok"}


@app.get("/")
def root():
    """Redirect the root URL to the chat UI."""
    return RedirectResponse(url="/static/index.html")


# Mount UI static files last so API routes take precedence
_ui_dir = Path(__file__).resolve().parent.parent / "ui"
app.mount("/static", StaticFiles(directory=str(_ui_dir)), name="static")
