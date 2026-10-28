from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from app.api.routes import HEALTH
from app.api.v1 import analytics, cameras
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.workers.video_worker import worker_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle."""
    setup_logging()
    logger.info("Initializing %s...", settings.APP_NAME)
    yield
    logger.info("Shutting down %s. Terminating all background video workers...", settings.APP_NAME)
    worker_manager.stop_all()


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Phase 1: AI-Based Visitor Detection and Entry Counting for Showrooms",
    lifespan=lifespan,
)

# Enable CORS for dashboard local development and cross-origin access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Centralized health check endpoint (using path constant from routes.py)
@app.get(HEALTH, tags=["Health"], summary="Check whether the API is running")
def health_check() -> Dict[str, str]:
    """Check whether the API is running."""
    return {"status": "ok", "service": "Showroom Visitor Intelligence - Phase 1"}


# Register Feature API Routers
app.include_router(cameras.router)
app.include_router(analytics.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback exception handler ensuring consistent error responses."""
    logger.error("Unhandled server exception: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please check server logs."},
    )
