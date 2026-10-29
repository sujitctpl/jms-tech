import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Dict
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from app.api.routes import HEALTH
from app.api.v1 import analytics, cameras, employees, identity
from app.core.config import settings
from app.core.logging import logger, setup_logging
from app.db.session import SessionLocal
from app.repositories.employee_repo import EmployeeRepository
from app.services.employee_service import global_employee_recognizer
from app.services.retention_service import retention_scheduler
from app.workers.video_worker import worker_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle."""
    setup_logging()
    logger.info("Initializing %s...", settings.APP_NAME)

    # Load active registered employee face embeddings into FAISS index
    try:
        db = SessionLocal()
        emp_repo = EmployeeRepository(db)
        active_embs = emp_repo.get_all_active_embeddings()
        global_employee_recognizer.load_active_employees(active_embs)
        logger.info("Loaded %d active employee face embeddings into Employee FAISS index.", len(active_embs))
        db.close()
    except Exception as e:
        logger.warning("Could not pre-load employee face embeddings on startup: %s", str(e))

    if settings.APP_ENV != "testing" and settings.RETENTION_AUTO_CLEANUP_ENABLED:
        retention_scheduler.start()
    yield
    logger.info("Shutting down %s. Terminating all background video workers...", settings.APP_NAME)
    retention_scheduler.stop()
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

# Mount static files directory for profile pictures and media assets
os.makedirs("static/uploads/employees", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Centralized health check endpoint (using path constant from routes.py)
@app.get(HEALTH, tags=["Health"], summary="Check whether the API is running")
def health_check() -> Dict[str, str]:
    """Check whether the API is running."""
    return {"status": "ok", "service": "Showroom Visitor Intelligence - Phase 1"}


# Register Feature API Routers
app.include_router(cameras.router)
app.include_router(analytics.router)
app.include_router(identity.router)
app.include_router(employees.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Fallback exception handler ensuring consistent error responses."""
    logger.error("Unhandled server exception: %s", str(exc), exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please check server logs."},
    )
