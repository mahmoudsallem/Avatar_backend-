import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.pipeline import models, step1
from app.routes.avatar import router as avatar_router

# Configure application logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("saytara")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler to load models once at startup and clean up at shutdown."""
    logger.info("Initializing Saytara Avatar backend...")
    logger.info("Environment: host=%s, port=%d", settings.HOST, settings.PORT)

    # Fail fast if CUDA is unavailable
    try:
        import torch
        assert torch.cuda.is_available(), "PyTorch cannot see the GPU - FLUX needs CUDA."
    except Exception as exc:
        logger.critical("Fatal: GPU check failed: %s", exc)
        raise

    # Load pipelines and start background workers
    try:
        models.load_models()
        logger.info("All models and workers initialized successfully.")
    except Exception as exc:
        logger.critical("Fatal: Failed to load models on startup: %s", exc, exc_info=True)
        raise

    yield

    # Clean shutdown
    logger.info("Shutting down Saytara Avatar backend...")
    try:
        models.shutdown_models()
    except Exception as exc:
        logger.warning("Error during model shutdown: %s", exc)
    logger.info("Shutdown complete.")

app = FastAPI(
    title="Saytara Avatar Generation API",
    description="High-performance backend for generating sci-fi avatars with FLUX and LoRA face swap.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for validation rejections
@app.exception_handler(step1.Rejected)
async def rejected_exception_handler(request: Request, exc: step1.Rejected):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error": str(exc)},
    )

# Fallback generic exception handler for unhandled errors
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server exception processing %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An unexpected internal server error occurred."},
    )

# Include routes
app.include_router(avatar_router)

@app.get("/", summary="Root status", include_in_schema=False)
async def root():
    return {
        "service": "Saytara Avatar Generation API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=False, workers=1)
