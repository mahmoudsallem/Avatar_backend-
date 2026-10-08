import sys
import time
import uuid
import base64
import logging
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app import pipeline

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("saytara")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, clean up workers at shutdown."""
    logger.info("Initializing Saytara Avatar backend...")
    logger.info("Environment: host=%s, port=%d", settings.HOST, settings.PORT)

    try:
        import torch
        assert torch.cuda.is_available(), "PyTorch cannot see the GPU - FLUX needs CUDA."
    except Exception as exc:
        logger.critical("Fatal: GPU check failed: %s", exc)
        raise

    try:
        pipeline.load_models()
        logger.info("All models and workers initialized successfully.")
    except Exception as exc:
        logger.critical("Fatal: Failed to load models on startup: %s", exc, exc_info=True)
        raise

    yield

    logger.info("Shutting down Saytara Avatar backend...")
    try:
        pipeline.shutdown_models()
    except Exception as exc:
        logger.warning("Error during model shutdown: %s", exc)
    logger.info("Shutdown complete.")

app = FastAPI(
    title="Saytara Avatar Generation API",
    description="High-performance backend for generating sci-fi avatars with FLUX and LoRA face swap.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "X-Tries-Used",
        "X-Best-Try",
        "X-Stop-Reason",
        "X-Generation-Seconds",
        "X-Identity-Similarity",
        "X-Visor-Status",
        "X-Bald-Status",
    ],
)

@app.exception_handler(pipeline.Rejected)
async def rejected_exception_handler(request: Request, exc: pipeline.Rejected):
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(exc)})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server exception processing %s %s", request.method, request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "An unexpected internal server error occurred."},
    )

@app.get("/health", summary="Health check endpoint")
async def health_check():
    import torch
    cuda_avail = False
    try:
        cuda_avail = torch.cuda.is_available()
    except Exception:
        pass

    clip_loaded = pipeline.clip is not None
    flux_loaded = pipeline.pipe is not None
    models_ready = clip_loaded and flux_loaded
    worker_ready = pipeline.is_val_worker_alive()
    free_gb, total_gb = pipeline.get_gpu_memory_info()

    return {
        "status": "ok" if (cuda_avail and models_ready and worker_ready) else "degraded",
        "cuda_available": cuda_avail,
        "device_name": pipeline.get_cuda_device_name() if cuda_avail else None,
        "models_loaded": models_ready,
        "clip_loaded": clip_loaded,
        "flux_loaded": flux_loaded,
        "val_worker_running": worker_ready,
        "lora_layers_attached": pipeline.lora_layers,
        "lora_mode": pipeline._LORA["mode"],
        "gpu_slots": len(pipeline.slots),
        "gpu_slots_requested": settings.GPU_SLOTS,
        "gpu_slot_errors": pipeline.slot_errors,
        "gpu_free_vram_gb": free_gb,
        "gpu_total_vram_gb": total_gb,
    }

async def _generate_avatar_from_bytes(content: bytes, filename: str):
    ext = Path(filename).suffix.lower()
    if ext not in pipeline.IMAGE_EXTENSIONS:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(pipeline.IMAGE_EXTENSIONS))}"},
        )

    settings.VAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = settings.VAL_TMP_DIR / f"upload_{uuid.uuid4().hex}{ext}"

    try:
        temp_path.write_bytes(content)

        # TIME_BUDGET covers the whole image (Step 1 analysis + Step 2 generation),
        # matching the notebook - start the clock before analyse_user, not after it.
        t0 = time.time()

        # Step 1 is CPU work: several requests analyse in parallel, without holding a GPU slot.
        async with pipeline.analysis_sem:
            try:
                info = await asyncio.to_thread(pipeline.analyse_user, temp_path)
            except pipeline.Rejected as r:
                logger.info("Image validation rejected for %s: %s", filename, r)
                return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(r)})

        # Step 2: take a free GPU slot (GPU_SLOTS avatars generate concurrently; extra requests wait here).
        # Returns the final avatar as JPEG bytes, already capped at settings.OUTPUT_MAX_BYTES.
        slot = await pipeline.acquire_slot()
        try:
            jpeg_bytes, metadata = await asyncio.to_thread(pipeline.generate_avatar, temp_path, info, t0, slot)
        finally:
            pipeline.release_slot(slot)

        headers = {
            "X-Tries-Used": str(metadata.get("tries", "")),
            "X-Best-Try": str(metadata.get("best_try", "")),
            "X-Stop-Reason": str(metadata.get("stopped", "")),
            "X-Generation-Seconds": str(metadata.get("seconds", "")),
            "X-Identity-Similarity": str(metadata.get("id_sim", "")),
            "X-Visor-Status": str(metadata.get("visor", "")),
            "X-Bald-Status": str(metadata.get("bald", "")),
        }
        return Response(content=jpeg_bytes, media_type="image/jpeg", headers=headers)

    except pipeline.Rejected as r:
        logger.info("Image validation rejected: %s", r)
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(r)})
    except Exception:
        logger.exception("Unexpected error during avatar generation for %s", filename)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "An internal server error occurred during avatar generation."},
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

AVATAR_RESPONSES = {
    200: {"content": {"image/jpeg": {}}, "description": "Generated avatar JPEG, at or under OUTPUT_MAX_BYTES."},
    400: {"description": "Validation rejected."},
    500: {"description": "Internal server error."},
}

@app.post(
    "/v1/avatar",
    responses=AVATAR_RESPONSES,
    summary="Generate Sci-Fi Avatar",
    description="Upload a photo, get back just the generated avatar JPEG. Serialized behind a GPU lock.",
)
async def create_avatar(
    file: UploadFile = File(..., description="User portrait photo"),
):
    filename = file.filename or "upload.jpg"
    # No server-side upload size limit - read whatever was sent.
    content = await file.read()
    return await _generate_avatar_from_bytes(content, filename)

class AvatarBase64Request(BaseModel):
    image: str  # base64-encoded image bytes (raw or "data:image/...;base64,..." data URL)
    filename: Optional[str] = "upload.jpg"  # only used to infer the file extension

@app.post(
    "/v1/avatar/base64",
    responses=AVATAR_RESPONSES,
    summary="Generate Sci-Fi Avatar (base64 input)",
    description="Same as /v1/avatar, but the photo is sent as a base64 string in a JSON body instead of multipart/form-data.",
)
async def create_avatar_base64(body: AvatarBase64Request):
    filename = body.filename or "upload.jpg"
    raw_b64 = body.image.split(",", 1)[-1] if body.image.startswith("data:") else body.image
    try:
        content = base64.b64decode(raw_b64, validate=True)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid base64 image data."},
        )
    return await _generate_avatar_from_bytes(content, filename)

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
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, reload=False, workers=1)
