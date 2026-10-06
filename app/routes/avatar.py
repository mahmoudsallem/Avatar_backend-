import io
import time
import uuid
import secrets
import logging
import asyncio
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    File,
    UploadFile,
    Header,
    HTTPException,
    Response,
    status,
)
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import HealthResponse, ErrorResponse
from app.pipeline import models, step1, generate

logger = logging.getLogger("saytara.api")

router = APIRouter()

def verify_api_key(x_api_key: Optional[str]) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, settings.API_KEY):
        logger.warning("Unauthorized request: missing or invalid X-API-Key header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-API-Key header",
        )

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    description="Reports process health, CUDA availability, VRAM stats, and model loading status.",
)
async def health_check():
    import torch
    cuda_avail = False
    try:
        cuda_avail = torch.cuda.is_available()
    except Exception:
        pass

    clip_loaded = models.clip is not None
    flux_loaded = models.pipe is not None
    models_ready = clip_loaded and flux_loaded
    worker_ready = models.is_val_worker_alive()
    free_gb, total_gb = models.get_gpu_memory_info()

    service_status = "ok" if (cuda_avail and models_ready and worker_ready) else "degraded"

    return HealthResponse(
        status=service_status,
        cuda_available=cuda_avail,
        device_name=models.get_cuda_device_name() if cuda_avail else None,
        models_loaded=models_ready,
        clip_loaded=clip_loaded,
        flux_loaded=flux_loaded,
        val_worker_running=worker_ready,
        gpu_free_vram_gb=free_gb,
        gpu_total_vram_gb=total_gb,
    )

@router.post(
    "/v1/avatar",
    responses={
        200: {
            "content": {"image/png": {}},
            "description": "Generated avatar image with metadata in response headers.",
        },
        400: {"model": ErrorResponse, "description": "Validation rejected."},
        401: {"model": ErrorResponse, "description": "Unauthorized."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
    },
    summary="Generate Sci-Fi Avatar",
    description="Upload a photo to generate an avatar. Serialized behind GPU lock; blocking work in thread pool.",
)
async def create_avatar(
    file: UploadFile = File(..., description="User portrait photo"),
    x_api_key: Optional[str] = Header(None, alias="X-API-Key"),
):
    verify_api_key(x_api_key)

    filename = file.filename or "upload.jpg"
    ext = Path(filename).suffix.lower()
    if ext not in step1.IMAGE_EXTENSIONS:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(step1.IMAGE_EXTENSIONS))}"},
        )

    # Read and enforce max upload size
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": f"Uploaded file exceeds maximum limit of {settings.MAX_UPLOAD_MB}MB"},
        )

    # Write temporarily to disk for CV2 / DeepFace processing
    settings.VAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_filename = f"upload_{uuid.uuid4().hex}{ext}"
    temp_path = settings.VAL_TMP_DIR / temp_filename

    try:
        temp_path.write_bytes(content)

        # Acquire GPU lock to serialize generation requests safely
        async with models.gpu_lock:
            # Step 1: Analyze user photo (run in thread pool)
            try:
                info = await asyncio.to_thread(step1.analyse_user, temp_path)
            except step1.Rejected as r:
                logger.info("Image validation rejected for %s: %s", filename, r)
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"error": str(r)},
                )

            # Step 2: Generate avatar (run in thread pool)
            out_img, metadata = await asyncio.to_thread(
                generate.generate_avatar, temp_path, info
            )

        # Encode image to PNG bytes
        buffer = io.BytesIO()
        out_img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        # Build metadata headers
        headers = {
            "X-Avatar-Type": str(info.get("avatar", "")),
            "X-Gender": str(info.get("gender", "")),
            "X-Glasses": str(info.get("glasses", False)).lower(),
            "X-Hijab": str(info.get("hijab", False)).lower(),
            "X-Beard": str(info.get("beard", False)).lower(),
            "X-Identity-Similarity": str(metadata.get("id_sim", "")),
            "X-Visor-Status": str(metadata.get("visor", "")),
            "X-Tries": str(metadata.get("tries", 1)),
            "X-Seconds-Elapsed": str(metadata.get("seconds", 0.0)),
        }

        return Response(content=png_bytes, media_type="image/png", headers=headers)

    except step1.Rejected as r:
        logger.info("Image validation rejected: %s", r)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": str(r)},
        )
    except Exception as exc:
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
