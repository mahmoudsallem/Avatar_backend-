import os
# Less allocator fragmentation = fewer CUDA OOMs with several concurrent GPU slots. Must be set before torch is imported.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
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
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import settings
from app import pipeline
from app import s3_watcher

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("saytara")

_models_loaded = False

def _sync_model_caches_from_s3():
    """Pull all three S3-cached model sets (HF hub, DeepFace weights, CLIP) into their local
    directories via install_models.py's own pull() logic, before pipeline.load_models() runs.
    A fresh container has none of these on disk - without this, load_models() would fall back to
    downloading FLUX.2 (gated) and DeepFace's weights straight from the internet on every boot."""
    import install_models

    s3 = install_models._client()
    for name, local_root, is_hf_cache in install_models.MODEL_SETS:
        s3_prefix = f"{settings.S3_MODEL_CACHE_PREFIX}{name}/"
        try:
            install_models.pull(s3, settings.S3_MODEL_BUCKET, name, local_root, s3_prefix, is_hf_cache, dry_run=False)
        except Exception as exc:
            logger.warning("Model cache pull failed for %s (will fall back to downloading on demand): %s", name, exc)


def _download_s3_prefix_if_needed():
    """Download all objects under s3://bucket/prefix (or tar.gz archive) into dest_dir if MODEL_S3_URI is set."""
    dest = Path(os.environ.get("LOCAL_MODEL_DIR", os.environ.get("HF_HOME", "/models/huggingface")))
    dest.mkdir(parents=True, exist_ok=True)

    # Check if SageMaker has already mounted models into /opt/ml/model
    sagemaker_model_dir = Path("/opt/ml/model")
    if sagemaker_model_dir.is_dir():
        model_files = list(sagemaker_model_dir.glob("*"))
        if model_files:
            logger.info("Found %d file(s) in SageMaker model directory /opt/ml/model.", len(model_files))

    s3_uri = os.environ.get("MODEL_S3_URI")
    if not s3_uri:
        return

    logger.info("MODEL_S3_URI is set (%s). Downloading model files from S3...", s3_uri)
    from urllib.parse import urlparse
    import boto3
    import tarfile
    from boto3.s3.transfer import TransferConfig

    u = urlparse(s3_uri)
    bucket = u.netloc
    prefix = u.path.lstrip("/")

    s3 = boto3.client("s3", region_name=getattr(settings, "S3_REGION", "us-east-1"))
    cfg = TransferConfig(max_concurrency=32, multipart_chunksize=64 * 1024 * 1024)

    # If URI points directly to a tar.gz / tar archive (SageMaker standard artifact format)
    if prefix.endswith((".tar.gz", ".tgz", ".tar")):
        archive_target = dest / Path(prefix).name
        logger.info("Downloading S3 model archive s3://%s/%s -> %s...", bucket, prefix, archive_target)
        s3.download_file(bucket, prefix, str(archive_target), Config=cfg)
        logger.info("Extracting model archive %s to %s...", archive_target, dest)
        with tarfile.open(archive_target, "r:*") as tar:
            tar.extractall(path=dest)
        archive_target.unlink(missing_ok=True)
        logger.info("Model extraction complete.")
        return

    count = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel_path = os.path.relpath(key, prefix)
            target = dest / rel_path
            if target.is_file() and target.stat().st_size == obj["Size"]:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading s3://%s/%s -> %s (%d bytes)", bucket, key, target, obj["Size"])
            s3.download_file(bucket, key, str(target), Config=cfg)
            count += 1
    logger.info("S3 model download complete. Transferred %d new/updated files.", count)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models once at startup, clean up workers at shutdown."""
    global _models_loaded
    _models_loaded = False

    logger.info("Initializing Saytara Avatar backend...")
    logger.info("Environment: host=%s, port=%d", settings.HOST, settings.PORT)

    try:
        import torch
        assert torch.cuda.is_available(), "PyTorch cannot see the GPU - FLUX needs CUDA."
    except Exception as exc:
        logger.critical("Fatal: GPU check failed: %s", exc)
        raise

    _sync_model_caches_from_s3()
    _download_s3_prefix_if_needed()

    try:
        pipeline.load_models()
        _models_loaded = True
        logger.info("All models and workers initialized successfully.")
    except Exception as exc:
        logger.critical("Fatal: Failed to load models on startup: %s", exc, exc_info=True)
        raise

    s3_watcher.start()

    yield

    logger.info("Shutting down Saytara Avatar backend...")
    await s3_watcher.stop()
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
        "X-Steps-Used",
        "X-Analysis-Wait-Seconds",
        "X-Analysis-Seconds",
        "X-Slot-Wait-Seconds",
        "X-Gen-Only-Seconds",
        "X-Score-Seconds",
    ],
)

# GET /metrics on :8080 - the default BYOC scrape target SageMaker's
# EnableDetailedObservability reads for container-level metrics.
Instrumentator().instrument(app).expose(app, include_in_schema=False)

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
        "flux_backend": settings.FLUX_BACKEND,
        "gpu_slots": len(pipeline.slots),
        "gpu_slots_requested": settings.GPU_SLOTS,
        "settings": {"steps": settings.STEPS, "cfg": settings.CFG, "style_mode": settings.STYLE_MODE,
                     "best_of_n": settings.BEST_OF_N, "time_budget": settings.TIME_BUDGET,
                     "allow_test_overrides": settings.ALLOW_TEST_OVERRIDES},
        "speed_options": {
            "cache_prompt_embeds": settings.CACHE_PROMPT_EMBEDS,
            "cache_ref_latents": settings.CACHE_REF_LATENTS,
            "fuse_lora_requested": settings.FUSE_LORA,
            "lora_fused": [s.fused for s in pipeline.slots],
            "attention_backend": settings.ATTENTION_BACKEND or "default",
            "compile_transformer": settings.COMPILE_TRANSFORMER,
            "cudnn_enabled": settings.CUDNN_ENABLED,
            "persistent_detect": settings.PERSISTENT_DETECT,
            "cache_stats": pipeline._CACHE_STATS,
        },
        "gpu_slot_errors": pipeline.slot_errors,
        "gpu_free_vram_gb": free_gb,
        "gpu_total_vram_gb": total_gb,
        "s3_watcher": {
            "enabled": settings.S3_ENABLED,
            "running": s3_watcher.is_running(),
            "bucket": settings.S3_BUCKET,
            "users_prefix": settings.S3_USERS_PREFIX,
            "generated_prefix": settings.S3_GENERATED_PREFIX,
        },
    }

def _test_overrides(steps, tries, time_budget, cfg, style) -> Optional[dict]:
    """Per-request A/B overrides; only honoured when ALLOW_TEST_OVERRIDES=true (otherwise ignored)."""
    if not settings.ALLOW_TEST_OVERRIDES:
        return None
    ov = {k: v for k, v in (
        ("steps", steps), ("tries", tries), ("time_budget", time_budget), ("cfg", cfg),
        ("style", style if style in ("semi_real", "comic", "off") else None),
    ) if v is not None}
    return ov or None

async def _generate_avatar_from_bytes(content: bytes, filename: str, overrides: Optional[dict] = None):
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

        try:
            jpeg_bytes, metadata = await pipeline.process_image_file(temp_path, overrides)
        except pipeline.Rejected as r:
            logger.info("Image validation rejected for %s: %s", filename, r)
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": str(r)})

        headers = {
            "X-Tries-Used": str(metadata.get("tries", "")),
            "X-Best-Try": str(metadata.get("best_try", "")),
            "X-Stop-Reason": str(metadata.get("stopped", "")),
            "X-Generation-Seconds": str(metadata.get("seconds", "")),
            "X-Identity-Similarity": str(metadata.get("id_sim", "")),
            "X-Visor-Status": str(metadata.get("visor", "")),
            "X-Bald-Status": str(metadata.get("bald", "")),
            "X-Steps-Used": str(metadata.get("steps", "")),
            "X-Cfg-Used": str(metadata.get("cfg", "")),
            "X-Style-Used": str(metadata.get("style", "")),
            "X-Analysis-Wait-Seconds": str(metadata.get("analysis_wait_s", "")),
            "X-Analysis-Seconds": str(metadata.get("analysis_s", "")),
            "X-Slot-Wait-Seconds": str(metadata.get("slot_wait_s", "")),
            "X-Gen-Only-Seconds": str(metadata.get("gen_s", "")),
            "X-Score-Seconds": str(metadata.get("val_s", "")),
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
    steps: Optional[int] = None, tries: Optional[int] = None, time_budget: Optional[float] = None,
    cfg: Optional[float] = None, style: Optional[str] = None,
):
    filename = file.filename or "upload.jpg"
    # No server-side upload size limit - read whatever was sent.
    content = await file.read()
    return await _generate_avatar_from_bytes(content, filename, _test_overrides(steps, tries, time_budget, cfg, style))

class AvatarBase64Request(BaseModel):
    image: str  # base64-encoded image bytes (raw or "data:image/...;base64,..." data URL)
    filename: Optional[str] = "upload.jpg"  # only used to infer the file extension

@app.post(
    "/v1/avatar/base64",
    responses=AVATAR_RESPONSES,
    summary="Generate Sci-Fi Avatar (base64 input)",
    description="Same as /v1/avatar, but the photo is sent as a base64 string in a JSON body instead of multipart/form-data.",
)
async def create_avatar_base64(
    body: AvatarBase64Request,
    steps: Optional[int] = None, tries: Optional[int] = None, time_budget: Optional[float] = None,
    cfg: Optional[float] = None, style: Optional[str] = None,
):
    filename = body.filename or "upload.jpg"
    raw_b64 = body.image.split(",", 1)[-1] if body.image.startswith("data:") else body.image
    try:
        content = base64.b64decode(raw_b64, validate=True)
    except Exception:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": "Invalid base64 image data."},
        )
    return await _generate_avatar_from_bytes(content, filename, _test_overrides(steps, tries, time_budget, cfg, style))

@app.get("/ping", summary="SageMaker health ping endpoint")
async def sagemaker_ping():
    """SageMaker ping endpoint: returns 200 when models are loaded, 503 while loading."""
    if _models_loaded and pipeline.is_val_worker_alive():
        return Response(status_code=status.HTTP_200_OK, content="OK")
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content="Model loading or unready")

@app.post("/invocations", summary="SageMaker inference invocation endpoint")
async def sagemaker_invocations(request: Request):
    """SageMaker invocations endpoint. Accepts JSON (base64 image), raw binary image, or multipart."""
    content_type = request.headers.get("content-type", "").lower()
    accept_header = request.headers.get("accept", "").lower()

    if "application/json" in content_type:
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Invalid JSON payload"})

        raw_b64 = body.get("image") or body.get("image_base64") or body.get("photo")
        filename = body.get("filename", "upload.jpg")
        if not raw_b64:
            if body.get("prompt"):
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={
                        "error": "Saytara Avatar Generation transforms portrait photos into sci-fi avatars. "
                                 "Please provide 'image' or 'image_base64' (base64-encoded JPEG/PNG) along with or instead of 'prompt'."
                    },
                )
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Missing 'image' or 'image_base64' in JSON body"})
        if raw_b64.startswith("data:"):
            raw_b64 = raw_b64.split(",", 1)[-1]
        try:
            content = base64.b64decode(raw_b64, validate=True)
        except Exception:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Invalid base64 image data"})

        res = await _generate_avatar_from_bytes(content, filename)
        if isinstance(res, JSONResponse):
            return res
        if "application/json" in accept_header or body.get("return_base64", False):
            img_b64 = base64.b64encode(res.body).decode("ascii")
            return JSONResponse(content={"image_base64": img_b64, "content_type": "image/jpeg"})
        return res

    elif any(t in content_type for t in ("image/", "application/octet-stream")):
        content = await request.body()
        if not content:
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Empty body"})
        ext = ".png" if "png" in content_type else ".jpg"
        return await _generate_avatar_from_bytes(content, f"upload{ext}")

    elif "multipart/form-data" in content_type:
        form = await request.form()
        file_obj = form.get("file") or form.get("image")
        if not file_obj or not hasattr(file_obj, "read"):
            return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Missing file in form data"})
        content = await file_obj.read()
        filename = getattr(file_obj, "filename", "upload.jpg")
        return await _generate_avatar_from_bytes(content, filename)

    else:
        body_bytes = await request.body()
        try:
            body = json.loads(body_bytes.decode("utf-8"))
            raw_b64 = body.get("image") or body.get("image_base64")
            if raw_b64:
                if raw_b64.startswith("data:"):
                    raw_b64 = raw_b64.split(",", 1)[-1]
                content = base64.b64decode(raw_b64)
                return await _generate_avatar_from_bytes(content, body.get("filename", "upload.jpg"))
        except Exception:
            pass
        return await _generate_avatar_from_bytes(body_bytes, "upload.jpg")

@app.get("/", summary="Root status")
async def root():
    return {
        "service": "Saytara Avatar Generation API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
        "ping": "/ping",
        "invocations": "/invocations",
    }

if __name__ == "__main__":
    import uvicorn
    # SageMaker passes "serve" as argv[1] - ignore extra CLI args and launch uvicorn on port 8080 by default
    port = int(os.environ.get("PORT", getattr(settings, "PORT", 8080)))
    host = os.environ.get("HOST", getattr(settings, "HOST", "0.0.0.0"))
    logger.info("Starting Saytara Avatar server on %s:%d (argv=%s)", host, port, sys.argv[1:])
    uvicorn.run(app, host=host, port=port, reload=False, workers=1)
