import os
import sys
import time
import atexit
import logging
import asyncio
import threading
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from app.config import settings

logger = logging.getLogger("saytara.models")

# Global pipeline singletons
clip = None
pipe = None

# LoRA dynamic scale state
_LORA = {"scale": 1.0}

def set_lora(weight: float) -> None:
    _LORA["scale"] = float(weight)

def lora_kwargs() -> dict:
    return {"scale": _LORA["scale"]}

# GPU serialization lock for async request handling
gpu_lock = asyncio.Lock()

# Persistent ArcFace embedding worker state
_VW = {"p": None}
_VLOCK = threading.Lock()
_POOL = ThreadPoolExecutor(max_workers=1)

def close_val_worker() -> None:
    with _VLOCK:
        p = _VW.get("p")
        _VW["p"] = None
        if p is not None and p.poll() is None:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass

atexit.register(close_val_worker)

def get_val_worker() -> subprocess.Popen:
    with _VLOCK:
        p = _VW.get("p")
        if p is not None and p.poll() is None:
            return p

        # Start a new worker
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "TF_USE_LEGACY_KERAS": "1",
            "TF_ENABLE_ONEDNN_OPTS": "0",
        }
        logger.info(
            "Starting persistent DeepFace validation worker (%s, %s)...",
            settings.VERIFY_MODEL,
            settings.VERIFY_DETECTOR,
        )

        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        settings.VAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

        log_file = open(settings.VAL_WORKER_LOG, "a", encoding="utf-8")
        p = subprocess.Popen(
            [
                sys.executable,
                str(settings.VAL_WORKER_PATH),
                settings.VERIFY_MODEL,
                settings.VERIFY_DETECTOR,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_file,
            text=True,
            bufsize=1,
            env=env,
        )

        # Wait for the __READY__ handshake
        ready = False
        t_start = time.time()
        for line in p.stdout:
            if line.startswith("__READY__"):
                ready = True
                break
            if time.time() - t_start > 120:
                break

        if not ready or p.poll() is not None:
            close_val_worker()
            raise RuntimeError(
                f"DeepFace val_worker failed to start - see {settings.VAL_WORKER_LOG}"
            )

        _VW["p"] = p
        logger.info("DeepFace validation worker ready.")
        return p

def get_gpu_memory_info() -> Tuple[Optional[float], Optional[float]]:
    try:
        import torch
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return round(free_bytes / 1e9, 2), round(total_bytes / 1e9, 2)
    except Exception:
        pass
    return None, None

def get_cuda_device_name() -> Optional[str]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None

def is_val_worker_alive() -> bool:
    p = _VW.get("p")
    return p is not None and p.poll() is None

def load_models() -> None:
    global clip, pipe
    import torch

    assert torch.cuda.is_available(), "PyTorch cannot see the GPU - FLUX needs CUDA."
    torch.backends.cudnn.enabled = False

    # Optional HF authentication
    if settings.effective_hf_token:
        try:
            import huggingface_hub
            huggingface_hub.login(token=settings.effective_hf_token)
            logger.info("Logged into Hugging Face Hub successfully.")
        except Exception as e:
            logger.warning("Hugging Face login failed: %s", e)

    free_gb, total_gb = get_gpu_memory_info()
    free_gb_val = free_gb if free_gb is not None else 0.0
    low_vram = free_gb_val < 45.0
    logger.info(
        "GPU free VRAM: %s / %s GB -> %s",
        f"{free_gb_val:.1f}",
        f"{total_gb:.1f}" if total_gb is not None else "?",
        "CPU-offload mode" if low_vram else "Full GPU mode",
    )

    # Load CLIP pipeline
    from transformers import pipeline as hf_pipeline
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()

    logger.info("Loading CLIP model: %s", settings.CLIP_MODEL)
    clip = hf_pipeline(
        "zero-shot-image-classification",
        model=settings.CLIP_MODEL,
        device=-1 if low_vram else 0,
    )
    logger.info("CLIP loaded successfully.")

    # Load FLUX.2 Klein + LoRA
    from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.utils import logging as df_logging
    df_logging.set_verbosity_error()

    logger.info("Loading FLUX model: %s", settings.FLUX_MODEL)
    pipe = Flux2KleinPipeline.from_pretrained(settings.FLUX_MODEL, dtype=torch.bfloat16)
    if low_vram:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    logger.info("Loading LoRA weights: %s / %s", settings.LORA, settings.LORA_FILE)
    pipe.load_lora_weights(settings.LORA, weight_name=settings.LORA_FILE, adapter_name="bfs")
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    logger.info("FLUX + LoRA pipeline loaded successfully.")

    # Start persistent ArcFace identity verification worker
    get_val_worker()

def shutdown_models() -> None:
    logger.info("Shutting down model resources...")
    close_val_worker()
    _POOL.shutdown(wait=False)
    logger.info("Model resources shut down cleanly.")
