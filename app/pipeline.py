"""Avatar generation pipeline: Step 1 photo validation/classification, prompt
assembly, and Step 2 FLUX.2-klein-9B + LoRA avatar generation.

Two DeepFace subprocess lifecycles are used against app/deepface_worker.py:
  - "detect": spawned fresh per request (run_deepface_worker) to keep
    TensorFlow out of this (PyTorch) process.
  - "embed": started once at startup and kept alive (get_val_worker) for
    ArcFace identity-similarity scoring during generation.
"""
import io
import os
import queue
import hashlib
import sys
import time
import json
import atexit
import asyncio
import logging
import threading
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.config import settings

logger = logging.getLogger("saytara.pipeline")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

class Rejected(Exception):
    """Image is blocked; the message is the reason shown to the user."""
    pass

# Manual fixes extension point (empty by default)
OVERRIDES: Dict[str, Dict[str, Any]] = {}

SCOPE_LABELS = {
    "person":  "a photo of a real human person showing their face",
    "animal":  "a photo of an animal, a pet, a cat, a dog or a bird",
    "text":    "a screenshot of computer code, text, a document or a website",
    "cartoon": "a cartoon, a drawing, anime, a 3D render or an illustration",
    "object":  "a photo of an object, food, a car, a building or a landscape with no people",
}

HIJAB_PROMPT_PAIRS = [
    ("a photo of a woman wearing a hijab that covers her hair and neck",
     "a photo of a woman with visible uncovered hair around her face"),
    ("a portrait of a woman in a headscarf wrapped around her head",
     "a portrait of a woman with her natural hair visible around her head"),
    ("a woman wearing a cloth scarf that frames her face and covers her hair",
     "a woman with an uncovered hairstyle visible on top and at the sides"),
]

# ============================================================================
# Model singletons, GPU lock, and the persistent ArcFace embedding worker
# ============================================================================

clip = None
pipe = None

# mode "kwargs"   : strength passed per call as attention_kwargs={"scale": w}   (what the notebook does)
# mode "adapters" : strength set with pipe.set_adapters(["bfs"], [w])            (fallback if the kwarg has no effect)
_LORA = {"scale": 1.0, "mode": "kwargs"}

class GpuSlot:
    """One independent FLUX pipeline (own transformer + LoRA state + scheduler + CUDA stream).
    A request owns a slot for the whole of Step 2, so several avatars generate at once."""

    def __init__(self, idx: int, slot_pipe, lora_state: Optional[dict] = None):
        self.idx = idx
        self.pipe = slot_pipe
        self.lora = lora_state if lora_state is not None else {"scale": 1.0, "mode": "kwargs"}
        self.stream = torch.cuda.Stream()
        self.fused = False          # True once the LoRA is merged into the weights (FUSE_LORA)
        self.fused_scale = None

    def set_lora(self, weight: float) -> None:
        self.lora["scale"] = float(weight)
        if self.fused:
            if abs(float(weight) - self.fused_scale) > 1e-6:
                raise RuntimeError(
                    f"LoRA strength {weight} requested but the LoRA is fused at {self.fused_scale}. "
                    "Use FUSE_LORA=false, or a uniform ID_LORA_SCHEDULE.")
            return
        if self.lora["mode"] == "adapters":
            self.pipe.set_adapters(["bfs"], adapter_weights=[float(weight)])

    def reset_lora(self) -> None:
        if not self.fused:
            self.set_lora(settings.LORA_STRENGTH)

    def lora_kwargs(self):
        if self.fused:
            return None
        return {"scale": self.lora["scale"]} if self.lora["mode"] == "kwargs" else None

    def fuse(self, strength: float) -> None:
        """Merge the BFS LoRA into the transformer weights at one fixed strength (no per-step LoRA matmuls)."""
        self.pipe.set_adapters(["bfs"], adapter_weights=[float(strength)])
        self.pipe.fuse_lora(components=["transformer"], adapter_names=["bfs"], lora_scale=1.0)
        self.fused, self.fused_scale = True, float(strength)

slots: List[GpuSlot] = []
slot_errors: List[str] = []   # why an extra slot could not be loaded (shown in /health)
_slot_q: Optional[asyncio.Queue] = None

async def acquire_slot() -> GpuSlot:
    """Wait until one of the GPU slots is free (replaces the old single global GPU lock)."""
    return await _slot_q.get()

def release_slot(slot: GpuSlot) -> None:
    _slot_q.put_nowait(slot)

def set_lora(weight: float) -> None:  # slot 0 helper, kept for older callers/tests
    slots[0].set_lora(weight)

def lora_kwargs():
    return slots[0].lora_kwargs()

# Step 1 (CPU: RetinaFace/DeepFace subprocess + CLIP) runs outside the GPU slots, this many at a time
analysis_sem = asyncio.Semaphore(settings.ANALYSIS_CONCURRENCY)
_CLIP_LOCK = threading.Lock()

# Persistent ArcFace embedding worker state
_VW = {"p": None}
_VLOCK = threading.RLock()
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

        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "-1",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "TF_USE_LEGACY_KERAS": "1",
            "TF_ENABLE_ONEDNN_OPTS": "0",
        }
        logger.info(
            "Starting persistent DeepFace embedding worker (%s, %s)...",
            settings.VERIFY_MODEL,
            settings.VERIFY_DETECTOR,
        )

        settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        settings.VAL_TMP_DIR.mkdir(parents=True, exist_ok=True)

        log_file = open(settings.WORKER_LOG, "a", encoding="utf-8")
        p = subprocess.Popen(
            [
                sys.executable,
                str(settings.WORKER_PATH),
                "embed",
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
                f"DeepFace embedding worker failed to start - see {settings.WORKER_LOG}"
            )

        _VW["p"] = p
        logger.info("DeepFace embedding worker ready.")
        return p

# ============================================================================
# Speed options (all controlled from .env, default OFF)
# ============================================================================

_PROMPT_CACHE: "OrderedDict[str, torch.Tensor]" = OrderedDict()
_PROMPT_LOCK = threading.Lock()
_REF_CACHE: "OrderedDict[str, torch.Tensor]" = OrderedDict()
_REF_SEEN: set = set()
_REF_LOCK = threading.Lock()
_CACHE_STATS = {"prompt_hit": 0, "prompt_miss": 0, "ref_hit": 0, "ref_miss": 0}

def prompt_inputs(slot: "GpuSlot", prompt: str) -> dict:
    """Either {"prompt": text} or, with CACHE_PROMPT_EMBEDS, {"prompt_embeds": cached encoder output}. Bit-identical."""
    if not settings.CACHE_PROMPT_EMBEDS:
        return {"prompt": prompt}
    with _PROMPT_LOCK:
        emb = _PROMPT_CACHE.get(prompt)
        if emb is not None:
            _PROMPT_CACHE.move_to_end(prompt)
    if emb is None:
        _CACHE_STATS["prompt_miss"] += 1
        with torch.no_grad():
            e, _ids = slot.pipe.encode_prompt(prompt=prompt, device=slot.pipe._execution_device)
        emb = e.detach().to("cpu")          # synchronous copy; kept on the CPU so it is safe across CUDA streams
        with _PROMPT_LOCK:
            _PROMPT_CACHE[prompt] = emb
            while len(_PROMPT_CACHE) > 256:
                _PROMPT_CACHE.popitem(last=False)
    else:
        _CACHE_STATS["prompt_hit"] += 1
    return {"prompt_embeds": emb.to(slot.pipe._execution_device)}

def _tensor_key(x: torch.Tensor) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(repr((tuple(x.shape), str(x.dtype))).encode())
    h.update(x.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def _install_ref_cache(p) -> None:
    """Cache the VAE latents of reference images that are seen repeatedly (avatar templates, visor refs).
    A photo seen only once (every user's face) is never stored. Deterministic (VAE 'argmax' mode) -> identical."""
    orig = p._encode_vae_image

    def cached(image, generator=None):
        key = _tensor_key(image)
        with _REF_LOCK:
            hit = _REF_CACHE.get(key)
            if hit is not None:
                _REF_CACHE.move_to_end(key)
        if hit is not None:
            _CACHE_STATS["ref_hit"] += 1
            return hit.to(image.device)
        _CACHE_STATS["ref_miss"] += 1
        out = orig(image=image, generator=generator)
        with _REF_LOCK:
            if key in _REF_SEEN:                       # second sighting -> worth keeping
                _REF_CACHE[key] = out.detach().to("cpu")
                while len(_REF_CACHE) > 64:
                    _REF_CACHE.popitem(last=False)
            else:
                _REF_SEEN.add(key)
                if len(_REF_SEEN) > 20000:
                    _REF_SEEN.clear()
        return out

    p._encode_vae_image = cached

def apply_speed_options(slot: "GpuSlot") -> None:
    p = slot.pipe
    if settings.FUSE_LORA:
        mults = [float(m) for m in settings.ID_LORA_SCHEDULE]
        if len(set(round(m, 6) for m in mults)) != 1:
            logger.warning("FUSE_LORA skipped: ID_LORA_SCHEDULE %s is not uniform (the LoRA strength changes per try, "
                           "and a fused LoRA has one fixed strength). Use e.g. [1.0,1.0] to enable it.", mults)
        else:
            strength = settings.LORA_STRENGTH * settings.ID_LORA_MULT * mults[0]
            slot.fuse(strength)
            logger.info("Slot %d: LoRA fused into the weights at strength %.3f", slot.idx, strength)
    if settings.ATTENTION_BACKEND:
        try:
            p.transformer.set_attention_backend(settings.ATTENTION_BACKEND)
            logger.info("Slot %d: attention backend = %s", slot.idx, settings.ATTENTION_BACKEND)
        except Exception as exc:
            logger.warning("Slot %d: could not set attention backend %r (%s) - keeping the default",
                           slot.idx, settings.ATTENTION_BACKEND, exc)
    if settings.CACHE_REF_LATENTS:
        _install_ref_cache(p)
    if settings.COMPILE_TRANSFORMER:
        if not slot.fused:
            logger.warning("COMPILE_TRANSFORMER without a fused LoRA recompiles for every LoRA strength - "
                           "enable FUSE_LORA with a uniform ID_LORA_SCHEDULE.")
        p.transformer.compile(mode=settings.COMPILE_MODE, dynamic=False)
        logger.info("Slot %d: transformer compiled (mode=%s) - first calls per shape are slow", slot.idx, settings.COMPILE_MODE)

def warmup_slots() -> None:
    """Run each template through each slot once so torch.compile builds every shape before the first request."""
    if not (settings.COMPILE_TRANSFORMER and settings.WARMUP_AT_STARTUP):
        return
    for slot in slots:
        for key in ("Man", "Woman", "Woman_Hijab"):
            try:
                t0 = time.time()
                avatar, _ = load_avatar(key)
                face = pad_square(avatar.resize((512, 512), Image.LANCZOS))
                visor = load_visor_ref(key)
                images = [avatar, face] + ([visor] if visor is not None else [])
                slot.set_lora(settings.LORA_STRENGTH * settings.ID_LORA_MULT * settings.ID_LORA_SCHEDULE[0])
                with torch.no_grad(), torch.cuda.stream(slot.stream):
                    slot.pipe(prompt="warmup", image=images, attention_kwargs=slot.lora_kwargs(),
                              width=avatar.size[0], height=avatar.size[1], num_inference_steps=2,
                              guidance_scale=settings.CFG, generator=torch.Generator("cuda").manual_seed(0))
                slot.stream.synchronize()
                logger.info("Warm-up slot %d / %s done in %.1fs", slot.idx, key, time.time() - t0)
            except Exception:
                logger.exception("Warm-up slot %d / %s failed (continuing)", slot.idx, key)
    torch.cuda.empty_cache()

# ---- persistent face-detection workers (PERSISTENT_DETECT) ----
_DETECT_Q: Optional["queue.Queue"] = None
_DETECT_ALL: List[subprocess.Popen] = []

def _spawn_detect_worker() -> subprocess.Popen:
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "TF_USE_LEGACY_KERAS": "1",
           "TF_CPP_MIN_LOG_LEVEL": "3", "TF_ENABLE_ONEDNN_OPTS": "0"}
    log_file = open(settings.WORKER_LOG, "a", encoding="utf-8")
    p = subprocess.Popen([sys.executable, str(settings.WORKER_PATH), "detect_server"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log_file,
                         text=True, bufsize=1, env=env)
    t0 = time.time()
    for line in p.stdout:
        if line.startswith("__READY__"):
            _DETECT_ALL.append(p)
            return p
        if time.time() - t0 > 180:
            break
    try:
        p.kill()
    except Exception:
        pass
    raise RuntimeError(f"Persistent detect worker failed to start - see {settings.WORKER_LOG}")

def start_detect_pool() -> None:
    global _DETECT_Q
    if not settings.PERSISTENT_DETECT:
        return
    _DETECT_Q = queue.Queue()
    for _ in range(max(1, settings.DETECT_WORKERS)):
        _DETECT_Q.put(_spawn_detect_worker())
    logger.info("Persistent face-detection pool ready (%d workers).", _DETECT_Q.qsize())

def close_detect_pool() -> None:
    for p in list(_DETECT_ALL):
        try:
            p.stdin.close()
            p.terminate()
            p.wait(timeout=2)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    _DETECT_ALL.clear()

atexit.register(close_detect_pool)

def _run_detect_persistent(image_path: Path, timeout: int) -> dict:
    p = _DETECT_Q.get()
    ok = False
    try:
        args = [str(image_path), settings.FACE_DETECTOR, str(settings.CROP_SCALE), str(settings.MIN_FACE_CONFIDENCE),
                str(settings.MIN_FACE_SIZE_PX), str(settings.MAIN_FACE_DOMINANCE)]
        p.stdin.write(json.dumps({"args": args}) + "\n")
        p.stdin.flush()
        timer = threading.Timer(timeout, p.kill)
        timer.start()
        try:
            while True:
                line = p.stdout.readline()
                if not line:
                    raise RuntimeError("detect worker died or timed out")
                if line.startswith("__JSON__"):
                    res = json.loads(line[len("__JSON__"):])
                    break
        finally:
            timer.cancel()
        ok = True
        return res
    finally:
        if ok:
            _DETECT_Q.put(p)
        else:                                    # replace a broken worker; callers fall back to the one-shot path
            try:
                p.kill()
            except Exception:
                pass
            if p in _DETECT_ALL:
                _DETECT_ALL.remove(p)
            try:
                _DETECT_Q.put(_spawn_detect_worker())
            except Exception:
                logger.exception("Could not respawn the detect worker")

def get_gpu_memory_info() -> Tuple[Optional[float], Optional[float]]:
    try:
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return round(free_bytes / 1e9, 2), round(total_bytes / 1e9, 2)
    except Exception:
        pass
    return None, None

def get_cuda_device_name() -> Optional[str]:
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None

def is_val_worker_alive() -> bool:
    p = _VW.get("p")
    return p is not None and p.poll() is None

lora_layers = 0

def _count_lora_layers() -> int:
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
        return sum(isinstance(m, BaseTunerLayer) for m in pipe.transformer.modules())
    except Exception:
        return sum(1 for m in pipe.transformer.modules() if hasattr(m, "lora_A"))

def _adapters():
    try:
        return pipe.get_active_adapters()
    except Exception:
        return None

def _log_versions() -> None:
    import importlib.metadata as md
    vs = []
    for pkg in ("torch", "diffusers", "transformers", "peft", "accelerate", "safetensors", "deepface", "tensorflow", "tf-keras"):
        try:
            vs.append(f"{pkg}={md.version(pkg)}")
        except Exception:
            vs.append(f"{pkg}=?")
    logger.info("Library versions: %s", ", ".join(vs))
    try:
        if int(md.version("transformers").split(".")[0]) >= 5:
            logger.warning(
                "transformers %s is a 5.x release - the notebook was built on 4.x. If avatars ignore the user's "
                "face/hair, run: pip install 'transformers>=4.56,<5' and restart.", md.version("transformers"))
    except Exception:
        pass

def _lora_selftest() -> None:
    """Generate the same tiny image with the LoRA off and on. If the output does not change, the face-swap LoRA is
    doing nothing (this is what makes avatars ignore the user's face). Tries the notebook's way (attention_kwargs
    scale) first, then pipe.set_adapters; raises if neither changes the output."""
    logger.info("LoRA self-test: generating 4 tiny images (a few seconds)...")
    av, _ = load_avatar("Man")
    base = av.resize((512, 512), Image.LANCZOS)
    face = pad_square(base.crop((140, 20, 400, 300)), 512)

    def gen(weight, kwargs_mode):
        _LORA["mode"] = "kwargs" if kwargs_mode else "adapters"
        if not kwargs_mode:
            pipe.set_adapters(["bfs"], adapter_weights=[float(weight)])
        else:
            pipe.set_adapters(["bfs"], adapter_weights=[1.0])
        with torch.no_grad():
            im = pipe(
                prompt="head_swap: start with Picture 1 as the base image, replace the head with the head from Picture 2.",
                image=[base, face],
                attention_kwargs={"scale": float(weight)} if kwargs_mode else None,
                width=512, height=512, num_inference_steps=4, guidance_scale=settings.CFG,
                generator=torch.Generator("cuda").manual_seed(1),
            ).images[0]
        torch.cuda.empty_cache()
        return np.asarray(im.convert("RGB")).astype(np.int16)

    thr = 1.0
    try:
        d_kwargs = float(np.abs(gen(0.0, True) - gen(1.1, True)).mean())
        logger.info("LoRA self-test: attention_kwargs scale 0 vs 1.1 -> mean pixel diff %.2f", d_kwargs)
        if d_kwargs > thr:
            _LORA["mode"] = "kwargs"
            logger.info("LoRA self-test PASSED (mode=kwargs, same as the notebook).")
            return
        d_ad = float(np.abs(gen(0.0, False) - gen(1.1, False)).mean())
        logger.info("LoRA self-test: set_adapters weight 0 vs 1.1 -> mean pixel diff %.2f", d_ad)
        if d_ad > thr:
            _LORA["mode"] = "adapters"
            logger.warning("attention_kwargs scale had NO effect with these library versions - switched to "
                           "pipe.set_adapters for LoRA strength (mode=adapters).")
            return
    finally:
        try:
            pipe.set_adapters(["bfs"], adapter_weights=[1.0])
        except Exception:
            pass
    raise RuntimeError(
        "LoRA self-test FAILED: the BFS LoRA does not change the output at any strength, so the face swap would "
        "ignore the user's photo. Check diffusers / peft / transformers versions (see env_report.py)."
    )

def load_models() -> None:
    global clip, pipe

    assert torch.cuda.is_available(), "PyTorch cannot see the GPU - FLUX needs CUDA."
    torch.backends.cudnn.enabled = bool(settings.CUDNN_ENABLED)

    free_gb, total_gb = get_gpu_memory_info()
    free_gb_val = free_gb if free_gb is not None else 0.0
    low_vram = free_gb_val < 45.0
    logger.info(
        "GPU free VRAM: %s / %s GB -> %s",
        f"{free_gb_val:.1f}",
        f"{total_gb:.1f}" if total_gb is not None else "?",
        "CPU-offload mode" if low_vram else "Full GPU mode",
    )

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

    from diffusers import Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler
    from diffusers.utils import logging as df_logging
    df_logging.set_verbosity_error()

    logger.info("Loading FLUX model: %s", settings.FLUX_MODEL)
    pipe = Flux2KleinPipeline.from_pretrained(settings.FLUX_MODEL, dtype=torch.bfloat16)
    if low_vram:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    # Decode the final image in tiles instead of one giant allocation - this is what was
    # tipping requests into CUDA OOM right at the end of generation.
    if settings.VAE_TILING:
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()

    logger.info("Loading LoRA weights: %s / %s", settings.LORA, settings.LORA_FILE)
    pipe.load_lora_weights(settings.LORA, weight_name=settings.LORA_FILE, adapter_name="bfs")
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        shift=1.0,
        use_dynamic_shifting=False,
    )
    global lora_layers
    lora_layers = _count_lora_layers()
    if settings.LORA_SELFTEST:
        _lora_selftest()
    _log_versions()
    logger.info("Active adapters: %s | LoRA layers attached to transformer: %d", _adapters(), lora_layers)
    if lora_layers == 0:
        raise RuntimeError(
            "BFS LoRA loaded but NO LoRA layers are attached to the FLUX transformer - face swap would silently "
            "ignore the user's face. Check diffusers / peft / transformers versions against the notebook."
        )
    logger.info("FLUX + LoRA pipeline loaded successfully.")

    # ---- GPU slots: slot 0 is the pipeline above; extra slots add concurrency ----
    global _slot_q
    slots.clear()
    slots.append(GpuSlot(0, pipe, _LORA))
    wanted = 1 if low_vram else (settings.GPU_SLOTS if settings.GPU_SLOTS > 0 else 8)  # 0 = auto-fill VRAM
    if low_vram and settings.GPU_SLOTS > 1:
        logger.warning("Low free VRAM - running with 1 GPU slot (CPU-offload mode cannot run concurrent jobs).")
    try:
        te_has_lora = any(hasattr(m, "lora_A") for m in pipe.text_encoder.modules())
    except Exception:
        te_has_lora = True
    shared = {} if te_has_lora else dict(text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer, vae=pipe.vae)
    slot_errors.clear()
    logger.info("GPU slots: GPU_SLOTS=%s (0=auto) SLOT_MIN_FREE_GB=%s -> trying to load up to %d slot(s); low_vram=%s",
                settings.GPU_SLOTS, settings.SLOT_MIN_FREE_GB, wanted, low_vram)
    if low_vram and settings.GPU_SLOTS != 1:
        slot_errors.append(f"low VRAM at startup ({free_gb_val:.1f} GB free < 45 GB) -> CPU-offload mode, 1 slot only")
    for i in range(1, wanted):
        free_now, _t = get_gpu_memory_info()
        if free_now is None or free_now < settings.SLOT_MIN_FREE_GB:
            msg = f"stopped at {len(slots)} slot(s): only {free_now} GB VRAM free (< SLOT_MIN_FREE_GB={settings.SLOT_MIN_FREE_GB})"
            logger.warning(msg)
            slot_errors.append(msg)
            break
        sp = None
        # try 1: share text encoder + VAE (saves ~16 GB); try 2: fully separate pipeline
        for attempt, comps in (("shared text encoder/VAE", shared), ("separate full copy", {})):
            if attempt == "separate full copy" and (not shared or (get_gpu_memory_info()[0] or 0) < 36):
                break
            try:
                logger.info("Loading GPU slot %d (%s)...", i, attempt)
                sp = Flux2KleinPipeline.from_pretrained(settings.FLUX_MODEL, dtype=torch.bfloat16, **comps)
                sp.to("cuda")
                sp.load_lora_weights(settings.LORA, weight_name=settings.LORA_FILE, adapter_name="bfs")
                sp.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                    sp.scheduler.config, shift=1.0, use_dynamic_shifting=False)
                st = GpuSlot(i, sp, {"scale": 1.0, "mode": _LORA["mode"]})
                if st.lora["mode"] == "adapters":
                    sp.set_adapters(["bfs"], adapter_weights=[1.0])
                slots.append(st)
                break
            except Exception as exc:
                sp = None
                logger.exception("GPU slot %d failed (%s)", i, attempt)
                slot_errors.append(f"slot {i} ({attempt}): {type(exc).__name__}: {str(exc)[:300]}")
                torch.cuda.empty_cache()
        if sp is None:
            logger.warning("Could not load GPU slot %d - continuing with %d slot(s).", i, len(slots))
            break
    for s in slots:
        apply_speed_options(s)
    _slot_q = asyncio.Queue()
    for s in slots:
        _slot_q.put_nowait(s)
    free_after, _t = get_gpu_memory_info()
    logger.info("GPU slots ready: %d concurrent avatar generations (free VRAM now %s GB).", len(slots), free_after)

    # Start persistent ArcFace identity verification worker
    get_val_worker()
    start_detect_pool()
    warmup_slots()

def shutdown_models() -> None:
    logger.info("Shutting down model resources...")
    close_val_worker()
    close_detect_pool()
    _POOL.shutdown(wait=False)
    logger.info("Model resources shut down cleanly.")

# ============================================================================
# Step 1 - photo validation and classification
# ============================================================================

def load_image(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise Rejected("Image cannot be read (not an image or corrupted)")
    return img_bgr

def clip_scores(pil_img: Image.Image, labels: List[str]) -> Dict[str, float]:
    if clip is None:
        raise RuntimeError("CLIP model is not loaded.")
    with _CLIP_LOCK:
        res = clip(pil_img, candidate_labels=list(labels))
    return {r["label"]: float(r["score"]) for r in res}

def check_scope(pil_img: Image.Image) -> Dict[str, float]:
    s = clip_scores(pil_img, list(SCOPE_LABELS.values()))
    scores = {k: s[v] for k, v in SCOPE_LABELS.items()}
    top = max(scores, key=scores.get)
    if top != "person" or scores["person"] < settings.SCOPE_MIN_PERSON_SCORE:
        raise Rejected(f"Out of scope: looks like '{top}' (person score {scores['person']:.0%})")
    return scores

def run_deepface_worker(image_path: Path, timeout: int = None) -> dict:
    timeout = timeout or settings.WORKER_TIMEOUT
    if _DETECT_Q is not None:
        try:
            return _run_detect_persistent(image_path, timeout)
        except Exception:
            logger.exception("Persistent detect failed - falling back to a one-shot process")
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "-1",
        "TF_USE_LEGACY_KERAS": "1",
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }
    process = subprocess.run(
        [
            sys.executable,
            str(settings.WORKER_PATH),
            "detect",
            str(image_path),
            settings.FACE_DETECTOR,
            str(settings.CROP_SCALE),
            str(settings.MIN_FACE_CONFIDENCE),
            str(settings.MIN_FACE_SIZE_PX),
            str(settings.MAIN_FACE_DOMINANCE),
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    payload = next(
        (
            line[len("__JSON__"):]
            for line in reversed(process.stdout.splitlines())
            if line.startswith("__JSON__")
        ),
        None,
    )
    if payload is None:
        raise RuntimeError(
            "DeepFace worker failed:\n" + (process.stderr or process.stdout)[-1500:]
        )
    return json.loads(payload)

def detect_main_face(user_path: Path) -> Tuple[dict, int]:
    res = run_deepface_worker(user_path)
    if res.get("error") == "unreadable":
        raise Rejected("Image cannot be read (not an image or corrupted)")
    if res.get("error") == "multiple_main_faces":
        raise Rejected(
            f"Multiple main faces ({res['main_faces']}) — upload a photo with one clear foreground person"
        )
    if not res.get("faces"):
        raise Rejected("No clear face detected (too small, blurry or covered)")
    return res["faces"][0], res.get("background_faces_ignored", 0)

def crop_face(img_bgr: np.ndarray, area: dict, scale: float = None) -> np.ndarray:
    scale = scale or settings.CROP_SCALE
    h_img, w_img = img_bgr.shape[:2]
    x, y, w, h = int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"])
    cx, cy = x + w // 2, y + h // 2
    nw, nh = int(w * scale), int(h * scale)
    x1, y1 = max(0, cx - nw // 2), max(0, cy - nh // 2)
    x2, y2 = min(w_img, cx + nw // 2), min(h_img, cy + nh // 2)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise Rejected("Face crop is empty")
    return crop

def detect_gender(face: dict, crop_bgr: np.ndarray, key: str) -> Tuple[str, float, str]:
    if "gender" in OVERRIDES.get(key, {}):
        return OVERRIDES[key]["gender"], 100.0, "override"
    conf = 0.0
    if face.get("gender"):
        g, conf = max(face["gender"].items(), key=lambda kv: kv[1])
        g = "Man" if g.lower().startswith("man") else "Woman"
        if conf >= settings.MIN_GENDER_CONFIDENCE:
            return g, conf, "deepface"

    pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    s = clip_scores(pil, ["a photo of a man", "a photo of a woman"])
    g2 = "Man" if s["a photo of a man"] >= s["a photo of a woman"] else "Woman"
    c2 = max(s.values())
    if c2 < settings.MIN_CLIP_GENDER_CONFIDENCE:
        raise Rejected(f"Gender uncertain (DeepFace {conf:.0f}%, CLIP {c2:.0%})")
    return g2, c2 * 100.0, "clip"

def create_image_views(image: Image.Image) -> List[Image.Image]:
    w, h = image.size
    return [
        image,
        ImageEnhance.Contrast(image).enhance(1.10),
        ImageEnhance.Sharpness(image).enhance(1.08),
        image.crop((0, 0, w, max(1, int(h * 0.78)))),
        image.crop((int(w * 0.06), int(h * 0.06), int(w * 0.94), int(h * 0.94))),
        image.crop((int(w * 0.18), int(h * 0.12), int(w * 0.82), int(h * 0.55))),
        image.crop((int(w * 0.08), int(h * 0.03), int(w * 0.92), int(h * 0.88))),
    ]

def classify_binary(
    views: List[Image.Image], pos_prompts: List[str], neg_prompts: List[str]
) -> Dict[str, float]:
    pos_all, neg_all = [], []
    for v in views:
        for p, n in zip(pos_prompts, neg_prompts):
            s = clip_scores(v, [p, n])
            tot = s[p] + s[n]
            pos_all.append(s[p] / tot if tot else 0.0)
            neg_all.append(s[n] / tot if tot else 0.0)
    pos, neg = float(np.mean(pos_all)), float(np.mean(neg_all))
    return {"positive": pos, "negative": neg, "margin": abs(pos - neg)}

def yes_no(scores: Dict[str, float], thr: float, margin: float) -> bool:
    if (
        scores["positive"] >= thr
        and scores["positive"] > scores["negative"]
        and scores["margin"] >= margin
    ):
        return True
    return False

def detect_glasses(views: List[Image.Image], gender: str) -> Tuple[bool, dict]:
    who = "man" if gender == "Man" else "woman"
    s = classify_binary(
        views,
        [
            f"a {who} wearing eyeglasses",
            f"a {who} wearing glasses",
            f"a {who} wearing prescription spectacles",
        ],
        [
            f"a {who} not wearing eyeglasses",
            f"a {who} without glasses",
            f"a {who} without prescription spectacles",
        ],
    )
    return yes_no(s, settings.GLASSES_POSITIVE_THRESHOLD, settings.GLASSES_MIN_SCORE_MARGIN), s

def detect_beard(image: Image.Image) -> Tuple[bool, dict]:
    width, height = image.size
    lower_face = image.crop((0, int(height * 0.28), width, height))
    scores = classify_binary(
        [image, lower_face, ImageEnhance.Contrast(lower_face).enhance(1.10)],
        [
            "a man with a visible beard",
            "a man with clear facial hair",
            "a man with a beard and moustache",
        ],
        [
            "a clean-shaven man with no beard",
            "a man with no moustache or stubble",
            "a man with a smooth bare chin and cheeks",
        ],
    )
    return yes_no(scores, settings.BEARD_POSITIVE_THRESHOLD, settings.BEARD_MIN_SCORE_MARGIN), scores

def _lab_pixels(region_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(region_bgr), cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float64)

def scalp_is_bare(crop_bgr: np.ndarray, rect: Tuple[int, int, int, int]) -> Optional[bool]:
    """Skin-colour test: is the strip of head ABOVE the face box (scalp) the same colour/brightness as the forehead
    (bald/shaved) instead of dark hair? None = the scalp area is outside the crop, cannot tell."""
    fx, fy, fw, fh = rect
    y0, y1 = max(0, int(fy - 0.32 * fh)), int(fy - 0.06 * fh)
    if y1 - y0 < 0.10 * fh:
        return None
    x0, x1 = max(0, int(fx + 0.25 * fw)), int(fx + 0.75 * fw)
    scalp = crop_bgr[y0:y1, x0:x1]
    fore = crop_bgr[max(0, int(fy + 0.02 * fh)):int(fy + 0.14 * fh), x0:x1]
    if scalp.size == 0 or fore.size == 0:
        return None
    s, f = _lab_pixels(scalp), _lab_pixels(fore)
    f_l = max(float(np.median(f[:, 0])), 1.0)
    d_e = float(np.linalg.norm(np.median(s, 0) - np.median(f, 0)))
    lum_ratio = float(np.median(s[:, 0])) / f_l
    # Background-aware: when the face box starts at the very top of a bald head (or the photo is a round/tight
    # crop on a black backdrop) the strip above the box is only background. That used to count as "dark = hair"
    # and a bald man was read as "has hair" -> the avatar kept the template's hair. Background pixels (colour of
    # the crop's top edge) are now excluded; if the strip is mostly background we cannot judge -> None (CLIP decides).
    bg = _lab_pixels(crop_bgr[0:max(2, int(0.04 * crop_bgr.shape[0])), :])
    bg_med = np.median(bg, 0)
    bg_like = np.linalg.norm(s - bg_med, axis=1) < 14.0
    if float(bg_like.mean()) > 0.50:
        logger.info("Scalp check: strip above the face is %.0f%% background -> cannot judge (None)", 100 * float(bg_like.mean()))
        return None
    if (~bg_like).sum() > 0:
        s = s[~bg_like]
    dark_frac = float((s[:, 0] < 0.55 * f_l).mean())
    # Only the share of pixels clearly DARKER than the forehead skin (= hair) is used. The old extra tests
    # (lum_ratio / colour distance to the forehead) failed real bald heads: a shiny scalp is brighter than the
    # forehead and the strip above a tight face box often lands on the wall behind the head, so a bald man was
    # read as "not bald". A dark backdrop only makes dark_frac high, which fails safe (not bald).
    logger.info("Scalp check: dark_frac=%.2f lum_ratio=%.2f dE=%.1f -> bare=%s", dark_frac, lum_ratio, d_e, dark_frac < 0.12)
    return dark_frac < 0.12

def detect_bald(image: Image.Image, crop_bgr: np.ndarray = None, rect=None) -> Tuple[bool, dict]:
    """Bald only if the scalp-colour test AND CLIP agree (a false positive removes a man's real hair)."""
    width, height = image.size
    top_head = image.crop((0, 0, width, int(height * 0.45)))
    scores = classify_binary(
        [image, top_head, ImageEnhance.Contrast(top_head).enhance(1.10)],
        [
            "a photo of a bald man with a bare shaved scalp",
            "a bald man with no hair on top of his head",
            "a man with a completely bald shaved head",
        ],
        [
            "a photo of a man with full hair on top of his head",
            "a man with a haircut and visible hair on top",
            "a man with thick hair on his head",
        ],
    )
    clip_yes = yes_no(scores, settings.BALD_POSITIVE_THRESHOLD, settings.BALD_MIN_SCORE_MARGIN)
    bare = scalp_is_bare(crop_bgr, rect) if (crop_bgr is not None and rect is not None) else None
    scores["scalp_bare"] = bare
    # A dark ceiling / backdrop above a face box that starts at the top of a bald head looks like "dark hair" to
    # the colour test, so a very confident CLIP verdict is allowed to override a "not bare" scalp test.
    strong = bool(clip_yes and scores["positive"] >= settings.BALD_STRONG_CLIP)
    if bare is None:
        bald = scores["positive"] >= settings.BALD_FALLBACK_CLIP and clip_yes
    else:
        bald = bool((bare and clip_yes) or strong)
    scores["strong_clip"] = strong
    return bald, scores

def classify_hijab(image: Image.Image) -> Tuple[bool, dict]:
    width, height = image.size
    head = image.crop((int(width * 0.05), 0, int(width * 0.95), int(height * 0.85)))
    labels = [label for pair in HIJAB_PROMPT_PAIRS for label in pair]
    view_scores = []
    for view in (image, head):
        scores = clip_scores(view, labels)
        view_scores.extend(
            scores[yes] / max(scores[yes] + scores[no], 1e-12)
            for yes, no in HIJAB_PROMPT_PAIRS
        )
    positive = float(np.mean(view_scores))
    report = {
        "positive": positive,
        "negative": 1.0 - positive,
        "margin": abs(2.0 * positive - 1.0),
    }
    is_hijab = (
        positive > 0.5
        and positive >= settings.HIJAB_POSITIVE_THRESHOLD
        and (2.0 * positive - 1.0) >= settings.HIJAB_MIN_SCORE_MARGIN
    )
    return is_hijab, report

def analyse_user(user_path: Path, crop_dir: Path = None) -> dict:
    user_path = Path(user_path)
    key = user_path.stem
    if user_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise Rejected(f"Unsupported file type: {user_path.suffix}")

    img_bgr = load_image(user_path)
    face, ignored_background_faces = detect_main_face(user_path)
    area = face["facial_area"]
    crop_bgr = crop_face(img_bgr, area)
    crop_pil = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    _ = check_scope(crop_pil)
    gender, g_conf, g_src = detect_gender(face, crop_bgr, key)

    views = create_image_views(crop_pil)
    glasses, glasses_s = detect_glasses(views, gender)
    beard, beard_s = detect_beard(crop_pil) if gender == "Man" else (False, None)
    bald = False
    if gender == "Man" and settings.DETECT_BALD:
        _x1 = max(0, (int(area["x"]) + int(area["w"]) // 2) - int(int(area["w"]) * settings.CROP_SCALE) // 2)
        _y1 = max(0, (int(area["y"]) + int(area["h"]) // 2) - int(int(area["h"]) * settings.CROP_SCALE) // 2)
        rect = (int(area["x"]) - _x1, int(area["y"]) - _y1, int(area["w"]), int(area["h"]))
        bald, bald_s = detect_bald(crop_pil, crop_bgr, rect)
        logger.info("Bald check for %s: bald=%s %s", key, bald, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in bald_s.items()})
    hijab, hijab_s = classify_hijab(crop_pil) if gender == "Woman" else (False, None)

    ov = OVERRIDES.get(key, {})
    if "glasses" in ov:
        glasses = bool(ov["glasses"])
    if "hijab" in ov and gender == "Woman":
        hijab = bool(ov["hijab"])
    if "beard" in ov and gender == "Man":
        beard = bool(ov["beard"])
    if "bald" in ov and gender == "Man":
        bald = bool(ov["bald"])

    tags = [gender] + (["Glasses"] if glasses else []) + (["Hijab"] if hijab else []) + (["Bald"] if bald else [])
    target_crop_dir = crop_dir or settings.CROP_DIR
    target_crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = target_crop_dir / f"{key}_{'_'.join(tags)}.jpg"
    cv2.imwrite(str(crop_path), crop_bgr)

    info = {
        "key": key,
        "user_path": str(user_path),
        "crop_path": str(crop_path),
        "face_box": [int(area["x"]), int(area["y"]), int(area["w"]), int(area["h"])],
        "faces": 1,
        "background_faces_ignored": ignored_background_faces,
        "gender": gender,
        "gender_conf": round(g_conf, 1),
        "gender_src": g_src,
        "glasses": glasses,
        "glasses_source": glasses_s.get("source", "step1_clip") if isinstance(glasses_s, dict) else "step1_clip",
        "hijab": hijab,
        "beard": beard,
        "bald": bald,
        "avatar": "Woman_Hijab" if hijab else gender,
    }
    logger.info(
        "Step 1 analysis complete for %s: avatar=%s, gender=%s (%.1f%% %s), glasses=%s, hijab=%s, beard=%s",
        key,
        info["avatar"],
        gender,
        g_conf,
        g_src,
        glasses,
        hijab,
        beard,
    )
    return info

# ============================================================================
# Prompt assembly (verbatim, tuned text - do not reword)
# ============================================================================

_SEMI_REAL = (
    "render the swapped head in the semi-realistic digital painting style of Picture 1: a realistic, true-to-life "
    "face with natural proportions and a recognisable likeness, painted with smooth airbrushed skin and soft "
    "realistic shading (not flat cartoon colors, not a photo and not a vector icon). subtle thin dark line work only "
    "around the eyes, lips, hair strands and jacket edges. detailed realistic eyes with crisp catchlights, soft "
    "glossy lips, natural eyebrows, fine individually painted hair strands with a glossy sheen, warm natural skin "
    "tones with gentle highlights, a strong red rim light along the hair, forehead and cheek edge, and cool blue "
    "fill light on the shadow side, matching the color palette and lighting of the jacket and emblem. the head "
    "blends seamlessly into the painting with no visible swap seam and no harsh photographic texture, "
    "and keeps exactly the face of Picture 2: same face shape, jaw, hairline, facial hair and the real eye color of "
    "Picture 2 (do not turn the eyes blue). "
    "keep the natural age, forehead lines, smile lines, under-eye detail and any grey in the beard or hair of "
    "Picture 2; the airbrushed finish must not slim the face, soften the jaw or change any proportion. "
    "sharp detail, high quality, 4k."
)

_COMIC = (
    "render the swapped head fully in the illustration style of Picture 1, not as a photo: a premium digital "
    "comic-book vector portrait with bold clean ink outlines of consistent line weight, cel-shaded skin with smooth "
    "airbrushed gradients and crisp highlight and shadow shapes, rich saturated warm skin tones, individually inked "
    "hair strands with a glossy sheen, glossy lips and sharp eye catchlights, a strong red rim light along the hair, "
    "forehead, cheek and jaw edge and cool blue fill shadows on the opposite side, matching the line quality, color "
    "palette and lighting of the jacket and emblem. the head blends seamlessly into the illustration with no "
    "photographic texture and no swap seam, while still looking exactly like the person in "
    "Picture 2: same face shape, hairline, facial hair and the real eye color of Picture 2 (do not turn the eyes "
    "blue). sharp detail, high quality, 4k."
)

IDENTITY = (
    "IDENTITY IS THE TOP PRIORITY: the result must look like the same real person as Picture 2, not like the person "
    "or the face shape in Picture 1. copy the exact face geometry of Picture 2: overall face width-to-height ratio, "
    "forehead height and width, cheekbone position, full cheeks, jaw width, jawline angle and chin shape (never slim, "
    "narrow, sharpen, shorten or elongate the face, never make it more handsome, younger or more glamorous), the exact "
    "nose length, bridge and nostril width, the exact lip shape and thickness and the same mouth expression (a "
    "closed-mouth smile stays closed, an open smile with visible teeth stays open with the same visible teeth), the "
    "same eyebrow shape, thickness and spacing, the same eye shape, eye size and eye distance, the exact eye color of "
    "Picture 2 (dark brown stays dark brown, never lighten to blue, green or grey), the same skin tone, apparent age, "
    "smile lines and under-eye detail. keep every facial landmark in the same relative position "
    "as Picture 2; only the painting style comes from Picture 1. "
)

# "id_focus" prompt variant (settings.PROMPT_VARIANT): identity text FIRST (the text encoder weights early tokens most;
# the tuned prompt opens with ~6 visor sentences), concrete landmark-by-landmark copy instructions, and an explicit
# "keep the real skin texture" rule (the airbrushed style wording erases wrinkles/pores/beard texture = looks like a
# younger, different person). Same content rules as IDENTITY, just shorter, ordered and repeated once at the end.
IDENTITY_FOCUS = (
    "FACE IDENTITY (highest priority, above style and above the visor): the face must be recognisable as the exact "
    "person in Picture 2 at first glance. Picture 2 is the ONLY source of facial structure; Picture 1 supplies only "
    "painting style, lighting, jacket, emblem and background. copy from Picture 2, landmark by landmark: head and face "
    "width-to-height ratio, forehead, cheekbones, cheek fullness, jaw width and jawline angle, chin, ear shape and "
    "size, nose length, bridge and nostrils, lip shape and thickness, the same mouth and smile (closed stays closed, "
    "visible teeth stay visible), eyebrow shape and thickness, eye shape, eye size, eye spacing and the real eye "
    "color. keep every landmark at the same relative position and distance as Picture 2. keep the real skin texture "
    "and age: forehead lines, smile lines, crow's feet, under-eye bags, pores, beard stubble and the true grey or "
    "white strands in the beard; do not airbrush them away. never slim, sharpen, lengthen, rejuvenate or beautify the "
    "face, never change the skin tone. "
)

IDENTITY_FOCUS_CLOSING = (
    "final identity check: the head must read as the same real person as Picture 2: same face shape, jaw, cheeks, "
    "ears, nose, mouth, eyes, eyebrows, beard shape and skin texture. "
)

def _id_focus_style(text: str) -> str:
    """Dial back the style wording that smooths the face / shifts the skin colour (id_focus variant only)."""
    return (
        text.replace("painted with smooth airbrushed skin and soft realistic shading",
                     "painted with softly shaded skin that keeps the visible skin texture")
        .replace("cel-shaded skin with smooth airbrushed gradients and crisp highlight and shadow shapes, "
                 "rich saturated warm skin tones,",
                 "cel-shaded skin with soft gradients and crisp highlight and shadow shapes that keep the visible "
                 "skin texture and the real skin tone of Picture 2,")
        .replace("warm natural skin tones with gentle highlights,", "the natural skin tone of Picture 2 with gentle highlights,")
    )

BUILD = (
    "BUILD: match the fullness of the person in Picture 2. if the face is round, full or heavy, keep it a little "
    "fuller: fuller cheeks, softer wider jaw, a fuller chin (even a soft double chin) and a thicker neck. if the face "
    "is thin or lean, keep it a little thinner: leaner cheeks, a defined narrower jaw, visible cheekbones and a "
    "slimmer neck. if average, keep it average. never slim down a full face and never fatten a thin face, and never "
    "copy the build of the person in Picture 1. "
)

FACE_CLEAN = (
    "FACE QUALITY: a clean, continuous, smooth jawline and chin contour from ear to chin on both sides, symmetrical and "
    "undistorted, joining the neck naturally, with no warped, melted, doubled, broken or jagged jaw outline. the skin "
    "is clean, smooth and evenly painted in one natural skin tone. "
)

VISOR_MANDATORY = (
    "CRITICAL REQUIREMENT: The face MUST wear the exact angular electric-blue wraparound visor "
    "from Picture 1 or 3. This is non-negotiable. No ordinary glasses, no sunglasses, no other eyewear. "
    "Only the futuristic visor with the clear transparent lens and blue frame. "
)

VISOR_LOOK_MAN = (
    "the visor is exactly the one worn in Picture 1: a single flat, angular, futuristic wraparound shield made of ONE "
    "continuous transparent panel, with a perfectly straight horizontal top edge running just under the eyebrows from "
    "the outer eye corner on one side, across the bridge of the nose, to the temple on the other side; sharp "
    "chamfered (cut, faceted) lower corners and a shallow notch for the nose; a thin light electric-blue frame line "
    "along its edges; a chunky faceted cyan-blue corner block with a white highlight at the outer end on the side "
    "nearest the camera; and a thick silver-white and blue arm running back along the temple to the ear on the far "
    "side. the lens is clear and transparent with only a very light icy-blue tint and a few crisp white diagonal "
    "glints, so the eyes, eyelashes, eyelids and eyebrows of Picture 2 stay sharp and fully visible through it. "
)

VISOR_LOOK_DEFAULT = (
    "the visor is exactly the large angular futuristic shield worn in Picture 1: ONE continuous transparent panel "
    "covering both eyes, with a straight thin electric-blue upper rim just under the eyebrows, broad faceted outer "
    "corners, a shallow V-shaped lower edge around the nose, cyan-blue side blocks and slim arms returning to both "
    "temples. the lens is clear and transparent with only a very light icy-blue tint and a few crisp white glints, "
    "so the eyes, eyelashes, eyelids and eyebrows of Picture 2 stay sharp and fully visible through it. do not turn "
    "it into ordinary eyeglasses or two separate lenses. "
)

VISOR_REF_TEXT = (
    "Picture 3 is a close-up of the exact visor glasses: copy this visor exactly onto the face (same shape, straight "
    "top edge, angular corners, blue frame, clear lens, corner blocks and side arms). use ONLY the glasses from "
    "Picture 3, never its face. "
)

VISOR_CORE = (
    "EYEWEAR (mandatory): the face wears the Picture 1 visor and NOTHING else. if Picture 2 wears eyeglasses of any "
    "kind (black, thick, round, rectangular, thin metal or clear), DELETE them: erase their frames, rims, nose pads, "
    "arms and lens reflections completely and paint bare natural skin where they were, then put the visor on top. "
)

VISOR_NEG = (
    "exactly ONE pair of glasses on the face: no black frame, no thick rims, no second frame, no double lines above "
    "or below the visor, no round or rectangular eyeglasses, no rounded goggles, no sunglasses, no opaque or solid "
    "blue lens. "
)

VISOR_FIT = (
    "fit the visor to the face of Picture 2: its width spans exactly from temple to temple of that face and is never "
    "wider than the face, it sits level across the bridge of the nose directly over both eyes with the eyes centered "
    "behind the lens, it follows the same head tilt and perspective as the face, and its arms end at the temples and "
    "ears. not oversized, not floating, not tilted, not sliding off the face. "
)

VISOR_RETRY = (
    "REMINDER: ordinary or thick eyeglasses must NOT appear anywhere; only the angular blue-framed clear wraparound "
    "visor of Picture 1 appears on the face, clearly visible. "
)

HIJAB_VISOR_NOTE = (
    "the visor sits on the face inside the hijab opening: it is no wider than the visible face between the two edges "
    "of the hijab, its arms and corner blocks tuck against the hijab at the temples (they may rest slightly over the "
    "fabric edge), the hijab edge above the eyebrows stays fully visible and is not covered, and the visor never "
    "sticks out beyond the hijab outline or covers the forehead fabric. the hijab fabric is not deformed by it. the "
    "visor MUST be clearly visible on her face. "
)

HIJAB_USER = (
    "she wears her own hijab from Picture 2, copied exactly: the same fabric colors (including any two-tone or "
    "lighter under-scarf), the same sheen and soft folds, wrapped the same way tightly around the face with the "
    "edge sitting at the same place on the forehead, cheeks and under the chin, covering all hair, both ears and the "
    "neck, then falling in soft drapes onto the shoulders and chest and flowing into the jacket collar of Picture 1. "
    "no hair visible, no pins, brooches or patterns added, the fabric is not changed to another color. only her head "
    "and hijab are taken from Picture 2; ignore her clothes, cardigan, shirt, body and background. "
)

HIJAB_FACE = (
    "keep her natural look from Picture 2: the full natural face with its full cheeks and soft jaw (do not slim or "
    "shrink it), her natural skin tone and smile lines, her natural makeup level and "
    "lip color only - do NOT add eyeliner, eyeshadow, long lashes, contouring or glamour retouching, and do not make "
    "her look younger. the hijab frames the face exactly like Picture 2, not looser and not further back. "
)

FEMALE_FACE = (
    "FEMALE FACE DETAILS: copy her face details exactly from Picture 2: the eyebrow shape, thickness and arch, the eye "
    "shape, size and eyelid fold, her natural lash level, the nose width, bridge and tip, the lip shape, fullness and "
    "natural lip color, the smile with the same visible teeth, the cheek fullness, the forehead, the face shape and "
    "chin, her skin tone and dimples. use only a subtle natural makeup level like Picture 2; do not "
    "add glamour makeup, strong lipstick, long lashes or contouring, and do not make her look younger or thinner. "
    "from Picture 1 take only the hairstyle, jacket, painting style and visor. "
)

HAIR_USER = (
    "HAIR: ignore the hairstyle of Picture 1 completely and keep the exact hair of Picture 2: the same color (never "
    "add red, orange or blond highlights), length, texture (curly, wavy or straight), volume, hairline, parting and "
    "side length. if Picture 2 has short hair keep it short; if Picture 2 is bald, shaved or has a receding hairline "
    "keep the scalp bald or receding exactly as in Picture 2 and do NOT add, grow or paint any hair. do not smooth, "
    "slick back, comb up, restyle or thicken the hair. the only change allowed on the hair is a thin red rim light. "
)

BALD_USER = (
    "BALD HEAD / NO HAIR: Picture 2 is a BALD man with a completely bare shaved scalp. "
    "Do NOT include or copy any hair from Picture 1 or anywhere else. "
    "The head MUST BE COMPLETELY BALD with a bare, smooth shaved scalp and NO HAIR whatsoever on top, forehead, sides or back. "
    "Completely remove the hair of Picture 1 and replace it with a smooth bare bald scalp matching Picture 2. "
    "Do not draw, paint, grow or add any hair, hairline, haircut, tufts or strands of hair. "
    "The top of the head must be clean smooth bare skin. "
)

BALD_RETRY = (
    "RETRY - the previous attempt wrongly kept hair. The swept, curly or styled hair of Picture 1 must NOT appear: "
    "paint the whole top and sides of the head above the forehead as one smooth, glossy bare scalp in skin tone with "
    "only a thin red rim light on its edge, exactly like the bald head of Picture 2. "
)

def visor_block(info, attempt=0, ref=False):
    look = VISOR_LOOK_MAN if info.get("avatar") == "Man" else VISOR_LOOK_DEFAULT
    return (
        VISOR_CORE
        + (VISOR_REF_TEXT if ref else "")
        + look
        + VISOR_NEG
        + VISOR_FIT
        + (VISOR_RETRY if attempt > 0 else "")
    )

def build_prompt(info, attempt=0, ref=False, style_mode=None, keep_user_expression=None):
    if style_mode is None:
        style_mode = settings.STYLE_MODE
    if keep_user_expression is None:
        keep_user_expression = settings.KEEP_USER_EXPRESSION

    # Only the MALE prompt has a bald branch; women / hijab prompts are untouched.
    is_bald = bool(info.get("bald", False)) and not info.get("hijab", False) and info.get("gender") != "Woman"

    expr = (
        "copy the head rotation and eye direction from Picture 1, but keep the facial expression and smile of Picture 2"
        if keep_user_expression
        else "copy the direction of the eye, head rotation, micro expressions from Picture 1"
    )
    hair_desc = "bare bald scalp (no hair)" if is_bald else "hair"
    bfs = (
        "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. "
        "remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly "
        f"preserving the face, {hair_desc}, eye color and nose structure of Picture 2. "
        + expr
        + ". "
    )

    base_style = (
        {"semi_real": _SEMI_REAL, "comic": _COMIC}[style_mode]
        if style_mode in ("semi_real", "comic")
        else "high quality, sharp details, 4k."
    )
    if is_bald and style_mode in ("semi_real", "comic"):
        base_style = (
            base_style.replace("around the eyes, lips, hair strands and jacket edges.", "around the eyes, lips, scalp contour and jacket edges.")
            .replace("fine individually painted hair strands with a glossy sheen,", "")
            .replace("individually inked hair strands with a glossy sheen,", "")
            .replace("along the hair,", "along the bald scalp,")
            .replace("along the hair", "along the bald scalp")
            .replace("beard or hair", "beard or scalp")
            .replace("same face shape, jaw, hairline,", "same face shape, jaw, bald scalp,")
            .replace("same face shape, hairline,", "same face shape, bald scalp,")
        )
    style_text = (
        ("the painted finish applies to the rendering only and must never change the face proportions. " + base_style)
        if style_mode in ("semi_real", "comic")
        else "high quality, sharp details, 4k."
    )
    V = visor_block(info, attempt, ref)
    closing = (
        "keep the jacket, emblem and solid black background of Picture 1 unchanged. "
        "final check: the face and build match Picture 2, the jawline is clean, and the angular blue clear "
        "visor of the avatar is on the face with no other glasses."
    )
    # Prompt variant: "tuned" (default) is byte-for-byte the notebook order; "id_focus" puts the identity text first.
    head, IDENTITY_TXT = f"{VISOR_MANDATORY}{bfs}", IDENTITY
    if settings.PROMPT_VARIANT == "id_focus":
        head, IDENTITY_TXT = f"{bfs}{IDENTITY_FOCUS}{VISOR_MANDATORY}", ""
        style_text = _id_focus_style(style_text)
        closing = IDENTITY_FOCUS_CLOSING + closing
    if info.get("hijab", False):
        return (
            f"{head}{IDENTITY_TXT}{BUILD}{V}{HIJAB_VISOR_NOTE}"
            f"{HIJAB_FACE}{HIJAB_USER}{FACE_CLEAN}{style_text} {closing} the visor is present and fits her face."
        )
    if info.get("gender") == "Woman":
        return (
            f"{head}{IDENTITY_TXT}{FEMALE_FACE}{BUILD}"
            "her hair is styled exactly like Picture 1: long black hair with red and orange highlights woven "
            "throughout, styled in a high voluminous bun or updo at the crown, sleek and professionally polished, "
            f"framing the face. {V}{FACE_CLEAN}{style_text} " + closing
        )
    hair_user_block = (BALD_USER + (BALD_RETRY if attempt > 0 else "")) if is_bald else HAIR_USER
    return (
        f"{head}{IDENTITY_TXT}{BUILD}{hair_user_block}"
        "FACIAL HAIR: keep the facial hair of Picture 2 exactly as it is: if Picture 2 has a moustache, goatee, "
        "beard or stubble keep the same shape, coverage, length, density and grey or black color with a neat "
        "natural edge; if Picture 2 is clean-shaven keep the skin smooth and add no facial hair. do not copy any "
        f"facial hair from Picture 1. {V}{FACE_CLEAN}{style_text} " + closing
    )

# ============================================================================
# Step 2 - avatar generation from the Step-1 face crop
# ============================================================================

AVATAR_PATHS = {
    "Man": "Saytara_male.jpg",
    "Woman": "Saytara_Femal.png",
    "Woman_Hijab": "Saytara_hijab.jpg",
}

TEST_AVATARS = {
    "Man": "Saytara_male.jpg",
    "Woman": "Saytara_Femal.png",
    "Woman_Hijab": "Saytara_hijab.jpg",
}

VISOR_REFS = {
    "Man":         {"file": "Saytara_male.jpg",   "box": (0.28, 0.22, 0.65, 0.40), "image": None},
    "Woman":       {"file": "Saytara_Femal.png", "box": (0.25, 0.29, 0.63, 0.49), "image": None},
    "Woman_Hijab": {"file": "Saytara_hijab.jpg",        "box": (0.26, 0.25, 0.64, 0.45), "image": None},
}

AVATAR_HEAD_BOX = {
    "Man": (0.25, 0.05, 0.68, 0.62),
    "Woman": (0.25, 0.05, 0.68, 0.62),
    "Woman_Hijab": (0.30, 0.08, 0.72, 0.72),
}

_AV: Dict[str, Tuple[Image.Image, str]] = {}
_VR: Dict[str, Optional[Image.Image]] = {}

def fit16(img: Image.Image) -> Image.Image:
    w, h = (img.width // 16) * 16, (img.height // 16) * 16
    return img if (w, h) == img.size else img.resize((w, h), Image.LANCZOS)

def to_black_bg(img: Image.Image, thresh: int = 40) -> Image.Image:
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    dist = (255 - a).max(axis=2)
    _, lab = cv2.connectedComponents((dist < thresh).astype(np.uint8), connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    bg = np.isin(lab, list(border)).astype(np.float32)
    bg = cv2.GaussianBlur(cv2.dilate(bg, np.ones((3, 3), np.uint8), 1), (5, 5), 0)
    alpha = np.maximum(
        bg,
        cv2.GaussianBlur(bg, (9, 9), 0) * np.clip(1.0 - dist / (thresh * 3), 0.0, 1.0),
    )
    return Image.fromarray((a * (1.0 - alpha[..., None])).clip(0, 255).astype(np.uint8))

def avatar_on_black(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA", "P") or "transparency" in img.info:
        img = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", img.size, (0, 0, 0, 255)), img)
    img = img.convert("RGB")
    a = np.asarray(img)
    if np.concatenate([a[:8, :8], a[:8, -8:], a[-8:, :8], a[-8:, -8:]]).mean() > 200:
        img = to_black_bg(img)
    return fit16(img)

def load_avatar(key: str) -> Tuple[Image.Image, str]:
    if key not in _AV:
        primary_filename = TEST_AVATARS.get(key)
        primary_path = (settings.AVATAR_DIR / primary_filename) if primary_filename else None
        if primary_path and primary_path.exists():
            chosen = primary_path
        else:
            fallback_filename = AVATAR_PATHS.get(key)
            if not fallback_filename:
                raise KeyError(f"Unknown avatar key '{key}'")
            fallback_path = settings.AVATAR_DIR / fallback_filename
            if not fallback_path.exists():
                raise FileNotFoundError(
                    f"Avatar template '{key}' not found in {settings.AVATAR_DIR} "
                    f"(looked for {primary_filename} and {fallback_filename})"
                )
            chosen = fallback_path
        logger.info("Loaded avatar template for %s: %s", key, chosen.name)
        _AV[key] = (avatar_on_black(chosen), chosen.name)
    return _AV[key]

def load_visor_ref(key: str) -> Optional[Image.Image]:
    if not settings.USE_VISOR_REF or key not in VISOR_REFS:
        return None
    if key not in _VR:
        cfg = VISOR_REFS[key]
        img = None
        if cfg.get("image") and Path(cfg["image"]).exists():
            crop = avatar_on_black(Path(cfg["image"]))
        else:
            primary_file = settings.AVATAR_DIR / cfg["file"]
            fallback_file = settings.AVATAR_DIR / AVATAR_PATHS.get(key, "")
            src_path = (
                primary_file
                if primary_file.exists()
                else (fallback_file if fallback_file.exists() else None)
            )
            if src_path:
                src = avatar_on_black(src_path)
                w, h = src.size
                x0, y0, x1, y1 = cfg["box"]
                crop = src.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))
            else:
                crop = None
                logger.warning(
                    "Visor reference file not found for %s (%s) - using text description only",
                    key,
                    cfg["file"],
                )
        if crop is not None:
            s = max(crop.size)
            canvas = Image.new("RGB", (s, s), "black")
            canvas.paste(crop, ((s - crop.width) // 2, (s - crop.height) // 2))
            img = canvas.resize((768, 768), Image.LANCZOS)
        _VR[key] = img
    return _VR[key]

def pad_square(img: Image.Image, size: int = None) -> Image.Image:
    size = size or settings.REF_SIZE
    s = max(img.size)
    bg = img.resize((s, s), Image.LANCZOS).filter(ImageFilter.GaussianBlur(max(2, s // 18)))
    bg.paste(img, ((s - img.width) // 2, (s - img.height) // 2))
    return bg.resize((size, size), Image.LANCZOS)

def load_face(info: dict) -> Image.Image:
    p = info.get("crop_path")
    if p and Path(p).exists():
        return pad_square(Image.open(p).convert("RGB"))
    img = ImageOps.exif_transpose(Image.open(info["user_path"])).convert("RGB")
    x, y, w, h = info["face_box"]
    s = max(w, h) * (settings.HIJAB_SCALE if info.get("hijab") else settings.FALLBACK_SCALE)
    cx, cy = x + w / 2, y + h / 2 - h * 0.05
    return pad_square(
        img.crop(
            (
                int(max(0, cx - s)),
                int(max(0, cy - s)),
                int(min(img.width, cx + s)),
                int(min(img.height, cy + s)),
            )
        )
    )

def check_visor(out_img: Image.Image) -> dict:
    a = np.asarray(out_img.convert("RGB"))
    h, w = a.shape[:2]
    regions = {
        "left":   a[int(h * 0.20):int(h * 0.46), int(w * 0.26):int(w * 0.49)],
        "bridge": a[int(h * 0.22):int(h * 0.43), int(w * 0.44):int(w * 0.57)],
        "right":  a[int(h * 0.20):int(h * 0.46), int(w * 0.52):int(w * 0.76)],
    }
    s = {}
    for name, r in regions.items():
        hsv = cv2.cvtColor(np.ascontiguousarray(r), cv2.COLOR_RGB2HSV)
        s[name] = float(
            (
                (hsv[..., 0] >= 85)
                & (hsv[..., 0] <= 130)
                & (hsv[..., 1] > 60)
                & (hsv[..., 2] > 90)
            ).mean()
        )
    ok = s["left"] > 0.006 and s["bridge"] > 0.002 and s["right"] > 0.006
    return {
        "visor": "ok" if ok else "weak",
        "visor_score": round(min(s["left"], s["right"]) + s["bridge"], 4),
    }

def check_bald_output(out_img: Image.Image, info: dict) -> dict:
    """Output guard for BALD users: did the avatar keep/grow hair (the template's hair) instead of a bare scalp?
    CLIP on the avatar's head (and its top part) with illustration wording. Returns hair="ok"|"hair" and bald_score
    (CLIP share for "bald"; higher = baldier, also used to rank candidates). Not bald / not male -> always ok."""
    if not (settings.VERIFY_BALD_OUTPUT and info.get("bald") and info.get("avatar") == "Man"):
        return {"hair": "ok", "bald_score": None}
    try:
        box = AVATAR_HEAD_BOX.get(info["avatar"], (0.25, 0.05, 0.68, 0.62))
        w, h = out_img.size
        head = out_img.convert("RGB").crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
        top = head.crop((0, 0, head.width, int(head.height * 0.55)))
        s = classify_binary(
            [head, top],
            [
                "an illustration of a bald man with a bare shaved scalp",
                "a stylized portrait of a man with no hair on his head",
                "a digital painting of a completely bald man",
            ],
            [
                "an illustration of a man with hair on his head",
                "a stylized portrait of a man with a haircut and styled hair",
                "a digital painting of a man with thick hair",
            ],
        )
    except Exception:
        logger.exception("Bald output check failed - treating the candidate as ok")
        return {"hair": "ok", "bald_score": None}
    ok = s["positive"] >= settings.OUT_BALD_MIN
    return {"hair": "ok" if ok else "hair", "bald_score": round(s["positive"], 3)}

def _prep(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    k = settings.VERIFY_MAX_SIDE / max(img.size)
    return (
        img.resize((max(1, int(img.width * k)), max(1, int(img.height * k))), Image.LANCZOS)
        if k < 1
        else img
    )

def embed(img: Image.Image, enforce: bool = True, name: str = "out") -> Optional[dict]:
    im = _prep(img)
    settings.VAL_TMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = (
        settings.VAL_TMP_DIR / f"{name}_{os.getpid()}_{threading.get_ident()}_{time.time_ns()}.png"
    )
    im.save(temp_path)
    try:
        with _VLOCK:
            try:
                p = get_val_worker()
                p.stdin.write(json.dumps({"img": str(temp_path), "enforce": bool(enforce)}) + "\n")
                p.stdin.flush()
                while True:
                    line = p.stdout.readline()
                    if not line:
                        close_val_worker()
                        return None
                    if line.startswith("__JSON__"):
                        r = json.loads(line[len("__JSON__"):])
                        break
            except BaseException:
                close_val_worker()
                raise
        if "error" in r or not r.get("faces"):
            return None
        return {"faces": r["faces"], "width": im.width}
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass

def embed_user(face: Image.Image) -> Optional[dict]:
    return embed(face, True, "user") or embed(face, False, "user")

def _ratio(f: dict, width: int) -> Optional[float]:
    w, h = f.get("w"), f.get("h")
    return (w / max(h, 1)) if (w and h and w < 0.95 * width) else None

def _cos_dist(a: list, b: list) -> float:
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    return 1.0 - float(arr_a @ arr_b / (np.linalg.norm(arr_a) * np.linalg.norm(arr_b) + 1e-12))

def compare(user: dict, out: dict) -> Tuple[float, Optional[float], Optional[float]]:
    d, u, o = min(
        (
            (_cos_dist(u["emb"], o["emb"]), u, o)
            for u in user["faces"]
            for o in out["faces"]
        ),
        key=lambda t: t[0],
    )
    return round(1.0 - d, 4), _ratio(u, user["width"]), _ratio(o, out["width"])

def score_output(user_emb: Optional[dict], out: Image.Image, info: dict) -> dict:
    vis = check_visor(out)
    vis.update(check_bald_output(out, info))
    sim = u_r = o_r = None
    if settings.VALIDATE and user_emb is not None:
        o = embed(out, True, "out")
        if o is None:
            box = AVATAR_HEAD_BOX.get(info["avatar"], (0.25, 0.05, 0.68, 0.62))
            w, h = out.size
            o = embed(
                out.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h))),
                False,
                "out",
            )
        if o is not None:
            sim, u_r, o_r = compare(user_emb, o)
    jaw = (abs(o_r - u_r) / u_r) if (u_r and o_r) else None
    score = (
        (sim or 0.0)
        - settings.W_JAW * (jaw or 0.0)
        - (settings.W_NO_VISOR if vis["visor"] == "weak" else 0.0)
        - (settings.W_HAIR if vis["hair"] == "hair" else 0.0)
        + (0.5 * (vis["bald_score"] or 0.0) if info.get("bald") else 0.0)   # among bald candidates prefer the baldest
    )
    return {
        "id_sim": None if sim is None else round(sim, 3),
        "jaw_diff": None if jaw is None else round(jaw, 3),
        "score": round(score, 3),
        **vis,
    }

def encode_jpeg_capped(img: Image.Image, max_bytes: int = None) -> bytes:
    """Encode `img` as JPEG, stepping quality down and then resolution down
    until the encoded size is at or under `max_bytes`. Falls back to the
    smallest encoding found (with a logged warning) if the budget still can't
    be met at the lowest quality/resolution tried."""
    max_bytes = max_bytes or settings.OUTPUT_MAX_BYTES
    rgb = img.convert("RGB")
    quality_steps = (95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20)
    scale = 1.0
    smallest: Optional[bytes] = None
    for _ in range(8):
        work = (
            rgb
            if scale >= 0.999
            else rgb.resize(
                (max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))),
                Image.LANCZOS,
            )
        )
        for quality in quality_steps:
            buf = io.BytesIO()
            work.save(buf, format="JPEG", quality=quality, optimize=True)
            data = buf.getvalue()
            if smallest is None or len(data) < len(smallest):
                smallest = data
            if len(data) <= max_bytes:
                return data
        scale *= 0.85
    logger.warning(
        "Could not encode output under %d bytes after quality/resolution search; "
        "returning smallest JPEG found (%d bytes).",
        max_bytes,
        len(smallest),
    )
    return smallest

def generate_avatar(user_image_path: Path, info: dict, t0: Optional[float] = None,
                    slot: Optional[GpuSlot] = None, overrides: Optional[dict] = None) -> Tuple[bytes, dict]:
    # t0 defaults to "now" (generation-only budget) but callers should pass the
    # timestamp from before Step 1 analysis so TIME_BUDGET covers the whole
    # image end-to-end, matching the notebook.
    if t0 is None:
        t0 = time.time()
    if slot is None:
        slot = slots[0]
    user_path = Path(user_image_path)
    rec: Dict[str, Any] = {"file": user_path.name}
    ov = overrides or {}                       # per-request test overrides (see ALLOW_TEST_OVERRIDES in app.py)
    steps = int(ov.get("steps", settings.STEPS))
    time_budget = float(ov.get("time_budget", settings.TIME_BUDGET))

    test_lora = settings.LORA_STRENGTH * settings.ID_LORA_MULT
    progress = None
    try:
        avatar, _ = load_avatar(info["avatar"])
        face = load_face(info)
        visor_ref = load_visor_ref(info["avatar"])
        images = [avatar, face] + ([visor_ref] if visor_ref is not None else [])
        seed = settings.SEED
        w, h = avatar.size
        max_k = int(ov.get("tries", settings.BEST_OF_N)) if settings.VALIDATE else 1
        extras = 0
        hair_extras = 0

        dbg = None
        if settings.DEBUG_DUMP:
            dbg = settings.OUTPUT_DIR / "debug" / f"{info['key']}_{info['avatar']}"
            dbg.mkdir(parents=True, exist_ok=True)
            avatar.save(dbg / "0_avatar.png")
            face.save(dbg / "1_face_ref.png")
            if visor_ref is not None:
                visor_ref.save(dbg / "2_visor_ref.png")
            (dbg / "info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
            (dbg / "prompt_try1.txt").write_text(build_prompt(info, 0, visor_ref is not None), encoding="utf-8")
            logger.info("DEBUG_DUMP -> %s (adapters=%s, lora_layers=%d)", dbg, _adapters(), lora_layers)

        # CPU embedding runs in thread while GPU generates try #1
        fut = _POOL.submit(embed_user, face) if settings.VALIDATE else None

        best, k, gen_s, val_s, stopped = None, 0, 0.0, 0.0, "all tries"
        t_loop = time.time()

        progress = tqdm(total=max_k, desc=f"Generating avatar ({user_path.name})", unit="try")
        while True:
            mult = settings.ID_LORA_SCHEDULE[k % len(settings.ID_LORA_SCHEDULE)]
            slot.set_lora(test_lora * mult)
            prompt = build_prompt(info, attempt=k, ref=visor_ref is not None)

            tg = time.time()
            with torch.no_grad(), torch.cuda.stream(slot.stream):
                out = slot.pipe(
                    **prompt_inputs(slot, prompt),
                    image=images,
                    attention_kwargs=slot.lora_kwargs(),
                    width=w,
                    height=h,
                    num_inference_steps=steps,
                    guidance_scale=settings.CFG,
                    generator=torch.Generator("cuda").manual_seed(seed + k),
                ).images[0]
            slot.stream.synchronize()
            torch.cuda.empty_cache()
            g = time.time() - tg

            tv = time.time()
            user_emb = fut.result() if fut is not None else None
            v = score_output(user_emb, out, info)
            s = time.time() - tv

            gen_s, val_s = gen_s + g, val_s + s
            if dbg is not None:
                out.save(dbg / f"try{k + 1}_seed{seed + k}_lora{mult}.png")
            logger.info(
                "Try %d (seed=%d, lora_mult=%.1f): id_sim=%s, jaw_diff=%s, visor=%s, hair=%s(bald_score=%s), score=%s (gen=%.1fs, score=%.1fs)",
                k + 1,
                seed + k,
                mult,
                v["id_sim"],
                v["jaw_diff"],
                v["visor"],
                v["hair"],
                v["bald_score"],
                v["score"],
                g,
                s,
            )

            if v["hair"] != "ok":
                logger.warning("Try %d: BALD user but the avatar has hair (bald_score=%s) - candidate penalised", k + 1, v["bald_score"])
            if best is None or (v["visor"] == "ok", v["hair"] == "ok", v["score"]) > (
                best["v"]["visor"] == "ok",
                best["v"]["hair"] == "ok",
                best["v"]["score"],
            ):
                best = {"out": out, "v": v, "k": k + 1, "mult": mult}

            k += 1
            progress.set_postfix(id_sim=v["id_sim"], visor=v["visor"], score=v["score"])
            progress.update(1)

            if settings.VALIDATE and (v["id_sim"] or 0) >= settings.STOP_ID and v["visor"] == "ok" and v["hair"] == "ok":
                stopped = "good"
                break

            if k >= max_k:
                if best["v"]["visor"] != "ok" and extras < settings.VISOR_EXTRA_TRIES:
                    extras += 1
                    max_k += 1
                    progress.total = max_k
                    progress.refresh()
                    logger.info("Visor not ok - adding extra try (if budget permits)")
                elif best["v"]["hair"] != "ok" and hair_extras < settings.HAIR_EXTRA_TRIES:
                    hair_extras += 1
                    max_k += 1
                    progress.total = max_k
                    progress.refresh()
                    logger.info("Bald user but every candidate still has hair - adding extra try (if budget permits)")
                else:
                    break

            per_try = (time.time() - t_loop) / k
            if time_budget > 0 and time.time() - t0 + per_try > time_budget:
                stopped = "budget"
                logger.info(
                    "Time budget reached (%.1fs elapsed, ~%.1fs per try) - stopping search",
                    time.time() - t0,
                    per_try,
                )
                break

        out, v = best["out"], best["v"]
        jpeg_bytes = encode_jpeg_capped(out, settings.OUTPUT_MAX_BYTES)
        rec.update(
            status="done",
            gender=info["gender"],
            hijab=info.get("hijab", False),
            avatar=info["avatar"],
            glasses=info.get("glasses", False),
            beard=info.get("beard", False),
            bald=info.get("bald", False),
            hair=v["hair"],
            bald_score=v["bald_score"],
            seconds=round(time.time() - t0, 1),
            gen_s=round(gen_s, 1),
            val_s=round(val_s, 1),
            tries=k,
            steps=steps,
            best_try=best["k"],
            stopped=stopped,
            lora_mult=best["mult"],
            id_sim=v["id_sim"],
            jaw_diff=v["jaw_diff"],
            score=v["score"],
            visor=v["visor"],
            visor_score=v["visor_score"],
            output_bytes=len(jpeg_bytes),
        )
        logger.info(
            "Best candidate selected: try %d of %d, id_sim=%s, visor=%s, score=%s in %.1fs, "
            "output=%d bytes (JPEG, cap=%d)",
            best["k"],
            k,
            v["id_sim"],
            v["visor"],
            v["score"],
            rec["seconds"],
            len(jpeg_bytes),
            settings.OUTPUT_MAX_BYTES,
        )
        return jpeg_bytes, rec
    finally:
        if progress is not None:
            progress.close()
        slot.reset_lora()
