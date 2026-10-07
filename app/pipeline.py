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

import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps

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

def set_lora(weight: float) -> None:
    _LORA["scale"] = float(weight)
    if _LORA["mode"] == "adapters" and pipe is not None:
        pipe.set_adapters(["bfs"], adapter_weights=[float(weight)])

def lora_kwargs():
    return {"scale": _LORA["scale"]} if _LORA["mode"] == "kwargs" else None

# GPU serialization lock for async request handling
gpu_lock = asyncio.Lock()

# Persistent ArcFace embedding worker state
_VW = {"p": None}
_VLOCK = threading.RLock()

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
    face = base.crop((140, 20, 400, 300)).resize((512, 512), Image.LANCZOS)

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
    torch.backends.cudnn.enabled = False

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

    # Start persistent ArcFace identity verification worker (only needed when VALIDATE is on -
    # off by default since ArcFace is unreliable on illustrated avatars)
    if settings.VALIDATE:
        get_val_worker()

def shutdown_models() -> None:
    logger.info("Shutting down model resources...")
    close_val_worker()
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
    hijab, hijab_s = classify_hijab(crop_pil) if gender == "Woman" else (False, None)

    ov = OVERRIDES.get(key, {})
    if "glasses" in ov:
        glasses = bool(ov["glasses"])
    if "hijab" in ov and gender == "Woman":
        hijab = bool(ov["hijab"])
    if "beard" in ov and gender == "Man":
        beard = bool(ov["beard"])

    tags = [gender] + (["Glasses"] if glasses else []) + (["Hijab"] if hijab else [])
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

_SEMI_REAL_STYLE_TEXT = (
    "render the swapped head in the semi-realistic digital painting style of Picture 1: a realistic, true-to-life "
    "face with natural proportions and a recognisable likeness, painted with smooth airbrushed skin and soft "
    "realistic shading (not flat cartoon colors, not a photo and not a vector icon). subtle thin dark line work only "
    "around the eyes, lips, hair strands and jacket edges. detailed realistic eyes with crisp catchlights, soft "
    "glossy lips, natural eyebrows, fine individually painted hair strands with a glossy sheen, warm natural skin "
    "tones with gentle highlights, a strong red rim light along the hair, forehead and cheek edge, and cool blue "
    "fill light on the shadow side, matching the color palette and lighting of the jacket and emblem. the head "
    "blends seamlessly into the painting with no visible swap seam, no harsh photographic texture, no visible pores, "
    "and keeps exactly the face of Picture 2: same face shape, jaw, hairline, facial hair and the real eye color of "
    "Picture 2 (do not turn the eyes blue). sharp detail, high quality, 4k."
)

_COMIC_STYLE_TEXT = (
    "render the swapped head fully in the illustration style of Picture 1, not as a photo: a premium digital "
    "comic-book vector portrait with bold clean ink outlines of consistent line weight, cel-shaded skin with smooth "
    "airbrushed gradients and crisp highlight and shadow shapes, rich saturated warm skin tones, individually inked "
    "hair strands with a glossy sheen, glossy lips and sharp eye catchlights, a strong red rim light along the hair, "
    "forehead, cheek and jaw edge and cool blue fill shadows on the opposite side, matching the line quality, color "
    "palette and lighting of the jacket and emblem. the head blends seamlessly into the illustration with no "
    "photographic texture, no visible pores and no swap seam, while still looking exactly like the person in "
    "Picture 2: same face shape, hairline, facial hair and the real eye color of Picture 2 (do not turn the eyes "
    "blue). sharp detail, high quality, 4k."
)

AVATAR_STYLE_TEXT = {"semi_real": _SEMI_REAL_STYLE_TEXT, "comic": _COMIC_STYLE_TEXT}

def _visor_text(info: dict) -> str:
    """The avatar's own visor glasses are ALWAYS present on the output (no Picture 3 reference image -
    the visor is described in text only)."""
    return (
        "the face wears the slim wraparound visor glasses of Picture 1: exactly the same shape, thin frame and "
        "the same electric blue color as Picture 1, sitting across the eyes with perfectly clear see-through "
        "lenses so the eyes, eyelashes and eyebrows of Picture 2 stay sharp and fully visible. the lenses carry crisp "
        "white and blue glass reflections near the temples and a thin clean ink outline, drawn in the same "
        "illustration style as the rest of the avatar. "
        "the Picture 1 visor is the ONLY eyewear on the face; do NOT copy any other glasses from Picture 2. "
    )

def _bfs_prompt(keep_user_expression: Optional[bool] = None) -> str:
    if keep_user_expression is None:
        keep_user_expression = settings.KEEP_USER_EXPRESSION
    expr = (
        "copy the head rotation and eye direction from Picture 1, but keep the facial expression and smile of Picture 2"
        if keep_user_expression
        else "copy the direction of the eye, head rotation, micro expressions from Picture 1"
    )
    style_text = AVATAR_STYLE_TEXT.get(settings.STYLE_MODE, "high quality, sharp details, 4k.")
    return (
        "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. "
        "remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly "
        f"preserving the hair, eye color, nose structure of Picture 2. {expr}. {style_text}"
    )

def build_hijab_prompt(info: dict) -> str:
    return (
        f"{_bfs_prompt()} "
        "exact face swap: keep the identity of Picture 2 - face shape, cheeks, chin, eyes, eyebrows, eyelashes, nose, "
        "lips, lipstick color, skin tone and her natural smile with the same teeth. "
        "her skin is clean, smooth and natural like Picture 2. "
        "she wears her own hijab from Picture 2: the same fabric color, pattern and folds, with the inner cap if "
        "visible, wrapped around her face and neck and flowing into the jacket collar of Picture 1. "
        "the hijab fully covers her hair, ears and neck. "
        "only her head and hijab are taken from Picture 2; ignore her clothes, shirt, body and background. "
        f"{_visor_text(info)}"
        "keep the jacket, emblem and solid black background of Picture 1 unchanged."
    )

def build_prompt(info: dict) -> str:
    if info.get("hijab", False):
        return build_hijab_prompt(info)

    if info.get("gender") == "Woman":
        return (
            f"{_bfs_prompt()} "
            "exact face swap: keep the identity of Picture 2 - the exact jawline, jaw width and chin shape, "
            "full cheeks, nose, eye shape, the exact eye color, eyebrows, lips, skin tone, skin texture and age. "
            "her hair is styled exactly like Picture 1: long black hair with red and orange highlights woven throughout, "
            "styled in a high voluminous bun or updo at the crown, sleek and professionally polished, framing the face. "
            f"{_visor_text(info)}"
            "keep the jacket, emblem and solid black background of Picture 1 unchanged. "
            "keep the same makeup and lipstick color as Picture 1."
        )

    if info.get("beard", False):
        beard = (
            "keep the exact beard style of Picture 2: its moustache, beard and goatee with the same shape, "
            "coverage, length, density and natural color, trimmed the same way along the cheeks, jaw and chin. "
            "keep grey hairs only where Picture 2 has them, do not add white hairs, do not shave, thin or "
            "shorten it. "
        )
    else:
        beard = (
            "Picture 2 shows a CLEAN-SHAVEN man with NO facial hair whatsoever. "
            "the man has a smooth, bare face: absolutely NO beard, NO moustache, NO goatee, NO stubble, NO sideburns. "
            "the cheeks, chin, jaw, neck and upper lip are smooth and hairless. "
            "do NOT copy any beard from Picture 1; completely remove it from the face if present. "
            "keep only the natural hair on the head. "
        )

    return (
        f"{_bfs_prompt()} "
        "exact face swap: keep the identity of Picture 2 - the exact jawline, jaw width and chin shape of Picture 2 "
        "(do not slim, narrow or lengthen the face), full cheeks, nose, eye shape, the exact eye color of Picture 2, "
        "eyebrows, lips, skin tone, skin texture and age. "
        "keep the exact hairstyle of Picture 2: the same hair color, length, volume, hairline and parting. "
        "keep the smile of Picture 2 with the same open mouth and teeth. "
        f"{beard}"
        f"{_visor_text(info)}"
        "keep the jacket, emblem and solid black background of Picture 1 unchanged."
    )

# ============================================================================
# Step 2 - avatar generation from the cropped face (single pass, no retries)
# ============================================================================

AVATAR_PATHS = {
    "Man": "Saytara_male.jpg",
    "Woman": "Saytara_Femal.png",
    "Woman_Hijab": "Saytara_hijab.jpg",
}

AVATAR_HEAD_BOX = {
    "Man": (0.25, 0.05, 0.68, 0.62),
    "Woman": (0.25, 0.05, 0.68, 0.62),
    "Woman_Hijab": (0.30, 0.08, 0.72, 0.72),
}

_AV: Dict[str, Tuple[Image.Image, str]] = {}

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
        filename = AVATAR_PATHS.get(key)
        if not filename:
            raise KeyError(f"Unknown avatar key '{key}'")
        path = settings.AVATAR_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Avatar template '{key}' not found: {path}")
        logger.info("Loaded avatar template for %s: %s", key, path.name)
        _AV[key] = (avatar_on_black(path), path.name)
    return _AV[key]

def head_crop(info: dict, scale: float, size: int = None) -> Image.Image:
    """Head crop straight from the user's original photo, using Step 1's face box - no
    eye-leveling rotation, no blur-padding, just a square crop resized to fit."""
    size = size or settings.REF_SIZE
    img = ImageOps.exif_transpose(Image.open(info["user_path"])).convert("RGB")
    x, y, w, h = info["face_box"]
    cx, cy, s = x + w / 2, y + h / 2 - h * 0.05, max(w, h) * scale
    box = (
        int(max(0, cx - s)),
        int(max(0, cy - s)),
        int(min(img.width, cx + s)),
        int(min(img.height, cy + s)),
    )
    return img.crop(box).resize((size, size), Image.LANCZOS)

def _crop_frac(img: Image.Image, box: Tuple[float, float, float, float]) -> Image.Image:
    w, h = img.size
    x0, y0, x1, y1 = box
    return img.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))

def check_visor(out_img: Image.Image) -> dict:
    a = np.asarray(out_img.convert("RGB"))
    h, w = a.shape[:2]
    band = a[int(h * 0.20):int(h * 0.50), int(w * 0.25):int(w * 0.70)]
    hsv = cv2.cvtColor(band, cv2.COLOR_RGB2HSV)
    m = (hsv[..., 0] >= 85) & (hsv[..., 0] <= 130) & (hsv[..., 1] > 80) & (hsv[..., 2] > 90)
    score = round(float(m.mean()), 4)
    return {"glasses_check": "ok" if score > 0.02 else "weak", "visor_score": score}

_CLIP_RAW: Dict[str, Any] = {}

def _load_clip_raw():
    if "m" not in _CLIP_RAW:
        from transformers import CLIPModel, CLIPProcessor
        _CLIP_RAW["m"] = CLIPModel.from_pretrained(settings.CLIP_MODEL).eval()
        _CLIP_RAW["p"] = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
    return _CLIP_RAW["m"], _CLIP_RAW["p"]

_BEARD_LABELS = [
    "a close-up photo of a man with a full thick beard",
    "a close-up photo of a man with a goatee and a moustache",
    "a close-up photo of a man with a short dense stubble beard on his cheeks and chin",
    "a close-up photo of a clean-shaven man with smooth bare skin and no facial hair",
]

def _hair_prob(head_img: Image.Image) -> Tuple[Optional[float], Optional[List[float]]]:
    try:
        w, h = head_img.size
        lower = head_img.convert("RGB").crop((int(w * 0.10), int(h * 0.40), int(w * 0.90), h))
        m, p = _load_clip_raw()
        inputs = p(text=_BEARD_LABELS, images=lower, return_tensors="pt", padding=True)
        with torch.no_grad():
            probs = m(**inputs).logits_per_image.softmax(-1)[0].tolist()
        return round(1.0 - probs[3], 3), probs
    except Exception as e:
        logger.warning("Beard CLIP check failed: %s", e)
        return None, None

def _jaw_score(head_img: Image.Image) -> float:
    g = np.asarray(head_img.convert("L")).astype(float)
    h, w = g.shape
    cheek = np.concatenate(
        [
            g[int(h * 0.48):int(h * 0.60), int(w * 0.18):int(w * 0.34)].ravel(),
            g[int(h * 0.48):int(h * 0.60), int(w * 0.66):int(w * 0.82)].ravel(),
        ]
    )
    jaw = g[int(h * 0.68):int(h * 0.92), int(w * 0.28):int(w * 0.72)]
    return round(float((jaw < np.median(cheek) * 0.55).mean()), 3)

def detect_beard_step2(head_img: Image.Image) -> Tuple[bool, float]:
    """Re-detects the beard on the head_crop used for generation (more accurate than
    Step 1's detect_beard, which runs on the raw Step-1 crop for template tagging only)."""
    jaw = _jaw_score(head_img)
    p_hair, probs = _hair_prob(head_img)
    if p_hair is None:
        return jaw > settings.JAW_BEARD, jaw
    clean_shaven_score = probs[3]
    beard_score = max(probs[0], probs[1], probs[2])
    has_beard = (beard_score > clean_shaven_score) or (jaw > settings.JAW_BEARD and beard_score > 0.4)
    return has_beard, max(beard_score, p_hair)

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

def verify_faces(
    user_img: Image.Image, out_img: Image.Image, box: Optional[Tuple[float, float, float, float]] = None
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    user_emb = embed_user(user_img)
    if user_emb is None:
        return None, None, None
    out_emb = embed(out_img, True, "out")
    if out_emb is None and box is not None:
        out_emb = embed(_crop_frac(out_img, box), False, "out")
    if out_emb is None:
        return None, None, None
    return compare(user_emb, out_emb)

def validate_output(user_img: Image.Image, out: Image.Image, info: dict) -> dict:
    """Validates the single generated image (identity, jaw, beard, visor). Only runs when
    settings.VALIDATE is True - off by default, since ArcFace gives unreliable scores on
    illustrated avatars. Used for logging/response metadata, not for retrying (single pass)."""
    box = AVATAR_HEAD_BOX.get(info["avatar"])
    id_sim, u_ratio, o_ratio = verify_faces(user_img, out, box)
    jaw_diff = (abs(o_ratio - u_ratio) / u_ratio) if (u_ratio and o_ratio) else None

    beard_ok = None
    if info["gender"] == "Man" and box:
        p_out, _ = _hair_prob(_crop_frac(out, box))
        if p_out is not None:
            beard_ok = (p_out > settings.BEARD_P) == bool(info.get("beard", False))

    vis = check_visor(out)
    visor_ok = vis["glasses_check"] == "ok"

    passed = (
        (id_sim is None or id_sim >= settings.ID_PASS)
        and (jaw_diff is None or jaw_diff <= settings.JAW_TOL)
        and (beard_ok is None or beard_ok)
        and visor_ok
    )
    score = (
        (id_sim or 0.0)
        - 0.5 * (jaw_diff or 0.0)
        - (0.3 if beard_ok is False else 0.0)
        - (0.4 if not visor_ok else 0.0)
    )
    return {
        "id_sim": None if id_sim is None else round(id_sim, 3),
        "jaw_diff": None if jaw_diff is None else round(jaw_diff, 3),
        "beard_ok": beard_ok,
        "passed": passed,
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

def generate_avatar(user_image_path: Path, info: dict, t0: Optional[float] = None) -> Tuple[bytes, dict]:
    # t0 defaults to "now" but callers should pass the timestamp from before Step 1
    # analysis so "seconds" reports the whole image end-to-end, matching the notebook.
    if t0 is None:
        t0 = time.time()
    user_path = Path(user_image_path)
    rec: Dict[str, Any] = {"file": user_path.name}

    try:
        avatar, _ = load_avatar(info["avatar"])
        scale = settings.HIJAB_SCALE if info.get("hijab") else settings.FALLBACK_SCALE
        face = head_crop(info, scale)
        seed = settings.SEED
        w, h = avatar.size

        has_beard, beard_score = (
            detect_beard_step2(face) if info["gender"] == "Man" else (False, 0.0)
        )
        info["beard"] = has_beard
        logger.info("Step 2 beard re-check: %s (score %.3f)", has_beard, beard_score or 0.0)

        prompt = build_prompt(info)

        if settings.DEBUG_DUMP:
            dbg = settings.OUTPUT_DIR / "debug" / f"{info['key']}_{info['avatar']}"
            dbg.mkdir(parents=True, exist_ok=True)
            avatar.save(dbg / "0_avatar.png")
            face.save(dbg / "1_face_ref.png")
            (dbg / "info.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
            (dbg / "prompt.txt").write_text(prompt, encoding="utf-8")
            logger.info("DEBUG_DUMP -> %s (adapters=%s, lora_layers=%d)", dbg, _adapters(), lora_layers)

        set_lora(settings.LORA_STRENGTH)
        tg = time.time()
        with torch.no_grad():
            out = pipe(
                prompt=prompt,
                image=[avatar, face],
                attention_kwargs=lora_kwargs(),
                width=w,
                height=h,
                num_inference_steps=settings.STEPS,
                guidance_scale=settings.CFG,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
        torch.cuda.empty_cache()
        gen_s = time.time() - tg

        tv = time.time()
        if settings.VALIDATE and info["avatar"] in settings.VALIDATE_AVATARS:
            v = validate_output(face, out, info)
        else:
            vis = check_visor(out)
            visor_ok = vis["glasses_check"] == "ok"
            v = {
                "id_sim": None,
                "jaw_diff": None,
                "beard_ok": None,
                "passed": visor_ok,
                "score": vis["visor_score"],
                **vis,
            }
        val_s = time.time() - tv

        logger.info(
            "id_sim=%s, jaw_diff=%s, beard_ok=%s, visor=%s, score=%s (gen=%.1fs, validate=%.1fs)",
            v["id_sim"],
            v["jaw_diff"],
            v["beard_ok"],
            v["glasses_check"],
            v["score"],
            gen_s,
            val_s,
        )

        jpeg_bytes = encode_jpeg_capped(out, settings.OUTPUT_MAX_BYTES)
        rec.update(
            status="done",
            gender=info["gender"],
            hijab=info.get("hijab", False),
            avatar=info["avatar"],
            glasses=info.get("glasses", False),
            beard=has_beard,
            beard_score=round(float(beard_score or 0.0), 3),
            seconds=round(time.time() - t0, 1),
            gen_s=round(gen_s, 1),
            val_s=round(val_s, 1),
            validated=v["passed"],
            id_sim=v["id_sim"],
            jaw_diff=v["jaw_diff"],
            beard_ok=v["beard_ok"],
            visor=v["glasses_check"],
            visor_score=v["visor_score"],
            score=v["score"],
            output_bytes=len(jpeg_bytes),
        )
        logger.info(
            "Done: id_sim=%s, visor=%s, passed=%s in %.1fs, output=%d bytes (JPEG, cap=%d)",
            v["id_sim"],
            v["glasses_check"],
            v["passed"],
            rec["seconds"],
            len(jpeg_bytes),
            settings.OUTPUT_MAX_BYTES,
        )
        return jpeg_bytes, rec
    finally:
        set_lora(settings.LORA_STRENGTH)
