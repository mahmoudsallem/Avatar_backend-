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

_LORA = {"scale": 1.0}

def set_lora(weight: float) -> None:
    _LORA["scale"] = float(weight)

def lora_kwargs() -> dict:
    return {"scale": _LORA["scale"]}

# GPU serialization lock for async request handling
gpu_lock = asyncio.Lock()

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

    expr = (
        "copy the head rotation and eye direction from Picture 1, but keep the facial expression and smile of Picture 2"
        if keep_user_expression
        else "copy the direction of the eye, head rotation, micro expressions from Picture 1"
    )
    bfs = (
        "head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. "
        "remove the head from Picture 1 completely and replace it with the head from Picture 2, strictly "
        "preserving the face, hair, eye color and nose structure of Picture 2. "
        + expr
        + ". "
    )
    style_text = (
        (
            "the painted finish applies to the rendering only and must never change the face proportions. "
            + {"semi_real": _SEMI_REAL, "comic": _COMIC}[style_mode]
        )
        if style_mode in ("semi_real", "comic")
        else "high quality, sharp details, 4k."
    )
    V = visor_block(info, attempt, ref)
    closing = (
        "keep the jacket, emblem and solid black background of Picture 1 unchanged. "
        "final check: the face and build match Picture 2, the jawline is clean, and the angular blue clear "
        "visor of the avatar is on the face with no other glasses."
    )
    if info.get("hijab", False):
        return (
            f"{VISOR_MANDATORY}{bfs}{IDENTITY}{BUILD}{V}{HIJAB_VISOR_NOTE}"
            f"{HIJAB_FACE}{HIJAB_USER}{FACE_CLEAN}{style_text} {closing} the visor is present and fits her face."
        )
    if info.get("gender") == "Woman":
        return (
            f"{VISOR_MANDATORY}{bfs}{IDENTITY}{FEMALE_FACE}{BUILD}"
            "her hair is styled exactly like Picture 1: long black hair with red and orange highlights woven "
            "throughout, styled in a high voluminous bun or updo at the crown, sleek and professionally polished, "
            f"framing the face. {V}{FACE_CLEAN}{style_text} " + closing
        )
    return (
        f"{VISOR_MANDATORY}{bfs}{IDENTITY}{BUILD}{HAIR_USER}"
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

def generate_avatar(user_image_path: Path, info: dict) -> Tuple[bytes, dict]:
    t0 = time.time()
    user_path = Path(user_image_path)
    rec: Dict[str, Any] = {"file": user_path.name}

    test_lora = settings.LORA_STRENGTH * settings.ID_LORA_MULT
    progress = None
    try:
        avatar, _ = load_avatar(info["avatar"])
        face = load_face(info)
        visor_ref = load_visor_ref(info["avatar"])
        images = [avatar, face] + ([visor_ref] if visor_ref is not None else [])
        seed = settings.SEED
        w, h = avatar.size
        max_k = settings.BEST_OF_N if settings.VALIDATE else 1
        extras = 0

        # CPU embedding runs in thread while GPU generates try #1
        fut = _POOL.submit(embed_user, face) if settings.VALIDATE else None

        best, k, gen_s, val_s, stopped = None, 0, 0.0, 0.0, "all tries"
        t_loop = time.time()

        progress = tqdm(total=max_k, desc=f"Generating avatar ({user_path.name})", unit="try")
        while True:
            mult = settings.ID_LORA_SCHEDULE[k % len(settings.ID_LORA_SCHEDULE)]
            set_lora(test_lora * mult)
            prompt = build_prompt(info, attempt=k, ref=visor_ref is not None)

            tg = time.time()
            with torch.no_grad():
                out = pipe(
                    prompt=prompt,
                    image=images,
                    attention_kwargs=lora_kwargs(),
                    width=w,
                    height=h,
                    num_inference_steps=settings.STEPS,
                    guidance_scale=settings.CFG,
                    generator=torch.Generator("cuda").manual_seed(seed + k),
                ).images[0]
            g = time.time() - tg

            tv = time.time()
            user_emb = fut.result() if fut is not None else None
            v = score_output(user_emb, out, info)
            s = time.time() - tv

            gen_s, val_s = gen_s + g, val_s + s
            logger.info(
                "Try %d (seed=%d, lora_mult=%.1f): id_sim=%s, jaw_diff=%s, visor=%s, score=%s (gen=%.1fs, score=%.1fs)",
                k + 1,
                seed + k,
                mult,
                v["id_sim"],
                v["jaw_diff"],
                v["visor"],
                v["score"],
                g,
                s,
            )

            if best is None or (v["visor"] == "ok", v["score"]) > (
                best["v"]["visor"] == "ok",
                best["v"]["score"],
            ):
                best = {"out": out, "v": v, "k": k + 1, "mult": mult}

            k += 1
            progress.set_postfix(id_sim=v["id_sim"], visor=v["visor"], score=v["score"])
            progress.update(1)

            if settings.VALIDATE and (v["id_sim"] or 0) >= settings.STOP_ID and v["visor"] == "ok":
                stopped = "good"
                break

            if k >= max_k:
                if best["v"]["visor"] != "ok" and extras < settings.VISOR_EXTRA_TRIES:
                    extras += 1
                    max_k += 1
                    progress.total = max_k
                    progress.refresh()
                    logger.info("Visor not ok - adding extra try (if budget permits)")
                else:
                    break

            per_try = (time.time() - t_loop) / k
            if time.time() - t0 + per_try > settings.TIME_BUDGET:
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
            seconds=round(time.time() - t0, 1),
            gen_s=round(gen_s, 1),
            val_s=round(val_s, 1),
            tries=k,
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
        set_lora(settings.LORA_STRENGTH)
