import os
import sys
import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple, List

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from app.config import settings
from app.pipeline import models

logger = logging.getLogger("saytara.step1")

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

def load_image(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise Rejected("Image cannot be read (not an image or corrupted)")
    return img_bgr

def clip_scores(pil_img: Image.Image, labels: List[str]) -> Dict[str, float]:
    if models.clip is None:
        raise RuntimeError("CLIP model is not loaded.")
    res = models.clip(pil_img, candidate_labels=list(labels))
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
            str(settings.DEEPFACE_WORKER_PATH),
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
