import os
import time
import json
import logging
import threading
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

from app.config import settings
from app.pipeline import models
from app.pipeline.prompts import build_prompt

logger = logging.getLogger("saytara.generate")

AVATAR_PATHS = {
    "Man": "Saytara_male.jpg",
    "Woman": "Saytara_Femal.png",
    "Woman_Hijab": "Saytara_hijab.jpg",
}

TEST_AVATARS = {
    "Man": "Saytara_male_v2.png",
    "Woman": "Saytara_Femal_clean.png",
    "Woman_Hijab": "Saytara_hijab.jpg",
}

VISOR_REFS = {
    "Man":         {"file": "Saytara_male_v2.png",         "box": (0.28, 0.22, 0.65, 0.40), "image": None},
    "Woman":       {"file": "Saytara_Femal_clean.png",       "box": (0.25, 0.29, 0.63, 0.49), "image": None},
    "Woman_Hijab": {"file": "Saytara_hijab.jpg", "box": (0.26, 0.25, 0.64, 0.45), "image": None},
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
        with models._VLOCK:
            try:
                p = models.get_val_worker()
                p.stdin.write(json.dumps({"img": str(temp_path), "enforce": bool(enforce)}) + "\n")
                p.stdin.flush()
                while True:
                    line = p.stdout.readline()
                    if not line:
                        models.close_val_worker()
                        return None
                    if line.startswith("__JSON__"):
                        r = json.loads(line[len("__JSON__"):])
                        break
            except BaseException:
                models.close_val_worker()
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

def generate_avatar(user_image_path: Path, info: dict) -> Tuple[Image.Image, dict]:
    t0 = time.time()
    user_path = Path(user_image_path)
    rec: Dict[str, Any] = {"file": user_path.name}

    test_lora = settings.LORA_STRENGTH * settings.ID_LORA_MULT
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
        fut = (
            models._POOL.submit(embed_user, face)
            if settings.VALIDATE
            else None
        )

        best, k, gen_s, val_s, stopped = None, 0, 0.0, 0.0, "all tries"
        t_loop = time.time()

        while True:
            mult = settings.ID_LORA_SCHEDULE[k % len(settings.ID_LORA_SCHEDULE)]
            models.set_lora(test_lora * mult)
            prompt = build_prompt(info, attempt=k, ref=visor_ref is not None)

            tg = time.time()
            with torch.no_grad():
                out = models.pipe(
                    prompt=prompt,
                    image=images,
                    attention_kwargs=models.lora_kwargs(),
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

            if settings.VALIDATE and (v["id_sim"] or 0) >= settings.STOP_ID and v["visor"] == "ok":
                stopped = "good"
                break

            if k >= max_k:
                if best["v"]["visor"] != "ok" and extras < settings.VISOR_EXTRA_TRIES:
                    extras += 1
                    max_k += 1
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
        )
        logger.info(
            "Best candidate selected: try %d of %d, id_sim=%s, visor=%s, score=%s in %.1fs",
            best["k"],
            k,
            v["id_sim"],
            v["visor"],
            v["score"],
            rec["seconds"],
        )
        return out, rec
    finally:
        models.set_lora(settings.LORA_STRENGTH)
