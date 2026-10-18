"""Recolour the HIJAB template's black hijab + neck drape to the user's own hijab colour BEFORE generation.

FLUX starts from the template (Picture 1). Its hijab and the cowl under the chin are black, and that black kept
winning: the neck drape stayed black and the head covering drifted to a saturated magenta instead of the user's
light lilac / pink. Giving the model a template that already wears the right colour fixes it at the source (the
old pixel post-fix painted flat patches afterwards). The hijab area is traced once by hand on the fixed template;
only dark, unsaturated fabric pixels inside it change, the red rim light and the face are kept, and the fabric's
own light/dark folds are preserved. Pure numpy + OpenCV, fail-soft.
"""
from typing import Optional, Sequence

import cv2
import numpy as np
from PIL import Image

# Hijab + neck-drape outline on Saytara_hijab.jpg, normalised (x, y in 0..1 of the template), traced by hand.
_POLY_1000 = [
    (510, 118), (580, 130), (640, 170), (680, 230), (705, 300), (710, 380), (705, 450), (695, 510), (680, 535),
    (530, 570), (520, 620), (510, 700), (500, 800), (420, 800), (405, 700), (395, 630), (390, 545), (375, 500),
    (330, 455), (316, 380), (312, 330), (322, 270), (350, 205), (420, 140),
]
# The face (skin, brows, eyes, lips) - never recoloured. Ellipse centre / radii, same normalised units.
_FACE_1000 = ((440, 395), (115, 150))


HIJAB_CHROMA_BOOST = 1.5


def boosted_chroma_factor(lab) -> float:
    """Kept for compatibility. NO boost any more: measured against the photos, the boost made a pale blush scarf about
    twice as pink as the user's (a 18 vs 3-10), so the target is exactly the colour sampled from the photo."""
    return 1.0


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def recolor_hijab_template(img: Image.Image, hijab_lab: Optional[Sequence[float]]) -> Image.Image:
    """img = the hijab template (any size). hijab_lab = the user's hijab colour (OpenCV float Lab: L 0..100)."""
    if hijab_lab is None:
        return img
    try:
        rgb = np.asarray(img.convert("RGB"))
        H, W = rgb.shape[:2]
        pts = np.array([[x / 1000 * W, y / 1000 * H] for x, y in _POLY_1000], np.int32)
        inside = np.zeros((H, W), np.uint8)
        cv2.fillPoly(inside, [pts], 1)
        (cx, cy), (rx, ry) = _FACE_1000
        face = np.zeros((H, W), np.uint8)
        cv2.ellipse(face, (int(cx / 1000 * W), int(cy / 1000 * H)), (int(rx / 1000 * W), int(ry / 1000 * H)),
                    0, 0, 360, 1, -1)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        hh, s, v = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
        red_rim = ((hh < 10) | (hh > 165)) & (s > 120) & (v > 60)        # the red rim light on the cloth stays
        # black cloth: dark and not the red rim. Very dark pixels have noisy saturation, so accept any s below v<50.
        fabric = (inside > 0) & (face == 0) & ~red_rim & (((v < 95) & (s < 140)) | (v < 50))
        fabric = cv2.morphologyEx(fabric.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        fabric = cv2.morphologyEx(fabric, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        fabric &= (inside > 0).astype(np.uint8) & (face == 0).astype(np.uint8)
        lab = _lab(rgb)
        L0 = lab[..., 0]
        lo, hi = np.percentile(L0[fabric > 0], [5, 97]) if fabric.sum() > 1000 else (0.0, 40.0)
        t = np.clip((L0 - lo) / max(hi - lo, 1.0), 0, 1)                   # fold shading 0 (deep) .. 1 (lit)
        tL, ta, tb = [float(c) for c in hijab_lab]
        # FLUX washes pastel cloth out towards grey/beige, so push the target chroma up a little
        _k = boosted_chroma_factor(hijab_lab)
        ta, tb = ta * _k, tb * _k
        new = lab.copy()
        new[..., 0] = np.clip(tL * (0.85 + 0.30 * t), 0, 100)             # keep folds, centred on the user's L
        new[..., 1] = ta * (0.95 + 0.10 * t)
        new[..., 2] = tb * (0.95 + 0.10 * t)
        m = cv2.GaussianBlur(fabric.astype(np.float32), (7, 7), 0)[..., None]
        out = lab * (1 - m) + new * m
        out = np.clip(cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    except Exception:
        return img


def quantize_lab(lab: Sequence[float], step: float = 4.0):
    """Round the colour so repeated colours give byte-identical templates (cache hits for the VAE latents)."""
    return tuple(float(round(c / step) * step) for c in lab)


def sample_user_hijab_lab(user: Image.Image, box) -> Optional[np.ndarray]:
    """User's hijab colour (OpenCV float Lab). box = (x, y, w, h) of the face in `user`.

    Samples the cloth over the top of the head (above the hairline, always hijab for a hijab photo) plus the cloth
    beside both cheeks, and returns the centre of the main colour cluster (k-means, k=3),
    so a strip of background or a little hair at the hairline cannot pull the colour. None when unsure.
    """
    try:
        rgb = np.asarray(user.convert("RGB"))
        Hh, Ww = rgb.shape[:2]
        x, y, w, h = [int(v) for v in box]
        sel = np.zeros((Hh, Ww), bool)

        def put(x0, x1, y0, y1):
            x0, x1 = max(0, int(x0)), min(Ww, int(x1))
            y0, y1 = max(0, int(y0)), min(Hh, int(y1))
            if x1 > x0 and y1 > y0:
                sel[y0:y1, x0:x1] = True

        # a thin ring hugging the face: the hijab wraps tightly, so cloth starts right at the face edge, while a
        # wider band would already reach the room behind her
        put(x + 0.20 * w, x + 0.80 * w, y - 0.16 * h, y - 0.01 * h)       # above the forehead
        put(x - 0.10 * w, x - 0.01 * w, y + 0.10 * h, y + 0.75 * h)       # left of the cheek
        put(x + 1.01 * w, x + 1.10 * w, y + 0.10 * h, y + 0.75 * h)       # right of the cheek
        # no skin filter: a light pink / beige hijab has skin-like chroma and would be thrown away (the strips are
        # already outside the face box)
        if sel.sum() < 400:
            return None
        lab = _lab(rgb)[sel].reshape(-1, 3).astype(np.float32)
        if len(lab) > 40000:
            lab = lab[np.random.default_rng(0).choice(len(lab), 40000, replace=False)]
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _c, labels, centers = cv2.kmeans(lab, 3, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=3)
        order = list(np.argsort(-counts))
        k = int(order[0])
        # a dark dominant cluster is often hair at the hairline / shadow: prefer a lighter cluster that still covers
        # a good share; a truly black hijab keeps the dark cluster because nothing else is big enough
        if centers[k][0] < 25:
            for j in order[1:]:
                if centers[j][0] >= 25 and counts[j] >= 0.30 * len(lab):
                    k = int(j)
                    break
        if counts[k] < 0.25 * len(lab):        # no dominant colour -> do not guess
            return None
        c = centers[k].copy()
        # the strips hug the face and are partly shaded: take lightness from the lit part of the same-coloured pixels
        same = np.hypot(lab[:, 1] - c[1], lab[:, 2] - c[2]) < 12.0
        if int(same.sum()) >= 200:
            c[0] = float(np.percentile(lab[same, 0], 70))
            c[1] = float(np.median(lab[same, 1]))
            c[2] = float(np.median(lab[same, 2]))
        # warm indoor light makes pale pink cloth read tan: when the cloth is a soft, rosy colour (a > 0) whose yellow
        # (b) is stronger than its red (a), pull b back towards a so it stays pink instead of turning beige
        if c[1] > 6.0 and c[2] > 0.6 * c[1] and np.hypot(c[1], c[2]) < 35.0:
            c[2] = 0.6 * c[1]
        return c
    except Exception:
        return None
