"""Deterministic clean-up for HIJAB avatars (runs after FLUX, on the chosen candidate).

FLUX copies the template's black hijab / black neck drape and its warm skin tone onto the user, and paints a rough,
glossy, aged-looking skin. Prompt wording alone did not fix that, so this module corrects it in pixel space:

  1. skin   - edge-preserving smoothing + highlight compression + tone match to the user's real skin (face ellipse only)
  2. hijab  - shift the hijab fabric colour to the colour of the user's own hijab (sampled above the forehead)
  3. drape  - the black cowl/drape under the chin that comes from the template is re-coloured as hijab fabric

Everything is fail-soft: any missing input / implausible measurement skips that step, and any exception returns the
untouched image. Pure numpy + OpenCV (already required by the backend).
"""
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

Box = Tuple[int, int, int, int]  # x, y, w, h in pixels


def _bgr(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def _pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _lab(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)  # L 0..100, a/b ~ -127..127


def _from_lab(lab: np.ndarray) -> np.ndarray:
    return np.clip(cv2.cvtColor(lab.astype(np.float32), cv2.COLOR_LAB2BGR) * 255.0, 0, 255).astype(np.uint8)


def _skin_like(bgr: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    cr, cb = ycc[..., 1], ycc[..., 2]
    ok = (cr > 133) & (cr < 178) & (cb > 78) & (cb < 135) & (hsv[..., 2] > 60) & (hsv[..., 1] > 22)
    ok &= bgr[..., 0] < bgr[..., 2]  # not bluish (visor, eyes through the lens)
    return ok


def _ellipse(shape, box: Box, sx: float, sy: float, cy: float = 0.52) -> np.ndarray:
    x, y, w, h = box
    m = np.zeros(shape[:2], np.float32)
    cv2.ellipse(m, (int(x + w / 2), int(y + h * cy)), (max(1, int(w * 0.5 * sx)), max(1, int(h * 0.5 * sy))), 0, 0, 360, 1.0, -1)
    return m


def _soft(mask: np.ndarray, px: float) -> np.ndarray:
    k = max(1, int(px) | 1)
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0), 0, 1)


def _fill_holes(mask: np.ndarray, max_area: float) -> np.ndarray:
    """Fill enclosed holes of a 0/1 float mask that are smaller than max_area pixels."""
    inv = (mask < 0.5).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(inv)
    out = mask.copy()
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < max_area:
            out[lab == k] = 1.0
    return out


def _median_lab(lab: np.ndarray, sel: np.ndarray, min_px: int) -> Optional[np.ndarray]:
    if int(sel.sum()) < min_px:
        return None
    return np.median(lab[sel], axis=0)


# --------------------------------------------------------------------------------------------- colour palette & helpers
_COLOUR_NAMES = [  # (name, sRGB) - nearest in Lab; used to tell the text prompt what colour the user's hijab is
    ("white", (240, 240, 238)), ("cream", (236, 224, 200)), ("beige", (214, 190, 160)), ("light grey", (190, 190, 192)),
    ("grey", (128, 128, 130)), ("charcoal", (60, 60, 64)), ("black", (20, 20, 22)), ("brown", (110, 75, 55)),
    ("light pink", (240, 200, 210)), ("dusty pink", (205, 150, 165)), ("hot pink", (215, 70, 130)), ("red", (190, 30, 40)),
    ("burgundy", (110, 25, 50)), ("orange", (225, 130, 50)), ("mustard", (205, 160, 50)), ("yellow", (240, 220, 90)),
    ("olive green", (110, 120, 60)), ("green", (50, 140, 80)), ("mint green", (170, 220, 195)), ("teal", (40, 130, 135)),
    ("light blue", (160, 200, 235)), ("blue", (50, 90, 190)), ("navy blue", (30, 40, 90)),
    ("lilac", (190, 160, 215)), ("purple", (110, 60, 150)), ("plum", (95, 40, 85)), ("mauve", (150, 105, 140)),
]


def hijab_colour_name(lab_med) -> Optional[str]:
    """Nearest plain colour word for a median Lab colour (None if no colour)."""
    if lab_med is None:
        return None
    try:
        pal = np.array([c for _, c in _COLOUR_NAMES], np.uint8).reshape(-1, 1, 3)
        pal_lab = cv2.cvtColor(pal.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB).reshape(-1, 3)
        diff = pal_lab - np.asarray(lab_med, np.float32)
        # Weight luminance (L) at 0.6 so fold shadows/highlights don't overpower chromatic hue/saturation
        dists = np.sqrt((diff[:, 0] * 0.6) ** 2 + np.sum(diff[:, 1:] ** 2, axis=1))
        i = int(np.argmin(dists))
        return _COLOUR_NAMES[i][0]
    except Exception:
        return None


class HijabLab(tuple):
    """2-tuple (median_lab, spread) that carries optional pattern metadata."""

    def __new__(cls, lab, spread, is_pattern=False, pattern_labs=None, pattern_names=None):
        return super().__new__(cls, (lab, spread))

    def __init__(self, lab, spread, is_pattern=False, pattern_labs=None, pattern_names=None):
        self.is_pattern = bool(is_pattern)
        self.pattern_labs = tuple(pattern_labs or ())
        self.pattern_names = tuple(pattern_names or ())


def _cluster_pattern(pixels: np.ndarray) -> Optional[dict]:
    """Detect whether a pixel population forms a multi-tone / patterned fabric.

    Distinguishes genuine multi-tone print/fabric from single-colour fold shading or noise:
      - 2 distinct, balanced clusters (secondary cluster >= 18% of pixels)
      - Significant chromatic or Euclidean separation between cluster centers
      - Tight within-cluster spread (both <= 36) to reject chaotic noise
      - Both clusters map to distinct named colours in _COLOUR_NAMES
    """
    if len(pixels) < 80:
        return None
    valid = (pixels[:, 0] >= 14.0) & (pixels[:, 0] <= 96.0)
    p = pixels[valid]
    if len(p) < 60:
        return None

    try:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
        ret, labels, centers = cv2.kmeans(p.astype(np.float32), 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
        labels = labels.flatten()
        n0, n1 = int((labels == 0).sum()), int((labels == 1).sum())
        c0, c1 = centers[0], centers[1]
        if n1 > n0:
            n0, n1 = n1, n0
            c0, c1 = c1, c0
            labels = 1 - labels

        w0 = n0 / len(p)
        w1 = n1 / len(p)

        # 1. Secondary cluster must be substantial (not stray outlier pixels)
        if w1 < 0.18:
            return None

        # 2. Distances
        dE = float(np.linalg.norm(c0 - c1))
        dC = float(np.hypot(c0[1] - c1[1], c0[2] - c1[2]))
        dL = abs(c0[0] - c1[0])

        # Significant chromatic separation or high-contrast luminance two-tone
        if dC < 9.0 and not (dL >= 18.0 and dE >= 22.0):
            return None

        # 3. Within-cluster compactness (rejection of random clutter/noise)
        p0 = p[labels == 0]
        p1 = p[labels == 1]
        sp0 = float(np.median(np.abs(p0 - c0).sum(axis=1)))
        sp1 = float(np.median(np.abs(p1 - c1).sum(axis=1)))
        if sp0 > 34.0 or sp1 > 36.0:
            return None

        # 4. Must map to distinct recognized colours
        name0 = hijab_colour_name(c0)
        name1 = hijab_colour_name(c1)
        if not name0 or name0 == name1:
            return None

        return {
            "c0": c0, "sp0": sp0, "name0": name0, "w0": w0,
            "c1": c1, "sp1": sp1, "name1": name1, "w1": w1,
            "dE": dE, "dC": dC,
        }
    except Exception:
        return None


# --------------------------------------------------------------------------------------------- user measurements
def user_skin_lab(user: Image.Image, box: Box) -> Optional[np.ndarray]:
    bgr = _bgr(user)
    sel = (_ellipse(bgr.shape, box, 0.62, 0.62) > 0.5) & _skin_like(bgr)
    return _median_lab(_lab(bgr), sel, 400)


def user_hijab_lab(user: Image.Image, box: Box) -> HijabLab:
    """Median colour of the hijab fabric hugging the face, with pattern detection.

    Samples three anatomically targeted regions:
      1. Top (forehead / crown wrap): most reliable fabric anchor directly on the head (cannot bleed
         into the room background beside the head).
      2. Left cheek strip: narrowed (0.075w) and vertically constrained to cheeks (0.12h..0.65h) so it
         stays on the fabric wrap hugging the cheek and cuts off above the jaw taper.
      3. Right cheek strip: mirrored narrow cheek strip.

    Cross-checks regional consistency so that in 3/4 poses or off-center framing, a side band hitting
    the background wall or heavy shadow is excluded rather than contaminating the median. Trims outlier
    pixels and returns (median_lab, spread).

    When solid cross-checking fails or within-sample spread exceeds solid thresholds, tests for
    multi-tone/patterned fabric clustering (e.g. pink with grey-blue watercolor print) and returns the
    dominant cluster Lab along with pattern metadata. Degrades safely to (None, None) when the sample
    is ambiguous, too small, or too noisy.
    """
    try:
        bgr = _bgr(user)
        H, W = bgr.shape[:2]
        if max(H, W) > 512:
            scale = 512.0 / max(H, W)
            bgr = cv2.resize(bgr, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA)
            box = (box[0] * scale, box[1] * scale, box[2] * scale, box[3] * scale)
            H, W = bgr.shape[:2]
        x, y, w, h = box
        if w < 10 or h < 10 or x >= W or y >= H:
            return HijabLab(None, None)

        # Targeted sampling boxes:
        # Top: forehead / crown wrap sitting squarely on the head
        top_box = (x + 0.15 * w, x + 0.85 * w, y - 0.22 * h, y + 0.02 * h)
        # Left / right: narrow cheek strips hugging the face silhouette, cutting off before the jaw narrows
        left_box = (x - 0.08 * w, x - 0.005 * w, y + 0.12 * h, y + 0.65 * h)
        right_box = (x + 1.005 * w, x + 1.08 * w, y + 0.12 * h, y + 0.65 * h)

        face = _ellipse(bgr.shape, box, 1.0, 1.0) > 0.5
        skin = _skin_like(bgr)
        lab = _lab(bgr)

        def get_pixels(x0, x1, y0, y1, is_top=False):
            x0, x1 = max(0, int(x0)), min(W, int(x1))
            y0, y1 = max(0, int(y0)), min(H, int(y1))
            if x1 <= x0 or y1 <= y0:
                return np.empty((0, 3), np.float32)
            m = ~face[y0:y1, x0:x1]
            if is_top:
                # On the crown well above the forehead (y < y - 0.02*h), do not discard light pink / cream fabrics as skin
                hairline_row = max(0, int(y - 0.02 * h) - y0)
                m[hairline_row:] &= ~skin[y0:y1, x0:x1][hairline_row:]
            else:
                m &= ~skin[y0:y1, x0:x1]
            if not np.any(m):
                return np.empty((0, 3), np.float32)
            return lab[y0:y1, x0:x1][m]

        p_top = get_pixels(*top_box, is_top=True)
        p_left = get_pixels(*left_box)
        p_right = get_pixels(*right_box)

        def region_stats(p):
            if len(p) < 40:
                return None, None, None
            m = np.median(p, axis=0)
            sp = float(np.median(np.abs(p - m).sum(axis=1)))
            ch = float(np.hypot(m[1], m[2]))
            return m, sp, ch

        med_top, sp_top, ch_top = region_stats(p_top)
        med_left, sp_left, ch_left = region_stats(p_left)
        med_right, sp_right, ch_right = region_stats(p_right)

        def matches(m1, sp1, m2, sp2):
            if m1 is None or m2 is None or sp1 > 42.0 or sp2 > 42.0:
                return False
            dE = float(np.linalg.norm(m1 - m2))
            dC = float(np.hypot(m1[1] - m2[1], m1[2] - m2[2]))
            dL = abs(m1[0] - m2[0])
            # Two regions of the same fabric: close chromaticity and reasonable total distance
            if dC <= 14.0 and dE <= 26.0:
                return True
            # Shadow/light variation of the same fabric: larger dL permitted if chromaticity is very tight
            if dL <= 22.0 and dC <= 9.0:
                return True
            return False

        has_top = med_top is not None and sp_top <= 40.0
        has_left = med_left is not None and sp_left <= 42.0
        has_right = med_right is not None and sp_right <= 42.0

        chosen = []
        sides_match = has_left and has_right and matches(med_left, sp_left, med_right, sp_right)

        # If top is a neutral under-scarf/cap (ch < 7) while cheeks show the coloured outer hijab (ch >= 12)
        if sides_match and ch_left >= 12.0 and (not has_top or ch_top < 7.0):
            chosen = [p_left, p_right]
        elif has_top:
            chosen = [p_top]
            if has_left and matches(med_top, sp_top, med_left, sp_left):
                chosen.append(p_left)
            if has_right and matches(med_top, sp_top, med_right, sp_right):
                chosen.append(p_right)
        elif sides_match:
            chosen = [p_left, p_right]
        elif has_left and sp_left <= 35.0 and len(p_left) >= 80:
            chosen = [p_left]
        elif has_right and sp_right <= 35.0 and len(p_right) >= 80:
            chosen = [p_right]

        if chosen:
            all_p = np.concatenate(chosen, axis=0)
            if len(all_p) >= 120:
                # Outlier trimming: drop the 15% furthest pixels from the preliminary median
                raw_med = np.median(all_p, axis=0)
                dists = np.linalg.norm(all_p - raw_med, axis=1)
                inliers = all_p[dists <= np.percentile(dists, 85)]
                med = np.median(inliers, axis=0)
                spread = float(np.median(np.abs(inliers - med).sum(axis=1)))

                # Check if the chosen inlier population splits into multiple distinct colour clusters (pattern)
                pat_inliers = _cluster_pattern(inliers)
                if pat_inliers:
                    return HijabLab(
                        pat_inliers["c0"],
                        pat_inliers["sp0"],
                        is_pattern=True,
                        pattern_labs=(pat_inliers["c0"], pat_inliers["c1"]),
                        pattern_names=(pat_inliers["name0"], pat_inliers["name1"]),
                    )

                if spread <= 42.0 and 12.0 <= med[0] <= 98.0:
                    return HijabLab(med, spread, is_pattern=False)

        # --- Pattern / Multi-tone Detection ---
        # When solid cross-checking fails or inlier spread exceeds solid threshold, test for pattern clustering
        pat = None
        # 1. First test forehead/crown wrap alone (safest fabric anchor squarely on the head)
        if len(p_top) >= 80:
            pat = _cluster_pattern(p_top)

        # 2. If not detected in top alone, test across combined reliable fabric candidates
        if not pat:
            candidates = []
            if len(p_top) >= 40:
                candidates.append(p_top)
            for p_side, med_side in [(p_left, med_left), (p_right, med_right)]:
                if len(p_side) >= 40 and med_side is not None and 14.0 <= med_side[0] <= 96.0:
                    candidates.append(p_side)
            if candidates:
                all_cand = np.concatenate(candidates, axis=0)
                if len(all_cand) >= 80:
                    pat = _cluster_pattern(all_cand)

        if pat:
            return HijabLab(
                pat["c0"],
                pat["sp0"],
                is_pattern=True,
                pattern_labs=(pat["c0"], pat["c1"]),
                pattern_names=(pat["name0"], pat["name1"]),
            )

        return HijabLab(None, None)
    except Exception:
        return HijabLab(None, None)


def user_hijab_colour(user: Image.Image, box: Box) -> Optional[str]:
    """Colour word or pattern description of the user's hijab.

    Returns:
      - Plain colour word for solid fabrics (e.g. 'plum', 'light pink')
      - Structural pattern description for multi-tone fabrics (e.g. 'patterned light pink and grey')
      - None when the sample is missing, black (template default), or unreliable noise
    """
    res = user_hijab_lab(user, box)
    med, spread = res
    if med is None:
        return None

    if getattr(res, "is_pattern", False):
        names = getattr(res, "pattern_names", ())
        if not names:
            return None
        name0 = names[0]
        name1 = names[1] if len(names) > 1 else None
        if name0 == "black" and (not name1 or name1 == "black"):
            return None
        if name1 and name1 != "black" and name1 != name0:
            return f"patterned {name0} and {name1}"
        elif name0 and name0 != "black":
            return f"patterned with a {name0} base"
        elif name1 and name1 != "black":
            return f"patterned with a {name1} print"
        return None

    if spread is not None and spread > 45:
        return None
    col = hijab_colour_name(med)
    if col == "black":
        return None  # the template's hijab is already black; avoid generating 'her hijab is black, not black'
    return col


# --------------------------------------------------------------------------------------------- the three fixes
def _fix_skin(bgr, lab, box: Box, skin_ref: Optional[np.ndarray], smooth: float, match: float) -> Tuple[np.ndarray, Dict]:
    info: Dict = {}
    try:
        # Zero-copy Rust acceleration (releases Python GIL during pixel-level tone matching)
        from saytara_rust_core import fast_tone_match_mask
        target_lab = skin_ref if skin_ref is not None else np.array([65.0, 15.0, 18.0], dtype=np.float32)
        m = fast_tone_match_mask(
            bgr,
            int(box[0]), int(box[1]), int(box[2]), int(box[3]),
            float(target_lab[0]), float(target_lab[1]), float(target_lab[2]),
            threshold=40.0,
            sx=0.86,
            sy=0.96,
        )
        info["rust_accelerated"] = True
    except (ImportError, Exception):
        m = _ellipse(bgr.shape, box, 0.86, 0.96) * _skin_like(bgr).astype(np.float32)
    m = cv2.erode(m, np.ones((3, 3), np.uint8))
    m = _soft(m, max(5, box[2] * 0.03))
    sel = m > 0.5
    if int(sel.sum()) < 800:
        return bgr, {"skin": "skipped (no skin found)"}

    out = bgr.astype(np.float32)
    if smooth > 0:
        sp = max(4.0, box[2] * 0.03)
        H, W = bgr.shape[:2]
        x, y, w, h = box
        x0, x1, y0, y1 = max(0, int(x - 0.2 * w)), min(W, int(x + 1.2 * w)), max(0, int(y - 0.3 * h)), min(H, int(y + 1.2 * h))
        low = bgr.copy()
        roi = bgr[y0:y1, x0:x1]
        for _ in range(2):
            roi = cv2.bilateralFilter(roi, 11, 38, sp)
        low[y0:y1, x0:x1] = roi
        mix = (m * smooth)[..., None]
        out = out * (1 - mix) + low.astype(np.float32) * mix
        info["skin_smooth"] = smooth

    lab2 = _lab(np.clip(out, 0, 255).astype(np.uint8))
    L = lab2[..., 0]
    # shine: remove local luminance bumps (oily highlights) relative to a heavily blurred skin luminance, then compress what is left
    wsel = sel.astype(np.float32)
    sg = max(9.0, box[2] * 0.05)
    num = cv2.GaussianBlur(L * wsel, (0, 0), sg)
    den = cv2.GaussianBlur(wsel, (0, 0), sg) + 1e-6
    Lbase = num / den
    bump = np.maximum(L - Lbase - 2.0, 0)
    L = L - bump * 0.9 * m
    p = float(np.percentile(L[sel], 75))
    lab2[..., 0] = L - np.maximum(L - p, 0) * 0.5 * m
    # shiny skin also washes out colour: bring a touch of chroma back in the corrected area
    lab2[..., 1] += 0.0

    cur = np.median(lab2[sel], axis=0)
    if skin_ref is not None and match > 0:
        d = (skin_ref - cur) * match
        d[0] = float(np.clip(d[0], -14, 14))
        d[1:] = np.clip(d[1:], -9, 9)
        for c in range(3):
            lab2[..., c] += d[c] * m
        info["skin_shift_Lab"] = [round(float(v), 1) for v in d]
    return _from_lab(lab2), info


def _hijab_mask(bgr, box: Box) -> np.ndarray:
    H, W = bgr.shape[:2]
    x, y, w, h = box
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    region = np.zeros((H, W), bool)
    region[max(0, int(y - 0.9 * h)):min(H, int(y + 1.75 * h)), max(0, int(x - 0.8 * w)):min(W, int(x + 1.8 * w))] = True
    face = _ellipse(bgr.shape, box, 1.12, 1.1) > 0.5
    red_jacket = (ss > 150) & ((hh < 9) | (hh > 172))
    blue = (hh > 95) & (hh < 135) & (ss > 110)
    return region & ~face & ~red_jacket & ~blue & (vv > 70)


def _fix_hijab_color(bgr, box: Box, hij_ref: Optional[np.ndarray], strength: float) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    cand = _hijab_mask(bgr, box)
    lab = _lab(bgr)
    cur = _median_lab(lab, cand, 1500)
    if cur is None:
        return bgr, None, {"hijab": "skipped (no hijab pixels)"}
    band, _sp = user_hijab_lab(_pil(bgr), box)  # bands right next to the face: the broad mask above also catches skin/neck
    if band is not None:
        cur = band
    lit_sel = cand & (lab[..., 0] >= np.percentile(lab[..., 0][cand], 55))
    lit = _median_lab(lab, lit_sel, 800)
    if lit is not None and band is None:
        cur_lit = lit
    else:
        cur_lit = cur
    if hij_ref is None or strength <= 0:
        return bgr, cur_lit, {"hijab": "skipped (no user hijab colour)"}
    gap = float(np.linalg.norm(hij_ref - cur))
    if gap > 32:  # the user sample is from another colour family (background / skin got in): trust the model's hijab
        return bgr, cur_lit, {"hijab": f"skipped (user colour sample too far from painted hijab: dE={gap:.0f})"}
    d = (hij_ref - cur) * strength
    d[0] = float(np.clip(d[0], -12, 12))   # only a SMALL correction of what the model painted
    ab = d[1:]
    n = float(np.linalg.norm(ab))
    if n > 16:
        d[1:] = ab * (16 / n)
    m = _soft(cand.astype(np.float32), 5)
    lab2 = lab.copy()
    for c in range(3):
        lab2[..., c] += d[c] * m
    new_med = cur_lit + d
    return _from_lab(lab2), new_med, {"hijab_shift_Lab": [round(float(v), 1) for v in d], "hijab_from": [round(float(v), 1) for v in cur]}


def _fix_drape(bgr, box: Box, hijab_lab: Optional[np.ndarray], strength: float) -> Tuple[np.ndarray, Dict]:
    if hijab_lab is None or strength <= 0:
        return bgr, {"drape": "skipped"}
    H, W = bgr.shape[:2]
    x, y, w, h = box
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    # dark cloth: not the pure-black canvas (V<=2), not the navy emblem, not the red jacket
    not_emblem = ~((hh > 100) & (hh < 135) & (ss > 120) & (vv >= 20)) & ~((ss > 150) & ((hh < 9) | (hh > 172)) & (vv >= 30))
    dark = (vv >= 3) & (vv < 105) & (ss < 150) & not_emblem                 # core: neutral dark cloth
    speck = (vv >= 3) & (vv < 45) & not_emblem                              # near-black pixels with a noisy saturation
    zone = np.zeros((H, W), bool)
    zone[int(y + 0.92 * h):min(H, int(y + 2.2 * h)), max(0, int(x - 0.30 * w)):min(W, int(x + 1.45 * w))] = True
    dark &= zone
    speck &= zone
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab_cc, st, _ = cv2.connectedComponentsWithStats(dark)
    # keep only the dark component(s) that touch the fabric directly above them (the template's cowl hangs from the hijab)
    fabric = _hijab_mask(bgr, box)
    fabric_d = cv2.dilate(fabric.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    keep = np.zeros((H, W), np.float32)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 0.004 * H * W:
            continue
        comp = lab_cc == k
        if (comp & fabric_d).sum() > 25:
            keep[comp] = 1.0
    # fill the pin-holes (the cowl's darkest pixels have a noisy saturation and were rejected above)
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    keep = _fill_holes(keep, 0.003 * H * W)
    keep = np.maximum(keep, (speck & (cv2.dilate(keep, np.ones((9, 9), np.uint8)) > 0)).astype(np.float32))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    keep = _fill_holes(keep, 0.003 * H * W)
    keep = cv2.erode(keep, np.ones((3, 3), np.uint8))                      # stay just inside the cloth edge (natural dark rim)
    keep = (cv2.GaussianBlur(keep, (25, 25), 0) > 0.5).astype(np.float32)  # round off the stair-steps / pixel jaggies
    area = float(keep.sum()) / (H * W)
    if area < 0.006:
        return bgr, {"drape": "skipped (no dark cowl found)"}
    if area > 0.12:
        return bgr, {"drape": "skipped (dark area too large to be the cowl)"}
    m = _soft(keep, 7) * strength
    lab = _lab(bgr)
    shade = np.clip((lab[..., 0] - 2.0) / 38.0, 0, 1)  # keep the folds: dark pixels stay darker
    tgt = lab.copy()
    f = (0.80 + 0.20 * shade)
    tgt[..., 0] = hijab_lab[0] * f
    tgt[..., 1] = hijab_lab[1] * (0.85 + 0.15 * shade)
    tgt[..., 2] = hijab_lab[2] * (0.85 + 0.15 * shade)
    mm = m[..., None]
    out = lab * (1 - mm) + tgt * mm
    return _from_lab(out), {"drape_area_pct": round(area * 100, 1)}


def _fix_grey_cowl(bgr, box: Box, hijab_lab: Optional[np.ndarray], strength: float) -> Tuple[np.ndarray, Dict]:
    """The model sometimes paints its own flat grey/taupe neck cowl. Recolour it to the (lit) hijab colour, keeping its folds."""
    if hijab_lab is None or strength <= 0:
        return bgr, {"cowl": "skipped"}
    H, W = bgr.shape[:2]
    x, y, w, h = box
    lab = _lab(bgr)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
    red = (ss > 110) & ((hh < 12) | (hh > 165))
    blue = (hh > 95) & (hh < 140) & (ss > 90)
    cowl_like = (ss < 70) & (vv > 70) & (vv < 215) & ~red & ~blue & (lab[..., 0] < hijab_lab[0] - 6)
    zone = np.zeros((H, W), bool)
    zone[int(y + 0.86 * h):min(H, int(y + 1.9 * h)), max(0, int(x - 0.15 * w)):min(W, int(x + 1.05 * w))] = True
    cm = (cowl_like & zone & ~(_skin_like(bgr) & (ss > 60))).astype(np.uint8)
    cm = cv2.morphologyEx(cm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    cm = cv2.morphologyEx(cm, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, cc, st, _ = cv2.connectedComponentsWithStats(cm)
    if n <= 1:
        return bgr, {"cowl": "skipped (no grey cowl found)"}
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    keep = (cc == k).astype(np.float32)
    keep = _fill_holes(keep, 0.004 * H * W)
    keep = cv2.erode(keep, np.ones((3, 3), np.uint8))
    keep = (cv2.GaussianBlur(keep, (21, 21), 0) > 0.5).astype(np.float32)
    area = float(keep.sum()) / (H * W)
    if area < 0.005 or area > 0.12:
        return bgr, {"cowl": f"skipped (area {area*100:.1f}% implausible)"}
    cur = _median_lab(lab, keep > 0.5, 500)
    if cur is None:
        return bgr, {"cowl": "skipped"}
    # move the cowl colour toward the hijab, keep its own light/dark variation (folds)
    tgt = lab.copy()
    tgt[..., 0] = lab[..., 0] + (hijab_lab[0] * 0.93 - cur[0])
    tgt[..., 1] = lab[..., 1] + (hijab_lab[1] - cur[1])
    tgt[..., 2] = lab[..., 2] + (hijab_lab[2] - cur[2])
    mm = (_soft(keep, 6) * strength)[..., None]
    out = lab * (1 - mm) + tgt * mm
    return _from_lab(out), {"cowl_area_pct": round(area * 100, 1), "cowl_from": [round(float(v), 1) for v in cur]}


def fix_hijab_avatar(
    out_img: Image.Image,
    out_face: Optional[Box],
    user_img: Optional[Image.Image] = None,
    user_face: Optional[Box] = None,
    *,
    skin_smooth: float = 0.8,
    skin_match: float = 0.6,
    hijab_match: float = 0.85,
    drape: float = 0.9,
) -> Tuple[Image.Image, Dict]:
    """Returns (fixed image, report). Never raises: on any problem the original image is returned."""
    rep: Dict = {}
    try:
        if not out_face or out_face[2] < 60 or out_face[3] < 60:
            return out_img, {"postfix": "skipped (no face box on avatar)"}
        bgr = _bgr(out_img)
        skin_ref = hij_ref = None
        if user_img is not None and user_face is not None:
            skin_ref = user_skin_lab(user_img, user_face)
            hij_res = user_hijab_lab(user_img, user_face)
            hij_ref, spread = hij_res
            if getattr(hij_res, "is_pattern", False):
                rep["user_hijab_pattern"] = list(getattr(hij_res, "pattern_names", ()))
                # Patterned fabric cannot be uniformly shifted to a single colour in pixel post-fix
                hij_ref = None
                rep["hijab_ref"] = "patterned fabric (pixel shift skipped)"
            rep["user_hijab_Lab"] = None if hij_ref is None else [round(float(v), 1) for v in hij_ref]
            rep["user_hijab_spread"] = None if spread is None else round(spread, 1)
            if hij_ref is not None and hij_ref[0] < 28:  # that is the room / black background, not fabric
                hij_ref = None
                rep['hijab_ref'] = 'rejected (too dark)'
            if hij_ref is not None and spread is not None and spread > 45:  # sample is mixed (background/skin) - do not trust it
                hij_ref = None
                rep["hijab_ref"] = "rejected (mixed sample)"
        bgr, r = _fix_skin(bgr, None, out_face, skin_ref, skin_smooth, skin_match)
        rep.update(r)
        bgr, hij_now, r = _fix_hijab_color(bgr, out_face, hij_ref, hijab_match)
        rep.update(r)
        if hij_now is not None and hij_ref is not None and float(np.linalg.norm(hij_now - hij_ref)) > 40:
            # the model kept its own (black) hijab and the user's colour was not applied: recolouring only the neck
            # produced a flat taupe patch. Leave the drape as painted so head and neck stay one fabric.
            rep["drape"] = "skipped (hijab colour not matched; drape left as painted)"
            return _pil(bgr), rep
        bgr, r = _fix_drape(bgr, out_face, hij_now, drape)
        rep.update(r)
        if str(r.get("drape", "")).startswith("skipped"):
            bgr, r = _fix_grey_cowl(bgr, out_face, hij_now, drape)
            rep.update(r)
        return _pil(bgr), rep
    except Exception as e:  # fail-soft by design
        return out_img, {"postfix": f"failed: {type(e).__name__}: {e}"}
