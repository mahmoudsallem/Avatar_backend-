"""Clean the visor reference image (Picture 3) so it shows ONLY the visor.

The raw reference is a close-up crop of the template avatar's face: the template's own eyes, eyebrows, nose and lips
seen through a dark-blue tinted lens. FLUX copied those into the user's avatar (template eyes/brows instead of the
user's, bluish/purple irises) together with an oversized visor. Here the visor outline is traced once per template
(the 3 templates are fixed assets), everything outside it becomes black, the lens interior becomes plain light clear
glass with no face behind it, and only the blue frame, corner blocks, arm and white glints are kept from the art.

Pure numpy + OpenCV, fail-soft: any problem returns the original image.
"""
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# Visor outline per template, in coordinates normalised to the VISOR_REFS crop box (x, y in 0..1), traced by hand.
VISOR_POLY = {
    "Woman": [(78, 238), (300, 165), (540, 85), (705, 105), (712, 165), (640, 200), (600, 198), (430, 308),
              (395, 300), (285, 240), (258, 258), (240, 350), (125, 390), (88, 340)],
    "Woman_Hijab": [(52, 205), (560, 66), (690, 90), (698, 165), (630, 192), (400, 300), (368, 292), (250, 228),
                    (238, 245), (215, 345), (100, 368), (58, 300)],
    "Man": [(42, 198), (500, 28), (648, 62), (655, 145), (590, 175), (400, 265), (365, 258), (235, 188),
            (222, 205), (205, 305), (95, 345), (45, 285)],
}
_TRACE_W, _TRACE_H = 800.0, 420.0


def isolate_visor(crop: Image.Image, key: str) -> Optional[Image.Image]:
    """crop = the template cut to VISOR_REFS[key]['box'] (before padding). Returns the cleaned crop, or None."""
    try:
        poly = VISOR_POLY.get(key)
        if not poly:
            return None
        rgb = np.asarray(crop.convert("RGB"))
        H, W = rgb.shape[:2]
        pts = np.array([[x / _TRACE_W * W, y / _TRACE_H * H] for x, y in poly], np.int32)
        inside = np.zeros((H, W), np.uint8)
        cv2.fillPoly(inside, [pts], 1)
        inside = inside.astype(bool)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
        # keep from the art: strongly saturated blue parts (corner blocks, arm) and white glints; the frame line is redrawn
        solid = (h >= 95) & (h <= 125) & (s >= 120) & (v >= 120)
        solid = cv2.morphologyEx(solid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, cc, st, _ = cv2.connectedComponentsWithStats(solid)
        blocks = np.zeros((H, W), bool)
        for k in range(1, n):  # only big solid parts = corner blocks / arm (eye-shadow specks are small)
            if st[k, cv2.CC_STAT_AREA] >= 0.004 * H * W:
                blocks |= cc == k
        glint = (v >= 240) & (s <= 60)
        lens = inside & ~(blocks | glint)
        # lens = clear glass on a black canvas: almost black with a faint icy tint + two thin diagonal glints, so the
        # model sees a see-through lens (an opaque light fill would teach it to hide the eyes)
        out = np.zeros_like(rgb)
        out[lens] = (14, 24, 36)
        gl = np.zeros((H, W), np.uint8)
        x0, x1 = int(pts[:, 0].min()), int(pts[:, 0].max())
        y0, y1 = int(pts[:, 1].min()), int(pts[:, 1].max())
        for fx, wd in ((0.32, 0.018), (0.38, 0.008)):
            cx = int(x0 + fx * (x1 - x0))
            cv2.line(gl, (cx, y1), (cx + int(0.18 * (x1 - x0)), y0), 1, max(1, int(wd * W)))
        gl = (gl > 0) & lens
        out[gl] = (150, 190, 225)
        out[blocks & inside] = rgb[blocks & inside]
        out[glint & inside] = rgb[glint & inside]
        cv2.polylines(out, [pts], True, (70, 165, 255), thickness=max(2, int(0.008 * W)), lineType=cv2.LINE_AA)
        return Image.fromarray(out)
    except Exception:
        return None


def detint_template_lens(img: Image.Image, key: str, box) -> Image.Image:
    """Remove the blue cast from the lens area of the TEMPLATE (Picture 1, the base image).

    The template's lens tints its eyes and skin blue/purple; FLUX starts from this image and keeps that tint on the
    user's eyes. Inside the traced visor outline (box = VISOR_REFS[key]['box'] in template fractions), every pixel that
    is not the blue frame / corner block / glint gets its colour shifted from the lens tint back to the skin colour just
    below the visor, so the lens reads as clear glass. Fail-soft: returns the input on any problem.
    """
    try:
        poly = VISOR_POLY.get(key)
        if not poly:
            return img
        rgb = np.asarray(img.convert("RGB"))
        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = box
        pts = np.array([[(x0 + x / _TRACE_W * (x1 - x0)) * W, (y0 + y / _TRACE_H * (y1 - y0)) * H] for x, y in poly],
                       np.int32)
        inside = np.zeros((H, W), np.uint8)
        cv2.fillPoly(inside, [pts], 1)
        inside = cv2.erode(inside, np.ones((5, 5), np.uint8)).astype(bool)   # keep the frame line itself untouched
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        h, s, v = hsv[..., 0].astype(np.int32), hsv[..., 1].astype(np.int32), hsv[..., 2].astype(np.int32)
        frame = ((h >= 95) & (h <= 125) & (s >= 120) & (v >= 120)) | ((v >= 240) & (s <= 60))
        frame = cv2.dilate(frame.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        lens = inside & ~frame
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
        # reference skin: a band just below the visor's lower edge (cheeks / nose), warm skin-like pixels only
        below = np.zeros((H, W), np.uint8)
        cv2.fillPoly(below, [pts + np.array([0, int(0.035 * H)])], 1)
        below = below.astype(bool) & ~inside
        skin = below & (rgb[..., 0].astype(int) > rgb[..., 2].astype(int) + 25) & (v > 90)
        lens_skin = lens & (v > 110) & (s < 120)            # mid-bright lens pixels = skin seen through the lens
        if skin.sum() < 300 or lens_skin.sum() < 300:
            return img
        ds = np.median(lab[skin], axis=0) - np.median(lab[lens_skin], axis=0)
        ds[0] = float(np.clip(ds[0], -6, 10))               # small brightness change, mostly a colour change
        m = cv2.GaussianBlur(lens.astype(np.float32), (9, 9), 0)[..., None]
        out = lab + 0.8 * ds[None, None, :] * m   # 0.8: full shift looked slightly too warm/orange
        out = np.clip(cv2.cvtColor(out.astype(np.float32), cv2.COLOR_LAB2RGB) * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    except Exception:
        return img


# Far-side face outline at visor height, per template, in the same 800x420 trace units as VISOR_POLY: everything of
# the visor LEFT of this line sticks out past the cheek / temple silhouette (physically it would be hidden behind the
# head). The model copied that overhang onto every user, so it is removed from the template and the visor reference.
VISOR_TRIM_LINE = {
    "Woman": ((152, 150), (205, 400)),
    "Woman_Hijab": ((122, 150), (172, 400)),
}


def _trim_mask(H: int, W: int, key: str, to_px) -> Optional[np.ndarray]:
    poly, line = VISOR_POLY.get(key), VISOR_TRIM_LINE.get(key)
    if not poly or not line:
        return None
    pts = np.array([to_px(x, y) for x, y in poly], np.int32)
    vis = np.zeros((H, W), np.uint8)
    cv2.fillPoly(vis, [pts], 1)
    (ax, ay), (bx, by) = line
    # half-plane left of the line (extended far up and down)
    t = 10.0
    p0 = to_px(ax - (bx - ax) * t, ay - (by - ay) * t)
    p1 = to_px(bx + (bx - ax) * t, by + (by - ay) * t)
    left = np.zeros((H, W), np.uint8)
    cv2.fillPoly(left, [np.array([p0, p1, (-10 * W, p1[1]), (-10 * W, p0[1])], np.int32)], 1)
    m = (vis & left).astype(np.uint8)
    return m if m.sum() > 50 else None


def trim_template_overhang(img: Image.Image, key: str, box) -> Image.Image:
    """Inpaint away the part of the TEMPLATE visor that sticks out past the far cheek. Fail-soft."""
    try:
        rgb = np.asarray(img.convert("RGB"))
        H, W = rgb.shape[:2]
        x0, y0, x1, y1 = box
        to_px = lambda x, y: (int((x0 + x / _TRACE_W * (x1 - x0)) * W), int((y0 + y / _TRACE_H * (y1 - y0)) * H))
        m = _trim_mask(H, W, key, to_px)
        if m is None:
            return img
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        out = cv2.inpaint(bgr, m * 255, max(5, int(0.01 * W)), cv2.INPAINT_TELEA)
        return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    except Exception:
        return img


def trim_ref_overhang(ref_crop: Image.Image, key: str) -> Image.Image:
    """Same cut on the cleaned visor reference crop (that crop is the VISOR_REFS box itself). Fail-soft."""
    try:
        rgb = np.array(ref_crop.convert("RGB"))
        H, W = rgb.shape[:2]
        to_px = lambda x, y: (int(x / _TRACE_W * W), int(y / _TRACE_H * H))
        m = _trim_mask(H, W, key, to_px)
        if m is None:
            return ref_crop
        # widen the cut a little so the redrawn outline of the removed part goes too, but only on the far side
        m2 = cv2.dilate(m, np.ones((11, 11), np.uint8))
        rgb[m2 > 0] = 0
        return Image.fromarray(rgb)
    except Exception:
        return ref_crop
