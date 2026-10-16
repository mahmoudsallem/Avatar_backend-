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
