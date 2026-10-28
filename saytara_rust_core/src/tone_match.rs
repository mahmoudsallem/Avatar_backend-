use ndarray::{Array2, Array3, ArrayView3, ArrayView2, Axis};
use rayon::prelude::*;

/// Fast sRGB to CIE L*a*b* conversion (D65 illuminant).
#[inline(always)]
pub fn bgr_to_lab(b: u8, g: u8, r: u8) -> (f32, f32, f32) {
    let srgb_to_linear = |v: u8| -> f32 {
        let val = v as f32 / 255.0;
        if val > 0.04045 {
            ((val + 0.055) / 1.055).powf(2.4)
        } else {
            val / 12.92
        }
    };

    let r_lin = srgb_to_linear(r);
    let g_lin = srgb_to_linear(g);
    let b_lin = srgb_to_linear(b);

    // D65 matrix transformation to CIE XYZ
    let x = r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375;
    let y = r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750;
    let z = r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041;

    let x_n = x / 0.95047;
    let y_n = y / 1.00000;
    let z_n = z / 1.08883;

    let f = |t: f32| -> f32 {
        if t > 0.008856 {
            t.cbrt()
        } else {
            7.787 * t + 16.0 / 116.0
        }
    };

    let fx = f(x_n);
    let fy = f(y_n);
    let fz = f(z_n);

    let l = (116.0 * fy - 16.0).clamp(0.0, 100.0);
    let a = 500.0 * (fx - fy);
    let b_val = 200.0 * (fy - fz);

    (l, a, b_val)
}

/// Fast CIE L*a*b* back to BGR.
#[inline(always)]
pub fn lab_to_bgr(l: f32, a: f32, b_val: f32) -> (u8, u8, u8) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b_val / 200.0;

    let f_inv = |t: f32| -> f32 {
        let t3 = t * t * t;
        if t3 > 0.008856 {
            t3
        } else {
            (t - 16.0 / 116.0) / 7.787
        }
    };

    let x = f_inv(fx) * 0.95047;
    let y = f_inv(fy) * 1.00000;
    let z = f_inv(fz) * 1.08883;

    let r_lin =  x * 3.2404542 - y * 1.5371385 - z * 0.4985314;
    let g_lin = -x * 0.9692660 + y * 1.8760108 + z * 0.0415560;
    let b_lin =  x * 0.0556434 - y * 0.2040259 + z * 1.0572252;

    let linear_to_srgb = |v: f32| -> u8 {
        let v = v.clamp(0.0, 1.0);
        let srgb = if v > 0.0031308 {
            1.055 * v.powf(1.0 / 2.4) - 0.055
        } else {
            12.92 * v
        };
        (srgb * 255.0).round().clamp(0.0, 255.0) as u8
    };

    (linear_to_srgb(b_lin), linear_to_srgb(g_lin), linear_to_srgb(r_lin))
}

/// Checks skin-like chromatic bounds in YCrCb and HSV color spaces.
#[inline(always)]
pub fn is_skin_like(b: u8, g: u8, r: u8) -> bool {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;

    // YCrCb bounds
    let y  = 0.299 * rf + 0.587 * gf + 0.114 * bf;
    let cr = (rf - y) * 0.713 + 128.0;
    let cb = (bf - y) * 0.564 + 128.0;

    let ok_ycc = cr > 133.0 && cr < 178.0 && cb > 78.0 && cb < 135.0;

    // HSV bounds: Value > 60, Saturation > 22
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let v = max;
    let s = if max > 0 { ((max - min) as f32 / max as f32) * 255.0 } else { 0.0 };

    let ok_hsv = v > 60 && s > 22.0;
    let not_bluish = b < r;

    ok_ycc && ok_hsv && not_bluish
}

/// Elliptical mask test: tests whether point (x, y) falls within the scaled face ellipse.
#[inline(always)]
pub fn in_ellipse(
    px: f32, py: f32,
    box_x: f32, box_y: f32, box_w: f32, box_h: f32,
    sx: f32, sy: f32, cy: f32,
) -> bool {
    let center_x = box_x + box_w * 0.5;
    let center_y = box_y + box_h * cy;
    let rx = (box_w * 0.5 * sx).max(1.0);
    let ry = (box_h * 0.5 * sy).max(1.0);

    let dx = (px - center_x) / rx;
    let dy = (py - center_y) / ry;
    (dx * dx + dy * dy) <= 1.0
}

/// Parallel zero-copy computation of the tone-matching mask across the image canvas.
pub fn compute_tone_match_mask(
    view: ArrayView3<u8>,
    box_x: usize,
    box_y: usize,
    box_w: usize,
    box_h: usize,
    target_l: f32,
    target_a: f32,
    target_b: f32,
    threshold: f32,
    sx: f32,
    sy: f32,
) -> Array2<f32> {
    let height = view.shape()[0];
    let width = view.shape()[1];

    let mut mask = Array2::<f32>::zeros((height, width));

    let bx = box_x as f32;
    let by = box_y as f32;
    let bw = box_w as f32;
    let bh = box_h as f32;

    // Parallel row iteration using Rayon
    mask.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(y, mut row)| {
            let py = y as f32;
            for x in 0..width {
                let px = x as f32;

                // 1. Check face ellipse boundary
                if !in_ellipse(px, py, bx, by, bw, bh, sx, sy, 0.52) {
                    continue;
                }

                // 2. Fetch BGR pixel
                let b = view[[y, x, 0]];
                let g = view[[y, x, 1]];
                let r = view[[y, x, 2]];

                // 3. Check skin-like chromatic bounds
                if !is_skin_like(b, g, r) {
                    continue;
                }

                // 4. Compute Delta E in CIE L*a*b* space
                let (pl, pa, pb) = bgr_to_lab(b, g, r);
                let dl = pl - target_l;
                let da = pa - target_a;
                let db = pb - target_b;

                // Weighted luminance (0.6) matching Saytara palette metrics
                let dist = ((dl * 0.6).powi(2) + da.powi(2) + db.powi(2)).sqrt();

                if dist <= threshold {
                    // Soft attenuation near threshold boundary
                    let val = 1.0 - (dist / threshold).clamp(0.0, 1.0);
                    row[x] = val;
                }
            }
        });

    mask
}

/// Applies a smooth Lab color shift to the target fabric/cowl region directly in Rust.
pub fn apply_lab_shift(
    mut img: Array3<u8>,
    mask: ArrayView2<f32>,
    shift_l: f32,
    shift_a: f32,
    shift_b: f32,
    strength: f32,
) -> Array3<u8> {
    let height = img.shape()[0];
    let width = img.shape()[1];

    img.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(y, mut row)| {
            for x in 0..width {
                let m = mask[[y, x]] * strength;
                if m < 0.01 {
                    continue;
                }

                let b = row[[x, 0]];
                let g = row[[x, 1]];
                let r = row[[x, 2]];

                let (l, a, b_val) = bgr_to_lab(b, g, r);

                // Blend in Lab space
                let target_l = (l + shift_l).clamp(0.0, 100.0);
                let target_a = a + shift_a;
                let target_b = b_val + shift_b;

                let blended_l = l * (1.0 - m) + target_l * m;
                let blended_a = a * (1.0 - m) + target_a * m;
                let blended_b = b_val * (1.0 - m) + target_b * m;

                let (new_b, new_g, new_r) = lab_to_bgr(blended_l, blended_a, blended_b);
                row[[x, 0]] = new_b;
                row[[x, 1]] = new_g;
                row[[x, 2]] = new_r;
            }
        });

    img
}
