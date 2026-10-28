mod tone_match;

use pyo3::prelude::*;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray2, PyReadonlyArray3};

/// Fast zero-copy tone matching and skin segmentation.
///
/// Releases the Python GIL immediately so the ASGI/FastAPI event loop and worker
/// threads remain completely unblocked during CPU-bound color processing.
#[pyfunction]
#[pyo3(signature = (img, box_x, box_y, box_w, box_h, target_l, target_a, target_b, threshold=35.0, sx=0.86, sy=0.96))]
fn fast_tone_match_mask<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, u8>,
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
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let view = img.as_array();

    // Release GIL for multi-threaded Rayon execution
    let mask = py.allow_threads(|| {
        tone_match::compute_tone_match_mask(
            view,
            box_x,
            box_y,
            box_w,
            box_h,
            target_l,
            target_a,
            target_b,
            threshold,
            sx,
            sy,
        )
    });

    Ok(mask.into_pyarray_bound(py))
}

/// Fast skin classifier across an entire BGR image canvas.
#[pyfunction]
fn fast_skin_mask<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let view = img.as_array();
    let height = view.shape()[0];
    let width = view.shape()[1];

    let mask = py.allow_threads(|| {
        use ndarray::{Array2, Axis};
        use rayon::prelude::*;

        let mut out = Array2::<u8>::zeros((height, width));
        out.axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(y, mut row)| {
                for x in 0..width {
                    let b = view[[y, x, 0]];
                    let g = view[[y, x, 1]];
                    let r = view[[y, x, 2]];
                    if tone_match::is_skin_like(b, g, r) {
                        row[x] = 255;
                    }
                }
            });
        out
    });

    Ok(mask.into_pyarray_bound(py))
}

/// Blends and shifts colors in CIE L*a*b* space without GIL contention.
#[pyfunction]
#[pyo3(signature = (img, mask, shift_l, shift_a, shift_b, strength=1.0))]
fn fast_apply_lab_shift<'py>(
    py: Python<'py>,
    img: PyReadonlyArray3<'py, u8>,
    mask: PyReadonlyArray2<'py, f32>,
    shift_l: f32,
    shift_a: f32,
    shift_b: f32,
    strength: f32,
) -> PyResult<Bound<'py, PyArray3<u8>>> {
    let img_owned = img.as_array().to_owned();
    let mask_view = mask.as_array();

    let recolored = py.allow_threads(|| {
        tone_match::apply_lab_shift(
            img_owned,
            mask_view,
            shift_l,
            shift_a,
            shift_b,
            strength,
        )
    });

    Ok(recolored.into_pyarray_bound(py))
}

/// The Saytara Rust Core native extension module.
#[pymodule]
fn saytara_rust_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_tone_match_mask, m)?)?;
    m.add_function(wrap_pyfunction!(fast_skin_mask, m)?)?;
    m.add_function(wrap_pyfunction!(fast_apply_lab_shift, m)?)?;
    Ok(())
}
