# Saytara Rust Core (`saytara_rust_core`)

High-performance native Rust extension for CPU-bound image processing and color transformations in Saytara's Avatar generation pipeline.

## Features
- **Zero-Copy NumPy Bridge**: Accepts `numpy.ndarray` via PyO3 without memory copies.
- **Immediate GIL Release**: Every heavy calculation runs inside `py.allow_threads(|| { ... })`, eliminating Python GIL contention.
- **SIMD & Multi-Threading**: Parallel pixel/row processing powered by `rayon`.
- **CIE L\*a\*b\* Color Space**: Fast vectorized conversions avoiding OpenCV / Python overhead.

## Building and Installation

### Local Development (with Rust & Maturin installed)
```bash
pip install maturin
cd saytara_rust_core
maturin develop --release
```

### Building Wheel for Production / Docker
```bash
maturin build --release --strip
pip install target/wheels/*.whl
```
