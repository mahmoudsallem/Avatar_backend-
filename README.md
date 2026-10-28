# Saytara Avatar Generation Backend

Production FastAPI backend service for generating stylized sci-fi avatars from user portrait photos using **FLUX.2-klein-9B** diffusion models and **BFS (Best Face Swap)** LoRA on EC2 GPU instances.

---

## 1. Overview & Architecture

The service executes a 2-step AI pipeline:

1. **Step 1: Face Validation & Classification (`analyse_user`)**
   - **Multi-face Dominance Filtering**: Runs RetinaFace face detection in an isolated CPU subprocess. Rejects images with multiple prominent faces or without a clear foreground person.
   - **Scope Check**: CLIP zero-shot classification blocks pets, cartoons, code screenshots, objects, and landscapes.
   - **Attribute Detection**: Classifies Gender (DeepFace with CLIP fallback), Eyeglasses (CLIP multi-view), Beard (CLIP multi-view), and Hijab (CLIP multi-view).
   - Generates an exact aligned face crop used by the diffusion model.

2. **Step 2: Sci-Fi Avatar Generation & Validation (`generate_avatar`)**
   - Prepares avatar template (`Man`, `Woman`, or `Woman_Hijab`), user face crop, and avatar visor reference image.
   - Assembles a detailed hand-tuned prompt enforcing facial likeness, build, and futuristic blue wraparound visor.
   - Executes FLUX.2-klein-9B with BFS face-swap LoRA.
   - **Identity & Visor Verification**: Evaluates generated candidates via ArcFace cosine similarity (in a persistent CPU worker) and OpenCV HSV visor detection. Retries up to `BEST_OF_N` candidates within a configured `TIME_BUDGET` (default: 40s), keeping the highest-scoring candidate.

3. **Concurrency Model**:
   - Generation runs through a pool of `GPU_SLOTS` FLUX pipelines (default 2) inside the single worker process. Each slot has its own transformer, LoRA state, scheduler and CUDA stream; the text encoder and VAE are shared. Up to `GPU_SLOTS` avatars generate at the same time, extra requests wait for a free slot. Step 1 (face analysis) runs outside the slots, `ANALYSIS_CONCURRENCY` at a time. Each extra slot costs ~18 GB VRAM, so raise `GPU_SLOTS` only after checking `gpu_free_vram_gb` in `/health` (a 96 GB card fits 2-3 slots). In CPU-offload (low VRAM) mode it stays at 1 slot.
   - Heavy blocking operations run in a thread pool (`asyncio.to_thread`), ensuring `GET /health` remains responsive even when generation is in progress.

### Code Layout
The whole service is 5 files:
```
app.py                  # FastAPI app, lifespan, /health and /v1/avatar routes
app/
  __init__.py
  config.py             # pydantic-settings, reads .env
  pipeline.py           # Step 1 validation + Step 2 generation + model/worker lifecycle
  deepface_worker.py    # CPU subprocess worker (modes: "detect" one-shot, "embed" persistent)
```

---

## 2. Hardware & License Requirements

### Optional Nunchaku trial on g7e.2xlarge

Set `FLUX_BACKEND=nunchaku` to use the quantized FLUX.2 Klein 9B transformer. The default
`diffusers` backend remains the reference. Nunchaku is an optional dependency and must be
installed in a separate Linux x86_64 environment with a wheel matching that environment's
Python version, PyTorch version and `torch.version.cuda`. The Amazon Linux 2023 DLAMI
20260929 has system Python 3.12; the published Vitoom Nunchaku x86_64 wheels do not list
Python 3.12. Use a Python 3.11 environment with a matching Torch build, following the
[Vitoom wheel instructions](https://huggingface.co/tonera/vitoom-nunchaku). Do not install
the DGX Spark `aarch64` wheel on EC2 x86_64.

On that AMI, create a separate runtime and install the published Python 3.11 / Torch 2.11 /
CUDA 13.0 wheel. Run the app with this environment's `python`, not `/usr/bin/python3.12`:

```bash
sudo dnf install -y python3.11 python3.11-pip
python3.11 -m venv .venv-nunchaku
source .venv-nunchaku/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install 'huggingface-hub[cli]'
hf download tonera/vitoom-nunchaku \
  'nunchaku-1.3.0.dev20260622+cu13.0torch2.11-cp311-cp311-linux_x86_64.whl' \
  --local-dir ./wheels
python -m pip install ./wheels/nunchaku-1.3.0.dev20260622+cu13.0torch2.11-cp311-cp311-linux_x86_64.whl
python -c 'import sys, torch, nunchaku; from nunchaku import NunchakuFlux2Transformer2DModel; print(sys.version); print(torch.__version__, torch.version.cuda); print(nunchaku.__version__)'
python app.py
```

If your service manager starts the app, point its executable at
`<checkout>/.venv-nunchaku/bin/python` as well. Installing into the new environment does not
make `nunchaku` importable by the old Python 3.12 service.

The quantized weights are loaded from `NUNCHAKU_MODEL` and the existing BFS LoRA is loaded
into each quantized transformer. Startup fails if the LoRA cannot load or the LoRA self-test
finds no effect. Keep `LORA_SELFTEST=true`. `FUSE_LORA` and `COMPILE_TRANSFORMER` must be false
with this backend. Set `GPU_SLOTS=2` initially, then benchmark 3 slots if VRAM permits.
The model's quantization can alter images, so compare identity scores, visor success and
retries before switching production traffic.

On the running instance, check the environment with:

```bash
python -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0))'
```

Run the existing concurrent test against both backend configurations, using the same photos
and settings for each. At least 50 requests provides a more useful steady-load comparison
than the default ten:

```bash
python tests/benchmarks/send_10_images.py --url http://localhost:8000 --count 50
```

The `throughput` line reports completed JPEG avatars per minute. The target is at least 10
avatars/minute with no increase in failed requests and acceptable identity and visor results.
The DGX Spark benchmark is not a throughput guarantee for the g7e GPU or this full pipeline.

### Recommended Hardware
- **EC2 Instance Type**: `g5.2xlarge`, `g5.4xlarge`, or `g6e.2xlarge` (NVIDIA A10G / L40S with >= 24GB VRAM).
- **Disk Space**: At least 50 GB root volume for models and container cache.
- **CUDA**: 12.1+ or 12.4+.

### Model Licenses
- **FLUX.2-klein-9B**: Gated model on Hugging Face under the **FLUX Non-Commercial License**. You must have a Hugging Face account, accept the license on the [black-forest-labs/FLUX.2-klein-9B repository page](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B), and authenticate once on the EC2 instance (see Step 4 below) so the weights can be downloaded and cached locally.
- **BFS LoRA**: Public MIT license repository (`sisniha/BFS-Best-Face-Swap`).

---

## 3. Installation & Setup on EC2

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone <your-repo-url>
cd Backend
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### Step 2: Install PyTorch with CUDA Support
**CRITICAL**: Do **NOT** install PyTorch from standard PyPI. Install the wheel matching your host CUDA driver first:
```bash
# Example for CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA is visible to PyTorch:
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print('CUDA OK:', torch.cuda.get_device_name(0))"
```

### Step 3: Install Remaining Dependencies
```bash
pip install -r requirements.txt
```

> **Note on Transformers**: FLUX.2-klein requires `transformers>=4.51.0` (with Qwen3 support). We pin `transformers>=4.56.0` in `requirements.txt`. If you encounter any text-encoder loading errors, run:
> ```bash
> pip install -U transformers
> ```

### Step 4: Hugging Face Authentication
Accept the license agreement at [huggingface.co/black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).
Then log in **once** on the EC2 instance with `huggingface-cli`:
```bash
huggingface-cli login
```
This caches the gated weights locally (`~/.cache/huggingface/hub`) and your credentials with the
Hub CLI. After that, `app/pipeline.py` loads `FLUX_MODEL` straight from that local cache on every
startup (`Flux2KleinPipeline.from_pretrained(settings.FLUX_MODEL, ...)`), the same way the
exploration notebooks do - no token needs to live in `.env` or be read by the app itself.

### Step 5: Avatar Template Images
The template images already ship inside `Backend/Avatar/` (copied in from the project's original
`Avatar/` folder), so there is nothing to do here for a checkout of this repo:
```
Backend/
  Avatar/
    Saytara_male.jpg    # Male template
    Saytara_Femal.png   # Female template
    Saytara_hijab.jpg   # Hijab template
```
If you ever replace these with new artwork, keep the same filenames (or update `AVATAR_DIR`/the
filename maps in `app/pipeline.py`) and redeploy `Backend/` as a self-contained folder.

### Step 6: Configure Environment
Copy the example configuration to `.env` and adjust values if needed (defaults work out of the box):
```bash
cp .env.example .env
nano .env
```

---

## 4. Running the Service

Start the backend:
```bash
python app.py
```

> **Why `--workers 1`?**
> The model is loaded once into GPU VRAM per process (occupying 20-30+ GB VRAM). Multiple Uvicorn workers would attempt to duplicate the 9B model and exhaust GPU memory. Serializing requests in a single worker process with `asyncio.to_thread` ensures optimal memory utilization and stability.

---

## 5. API Reference

### `GET /health`
No authentication required. Returns service status and GPU VRAM statistics.

**Example Request**:
```bash
curl -X GET http://localhost:8000/health
```

**Example Response (200 OK)**:
```json
{
  "status": "ok",
  "cuda_available": true,
  "device_name": "NVIDIA A10G",
  "models_loaded": true,
  "clip_loaded": true,
  "flux_loaded": true,
  "val_worker_running": true,
  "gpu_free_vram_gb": 18.42,
  "gpu_total_vram_gb": 23.69
}
```

---

### `POST /v1/avatar`
Generates a sci-fi avatar from a portrait photo. No authentication required.

- **Form Data**:
  - `file`: Image file (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`). No server-side size limit is
    enforced - if you want one, add it at the reverse proxy/ALB layer in front of this service.

**Example Request**:
```bash
curl -X POST http://localhost:8000/v1/avatar \
  -F "file=@photo.jpg" \
  --output avatar.jpg
```

**Success Response (200 OK)**:
- The response body is *just* the generated avatar - raw JPEG bytes (`Content-Type: image/jpeg`),
  nothing else attached. It is always re-encoded as JPEG and guaranteed to be at or under
  `OUTPUT_MAX_BYTES` (default **300 KB** / `307200` bytes): quality is stepped down from 95 towards
  20 first, and if it's still over budget at the lowest quality, the image is downscaled and the
  quality ladder is retried, repeating until it fits.
- A few scoring/progress fields are echoed back as response headers (the rest is still only
  logged server-side for debugging):
  - `X-Tries-Used`: how many generation attempts were run.
  - `X-Best-Try`: which attempt (1-based) was picked as the best candidate.
  - `X-Stop-Reason`: why the retry loop stopped (`good`, `all tries`, or `budget`).
  - `X-Generation-Seconds`: total time spent generating, in seconds.
  - `X-Identity-Similarity`: ArcFace cosine similarity of the best candidate to the user's face.
  - `X-Visor-Status`: whether the visor was detected correctly on the best candidate.

**Validation Rejection (400 Bad Request)**:
Returned when a photo is rejected by Step 1 validation (multiple faces, no face, non-human subject, uncertain gender, etc.):
```json
{
  "error": "Multiple main faces (2) — upload a photo with one clear foreground person"
}
```

**Internal Error (500 Internal Server Error)**:
Returned on unexpected server errors. Full tracebacks are logged server-side only:
```json
{
  "error": "An internal server error occurred during avatar generation."
}
```

---

### `POST /v1/avatar/base64`
Same as `/v1/avatar`, but the photo is sent as a base64 string in a JSON body instead of
multipart/form-data. Same responses as above.

- **JSON Body**:
  - `image` (required): base64-encoded image bytes. A `data:image/...;base64,...` data URL is
    also accepted - the `data:...;base64,` prefix is stripped automatically.
  - `filename` (optional, default `upload.jpg`): only used to infer the file extension
    (`.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`).

**Example Request**:
```bash
curl -X POST http://localhost:8000/v1/avatar/base64 \
  -H "Content-Type: application/json" \
  -d "{\"image\": \"$(base64 -w0 photo.jpg)\", \"filename\": \"photo.jpg\"}" \
  --output avatar.jpg
```
