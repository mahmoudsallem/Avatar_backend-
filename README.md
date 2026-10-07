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

2. **Step 2: Sci-Fi Avatar Generation (`generate_avatar`)**
   - Prepares the avatar template (`Man`, `Woman`, or `Woman_Hijab`) and a head crop taken directly from the user's
     original photo (no Picture 3 visor reference image - the visor is described in the prompt text only).
   - Re-checks beard presence specifically on that head crop (CLIP + jaw-darkness heuristic), overriding Step 1's
     beard tag for prompt purposes.
   - Assembles a detailed hand-tuned prompt enforcing facial likeness and the futuristic blue wraparound visor.
   - Executes FLUX.2-klein-9B with BFS face-swap LoRA in a **single pass** (no retries).
   - **Optional validation** (`VALIDATE`, off by default): scores the one generated image via ArcFace cosine
     similarity (in a persistent CPU worker), jaw ratio, beard match, and OpenCV HSV visor detection - for response
     metadata/logging only, since it's a single pass there's nothing to retry into. Off by default because ArcFace
     gives unreliable scores on the illustrated avatar style.

3. **Concurrency Model**:
   - Because FLUX.2-klein-9B is a ~9B parameter model requiring significant GPU VRAM, generation requests are serialized behind an asynchronous GPU lock (`pipeline.gpu_lock`).
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
- A few scoring fields are echoed back as response headers (the rest is still only logged
  server-side for debugging):
  - `X-Generation-Seconds`: total time spent on the image end-to-end (Step 1 + Step 2), in seconds.
  - `X-Identity-Similarity`: ArcFace cosine similarity to the user's face (empty unless `VALIDATE=true`).
  - `X-Visor-Status`: whether the visor was detected on the generated image (`ok` or `weak`).
  - `X-Validated`: whether the image passed validation (`true`/`false`) - always reflects at least the
    visor check, even with `VALIDATE=false`.

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
