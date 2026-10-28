# ==============================================================================
# Saytara Avatar Generation Backend - Production Dockerfile
# ==============================================================================
# Base image: Python 3.11 with Debian Bookworm
FROM python:3.11-slim-bookworm

# System configuration & performance flags
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Model paths inside container
ENV HF_HOME=/models/huggingface \
    DEEPFACE_HOME=/models/deepface \
    CLIP_DIR=/models/clip-vit-base-patch32

# Application networking (Port 8080 is standard for SageMaker)
ENV HOST=0.0.0.0 \
    PORT=8080

# Install required system packages for OpenCV, DeepFace, Git, and build tools
# apt-get upgrade picks up Debian security patches already released for the base image
# (e.g. libssl3) that aren't baked into the python:3.11-slim-bookworm tag yet.
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ninja-build \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install PyTorch 2.11 with CUDA 13.0 (Required for Blackwell sm_120 and transformers >= v2.6 requirement)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Optional build arg: install prebuilt Vitoom Nunchaku wheel for FLUX_BACKEND=nunchaku
ARG INSTALL_NUNCHAKU=true
RUN if [ "$INSTALL_NUNCHAKU" = "true" ]; then \
        pip install --no-cache-dir huggingface-hub && \
        python -c "from huggingface_hub import hf_hub_download; import subprocess; \
whl = hf_hub_download(repo_id='tonera/vitoom-nunchaku', filename='nunchaku-1.3.0.dev20260622+cu13.0torch2.11-cp311-cp311-linux_x86_64.whl'); \
subprocess.check_call(['pip', 'install', '--no-deps', whl])" || \
        echo "Warning: Nunchaku wheel install skipped or failed. Falling back to diffusers backend."; \
    fi

# 2026-10-24: flash-attn-4/flash-attn/xformers deliberately NOT installed. Confirmed this session
# (live Dynamo trace) that FLUX_BACKEND=nunchaku (current production setting) never touches
# PyTorch's attention dispatch at all - Nunchaku's transformer uses its own fused quantized kernels
# (attention_fp16/gemm_w4a4). "flash-attn-4" isn't a real PyPI package (always 404s); building
# mainline flash-attn from source takes 20-40+ min with zero runtime benefit here, and measurably
# degraded live EC2 throughput (8.7->2.6 avatars/min) when it ran as a background job earlier today.
# Only revisit this if FLUX_BACKEND is ever switched to "diffusers".

# Create directory structure for models and runtime storage
# Symlink /Models to /models so legacy default paths resolve automatically
RUN mkdir -p /models/huggingface \
             /models/deepface \
             /models/clip-vit-base-patch32 \
             /Models \
             /app/crop \
             /app/output \
             /app/rejected \
             /app/output/_val_tmp \
    && ln -sf /models/clip-vit-base-patch32 /Models/clip-vit-base-patch32

# Copy application source code and avatar templates
COPY app/ ./app/
COPY Avatar/ ./Avatar/
COPY app.py ./
COPY install_models.py ./
COPY saytara_rust_core/ ./saytara_rust_core/

# Compile native Rust extension module with Maturin
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal && \
    export PATH="$HOME/.cargo/bin:$PATH" && \
    pip install --no-cache-dir maturin && \
    cd saytara_rust_core && \
    maturin build --release --strip && \
    pip install --no-cache-dir target/wheels/*.whl && \
    cd .. && rm -rf "$HOME/.cargo" "$HOME/.rustup" ./saytara_rust_core/target

# Run as a non-root user. The GPU device nodes NVIDIA's container runtime exposes are accessible
# to any container user, so this doesn't need special GPU group membership.
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser \
    && chown -R appuser:appuser /app /models /Models
USER appuser

# Expose SageMaker service port
EXPOSE 8080

# Container healthcheck (SageMaker /ping endpoint)
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/ping || exit 1

# SageMaker executes: docker run <image> serve
ENTRYPOINT ["python", "app.py"]
CMD ["serve"]
