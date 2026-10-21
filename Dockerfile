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
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install PyTorch 2.11 with CUDA 13.0 (Required for Blackwell sm_120 and transformers >= v2.6 requirement)
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130

# Install Python requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Optional build arg: install prebuilt Vitoom Nunchaku wheel for FLUX_BACKEND=nunchaku
ARG INSTALL_NUNCHAKU=true
RUN if [ "$INSTALL_NUNCHAKU" = "true" ]; then \
        pip install --no-cache-dir huggingface-hub && \
        python -c "from huggingface_hub import hf_hub_download; import subprocess; \
whl = hf_hub_download(repo_id='tonera/vitoom-nunchaku', filename='nunchaku-1.3.0.dev20260622+cu13.0torch2.11-cp311-cp311-linux_x86_64.whl'); \
subprocess.check_call(['pip', 'install', '--no-deps', whl])" || \
        echo "Warning: Nunchaku wheel install skipped or failed. Falling back to diffusers backend."; \
    fi

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

# Expose SageMaker service port
EXPOSE 8080

# Container healthcheck (SageMaker /ping endpoint)
HEALTHCHECK --interval=30s --timeout=15s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

# SageMaker executes: docker run <image> serve
ENTRYPOINT ["python", "app.py"]
CMD ["serve"]
