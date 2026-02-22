# Docling RAG Dockerfile
# Base: NVIDIA PyTorch container (includes PyTorch, CUDA support)
FROM nvcr.io/nvidia/pytorch:26.01-py3

# Install system dependencies for OpenCV/RapidOCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libxcb-shm0 \
    libxcb-render0 \
    libgl1 \
    libglib2.0-0t64 \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv (fast Rust-based package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Create directories for venv and models (persist inside container)
RUN mkdir -p /opt/venv /opt/models

# Create venv with system-site-packages (inherits PyTorch from base image)
RUN uv venv /opt/venv --system-site-packages --python python3

# Set venv as default Python
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Set models directory
ENV MODELS_DIR="/opt/models"

# Copy requirements and constraints first (for better Docker layer caching)
WORKDIR /workspace/app
COPY requirements.txt constraints.txt ./

# Verify PyTorch from base image is intact before installing anything
RUN python -c "import torch; print(f'Base image PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Install Python dependencies using uv (much faster than pip under QEMU)
RUN uv pip install --no-cache -r requirements.txt

# Remove PyPI torch/torchvision/torchaudio from venv — the NVIDIA CUDA-optimized
# builds in system-site-packages must take precedence
RUN uv pip uninstall torch torchvision torchaudio 2>/dev/null || true

# Install OCR support
RUN uv pip install --no-cache rapidocr-onnxruntime

# Replace opencv with headless version (for container/server use)
RUN uv pip uninstall opencv-python 2>/dev/null || true && \
    uv pip install --no-cache opencv-python-headless

# Models are NOT baked in — mount them at runtime via volume:
#   -v /path/to/models:/opt/models

# Verify PyTorch was NOT overridden by pip dependency resolution
RUN python -c "import torch; print(f'Final PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# App code is also volume-mounted at runtime — don't COPY it
# This keeps the image lean (only base + venv) and allows code updates without rebuild

# Set environment variables
ENV TOKENIZERS_PARALLELISM=false

# Default working directory
WORKDIR /workspace/app

# Default command (can be overridden)
CMD ["bash"]
