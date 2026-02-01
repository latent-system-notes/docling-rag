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
    && rm -rf /var/lib/apt/lists/*

# Create directories for venv and models (persist inside container)
RUN mkdir -p /opt/venv /opt/models

# Create venv with system-site-packages (inherits PyTorch from base image)
RUN python3 -m venv /opt/venv --system-site-packages

# Set venv as default Python
ENV PATH="/opt/venv/bin:$PATH"
ENV VIRTUAL_ENV="/opt/venv"

# Set models directory
ENV MODELS_DIR="/opt/models"

# Copy requirements first (for better Docker layer caching)
WORKDIR /workspace/app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install OCR support
RUN pip install --no-cache-dir rapidocr-onnxruntime

# Replace opencv with headless version (for container/server use)
RUN pip uninstall -y opencv-python 2>/dev/null || true && \
    pip install --no-cache-dir opencv-python-headless

# Download models during build (optional - comment out to download at runtime)
# Note: This makes the image larger but faster to start
RUN pip install --no-cache-dir huggingface_hub && \
    python -c "from sentence_transformers import SentenceTransformer; \
    import os; os.makedirs('/opt/models/embedding/paraphrase-multilingual-mpnet-base-v2', exist_ok=True); \
    m = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2'); \
    m.save('/opt/models/embedding/paraphrase-multilingual-mpnet-base-v2')"

RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download(repo_id='docling-project/docling-layout-heron', revision='main'); \
    snapshot_download(repo_id='docling-project/docling-models', revision='v2.3.0')"

# Copy application code
COPY . .

# Set environment variables for offline operation
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1
ENV TOKENIZERS_PARALLELISM=false

# Default working directory
WORKDIR /workspace/app

# Default command (can be overridden)
CMD ["bash"]
