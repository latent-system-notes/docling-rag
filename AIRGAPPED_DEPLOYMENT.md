# Airgapped Deployment Guide

This document explains how to deploy the docling-rag system in an airgapped (offline) environment.

## Overview

The system requires three types of models to work offline:

| Model | Purpose | Size |
|-------|---------|------|
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | Text embeddings | ~420 MB |
| `docling-project/docling-layout-heron` | PDF layout analysis | ~50 MB |
| `docling-project/docling-models` | Table structure detection | ~30 MB |

## Prerequisites

- Python 3.11+
- All wheel files from `requirements.txt`
- Model files (see below)

## Step 1: Export Wheels from Internet-Connected Machine

On a machine with internet access:

```bash
# Create wheels directory
mkdir -p wheels/cuda

# Download CPU wheels (default, ~0.5 GB)
pip download -r requirements.txt -d wheels/

# Download build tools (required for editable install)
pip download wheel pip hatchling editables -d wheels/
```

### For CUDA GPU Support (Optional)

If deploying to machines with NVIDIA GPUs, also download CUDA wheels:

```bash
# Download CUDA 12.1 PyTorch (~2.3 GB)
pip download torch torchvision --index-url https://download.pytorch.org/whl/cu121 -d wheels/cuda/

# Or for CUDA 11.8 (older GPUs)
pip download torch torchvision --index-url https://download.pytorch.org/whl/cu118 -d wheels/cuda/
```

### Wheels Directory Structure

```
wheels/
├── *.whl                    # CPU wheels (~0.5 GB, 300 files)
├── torch-2.10.0-*.whl       # CPU PyTorch (~108 MB)
└── cuda/                    # CUDA wheels (optional, ~2.3 GB)
    ├── torch-2.5.1+cu121-*.whl      # CUDA PyTorch (~2.3 GB)
    └── torchvision-0.20.1+cu121-*.whl
```

**Important:** CPU and CUDA PyTorch are mutually exclusive. You install ONE or the OTHER, not both.

| Target Machine | PyTorch Wheel | Total Size |
|----------------|---------------|------------|
| CPU only | `wheels/torch-*.whl` | ~0.5 GB |
| NVIDIA GPU | `wheels/cuda/torch-*.whl` | ~2.8 GB |

## Step 2: Export Models

On the internet-connected machine, run the model download:

```bash
# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Download all models using CLI
rag models --download
```

Then copy the entire `models/` directory:

```
models/
├── .cache/
│   └── hub/
│       ├── models--docling-project--docling-layout-heron/
│       │   └── snapshots/
│       │       └── 54100edecdceb65a9d8204d2478ac4cc8d4ca68b/
│       │           ├── config.json
│       │           ├── model.safetensors
│       │           └── preprocessor_config.json
│       ├── models--docling-project--docling-models/
│       │   └── snapshots/
│       │       └── fc0f2d45e2218ea24bce5045f58a389aed16dc23/
│       │           └── config.json
│       └── models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2/
│           └── snapshots/
│               └── 4328cf26390c98c5e3c738b4460a05b95f4911f5/
│                   ├── config.json
│                   ├── model.safetensors
│                   ├── modules.json
│                   ├── sentence_bert_config.json
│                   ├── special_tokens_map.json
│                   ├── tokenizer.json
│                   ├── tokenizer_config.json
│                   └── vocab.txt
└── embedding/
    └── paraphrase-multilingual-mpnet-base-v2/
        └── (same files as above)
```

## Step 3: Transfer to Airgapped Machine

Transfer the following to the airgapped machine:

| Item | Size | Required |
|------|------|----------|
| `wheels/` directory (CPU) | ~0.5 GB | Yes |
| `wheels/cuda/` directory | ~2.3 GB | Only for NVIDIA GPU |
| `models/` directory | ~500 MB | Yes |
| Project source code | ~1 MB | Yes |
| `requirements.txt` | ~10 KB | Yes |

## Step 4: Install on Airgapped Machine

### Option A: CPU Installation (Default)

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Upgrade pip and install build tools
pip install --no-index --find-links=wheels/ pip wheel setuptools hatchling editables

# Install all dependencies from local wheels
pip install --no-index --find-links=wheels/ -r requirements.txt

# Install the project itself
pip install --no-index --find-links=wheels/ -e .
```

### Option B: CUDA GPU Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Upgrade pip and install build tools
pip install --no-index --find-links=wheels/ pip wheel setuptools hatchling editables

# Install CUDA PyTorch FIRST (from cuda folder)
pip install --no-index --find-links=wheels/cuda/ torch torchvision

# Install remaining dependencies (skips torch since already installed)
pip install --no-index --find-links=wheels/ -r requirements.txt

# Install the project itself
pip install --no-index --find-links=wheels/ -e .

# Configure GPU usage
echo "RAG_DEVICE=cuda" >> .env
```

## Step 5: Configure Environment

Create `.env` file:

```bash
# Offline mode is ENABLED BY DEFAULT
# The system automatically sets:
#   HF_HUB_OFFLINE=1
#   TRANSFORMERS_OFFLINE=1
#   HF_DATASETS_OFFLINE=1
# To disable (for model downloads), set:
# RAG_OFFLINE_MODE=false

# Device configuration (CPU/GPU)
# Options: cpu, cuda (NVIDIA), mps (Apple Silicon), auto
RAG_DEVICE=cpu

# ChromaDB storage location
RAG_CHROMA_PERSIST_DIR=./data/chroma

# Disable OCR if not needed (faster processing)
RAG_ENABLE_OCR=false

# Or configure OCR engine (auto, rapidocr, tesseract)
RAG_OCR_ENGINE=auto
RAG_OCR_LANGUAGES=eng+ara
```

**Note:** Offline mode is enabled by default. The `rag models --download` command temporarily disables it to download models.

## Step 6: Verify Installation

```bash
# Check device configuration
rag device

# Verify models
rag models

# Expected output:
# Models Status
# ├── Embedding Model: OK
# ├── Docling Layout: OK
# └── Docling Tables: OK

# Test ingestion
rag ingest test.pdf --dry-run

# Test query
rag query "test query"
```

## Offline Mode Enforcement

The system enforces offline mode through multiple mechanisms:

1. **Environment Variables**:
   - `HF_HUB_OFFLINE=1` - Prevents HuggingFace Hub access
   - `TRANSFORMERS_OFFLINE=1` - Prevents Transformers downloads

2. **Code-Level Enforcement**:
   - `local_files_only=True` when loading embedding model
   - `enable_remote_services=False` in Docling pipeline options
   - `HF_HOME` set to local `models/.cache` directory

3. **Model Path Configuration**:
   - Layout model configured with explicit `model_path`
   - All models loaded from local cache

## Directory Structure for Deployment

```
docling-rag/
├── .env                    # Environment configuration
├── cli/                    # CLI application
├── src/                    # Source code
├── data/
│   ├── chroma/            # ChromaDB storage
│   ├── checkpoints/       # Ingestion checkpoints
│   └── bm25_index.db      # BM25 search index
├── models/
│   ├── .cache/hub/        # HuggingFace model cache
│   └── embedding/         # Local embedding model
├── docs/                   # Documents to ingest
├── requirements.txt        # Python dependencies
└── wheels/                 # Local wheel files
    ├── *.whl              # CPU wheels
    └── cuda/              # CUDA wheels (optional)
```

## Troubleshooting

### "Cannot find model" errors

Verify models exist:
```bash
ls -la models/.cache/hub/
```

### Network connection errors

Ensure environment variables are set:
```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

### Missing dependencies

Check all wheels were transferred:
```bash
pip check
```

### CUDA not detected

Verify CUDA PyTorch was installed:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'PyTorch version: {torch.__version__}')"
```

If output shows `torch==X.X.X+cpu`, you installed CPU PyTorch. Reinstall with CUDA wheels:
```bash
pip uninstall torch torchvision -y
pip install --no-index --find-links=wheels/cuda/ torch torchvision
```

## Model Verification Script

Save as `verify_offline.py`:

```python
from pathlib import Path
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

models_dir = Path("models/.cache/hub")

required_models = [
    "models--docling-project--docling-layout-heron",
    "models--docling-project--docling-models",
    "models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2"
]

print("Verifying offline models...")
for model in required_models:
    model_path = models_dir / model / "snapshots"
    if model_path.exists() and list(model_path.iterdir()):
        print(f"  [OK] {model}")
    else:
        print(f"  [MISSING] {model}")

# Test model loading
print("\nTesting model loading...")
try:
    from sentence_transformers import SentenceTransformer
    embed_path = Path("models/embedding/paraphrase-multilingual-mpnet-base-v2")
    model = SentenceTransformer(str(embed_path), local_files_only=True)
    result = model.encode(["test"])
    print(f"  [OK] Embedding model loaded, vector dim: {len(result[0])}")
except Exception as e:
    print(f"  [FAIL] Embedding model: {e}")

# Test PyTorch/CUDA
print("\nTesting PyTorch...")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  [FAIL] PyTorch: {e}")

print("\nVerification complete.")
```

Run with:
```bash
python verify_offline.py
```

## GPU Acceleration

The system supports GPU acceleration for embedding generation, which significantly speeds up ingestion.

### Check Device Status

```bash
rag device
```

Output (CPU):
```
         Device Configuration
 Setting                Value
 Current Device Setting cpu
 CUDA (NVIDIA GPU)      Not available
 MPS (Apple Silicon)    Not available
 CPU                    Available
 PyTorch Default Device cpu
```

Output (CUDA):
```
         Device Configuration
 Setting                Value
 Current Device Setting cuda
 CUDA (NVIDIA GPU)      Available (1 GPU(s))
   GPU 0                NVIDIA GeForce RTX 3080 (10.0 GB)
 MPS (Apple Silicon)    Not available
 CPU                    Available
 PyTorch Default Device cuda
```

### Enable GPU

Set in `.env` file or environment variable:

```bash
# NVIDIA GPU (CUDA)
RAG_DEVICE=cuda

# Apple Silicon GPU (MPS)
RAG_DEVICE=mps

# Auto-detect best available device
RAG_DEVICE=auto

# CPU only (default)
RAG_DEVICE=cpu
```

### Performance Comparison

| Device | Embedding Speed | Recommended For |
|--------|----------------|-----------------|
| CPU | ~50 chunks/sec | Small documents, testing |
| CUDA | ~500+ chunks/sec | Large document sets |
| MPS | ~200+ chunks/sec | Apple Silicon Macs |

## OCR Support (Airgapped)

The system supports OCR for image-based PDFs using RapidOCR. All required models are embedded in the wheel files.

### OCR Libraries Included

| Library | Purpose | Size | Airgap Ready |
|---------|---------|------|--------------|
| `rapidocr` | OCR engine (ONNX-based) | 14.4 MB | Yes (models embedded) |
| `onnxruntime` | Model inference | 12.8 MB | Yes |
| `opencv-python` | Image processing | 38.3 MB | Yes |
| `pillow` | Image handling | 6.7 MB | Yes |
| `pytesseract` | Tesseract wrapper | 0.1 MB | Requires Tesseract binary |

### RapidOCR vs Tesseract

| Feature | RapidOCR | Tesseract |
|---------|----------|-----------|
| Airgap ready | Yes (all models in wheel) | No (needs binary install) |
| Setup | Just pip install | Install binary + pip |
| Languages | Chinese, English (built-in) | 100+ languages |
| Accuracy | Good for printed text | Better for complex layouts |

**Recommendation:** Use RapidOCR for airgapped environments (default).

### Enable OCR

In `.env` file:

```bash
# Enable OCR for image-based PDFs
RAG_ENABLE_OCR=true

# OCR engine (auto uses rapidocr if available)
RAG_OCR_ENGINE=auto  # or: rapidocr, tesseract

# Languages (for tesseract only)
RAG_OCR_LANGUAGES=eng+ara
```

### Verify OCR Installation

```bash
python -c "from rapidocr import RapidOCR; ocr = RapidOCR(); print('RapidOCR ready')"
```

## PDF Processing Libraries

All PDF processing libraries are included and work offline:

| Library | Purpose | Size |
|---------|---------|------|
| `pymupdf` | Fast PDF parsing, rendering | 17.6 MB |
| `pypdf` | Pure Python PDF parsing | 0.3 MB |
| `pypdfium2` | PDF rendering (Chrome's engine) | 3.0 MB |

These are used by Docling for document conversion and require no additional downloads.

## Quick Reference: Installation Commands

### CPU Machine
```bash
python -m venv .venv && .venv\Scripts\activate
pip install --no-index --find-links=wheels/ pip wheel setuptools hatchling editables
pip install --no-index --find-links=wheels/ -r requirements.txt
pip install --no-index --find-links=wheels/ -e .
```

### CUDA Machine
```bash
python -m venv .venv && .venv\Scripts\activate
pip install --no-index --find-links=wheels/ pip wheel setuptools hatchling editables
pip install --no-index --find-links=wheels/cuda/ torch torchvision
pip install --no-index --find-links=wheels/ -r requirements.txt
pip install --no-index --find-links=wheels/ -e .
echo "RAG_DEVICE=cuda" >> .env
```
