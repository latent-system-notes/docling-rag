# Offline Mode Implementation - Complete Fix

## Problem Summary

The application claimed to support offline mode but had **3 critical issues**:

1. **SentenceTransformer** - Missing `local_files_only=True` parameter
2. **HybridChunker Tokenizer** - Loading tokenizer from HuggingFace ID instead of local path
3. **Docling Layout Models** - Trying to download models from HuggingFace at runtime

## Solution Overview

We implemented a **multi-layer offline enforcement** strategy that works at the code level rather than relying on environment variables.

## Changes Made

### 1. src/config.py
**REMOVED** global offline environment variables that were preventing downloads:
```python
# BEFORE (prevented downloads):
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# AFTER (allows downloads when needed):
# Offline mode enforced via local_files_only=True in loading functions
```

### 2. src/utils.py
**ADDED** offline enforcement to model loading:
```python
# Embedding model loading
model = SentenceTransformer(
    str(local_path),
    device=settings.device,
    local_files_only=True,  # Enforces offline mode
    trust_remote_code=False  # Security
)
```

**ADDED** Docling model download function:
```python
def download_docling_models() -> None:
    """Download Docling layout models for offline PDF processing."""
    snapshot_download(
        repo_id="docling-project/docling-layout-heron",
        revision="main",
        local_dir=str(layout_path),
        local_dir_use_symlinks=False
    )
```

**UPDATED** model paths to include Docling:
```python
return {
    "embedding": settings.models_dir / "embedding" / embedding_name,
    "docling_layout": settings.models_dir / "docling" / "layout"
}
```

### 3. src/ingestion/chunker.py
**FIXED** tokenizer to load from local path:
```python
# Load tokenizer with offline mode
hf_tokenizer = AutoTokenizer.from_pretrained(
    str(local_model_path),
    local_files_only=True,  # Enforces offline
    trust_remote_code=False
)

# Wrap in Docling's tokenizer
tokenizer = HuggingFaceTokenizer(
    tokenizer=hf_tokenizer,
    max_tokens=max_tokens
)
```

### 4. src/ingestion/document.py
**ADDED** offline configuration for Docling:
```python
# Disable remote services
pipeline_options.enable_remote_services = False

# Configure local model path
if local_docling_path.exists():
    pipeline_options.layout_options.model_spec = LayoutModelConfig(
        name="docling_layout_heron",
        repo_id="docling-project/docling-layout-heron",
        revision="main",
        model_path=str(local_docling_path),  # Use local path
        supported_devices=[AcceleratorDevice.CPU]
    )
```

### 5. cli/cli.py
**UPDATED** model download command to download all models:
```python
# Download embedding model
download_embedding_model()

# Download Docling layout models
download_docling_models()
```

### 6. README.md
**UPDATED** offline mode documentation with accurate implementation details.

## Offline Mode Enforcement Layers

| Layer | Mechanism | Location |
|-------|-----------|----------|
| **Embedding Loading** | `local_files_only=True` | utils.py:76 |
| **Tokenizer Loading** | `local_files_only=True` | chunker.py:31 |
| **Security** | `trust_remote_code=False` | utils.py:77, chunker.py:32 |
| **Docling Services** | `enable_remote_services=False` | document.py:160 |
| **Docling Models** | `model_path=str(local_path)` | document.py:171 |

## How to Use

### Step 1: Download Models (One-Time, Requires Internet)
```bash
rag models --download
```

This downloads:
- **Embedding model**: ~420MB (`paraphrase-multilingual-mpnet-base-v2`)
- **Docling layout model**: ~500MB (`docling-layout-heron`)

### Step 2: Verify Models
```bash
rag models --verify
```

Expected output:
```
Model Status
 Model           Status
 embedding       Downloaded
 docling_layout  Downloaded
```

### Step 3: Use Offline
```bash
# Disconnect from internet or enable airplane mode

# Ingest documents
rag sync document.pdf

# Query
rag query "What is this about?"
```

## Error Messages

The system now provides clear error messages if models are missing:

```
OFFLINE MODE: Cannot download models because local_files_only=True.
Run 'rag models --download' first (requires internet connection).
```

## Technical Details

### Why Code-Level Enforcement?

We initially tried using environment variables (`HF_HUB_OFFLINE=1`) globally, but this prevented the download command from working. The solution is to:

1. **NOT set global offline environment variables**
2. **Enforce offline mode via `local_files_only=True` parameter** in each loading function
3. **Allow downloads to work normally** when explicitly requested

This approach gives us fine-grained control: downloads work when needed, but all other operations strictly use local files.

### Model Storage

```
./models/
├── embedding/
│   └── paraphrase-multilingual-mpnet-base-v2/
│       ├── config.json
│       ├── tokenizer.json
│       ├── model.safetensors
│       └── ...
└── docling/
    └── layout/
        ├── config.json
        ├── model.safetensors
        └── ...
```

## Testing Checklist

- [x] Models download successfully
- [x] Models verify correctly
- [x] Embedding loading works offline
- [x] Tokenizer loading works offline
- [x] PDF ingestion works offline (Docling models)
- [x] Querying works offline
- [x] Clear error messages when models missing

## Future Improvements

1. Add progress bars for large model downloads
2. Support for model updates (check for newer versions)
3. Model size optimization (quantization, pruning)
4. Automatic cleanup of old model versions
