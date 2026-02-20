# Docling RAG

A document ingestion and retrieval system using Docling for document processing and ChromaDB for vector storage.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download required models:
   ```bash
   python -m cli models --download
   ```

3. Create an environment file (e.g., `.env.safety`):
   ```bash
   DOCUMENTS_DIR=./docs/my-project
   DATA_DIR=./data
   MODELS_DIR=./models
   ```

## CLI Commands

All commands are run via `python -m cli <command> <env>`, where `<env>` refers to the suffix of your `.env.<env>` file.

```bash
python -m cli ingest <env>              # Ingest documents from DOCUMENTS_DIR
python -m cli list <env>                # List indexed documents
python -m cli query <env> "question"    # Query the knowledge base
python -m cli remove <env> <doc_id>     # Remove a document
python -m cli stats <env>               # Show database statistics
python -m cli reset <env>               # Reset the database
python -m cli mcp <env>                 # Start MCP server
python -m cli models --download         # Download models
python -m cli models --verify           # Verify models are installed
```

### Ingest Options

```bash
python -m cli ingest <env> --dry-run              # Preview which files would be ingested
python -m cli ingest <env> --force                 # Re-ingest already indexed documents
python -m cli ingest <env> --no-recursive          # Only process top-level files
python -m cli ingest <env> --folders "Folder A|Folder B"  # Only ingest from specific folders
```

### List Options

```bash
python -m cli list <env> --full-path               # Show full file paths
python -m cli list <env> --limit 10 --offset 20    # Paginate results
```

## Configuration

All configuration is done via environment variables in `.env.<env>` files.

| Variable | Default | Description |
|---|---|---|
| `DOCUMENTS_DIR` | `./documents` | Path to the documents directory |
| `DATA_DIR` | `./data` | Path to persistent data (ChromaDB, BM25 index) |
| `MODELS_DIR` | `/opt/models` | Path to downloaded models |
| `INCLUDE_FOLDERS` | *(unset)* | Pipe-separated folder names to include during ingestion |
| `MCP_SERVER_NAME` | `docling-rag` | MCP server display name |
| `MCP_PORT` | `9090` | MCP server port |
| `RAG_DEVICE` | `cuda` | Compute device: `cuda`, `mps`, or `cpu` |

### Folder Filtering

When `DOCUMENTS_DIR` contains many subfolders but only specific ones should be ingested, use `INCLUDE_FOLDERS` to filter by folder name. Folder names are pipe-separated (`|`) because names may contain spaces or commas.

In your `.env` file:
```bash
INCLUDE_FOLDERS="01 Vendor Publication|02 Document Publications"
```

Or override per-run via CLI:
```bash
python -m cli ingest safety --folders "01 Vendor Publication|02 Document Publications"
```

When `INCLUDE_FOLDERS` is not set, all files under `DOCUMENTS_DIR` are processed (backward compatible). When set, only files inside matching folders are included — files sitting directly in the `DOCUMENTS_DIR` root are excluded. Folder names are matched using glob patterns, so wildcards like `01*` are supported.

## GPU Acceleration

By default, the system runs on CPU. To enable GPU acceleration, add to your `.env` file:

```bash
# For NVIDIA GPU (CUDA)
RAG_DEVICE=cuda

# For Apple Silicon (M1/M2/M3)
RAG_DEVICE=mps
```

### CUDA Prerequisites

1. NVIDIA GPU with CUDA support
2. CUDA toolkit installed
3. PyTorch with CUDA support:
   ```bash
   pip uninstall torch
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

### Verify GPU Availability

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## Supported File Types

PDF, DOCX, PPTX, XLSX, HTML, Markdown, PNG, JPG, TIFF, WAV, MP3
