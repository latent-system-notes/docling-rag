# Docling RAG

A document ingestion and retrieval system using Docling for document processing and PostgreSQL + pgvector for vector storage with built-in full-text search.

## Setup

### PostgreSQL

Start PostgreSQL with pgvector:
```bash
docker compose up postgres -d
```

This automatically creates the database schema via `docker/init.sql`.

For local development without Docker, install PostgreSQL with the pgvector extension and run `docker/init.sql` manually.

### Application

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download required models:
   ```bash
   python -m cli models --download
   ```

3. Create an environment file (e.g., `.env.test`):
   ```bash
   DOCUMENTS_DIR=./docs/my-project
   DATA_DIR=./data
   MODELS_DIR=./models

   POSTGRES_HOST=localhost
   POSTGRES_PORT=5432
   POSTGRES_DB=docling_rag
   POSTGRES_USER=docling
   POSTGRES_PASSWORD=docling
   POSTGRES_POOL_MIN=2
   POSTGRES_POOL_MAX=10
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

## Docker Deployment

```bash
# Start everything (PostgreSQL + services)
docker compose up -d

# Start only PostgreSQL
docker compose up postgres -d

# Ingest documents
docker compose exec techpub python -m cli ingest techpub
```

Services (`techpub`, `safety`) automatically wait for PostgreSQL to be healthy before starting.

## Configuration

All configuration is done via environment variables in `.env.<env>` files.

| Variable | Default | Description |
|---|---|---|
| `DOCUMENTS_DIR` | `./documents` | Path to the documents directory |
| `DATA_DIR` | `./data` | Path to persistent data |
| `MODELS_DIR` | `/opt/models` | Path to downloaded models |
| `INCLUDE_FOLDERS` | *(unset)* | Pipe-separated folder names to include during ingestion |
| `MCP_SERVER_NAME` | `docling-rag` | MCP server display name |
| `MCP_PORT` | `9090` | MCP server port |
| `RAG_DEVICE` | `cuda` | Compute device: `cuda`, `mps`, or `cpu` |
| `POSTGRES_HOST` | `localhost` | PostgreSQL host (`postgres` in Docker) |
| `POSTGRES_PORT` | `5432` | PostgreSQL port |
| `POSTGRES_DB` | `docling_rag` | Database name |
| `POSTGRES_USER` | `docling` | Database user |
| `POSTGRES_PASSWORD` | `docling` | Database password |
| `POSTGRES_POOL_MIN` | `2` | Minimum connection pool size |
| `POSTGRES_POOL_MAX` | `10` | Maximum connection pool size |

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
