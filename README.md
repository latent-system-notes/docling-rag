# Docling RAG System

Production-ready RAG system with IBM Docling and ChromaDB.

## Features

- **Multi-format document processing**: PDF, DOCX, PPTX, XLSX, HTML, images (OCR), audio (ASR)
- **Advanced chunking**: Docling's HybridChunker for intelligent document segmentation
- **Vector search**: ChromaDB for efficient similarity search
- **Multilingual embeddings**: Support for 50+ languages including Arabic
- **Context retrieval**: Returns relevant chunks for external LLM reasoning (e.g., Claude)
- **CPU-only operation**: Explicitly configured to use CPU only (no GPU/CUDA)
- **Dual interfaces**: MCP server + CLI

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .

# For development
pip install -e ".[dev]"
```

### Configuration

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key settings:
- `RAG_ENABLE_OCR/ASR`: Enable/disable OCR and ASR processing
- `RAG_MODELS_DIR`: Directory for offline models (default: `./models`)
- `RAG_CHROMA_MODE`: Choose `persistent` (local SQLite) or `http` (server mode)
- `RAG_DEFAULT_TOP_K`: Number of context chunks to retrieve (default: 5)
- `RAG_DEVICE`: Always set to `cpu` (GPU usage is disabled)

### CPU-Only Mode

**This system is explicitly configured to use CPU only.** All processing operations are forced to run on CPU:

✅ **Enforced at multiple levels:**
1. **PyTorch**: Environment variables set to disable CUDA (`CUDA_VISIBLE_DEVICES=""`)
2. **Sentence Transformers**: Device explicitly set to `cpu`
3. **Docling**: AcceleratorOptions configured with `AcceleratorDevice.CPU`

Even if you have a GPU/CUDA available, the system will **not** use it. This ensures:
- Consistent behavior across all environments
- No unexpected GPU memory usage
- Simpler deployment (no CUDA dependencies)
- Lower costs in cloud environments

### ChromaDB Storage Modes

The system supports two storage modes for ChromaDB:

#### Persistent Mode (Default)

Direct SQLite file access. Simple setup, but has limitations:
- Single process access only
- File locking issues with concurrent access
- Slower shutdown times

```bash
# In .env:
RAG_CHROMA_MODE=persistent
RAG_CHROMA_PERSIST_DIR=./data/chroma
```

#### HTTP Mode (Recommended)

Client-server architecture for production use:
- **No file locking** - Multiple processes can access simultaneously
- **Fast shutdown** - Instant client disconnect
- **Better reliability** - Server handles all file access
- **Concurrent operations** - MCP server + CLI can run together

**Setup:**

1. **Start ChromaDB server:**
```bash
# Install ChromaDB server (one-time)
pip install chromadb

# Start server (point to existing data directory)
chroma run --host localhost --port 8000 --path ./data/chroma
```

2. **Configure your application:**
```bash
# In .env:
RAG_CHROMA_MODE=http
RAG_CHROMA_SERVER_HOST=localhost
RAG_CHROMA_SERVER_PORT=8000
RAG_CHROMA_SERVER_SSL=false
RAG_CHROMA_SERVER_API_KEY=  # Optional authentication
```

3. **Test concurrent access:**
```bash
# Terminal 1: Start MCP server
rag serve --mcp

# Terminal 2: Run queries simultaneously
rag query-cmd "test query"
```

**Docker Setup (Production):**
```bash
docker run -d \
  --name chromadb \
  -p 8000:8000 \
  -v ./data/chroma:/chroma/chroma \
  chromadb/chroma:latest
```

**Benefits:**
- Eliminates SQLite file locking errors
- MCP server shuts down instantly (no database cleanup)
- Multiple clients can ingest/query concurrently
- Can run ChromaDB on a remote server for scalability

**Migration:**
No data migration needed! The ChromaDB server reads your existing SQLite database from `./data/chroma`.

### Offline Model Setup

This system runs completely offline after initial setup. Download the embedding model once:

```bash
# Download embedding model (one-time setup, ~420MB)
rag models --download

# Verify model is downloaded
rag models --verify

# View model path
rag models --info
```

**Model Storage (based on .env settings):**
- Embedding: `./models/embedding/{model_name}/` (~420MB with default model)

**How it works:**
1. Model identifier in `.env` (e.g., `RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) is used to download from HuggingFace
2. Model is saved locally to `./models/` directory
3. The system extracts the model name from the identifier and loads from local path
4. After download, all operations work without internet connection

### CLI Usage

```bash
# Sync documents (works with files or folders)
rag sync document.pdf                   # Single file
rag sync ./documents                    # Entire folder (auto-detected, recursive)
rag sync ./docs --no-recursive          # Folder without subdirectories
rag sync ./documents --force            # Force re-ingest all files
rag sync ./documents --dry-run          # Preview changes without applying

# Query and retrieve relevant context
rag query "What is machine learning?"
rag query "Explain the concept" --top-k 10

# View statistics
rag stats

# Show configuration
rag config-show

# Reset system
rag reset

# Start MCP server
rag serve --mcp
```

### Smart Syncing

The `sync` command automatically:
- **Handles files and folders** - Works with single files or entire directories
- **Filters file types** - Only processes supported formats (PDF, DOCX, PPTX, XLSX, HTML, MD, images, audio)
- **Detects changes** - Finds new, modified, and deleted files
- **Excludes system files** - Automatically ignores hidden files, temp files, and backups
- **Shows detailed progress** - Per-file status with chunk counts
- **Preview mode** - Use `--dry-run` to see changes before applying

**Supported file types:**
- Documents: `.pdf`, `.docx`, `.pptx`, `.xlsx`
- Web: `.html`, `.htm`
- Markup: `.md`
- Images (OCR): `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`
- Audio (ASR): `.wav`, `.mp3`

**Automatically excluded:**
- Hidden files (`.*, __*`)
- Temp files (`*.tmp`, `*.temp`, `~*`)
- Backups (`*.bak`, `*.backup`)
- System files (`Thumbs.db`, `.DS_Store`)

## Project Status

**Phase 1: Core Infrastructure** ✅ Complete
**Phase 2: Document Processing** ✅ Complete
**Phase 3: Embeddings & Storage** ✅ Complete
**Phase 4: Retrieval & Generation** ✅ Complete
**Phase 5: Ingestion Pipeline** ✅ Complete
**Phase 6: Interfaces (MCP & CLI)** ✅ Complete

🎉 **Implementation Complete!** Ready for testing and deployment.

## Architecture

The codebase is organized into **4 modules** with **10 Python files**, optimized for beginner readability with extensive inline documentation:

```
src/
├── models.py              # Data models + Exceptions
├── config.py              # Settings (ChromaDB modes) + Logger
├── utils.py               # Utilities: Language, Embeddings, Model downloading
├── query.py               # 🎯 Main entry point - Query orchestration
│
├── ingestion/             # Document processing pipeline (3 files)
│   ├── document.py        # Load documents + extract metadata
│   ├── chunker.py         # Break docs into chunks
│   └── pipeline.py        # Orchestrate: load → chunk → embed → store
│
├── storage/               # Vector database (1 file)
│   └── chroma_client.py   # ChromaDB client factory (HTTP/persistent modes) + operations
│
├── retrieval/             # Search (1 file)
│   └── search.py          # Vector similarity search
│
└── mcp/                   # MCP server (1 file)
    └── server.py          # FastMCP server endpoints
```

### Design Principles

- **Beginner-friendly**: Extensive inline comments explaining WHY, not just WHAT
- **Consolidated**: Related functionality grouped (loader + converter = document.py)
- **Clear flow**: Linear pipeline easy to follow (load → chunk → embed → store)
- **Minimal abstraction**: No unnecessary layers, direct and simple
- **Good documentation**: See `docs/HOW_IT_WORKS.md` for detailed explanations

### Module Overview

| Module | Files | Purpose |
|--------|-------|---------|
| **ingestion** | 3 | Document loading, chunking, and pipeline |
| **storage** | 1 | ChromaDB client factory (HTTP/persistent) + operations |
| **retrieval** | 1 | Vector similarity search |
| **mcp** | 1 | MCP server for Claude Desktop |

**Top-level files:**
- `models.py`: All data structures and exceptions
- `config.py`: Settings (including ChromaDB modes) and logger
- `utils.py`: Language detection, embeddings, model downloading
- `query.py`: **🎯 Main entry point** - start reading here!

### For Beginners

**New to RAG?** Start here:
1. Read `docs/HOW_IT_WORKS.md` for a complete beginner's guide
2. Read `src/query.py` - it's the main orchestrator with clear comments
3. Follow the code flow: query.py → search.py
4. Try it: `rag sync paper.pdf` then `rag query "what is this about?"`

## Configuration Reference

See `.env.example` for all available settings. All settings can be overridden via:
1. Environment variables (prefix: `RAG_`)
2. Command-line arguments
3. `.env` file

## Development

```bash
# Lint
ruff check src/

# Type check
mypy src/
```

## License

MIT
