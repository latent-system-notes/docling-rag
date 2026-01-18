# Dockling RAG System

Production-ready RAG system with IBM Docling, ChromaDB, and IBM Granite.

## Features

- **Multi-format document processing**: PDF, DOCX, PPTX, XLSX, HTML, images (OCR), audio (ASR)
- **Advanced chunking**: Docling's HybridChunker for intelligent document segmentation
- **Vector search**: ChromaDB for efficient similarity search
- **Multilingual embeddings**: Support for 50+ languages including Arabic
- **Configurable answer modes**:
  - `granite`: Granite generates final answer
  - `context_only`: Return chunks for external LLM reasoning (e.g., Claude)
  - `both`: Return context + Granite's answer
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
- `RAG_ANSWER_MODE`: Choose `granite`, `context_only`, or `both`
- `RAG_GRANITE_QUANTIZATION`: Use `4bit` or `8bit` for lower memory
- `RAG_ENABLE_OCR/ASR`: Enable/disable OCR and ASR processing
- `RAG_MODELS_DIR`: Directory for offline models (default: `./models`)
- `RAG_CHROMA_MODE`: Choose `persistent` (local SQLite) or `http` (server mode)

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

This system runs completely offline after initial setup. Download models once:

```bash
# Download all models (one-time setup, ~5.5GB total)
rag models --download

# Verify models are downloaded
rag models --verify

# View model paths
rag models --info
```

**Model Storage (based on .env settings):**
- Embedding: `./models/embedding/{model_name}/` (~420MB with default model)
- Granite LLM: `./models/granite/{model_name}/` (~5GB with default model)

**Total:** ~5.5GB disk space required (with default models)

**How it works:**
1. Model identifiers in `.env` (e.g., `RAG_EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2`) are used to download from HuggingFace
2. Models are saved locally to `./models/` directory
3. The system extracts the model name from the identifier and loads from local paths
4. After download, all operations work without internet connection

### CLI Usage

```bash
# Ingest documents
rag ingest document.pdf                 # Single file
rag ingest ./documents                  # Entire folder (auto-detected, recursive)
rag ingest ./docs --no-recursive        # Folder without subdirectories
rag ingest ./documents --force          # Re-ingest all files (ignore duplicates)

# Query (with different modes)
rag query-cmd "What is machine learning?" --mode granite
rag query-cmd "Explain the concept" --mode context_only
rag query-cmd "Summarize" --mode both

# Search without generation
rag search-cmd "machine learning"

# View statistics
rag stats

# Show configuration
rag config-show

# Reset system
rag reset

# Start MCP server
rag serve --mcp
```

### Smart Folder Ingestion

The `ingest` command automatically:
- **Detects folders vs files** - No `--batch` flag needed
- **Filters file types** - Only ingests supported formats (PDF, DOCX, PPTX, XLSX, HTML, MD, images, audio)
- **Skips duplicates** - Detects already-ingested files by file path
- **Excludes system files** - Automatically ignores hidden files, temp files, and backups
- **Shows detailed progress** - Per-file status with chunk counts

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

The codebase is organized into **5 modules** with **11 Python files** (~1,360 lines), optimized for beginner readability with extensive inline documentation:

```
src/
├── models.py              # Data models + Exceptions (94 lines)
├── config.py              # Settings (ChromaDB modes) + Logger (84 lines)
├── utils.py               # Utilities: Language, Embeddings, Model downloading (190 lines)
├── query.py               # 🎯 Main entry point - Query orchestration (65 lines)
│
├── ingestion/             # Document processing pipeline (3 files)
│   ├── document.py        # Load documents + extract metadata (170 lines)
│   ├── chunker.py         # Break docs into chunks (60 lines)
│   └── pipeline.py        # Orchestrate: load → chunk → embed → store (69 lines)
│
├── storage/               # Vector database (1 file)
│   └── chroma_client.py   # ChromaDB client factory (HTTP/persistent modes) + operations (351 lines)
│
├── retrieval/             # Search (1 file)
│   └── search.py          # Vector similarity search (50 lines)
│
├── generation/            # LLM generation (1 file)
│   └── granite.py         # Granite model + prompt templates (131 lines)
│
└── mcp/                   # MCP server (1 file)
    └── server.py          # FastMCP server endpoints (55 lines)
```

### Design Principles

- **Beginner-friendly**: Extensive inline comments explaining WHY, not just WHAT
- **Consolidated**: Related functionality grouped (loader + converter = document.py)
- **Clear flow**: Linear pipeline easy to follow (load → chunk → embed → store)
- **Minimal abstraction**: No unnecessary layers, direct and simple
- **Good documentation**: See `docs/HOW_IT_WORKS.md` for detailed explanations

### Module Overview

| Module | Files | Purpose | Lines |
|--------|-------|---------|-------|
| **ingestion** | 3 | Document loading, chunking, and pipeline | ~300 |
| **storage** | 1 | ChromaDB client factory (HTTP/persistent) + operations | ~351 |
| **retrieval** | 1 | Vector similarity search | ~50 |
| **generation** | 1 | Granite LLM + prompt engineering | ~131 |
| **mcp** | 1 | MCP server for Claude Desktop | ~55 |

**Top-level files:**
- `models.py` (94 lines): All data structures and exceptions
- `config.py` (84 lines): Settings (including ChromaDB modes) and logger
- `utils.py` (190 lines): Language detection, embeddings, model downloading
- `query.py` (65 lines): **🎯 Main entry point** - start reading here!

### For Beginners

**New to RAG?** Start here:
1. Read `docs/HOW_IT_WORKS.md` for a complete beginner's guide
2. Read `src/query.py` - it's the main orchestrator with clear comments
3. Follow the code flow: query.py → search.py → granite.py
4. Try it: `rag ingest paper.pdf` then `rag query-cmd "what is this about?"`

## Configuration Reference

See `.env.example` for all available settings. All settings can be overridden via:
1. Environment variables (prefix: `RAG_`)
2. Command-line arguments
3. `.env` file

## Answer Modes

- **granite**: Self-contained RAG, Granite generates the answer
- **context_only**: Return only retrieved chunks, let external LLM (Claude) reason
- **both**: Return context + Granite's answer for flexibility

## Development

```bash
# Lint
ruff check src/

# Type check
mypy src/
```

## License

MIT
