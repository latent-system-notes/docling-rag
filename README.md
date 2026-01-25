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
rag mcp serve

# Terminal 2: Run queries simultaneously
rag query "test query"
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

This system is **100% offline-capable** after initial setup. All operations work without internet access.

```bash
# Download embedding model (one-time setup, ~420MB)
rag config models --download

# Verify model is downloaded
rag config models --verify

# View model path
rag config models --info
```

**Model Storage (based on .env settings):**
- Embedding: `./models/embedding/{model_name}/` (~420MB with default model)
- Docling Layout: `./models/docling/layout/` (~500MB for PDF layout detection)

**How it works:**
1. Models are downloaded from HuggingFace Hub:
   - **Embedding model**: Specified in `.env` (default: `paraphrase-multilingual-mpnet-base-v2`)
   - **Docling layout model**: Required for PDF processing (`docling-project/docling-layout-heron`)
2. Models are saved locally to `./models/` directory
3. All subsequent operations load models from local paths only
4. After download, **zero internet connectivity required**

**Offline Mode Enforcement:**
The system enforces strict offline operation through code-level parameters (not environment variables):
- **HuggingFace Embeddings**: `local_files_only=True` prevents any network access when loading models
- **HuggingFace Tokenizers**: `local_files_only=True` prevents downloading tokenizer files
- **Docling**: `enable_remote_services=False` prevents external service calls
- **Docling Layout**: Uses `model_path` parameter to load from local directory
- **Security**: `trust_remote_code=False` prevents downloading and executing remote code

**Verify Offline Mode:**
```bash
# 1. Download models once (requires internet)
rag config models --download
rag config models --verify

# 2. Test offline operation (disconnect internet or use airplane mode)
rag ingestion start document.pdf
rag query "test query"

# All operations should work without internet connection
# If you see any network errors, please report them as a bug
```

### CLI Usage

```bash
# Ingest documents (works with files or folders)
rag ingestion start document.pdf              # Single file
rag ingestion start ./documents               # Entire folder (auto-detected, recursive)
rag ingestion start ./docs --no-recursive     # Folder without subdirectories
rag ingestion start ./documents --force       # Force re-ingest all files
rag ingestion start ./documents --dry-run     # Preview changes without applying

# Monitor and control ingestion
rag ingestion status                          # Live dashboard
rag ingestion stop                            # Stop running ingestion
rag ingestion log                             # View ingestion history

# Query and retrieve relevant context
rag query "What is machine learning?"
rag query "Explain the concept" --top-k 10

# View statistics
rag stats

# Show configuration
rag config show

# Device and model management
rag config device                             # Show device info
rag config models --verify                    # Verify models

# Reset system
rag reset

# Start MCP server
rag mcp serve
rag mcp status                                # Show server status
rag mcp metrics                               # Show detailed metrics
```

## MCP Server

The system includes an MCP (Model Context Protocol) server for integration with AI assistants using streamable-http transport.

### Configuration

```bash
# In .env:
RAG_MCP_TRANSPORT=streamable-http
RAG_MCP_HOST=127.0.0.1
RAG_MCP_PORT=8080

# Start server:
rag mcp serve
# Server runs on http://127.0.0.1:8080
```

### Available Tools

1. **query_rag** - Query documents and retrieve relevant chunks
   - Parameters: `query_text` (str), `top_k` (int, default=5)
   - Returns: QueryResult with ranked chunks, scores, and metadata
   - Use case: Retrieve context for answering questions

2. **list_all_documents** - Browse indexed documents with pagination
   - Parameters: `limit` (int, optional), `offset` (int, default=0)
   - Returns: List of documents with metadata (file paths, types, languages, chunk counts, timestamps)
   - Use case: Explore available documents

### Environment Variables

All MCP settings are configurable via environment variables:

- `RAG_MCP_TRANSPORT` - Transport protocol (default: streamable-http)
- `RAG_MCP_HOST` - Server host (default: 127.0.0.1)
- `RAG_MCP_PORT` - Server port (default: 8080)
- `RAG_MCP_INSTRUCTIONS` - Instructions shown to MCP clients
- `RAG_MCP_TOOL_QUERY_DESCRIPTION` - Custom query tool description
- `RAG_MCP_TOOL_LIST_DOCS_DESCRIPTION` - Custom list documents tool description
- `RAG_MCP_ENABLE_CLEANUP` - Enable resource cleanup on shutdown (default: true)

See `.env.example` for complete configuration options.

### Smart Ingestion

The `ingestion start` command automatically:
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
4. Try it: `rag ingestion start paper.pdf` then `rag query "what is this about?"`

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
