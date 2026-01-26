# Docling RAG System

Production-ready RAG system with IBM Docling and ChromaDB.

## Features

- **Multi-format document processing**: PDF, DOCX, PPTX, XLSX, HTML, Markdown, images
- **Advanced chunking**: Docling's HybridChunker for intelligent document segmentation
- **Hybrid search**: ChromaDB (vector) + BM25 (keyword) with reciprocal rank fusion
- **Multilingual embeddings**: Support for 50+ languages
- **Parallel ingestion**: Multi-worker processing for large document sets
- **Checkpointing**: Resume interrupted ingestion without data loss
- **MCP server**: Integration with AI assistants (Claude Desktop, etc.)
- **CPU-only operation**: No GPU/CUDA required

## Quick Start

### Installation

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

### Download Models

```bash
# Download required models (one-time, ~1GB)
rag models --download
```

### Ingest Documents

```bash
# Ingest a single file
rag ingest document.pdf

# Ingest a folder (parallel processing)
rag ingest ./documents --workers 4

# Preview without ingesting
rag ingest ./documents --dry-run
```

### Query Documents

```bash
# Query with JSON output (default)
rag query "What is machine learning?"

# Query with formatted text output
rag query "Explain the concept" --format text --top-k 10
```

## CLI Commands

```bash
rag ingest <path>           # Ingest file or directory
rag query "question"        # Query documents
rag list                    # List indexed documents
rag remove <doc_id>         # Remove a document
rag stats                   # Show statistics
rag reset                   # Clear all data
rag models --download       # Download required models
rag models --verify         # Verify models exist
rag mcp                     # Start MCP server
```

## MCP Server

The system includes an MCP (Model Context Protocol) server for integration with AI assistants.

### Start Server

```bash
rag mcp
# Server runs on http://0.0.0.0:9090
```

### Available Tools

1. **query_rag** - Query documents and retrieve relevant chunks
   - Parameters: `query_text` (str), `top_k` (int, default=5)
   - Returns: QueryResult with ranked chunks, scores, and metadata

2. **list_all_documents** - Browse indexed documents with pagination
   - Parameters: `limit` (int, optional), `offset` (int, default=0)
   - Returns: List of documents with metadata

## Configuration

Environment variables:
- `RAG_DEVICE`: Processing device (`cpu`, `cuda`, `mps`)

Data is stored in:
- `./data/chroma` - Vector database
- `./data/checkpoints` - Ingestion checkpoints
- `./models` - Downloaded models

## Architecture

```
src/
├── config.py              # Constants + Logger
├── models.py              # Data models + Exceptions
├── query.py               # Main query orchestration
├── utils.py               # Embeddings, language detection
│
├── ingestion/             # Document processing
│   ├── document.py        # Load documents + metadata
│   ├── chunker.py         # Break docs into chunks
│   ├── pipeline.py        # Single-file ingestion
│   ├── parallel_pipeline.py # Multi-worker ingestion
│   └── checkpoint.py      # Resume interrupted ingestion
│
├── storage/               # Data storage
│   ├── chroma_client.py   # ChromaDB vector store
│   └── bm25_index.py      # BM25 keyword index
│
├── retrieval/             # Search
│   └── search.py          # Hybrid search (vector + BM25)
│
└── mcp/                   # MCP server
    └── server.py          # FastMCP endpoints
```

### Key Features

| Feature | Description |
|---------|-------------|
| **Parallel Ingestion** | Multi-worker document processing for speed |
| **Checkpointing** | Resume interrupted ingestion without re-processing |
| **Hybrid Search** | Combines vector similarity with BM25 keyword matching |

## Supported File Types

- Documents: `.pdf`, `.docx`, `.pptx`, `.xlsx`
- Text: `.md`, `.html`
- Images: `.png`, `.jpg`, `.jpeg`, `.tiff`
- Audio: `.wav`, `.mp3`

## License

MIT
