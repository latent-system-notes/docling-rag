# Docling RAG

A document ingestion and retrieval system using Docling for document processing and ChromaDB for vector storage.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download required models:
   ```bash
   rag models --download
   ```

3. Create an environment file (e.g., `.env.safety`):
   ```bash
   DOCUMENTS_DIR=./docs/my-project
   DATA_DIR=./data
   MODELS_DIR=./models
   ```

## CLI Commands

```bash
rag ingest <env>              # Ingest documents from DOCUMENTS_DIR
rag list <env>                # List indexed documents
rag query <env> "question"    # Query the knowledge base
rag remove <env> <doc_id>     # Remove a document
rag stats <env>               # Show database statistics
rag reset <env>               # Reset the database
rag mcp <env>                 # Start MCP server
rag models --download         # Download models
rag models --verify           # Verify models are installed
```

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
