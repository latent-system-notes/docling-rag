# How the RAG System Works (Beginner's Guide)

## What is RAG?

**RAG = Retrieval-Augmented Generation**

Think of it as giving an AI a smart filing cabinet:
1. **Store**: Put your documents in the cabinet (with a clever indexing system)
2. **Search**: When you ask a question, find the most relevant documents
3. **Provide Context**: Return those documents to an external AI (like Claude) to write an answer

## The Problem RAG Solves

**Without RAG:**
- AI has general knowledge but doesn't know YOUR specific documents
- Can't answer questions like "What did our Q3 report say about sales?"

**With RAG:**
- AI can search YOUR documents and answer questions about them
- Like having a super-smart assistant who's read all your files

## How This System Works (Step by Step)

### Part 1: Ingestion (One Time Setup)

**What**: Load your documents into the system
**Where**: `src/ingestion/` folder

```
Your PDF → [Load] → [Chunk] → [Embed] → [Store in Database]
```

**Steps:**
1. **Load Document** (`document.py`)
   - Reads PDF, DOCX, images, etc.
   - Uses IBM Docling to extract text

2. **Chunk Document** (`chunker.py`)
   - Breaks document into smaller pieces (~512 words each)
   - Why? Smaller chunks = better search results

3. **Create Embeddings** (`utils.py:embed()`)
   - Converts each chunk to a list of numbers (a "vector")
   - Magic: Similar text → Similar numbers
   - Example: "dog" and "puppy" have similar vectors

4. **Store in ChromaDB** (`storage/chroma_client.py`)
   - Saves vectors in a database optimized for similarity search
   - Like a filing cabinet organized by meaning, not alphabetically
   - Supports two storage modes: **Persistent** (local SQLite) and **HTTP** (client-server)

### ChromaDB Storage Modes

The system supports two ways to store and access vectors:

#### Persistent Mode (Default)
```
Your App → ChromaDB → SQLite File (./data/chroma/chroma.sqlite3)
```

**How it works:**
- Your application directly accesses a SQLite database file
- Simple setup - no server needed
- Good for: Single-user, development, simple projects

**Limitations:**
- ❌ Only ONE process can access the database at a time (file locking)
- ❌ Slow shutdown (must close database connections properly)
- ❌ Can't run MCP server and CLI simultaneously

#### HTTP Mode (Recommended for Production)
```
Your App → HTTP Client → ChromaDB Server → SQLite File
```

**How it works:**
- ChromaDB runs as a separate server process
- Your application connects via HTTP (like accessing a website)
- Server handles all database access
- Multiple clients can connect simultaneously

**Benefits:**
- ✅ NO file locking - unlimited concurrent clients
- ✅ Fast shutdown - just close HTTP connection
- ✅ Run MCP server + CLI at the same time
- ✅ Better reliability - server manages all file access
- ✅ Can run on remote server for scalability

**Setup:**
```bash
# Terminal 1: Start ChromaDB server
chroma run --host localhost --port 8000 --path ./data/chroma

# Terminal 2: Configure your app (.env)
RAG_CHROMA_MODE=http
RAG_CHROMA_SERVER_HOST=localhost
RAG_CHROMA_SERVER_PORT=8000

# Now run your app
rag stats  # Works!
```

**Why HTTP mode solves file locking:**
- Persistent mode: Both MCP server AND CLI try to open the same SQLite file → ❌ Locking error
- HTTP mode: Both connect to ChromaDB server via HTTP → ✅ Server handles everything

**Migration:** Zero effort! ChromaDB server reads your existing SQLite database. Just start the server and change `RAG_CHROMA_MODE=http`.

### Part 2: Querying (Every Time You Ask)

**What**: Ask a question and retrieve relevant context
**Where**: `src/query.py` orchestrates everything

```
Your Question → [Embed] → [Search] → Return Context
```

**Steps:**
1. **Convert Question to Vector** (`utils.py:embed()`)
   - Your question becomes numbers too
   - Example: "What is a dog?" → [0.2, 0.8, 0.1, ...]

2. **Vector Search** (`retrieval/search.py`)
   - Find chunks with similar vectors to your question
   - Uses cosine similarity (measures angle between vectors)
   - Returns top 5 most relevant chunks

3. **Return Context**
   - Returns the relevant chunks to the user
   - User can feed these to external LLM (like Claude) for answer generation
   - LLM reads the context and writes an answer

## Key Concepts for Beginners

### Embeddings (Vectors)

**Simple explanation**:
- Text → List of numbers
- Similar meaning → Similar numbers

**Example**:
```python
"The cat sat on the mat" → [0.1, 0.3, 0.8, ...]
"A feline rested on the rug" → [0.1, 0.3, 0.7, ...]  # Similar!
"Pizza delivery is fast" → [0.9, 0.1, 0.2, ...]  # Different!
```

### Chunks

**Why chunk?**
- Full documents are too big for embeddings
- Smaller chunks = more precise search
- Default: ~512 tokens (roughly 2-3 paragraphs)

**Example**:
```
Document: "The Ultimate Guide to Coffee (50 pages)"
↓
Chunks:
- Chunk 1: "Coffee beans come from..."
- Chunk 2: "Espresso is made by..."
- Chunk 3: "Cold brew requires..."
... (100 chunks total)
```

### Vector Search

**How it works**:
1. Calculate similarity between query vector and all chunk vectors
2. Return the most similar chunks
3. Similarity = cosine of angle between vectors (closer angle = more similar)

**Visual**:
```
Query: "How to make espresso?"
        ↓ (embed)
    [0.2, 0.7, 0.1]
        ↓ (search)
    Compare to all chunks
        ↓
    Find closest matches
        ↓
    Return: "Espresso is made by..." (chunk 2)
```

## Code Flow (Follow Along)

### When you run: `rag sync paper.pdf`

1. `cli/cli.py` → Calls `ingest_document()`
2. `ingestion/document.py:load_document()` → Parses PDF
3. `ingestion/chunker.py:chunk_document()` → Creates chunks
4. `utils.py:embed_batch()` → Converts chunks to vectors
5. `storage/chroma_client.py:add_vectors()` → Saves to database

### When you run: `rag query "What is machine learning?"`

1. `cli/cli.py` → Calls `query()`
2. `query.py:query()` → Orchestrates everything:
   - `retrieval/search.py:search()` → Find relevant chunks
   - `utils.py:embed()` → Convert query to vector
   - `storage/chroma_client.py:search_vectors()` → Search database
3. Returns: Relevant context chunks for external LLM reasoning

## File Structure (Simplified)

```
src/
├── models.py        → Data structures (what is a Chunk? a SearchResult?)
├── config.py        → Settings (model names, ChromaDB modes, file paths, etc.)
├── utils.py         → Helper functions (embed text, download models)
├── query.py         → 🎯 MAIN ENTRY POINT for asking questions
│
├── ingestion/       → Load and process documents
│   ├── document.py  → Load PDFs, extract metadata
│   ├── chunker.py   → Break docs into chunks
│   └── pipeline.py  → Orchestrate: load → chunk → embed → store
│
├── storage/         → ChromaDB vector database
│   └── chroma_client.py → Client factory (HTTP/persistent modes) + save/search vectors
│
└── retrieval/       → Find relevant chunks
    └── search.py    → Vector similarity search
```

## Smart Syncing

### What It Does

The system can automatically sync files and folders, intelligently filtering and processing only supported file types while detecting changes.

### How It Works

**Example**: `rag sync ./documents`

```
./documents/
├── report.pdf          ✓ Supported, ingest
├── slides.pptx         ✓ Supported, ingest
├── data.xlsx           ✓ Supported, ingest
├── notes.txt           ✗ Not supported, skip
├── .hidden.pdf         ✗ Hidden file, skip
├── ~temp.docx          ✗ Temp file, skip
└── old.pdf.bak         ✗ Backup, skip
```

**Result**: Only `report.pdf`, `slides.pptx`, and `data.xlsx` are ingested.

### File Filtering Process

When you run `rag sync ./folder`, the system:

1. **Scans the directory** recursively (or use `--no-recursive` for flat scan)
2. **Filters by extension** - Only processes supported file types
3. **Excludes patterns** - Skips hidden files, temp files, backups
4. **Detects changes** - Finds new, modified, and deleted files
5. **Syncs files** - Adds new, updates modified, removes deleted
6. **Shows summary** - Reports added/updated/removed counts

### Supported File Types

The system automatically recognizes these formats:

- **Documents**: `.pdf`, `.docx`, `.pptx`, `.xlsx`
- **Web**: `.html`, `.htm`
- **Markup**: `.md`
- **Images** (with OCR): `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`
- **Audio** (with ASR): `.wav`, `.mp3`

All other files are silently skipped.

### Automatically Excluded Files

These patterns are always excluded:
- Hidden files: `.*`, `__*`
- Temp files: `*.tmp`, `*.temp`, `~*`
- Backups: `*.bak`, `*.backup`
- System files: `Thumbs.db`, `.DS_Store`

### Change Detection

**How it works:**
1. Each file path is converted to an MD5 hash (document ID)
2. The system checks ChromaDB for existing documents
3. Compares file modification timestamps to detect changes
4. Identifies files in DB but not on disk (deleted)

**Example:**
```bash
# First sync: Adds 10 files
rag sync ./documents
# Added: 10 files

# Second sync: Everything up to date
rag sync ./documents
# ✓ Everything is up to date!

# After editing a file and deleting another
rag sync ./documents
# Updated: 1 file (modified)
# Removed: 1 file (deleted from disk)

# Force re-sync (re-ingest all)
rag sync ./documents --force
# Updated: 10 files
```

### Usage Examples

**Basic folder sync** (recursive):
```bash
rag sync ./documents
```

**Output:**
```
Analyzing ./documents...

Sync Analysis:
  New files: 5
  Modified files: 2
  Deleted files: 1

Syncing...
  ✓ Added report.pdf (12 chunks)
  ✓ Added slides.pptx (8 chunks)
  ↻ Updated data.xlsx (5 chunks)
  ✗ Removed old-file.pdf (10 chunks)

Summary:
  Added: 5 files
  Updated: 2 files
  Removed: 1 file
```

**Preview changes** (dry run):
```bash
rag sync ./documents --dry-run
```

**Non-recursive** (only current folder):
```bash
rag sync ./documents --no-recursive
```

**Force re-sync** (re-ingest all):
```bash
rag sync ./documents --force
```

**Single file**:
```bash
rag sync paper.pdf
```

### Technical Details

**File Discovery** (`src/utils.py`):
- `discover_files()` - Scans directory with filtering
- `is_supported_file()` - Checks file extension
- `should_exclude_file()` - Matches exclusion patterns

**Duplicate Detection** (`src/storage/chroma_client.py`):
- `document_exists()` - Queries ChromaDB for existing document ID

**CLI Command** (`cli/cli.py`):
- Auto-detects folders vs files
- Shows detailed per-file progress
- Provides summary statistics

## Common Beginner Questions

### Q: Why do we need embeddings? Can't we just search for keywords?

**A**: Keyword search misses synonyms and context.

- Keyword search: "dog" won't find "puppy" or "canine"
- Vector search: Understands meaning, finds all related content

### Q: What is ChromaDB?

**A**: A specialized database for storing and searching vectors.

- Regular DB: Good for exact matches (name = "John")
- Vector DB: Good for similarity (find similar to "dog")

### Q: Why chunk documents?

**A**: Three reasons:
1. Embeddings work better on focused text (not full books)
2. Search returns precise passages, not entire documents
3. LLMs have token limits, can't read full documents

### Q: How do I add more documents?

**A**: Simple! Run:
```bash
rag sync /path/to/document.pdf
```

The system handles everything else automatically.

### Q: Can I use my own LLM?

**A**: Yes! The system returns relevant context chunks that you can feed to Claude, GPT, or any LLM you prefer for answer generation.

### Q: What's the difference between persistent and HTTP mode for ChromaDB?

**A**: Think of it like file access vs web access:

**Persistent mode:**
- Your app opens the SQLite database file directly
- Like opening a Word document - only one program can edit it at once
- Problem: If MCP server has the file open, CLI gets locked out ❌

**HTTP mode:**
- ChromaDB runs as a server (like a website)
- Your app connects via HTTP (like visiting a website)
- Multiple clients can connect simultaneously ✅
- Server handles all file access - no locking!

**When to use each:**
- **Persistent**: Simple projects, single user, development
- **HTTP**: Production, concurrent access (MCP + CLI together), better reliability

### Q: Do I need to migrate my data to use HTTP mode?

**A**: No! Just:
1. Start ChromaDB server: `chroma run --path ./data/chroma`
2. Change config: `RAG_CHROMA_MODE=http`

The server reads your existing SQLite database. Zero data migration needed!

### Q: Why does my MCP server shut down slowly in persistent mode?

**A**: SQLite connections take time to close properly (vacuuming, finalizing transactions). In HTTP mode, the client just closes the HTTP connection instantly - the server handles database cleanup in the background.

## Next Steps

1. **Try it**: Run `rag sync` and `rag query`
2. **Read the code**: Start with `src/query.py` (it's well-commented!)
3. **Experiment**: Try different top_k values for more or fewer context chunks
4. **Customize**: Edit `src/config.py` to change behavior

## Resources

- **What are embeddings?**: [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- **RAG explained**: [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- **Vector databases**: [ChromaDB Documentation](https://docs.trychroma.com/)
