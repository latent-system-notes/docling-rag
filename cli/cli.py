from pathlib import Path
import sys, os

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.system('chcp 65001 > nul 2>&1')

import logging
import typer
from contextlib import contextmanager
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

app = typer.Typer()

@contextmanager
def quiet_logging():
    """Temporarily suppress all verbose logs for clean spinner display."""
    import warnings
    old_filter = warnings.filters[:]
    old_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)  # Globally disable INFO and below
    warnings.filterwarnings("ignore")
    try:
        yield
    finally:
        logging.disable(old_disable)
        warnings.filters[:] = old_filter
console = Console(force_terminal=True, legacy_windows=False)

def _truncate(s: str, max_len: int = 55) -> str:
    if len(s) <= max_len: return s
    p = Path(s)
    avail = max_len - len(p.suffix) - 3
    return (p.stem[:avail] + "..." + p.suffix) if avail > 10 else s[:max_len-3] + "..."

def _load_env(env: str):
    """Load .env.{env} file."""
    env_file = Path(f".env.{env}")
    if not env_file.exists():
        console.print(f"[red]Environment file not found: {env_file}[/red]")
        raise typer.Exit(1)
    load_dotenv(env_file, override=True)
    from src.config import _setup_hf_env
    _setup_hf_env()


@app.command("list")
def list_docs(env: str, full_path: bool = False, limit: int = None, offset: int = 0):
    """List indexed documents."""
    _load_env(env)
    from datetime import datetime
    from src.storage.chroma_client import initialize_collections, list_documents
    initialize_collections()
    docs = list_documents(limit=limit, offset=offset)
    total = len(list_documents())

    if not docs:
        console.print(f"[yellow]No documents found{f' at offset {offset}' if offset else ''}.")
        return

    title = f"Indexed Documents ({offset+1}-{offset+len(docs)} of {total})" if limit or offset else f"Indexed Documents ({total})"
    t = Table(title=title, box=None)
    t.add_column("ID", style="dim", width=6)
    t.add_column("File", style="cyan")
    t.add_column("Type", style="magenta", width=6)
    t.add_column("Lang", style="blue", width=6)
    t.add_column("Chunks", style="green", justify="right", width=8)
    t.add_column("Ingested", style="yellow", width=19)

    for d in docs:
        ts = d['ingested_at']
        try: ts = datetime.fromisoformat(ts).strftime("%Y-%m-%d %H:%M:%S")
        except: pass
        t.add_row(d['doc_id'][-6:], _truncate(Path(d['file_path']).name) if not full_path else d['file_path'],
                  d['doc_type'], d['language'], str(d['num_chunks']), ts)
    console.print(t)
    console.print(f"\n[bold]Total:[/bold] {total} documents, {sum(d['num_chunks'] for d in docs)} chunks")


@app.command()
def remove(env: str, doc_id: str, yes: bool = typer.Option(False, "-y")):
    """Remove a document by ID."""
    _load_env(env)
    from src.storage.chroma_client import initialize_collections, remove_document_by_id
    initialize_collections()
    if not yes and not typer.confirm(f"Remove document '{doc_id[-6:]}'?"): return
    n = remove_document_by_id(doc_id)
    console.print(f"[green]Removed ({n} chunks)" if n else f"[yellow]Not found: {doc_id}")


@app.command()
def ingest(env: str, recursive: bool = True, dry_run: bool = False, force: bool = False):
    """Ingest documents from DOCUMENTS_DIR (file or directory)."""
    _load_env(env)
    from src.config import config, get_logger
    from src.ingestion.pipeline import ingest_document
    from src.utils import discover_files, is_supported_file
    from src.storage.chroma_client import initialize_collections, document_exists

    logger = get_logger(__name__)
    initialize_collections()
    doc_path = config("DOCUMENTS_DIR")

    if not doc_path.exists():
        console.print(f"[red]Path not found: {doc_path}[/red]")
        raise typer.Exit(1)

    # Support both single file and directory
    if doc_path.is_file():
        if not is_supported_file(doc_path):
            console.print(f"[red]Unsupported file type: {doc_path.suffix}[/red]")
            raise typer.Exit(1)
        files = [doc_path]
        console.print(f"[cyan]Processing file: {doc_path.name}[/cyan]")
    else:
        files = discover_files(doc_path, recursive=recursive)
        console.print(f"[cyan]Scanning {doc_path}...[/cyan]")

    processed, skipped, failed = 0, 0, 0
    for f in files:
        if dry_run:
            console.print(f"  + {f.name}")
            processed += 1
            if processed >= 20:
                console.print("  ... (dry-run limited to 20)")
                break
            continue

        if not force and document_exists(f):
            skipped += 1
            continue
        try:
            with console.status(f"Processing {f.name}..."):
                meta = ingest_document(f)
            processed += 1
            console.print(f"[green]{f.name} ({meta.num_chunks} chunks)")
        except Exception as e:
            failed += 1
            logger.error(f"{f.name}: {e}")
            console.print(f"[red]{f.name}: {e}")

    console.print(f"\n[bold]Done:[/bold] {processed} processed, {skipped} skipped, {failed} failed")


@app.command()
def query(env: str, query_text: str, top_k: int = None, format: str = "json"):
    """Query documents."""
    _load_env(env)
    from src.config import DEFAULT_TOP_K
    from src.query import query as query_fn
    import json

    top_k = top_k or DEFAULT_TOP_K
    r = query_fn(query_text, top_k)
    if format == "json":
        out = {"query": r.query, "total_results": len(r.context), "results": [
            {"rank": i, "score": round(x.score, 4), "text": x.chunk.text,
             "file": Path(x.chunk.metadata.get("file_path", "")).name, "page": x.chunk.page_num}
            for i, x in enumerate(r.context, 1)]}
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        console.print(f"[bold]Query:[/bold] {r.query}\n")
        for i, x in enumerate(r.context, 1):
            console.print(f"[cyan]{i}[/cyan] ({x.score:.3f}) {Path(x.chunk.metadata.get('file_path','')).name}")
            console.print(f"  {x.chunk.text[:200]}...\n")


@app.command()
def stats(env: str):
    """Show database statistics."""
    _load_env(env)
    from src.storage.chroma_client import initialize_collections, get_stats, list_documents
    initialize_collections()
    s = get_stats()
    docs = list_documents()
    t = Table(title="Stats", box=None)
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="green")
    t.add_row("documents", str(len(docs)))
    for k, v in s.items(): t.add_row(k, str(v))
    console.print(t)


@app.command()
def reset(env: str):
    """Reset the database."""
    _load_env(env)
    from src.storage.chroma_client import reset_collection
    if typer.confirm("Reset system?"): reset_collection(); console.print("[green]Reset complete")


@app.command()
def mcp(env: str):
    """Start MCP server."""
    _load_env(env)
    from src.config import config, MCP_HOST
    from src.mcp.server import run_server
    server_name = config("MCP_SERVER_NAME")
    port = config("MCP_PORT")
    console.print(f"[green]Starting MCP ({server_name}) on http://{MCP_HOST}:{port}")
    try: run_server()
    except KeyboardInterrupt: console.print("\n[yellow]Stopped[/yellow]")


@app.command()
def models(download: bool = False, verify: bool = False):
    """Manage models (no env needed)."""
    from src.config import _setup_hf_env, config
    _setup_hf_env()
    models_dir = config("MODELS_DIR")

    if download:
        for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]: os.environ.pop(k, None)
        from src.utils import download_embedding_model, download_docling_models
        with quiet_logging(), console.status("Downloading embedding model..."):
            download_embedding_model()
        with quiet_logging(), console.status("Downloading Docling models..."):
            download_docling_models()
        for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]: os.environ[k] = "1"
        console.print(f"[green]Done! Models in {models_dir}")
    elif verify:
        from src.utils import verify_models_exist
        status = verify_models_exist()
        t = Table(title="Model Status", box=None)
        t.add_column("Model", style="cyan")
        t.add_column("Status", style="green")
        for name, exists in status.items():
            t.add_row(name, "[green]OK" if exists else "[red]Missing")
        console.print(t)
    else:
        console.print("[yellow]Use --download to download models or --verify to check status")


if __name__ == "__main__":
    try: app()
    except KeyboardInterrupt: raise typer.Exit(0)
