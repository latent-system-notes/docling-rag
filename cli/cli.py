from pathlib import Path
import sys, os

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.system('chcp 65001 > nul 2>&1')

import typer
from rich.console import Console
from rich.table import Table

from src.config import device, DEFAULT_TOP_K, MCP_HOST, MCP_PORT, MODELS_DIR, get_logger

logger = get_logger(__name__)
app = typer.Typer()
config_app = typer.Typer()
mcp_app = typer.Typer()
console = Console(force_terminal=True, legacy_windows=False)

def _truncate(s: str, max_len: int = 55) -> str:
    if len(s) <= max_len: return s
    p = Path(s)
    avail = max_len - len(p.suffix) - 3
    return (p.stem[:avail] + "..." + p.suffix) if avail > 10 else s[:max_len-3] + "..."


@app.command("list")
def list_docs(full_path: bool = False, limit: int = None, offset: int = 0):
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
def remove(doc_id: str, yes: bool = typer.Option(False, "-y")):
    from src.storage.chroma_client import initialize_collections, remove_document_by_id
    initialize_collections()
    if not yes and not typer.confirm(f"Remove document '{doc_id[-6:]}'?"): return
    n = remove_document_by_id(doc_id)
    console.print(f"[green]Removed ({n} chunks)" if n else f"[yellow]Not found: {doc_id}")


@app.command()
def ingest(path: str = typer.Argument(...), recursive: bool = True, dry_run: bool = False, force: bool = False,
           resume: bool = True, workers: int = 4):
    from src.ingestion.pipeline import ingest_document
    from src.ingestion.parallel_pipeline import parallel_ingest_documents, ParallelIngestionConfig, collect_files
    from src.storage.chroma_client import initialize_collections, document_exists

    file_path = Path(path)
    initialize_collections()

    if file_path.is_file():
        if not force and document_exists(file_path):
            console.print("[yellow]Already exists. Use --force")
            return
        meta = ingest_document(file_path, resume=resume)
        console.print(f"[green]Ingested: {file_path.name} ({meta.num_chunks} chunks)")
        return

    if not file_path.is_dir():
        console.print(f"[red]Not found: {path}[/red]")
        raise typer.Exit(1)

    cfg = ParallelIngestionConfig(num_workers=workers, recursive=recursive, force=force, resume=resume)
    files = collect_files(file_path, cfg.extensions, recursive)
    if not files:
        console.print(f"[yellow]No files in {file_path}")
        return

    console.print(f"[cyan]Found {len(files)} documents ({workers} workers)[/cyan]")
    if dry_run:
        for f in files[:20]: console.print(f"  + {f.name}")
        if len(files) > 20: console.print(f"  ... and {len(files) - 20} more")
        return

    r = parallel_ingest_documents(file_path, cfg)
    console.print(f"\n[bold]Done:[/bold] {r.processed} files, {r.total_chunks} chunks, {r.duration_seconds/60:.1f}m")
    if r.failed: console.print(f"[red]Failed: {r.failed}[/red]")


@app.command()
def query(query_text: str, top_k: int = DEFAULT_TOP_K, format: str = "json"):
    from src.query import query as query_fn
    import json

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
def stats():
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
def reset():
    from src.storage.chroma_client import reset_collection
    if typer.confirm("Reset system?"): reset_collection(); console.print("[green]Reset complete")


@mcp_app.command("serve")
def mcp_serve():
    from src.mcp.server import run_server
    console.print(f"[green]Starting MCP on http://{MCP_HOST}:{MCP_PORT}")
    try: run_server()
    except KeyboardInterrupt: console.print("\n[yellow]Stopped[/yellow]")


@mcp_app.command("stop")
def mcp_stop(yes: bool = False):
    console.print("[yellow]Use Ctrl+C in the terminal running 'rag mcp serve' to stop the server[/yellow]")


@config_app.command("show")
def config_show():
    console.print(f"[bold]Device:[/bold] {device} [dim](RAG_DEVICE)[/dim]")
    console.print(f"[bold]Models:[/bold] {MODELS_DIR}")


@config_app.command("models")
def models_download():
    for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]: os.environ.pop(k, None)
    from src.utils import download_embedding_model, download_docling_models
    console.print("[cyan]Downloading embedding model...")
    download_embedding_model()
    console.print("[cyan]Downloading Docling models...")
    download_docling_models()
    for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]: os.environ[k] = "1"
    console.print(f"[green]Done! Models in {MODELS_DIR}")


app.add_typer(config_app, name="config")
app.add_typer(mcp_app, name="mcp")

if __name__ == "__main__":
    try: app()
    except KeyboardInterrupt: raise typer.Exit(0)
