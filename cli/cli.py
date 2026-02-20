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
    import warnings
    old_filter = warnings.filters[:]
    old_disable = logging.root.manager.disable
    logging.disable(logging.WARNING)
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
    env_file = Path(f".env.{env}")
    if not env_file.exists():
        console.print(f"[red]Environment file not found: {env_file}[/red]")
        raise typer.Exit(1)
    load_dotenv(env_file, override=True)
    from src.config import _setup_hf_env
    _setup_hf_env()


@app.command("list")
def list_docs(env: str, full_path: bool = False, limit: int = None, offset: int = 0):
    _load_env(env)
    from datetime import datetime
    from src.storage.chroma_client import create_collection, list_documents
    create_collection()
    all_docs = list_documents()
    total = len(all_docs)
    docs = all_docs[offset:]
    if limit: docs = docs[:limit]

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
    _load_env(env)
    from src.storage.chroma_client import create_collection, remove_document_by_id
    create_collection()
    if not yes and not typer.confirm(f"Remove document '{doc_id[-6:]}'?"): return
    n = remove_document_by_id(doc_id)
    console.print(f"[green]Removed ({n} chunks)" if n else f"[yellow]Not found: {doc_id}")


@app.command()
def ingest(env: str, recursive: bool = True, dry_run: bool = False, force: bool = False):
    _load_env(env)
    from src.config import config, get_logger
    from src.ingestion.pipeline import ingest_document
    from src.utils import discover_files, is_supported_file, managed_resources
    from src.storage.chroma_client import create_collection, document_exists

    logger = get_logger(__name__)
    create_collection()
    doc_path = config("DOCUMENTS_DIR")

    if not doc_path.exists():
        console.print(f"[red]Path not found: {doc_path}[/red]")
        raise typer.Exit(1)

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
    with managed_resources():
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
    _load_env(env)
    from src.config import DEFAULT_TOP_K
    from src.query import query as query_fn
    import json

    r = query_fn(query_text, top_k or DEFAULT_TOP_K)
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
    _load_env(env)
    from src.storage.chroma_client import create_collection, get_stats, list_documents
    create_collection()
    s = get_stats()
    t = Table(title="Stats", box=None)
    t.add_column("Metric", style="cyan")
    t.add_column("Value", style="green")
    t.add_row("documents", str(len(list_documents())))
    for k, v in s.items(): t.add_row(k, str(v))
    console.print(t)


@app.command()
def reset(env: str):
    _load_env(env)
    from src.storage.chroma_client import reset_collection
    if typer.confirm("Reset system?"): reset_collection(); console.print("[green]Reset complete")


@app.command()
def mcp(env: str):
    _load_env(env)
    from src.config import config, MCP_HOST
    from src.mcp.server import run_server
    port = config("MCP_PORT")
    console.print(f"[green]Starting MCP ({config('MCP_SERVER_NAME')}) on http://{MCP_HOST}:{port}")
    try: run_server()
    except KeyboardInterrupt: console.print("\n[yellow]Stopped[/yellow]")


@app.command()
def models(download: bool = False, verify: bool = False):
    from src.config import _setup_hf_env
    _setup_hf_env()
    from src.utils import get_embedding_model_path

    if download:
        for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]: os.environ.pop(k, None)
        from src.utils import download_embedding_model
        with quiet_logging(), console.status("Downloading embedding model..."):
            download_embedding_model()
        for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]: os.environ[k] = "1"
        console.print(f"[green]Done! Model saved to {get_embedding_model_path()}")
    elif verify:
        path = get_embedding_model_path()
        status = "[green]OK" if path.exists() else "[red]Missing"
        console.print(f"Embedding model ({path}): {status}")
    else:
        console.print("[yellow]Use --download to download models or --verify to check status")


if __name__ == "__main__":
    try: app()
    except KeyboardInterrupt: raise typer.Exit(0)
