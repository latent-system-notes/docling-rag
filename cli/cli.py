from pathlib import Path
import sys, os

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.system('chcp 65001 > nul 2>&1')

import typer
from rich.console import Console
from rich.table import Table

from src.config import settings, get_logger
from src.project import get_project_manager

logger = get_logger(__name__)
app = typer.Typer()
project_app = typer.Typer()
config_app = typer.Typer()
mcp_app = typer.Typer()
console = Console(force_terminal=True, legacy_windows=False)

def _show_project():
    from src.config import apply_project_settings
    if apply_project_settings():
        active = get_project_manager().get_active_project()
        if active: console.print(f"[dim]Project: {active.name}[/dim]")

def _truncate(s: str, max_len: int = 55) -> str:
    if len(s) <= max_len: return s
    p = Path(s)
    avail = max_len - len(p.suffix) - 3
    return (p.stem[:avail] + "..." + p.suffix) if avail > 10 else s[:max_len-3] + "..."


@app.command("list")
def list_docs(full_path: bool = False, limit: int = None, offset: int = 0):
    from datetime import datetime
    from src.storage.chroma_client import initialize_collections, list_documents
    _show_project()
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
    _show_project()
    initialize_collections()
    if not yes and not typer.confirm(f"Remove document '{doc_id[-6:]}'?"): return
    n = remove_document_by_id(doc_id)
    console.print(f"[green]Removed ({n} chunks)" if n else f"[yellow]Not found: {doc_id}")


@app.command()
def ingest(path: str = None, recursive: bool = True, dry_run: bool = False, force: bool = False,
           resume: bool = True, workers: int = 4):
    from src.ingestion.pipeline import ingest_document
    from src.ingestion.parallel_pipeline import parallel_ingest_documents, ParallelIngestionConfig, collect_files
    from src.ingestion.lock import IngestionLockError
    from src.config import apply_project_settings
    from src.storage.chroma_client import initialize_collections, document_exists

    pm = get_project_manager()
    active = pm.get_active_project() if apply_project_settings() else None
    if active: console.print(f"[dim]Project: {active.name}[/dim]")

    if path is None:
        if not active:
            console.print("[red]No path and no active project[/red]")
            raise typer.Exit(1)
        file_path = pm.get_project_paths(active.name)["docs_dir"]
    else:
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

    try:
        r = parallel_ingest_documents(file_path, cfg)
    except IngestionLockError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Done:[/bold] {r.processed} files, {r.total_chunks} chunks, {r.duration_seconds/60:.1f}m")
    if r.failed: console.print(f"[red]Failed: {r.failed}[/red]")


@app.command()
def query(query_text: str, top_k: int = settings.default_top_k, format: str = "json"):
    from src.query import query as query_fn
    from src.config import apply_project_settings
    import json

    if apply_project_settings() and format == "text":
        _show_project()

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
    _show_project()
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
    _show_project()
    if typer.confirm("Reset system?"): reset_collection(); console.print("[green]Reset complete")


# === MCP Commands ===

@mcp_app.command("serve")
def mcp_serve():
    from src.mcp.server import run_server
    from src.config import apply_project_settings

    apply_project_settings()
    pm = get_project_manager()
    active = pm.get_active_project()
    if active: console.print(f"[dim]Project: {active.name}[/dim]")

    console.print(f"[green]Starting MCP on http://{settings.mcp_host}:{settings.mcp_port}")
    try: run_server()
    except KeyboardInterrupt: console.print("\n[yellow]Stopped[/yellow]")


@mcp_app.command("stop")
def mcp_stop(yes: bool = False):
    console.print("[yellow]Use Ctrl+C in the terminal running 'rag mcp serve' to stop the server[/yellow]")


# === Config Commands ===

@config_app.command("show")
def config_show():
    pm = get_project_manager()
    active = pm.get_active_project()

    if active:
        paths = pm.get_project_paths(active.name)
        console.print("[bold]Active Project[/bold]")
        t = Table(box=None)
        t.add_column("Setting", style="cyan")
        t.add_column("Value", style="green")
        t.add_row("Name", active.name)
        t.add_row("Port", str(active.port))
        t.add_row("MCP Server", active.mcp_server_name)
        t.add_row("Data Dir", str(paths["data_dir"]))
        t.add_row("Docs Dir", str(paths["docs_dir"]))
        console.print(t)
    else:
        console.print("[yellow]No active project[/yellow]")

    console.print(f"\n[bold]Device:[/bold] {settings.device} [dim](RAG_DEVICE)[/dim]")
    console.print(f"[bold]Models:[/bold] {settings.models_dir}")


@config_app.command("models")
def models_download():
    for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]: os.environ.pop(k, None)
    from src.utils import download_embedding_model, download_docling_models
    console.print("[cyan]Downloading embedding model...")
    download_embedding_model()
    console.print("[cyan]Downloading Docling models...")
    download_docling_models()
    for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]: os.environ[k] = "1"
    console.print(f"[green]Done! Models in {settings.models_dir}")


# === Project Commands ===

@project_app.command("create")
def project_create(name: str, port: int = 9090):
    pm = get_project_manager()
    try:
        cfg = pm.create_project(name=name, port=port)
        paths = pm.get_project_paths(name)
        console.print(f"[green]Created '{name}'[/green]")
        console.print(f"  Port: {cfg.port} | MCP: {cfg.mcp_server_name}")
        console.print(f"  Data: {paths['data_dir']}")
        console.print(f"  Docs: {paths['docs_dir']}")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@project_app.command("list")
def project_list():
    pm = get_project_manager()
    projects = pm.list_projects()
    active = pm.get_active_project_name()
    if not projects:
        console.print("[yellow]No projects. Create with: rag project create <name>")
        return
    for p in projects:
        marker = "[green]*" if p.name == active else " "
        console.print(f"{marker} {p.name} (:{p.port})")


@project_app.command("switch")
def project_switch(name: str):
    try:
        cfg = get_project_manager().switch_project(name)
        console.print(f"[green]Switched to '{name}' (:{cfg.port})[/green]")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


@project_app.command("delete")
def project_delete(name: str, data: bool = False, yes: bool = typer.Option(False, "-y")):
    pm = get_project_manager()
    if not pm.project_exists(name):
        console.print(f"[red]Not found: {name}[/red]")
        raise typer.Exit(1)
    if not yes and not typer.confirm(f"Delete '{name}'{'and data' if data else ''}?"): return
    pm.delete_project(name, delete_data=data)
    console.print(f"[green]Deleted '{name}'[/green]")


app.add_typer(project_app, name="project")
app.add_typer(config_app, name="config")
app.add_typer(mcp_app, name="mcp")

if __name__ == "__main__":
    try: app()
    except KeyboardInterrupt: raise typer.Exit(0)
