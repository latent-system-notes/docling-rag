from pathlib import Path
import sys, os

if sys.platform == "win32":
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'): sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    os.system('chcp 65001 > nul 2>&1')

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from src.config import settings, get_logger
from src.project import get_project_manager

logger = get_logger(__name__)
app = typer.Typer()
project_app = typer.Typer()
ingestion_app = typer.Typer()
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

def _fmt_duration(s):
    if s is None: return "-"
    if s < 60: return f"{s:.0f}s"
    if s < 3600: return f"{s/60:.1f}m"
    return f"{s/3600:.1f}h"


@app.command()
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


def _ingest_parallel(directory: Path, recursive: bool, dry_run: bool, force: bool, resume: bool, workers: int):
    from src.ingestion.parallel_pipeline import parallel_ingest_documents, ParallelIngestionConfig, collect_files
    from src.ingestion.lock import IngestionLockError

    cfg = ParallelIngestionConfig(num_workers=workers, recursive=recursive, force=force, resume=resume)
    files = collect_files(directory, cfg.extensions, recursive)
    if not files:
        console.print(f"[yellow]No files in {directory}")
        return

    console.print(f"[cyan]Found {len(files)} documents ({workers} workers)[/cyan]")
    if dry_run:
        for f in files[:20]: console.print(f"  + {f.name}")
        if len(files) > 20: console.print(f"  ... and {len(files) - 20} more")
        return

    try:
        r = parallel_ingest_documents(directory, cfg)
    except IngestionLockError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Done:[/bold] {r.processed} files, {r.total_chunks} chunks, {r.duration_seconds/60:.1f}m")
    if r.failed: console.print(f"[red]Failed: {r.failed}[/red]")


@ingestion_app.command("start")
def ingest(path: str = None, recursive: bool = True, dry_run: bool = False, force: bool = False,
           resume: bool = True, workers: int = 4, save_path: bool = False):
    from src.ingestion.pipeline import ingest_document
    from src.config import apply_project_settings

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
        if save_path and active:
            pm.update_project(active.name, docs_dir=str(file_path.absolute()))

    from src.storage.chroma_client import initialize_collections, document_exists
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
    _ingest_parallel(file_path, recursive, dry_run, force, resume, workers)


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


@mcp_app.command("serve")
def mcp_serve(force: bool = False):
    from src.mcp.server import run_server
    from src.mcp.status import get_mcp_status_manager
    from src.config import apply_project_settings
    import time

    apply_project_settings()
    pm = get_project_manager()
    active = pm.get_active_project()
    if active: console.print(f"[dim]Project: {active.name}[/dim]")

    mgr = get_mcp_status_manager()
    if mgr.is_server_running():
        info = mgr.get_server_info()
        if force:
            if sys.platform == "win32":
                import subprocess
                subprocess.run(["taskkill", "/PID", str(info.pid), "/F"], capture_output=True, timeout=5)
            else:
                import signal; os.kill(info.pid, signal.SIGKILL)
            mgr.mark_server_stopped()
            time.sleep(1)
        else:
            console.print(f"[red]Already running (PID {info.pid}). Use --force[/red]")
            raise typer.Exit(1)

    console.print(f"[green]Starting MCP on http://{settings.mcp_host}:{settings.mcp_port}")
    try: run_server()
    except KeyboardInterrupt: console.print("\n[yellow]Stopped[/yellow]")


@mcp_app.command("stop")
def mcp_stop(yes: bool = False, force: bool = False):
    from src.mcp.status import get_mcp_status_manager
    from src.config import apply_project_settings
    import signal as sig

    apply_project_settings()
    mgr = get_mcp_status_manager()
    info = mgr.get_server_info()
    if not info or not mgr.is_server_running():
        console.print("[yellow]Server not running[/yellow]")
        return
    if not yes and not typer.confirm(f"Stop server (PID {info.pid})?"): return

    try:
        if sys.platform == "win32":
            import subprocess
            subprocess.run(["taskkill", "/PID", str(info.pid), "/F"], capture_output=True, timeout=10)
        else:
            os.kill(info.pid, sig.SIGKILL if force else sig.SIGTERM)
        mgr.mark_server_stopped()
        console.print(f"[green]Stopped (PID {info.pid})[/green]")
    except Exception as e:
        console.print(f"[red]Failed: {e}[/red]")


@mcp_app.command("status")
def mcp_status(live: bool = True, refresh: float = 2.0):
    from rich.live import Live
    from rich.panel import Panel
    from src.mcp.status import get_mcp_status_manager
    from src.config import apply_project_settings
    import time as t

    apply_project_settings()
    mgr = get_mcp_status_manager()

    def build():
        info = mgr.get_server_info()
        if not info:
            return Panel("[yellow]Server not started[/yellow]", title="MCP Status", border_style="dim")
        running = mgr.is_server_running()
        s = mgr.get_metrics_summary(60)
        txt = f"Status: {'[green]running[/green]' if running else '[yellow]stopped[/yellow]'}\n"
        txt += f"PID: {info.pid} | Port: {info.port}\n"
        txt += f"Uptime: {_fmt_duration(info.uptime_seconds) if running else '-'}\n\n"
        txt += f"Queries (1h): {s.total_queries} | {s.queries_per_minute:.1f}/min\n"
        txt += f"Avg: {s.avg_response_time_ms:.0f}ms | Errors: {s.error_rate*100:.1f}%"
        return Panel(txt, title="MCP Status", border_style="green" if running else "dim")

    if not live:
        console.print(build())
        return

    try:
        with Live(build(), refresh_per_second=1/refresh, console=console) as lv:
            while True: t.sleep(refresh); lv.update(build())
    except KeyboardInterrupt: pass


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
def models(download: bool = False, verify: bool = False, info: bool = False):
    if download:
        for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]: os.environ.pop(k, None)
        from src.utils import download_embedding_model, download_docling_models
        console.print("[cyan]Downloading embedding model...")
        download_embedding_model()
        console.print("[cyan]Downloading Docling models...")
        download_docling_models()
        for k in ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"]: os.environ[k] = "1"
        console.print(f"[green]Done! Models in {settings.models_dir}")
    elif verify:
        from src.utils import verify_models_exist
        for m, ok in verify_models_exist().items():
            console.print(f"  {m}: {'[green]OK' if ok else '[yellow]Missing'}")
    elif info:
        from src.utils import get_model_paths
        for n, p in get_model_paths().items(): console.print(f"  {n}: {p}")


@config_app.command("cleanup")
def cleanup():
    from src.utils import cleanup_all_resources
    cleanup_all_resources()
    console.print("[green]Cleanup complete")


@config_app.command("device")
def device():
    import torch
    console.print(f"[bold]Device:[/bold] {settings.device}")
    console.print(f"CUDA: {'[green]Yes' if torch.cuda.is_available() else '[dim]No'}")
    if hasattr(torch.backends, 'mps'):
        console.print(f"MPS: {'[green]Yes' if torch.backends.mps.is_available() else '[dim]No'}")


@ingestion_app.command("status")
def ingestion_status(live: bool = True, refresh: float = 2.0, history: bool = False):
    from rich.live import Live
    from rich.panel import Panel
    from src.ingestion.status import get_status_manager
    from src.config import apply_project_settings
    import time as t

    apply_project_settings()
    mgr = get_status_manager()

    if history:
        for s in mgr.get_recent_sessions(10):
            console.print(f"{s.session_id} | {Path(s.source_path).name} | {s.processed_files} files | {s.status}")
        return

    def build():
        sess = mgr.get_active_session()
        if not sess:
            return Panel("[yellow]No active ingestion[/yellow]", title="Ingestion Status", border_style="dim")
        pct = (sess.processed_files + sess.failed_files) / max(sess.total_files - sess.skipped_files, 1) * 100
        txt = f"Session: {sess.session_id}\n"
        txt += f"Progress: {sess.processed_files}/{sess.total_files} ({pct:.0f}%)\n"
        txt += f"Rate: {sess.rate:.2f} docs/sec | Chunks: {sess.total_chunks}\n"
        txt += f"Elapsed: {_fmt_duration(sess.elapsed_seconds)}"
        if sess.failed_files: txt += f"\n[red]Failed: {sess.failed_files}[/red]"
        return Panel(txt, title="Ingestion Status", border_style="green")

    if not live:
        console.print(build())
        return

    try:
        with Live(build(), refresh_per_second=1/refresh, console=console) as lv:
            while True:
                t.sleep(refresh)
                lv.update(build())
                if not mgr.get_active_session(): break
    except KeyboardInterrupt: pass


@ingestion_app.command("stop")
def ingestion_stop(yes: bool = False, force: bool = False):
    from src.ingestion.status import get_status_manager
    from src.config import apply_project_settings
    import signal as sig

    apply_project_settings()
    mgr = get_status_manager()
    sess = mgr.get_active_session()
    if not sess:
        console.print("[yellow]No active session[/yellow]")
        return
    if not yes and not typer.confirm(f"Stop session {sess.session_id}?"): return

    mgr.send_stop_signal(sess.session_id)
    if force:
        for pid in mgr.get_worker_pids(sess.session_id):
            try:
                if sys.platform == "win32":
                    import subprocess; subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
                else: os.kill(pid, sig.SIGKILL)
            except: pass
        if main := mgr.get_main_pid(sess.session_id):
            try:
                if sys.platform == "win32":
                    import subprocess; subprocess.run(["taskkill", "/PID", str(main), "/F"], capture_output=True)
                else: os.kill(main, sig.SIGKILL)
            except: pass
    mgr.complete_session(sess.session_id, "stopped")
    console.print("[green]Stopped[/green]")


@ingestion_app.command("log")
def ingestion_log(limit: int = 20, export: str = None):
    import csv, shutil
    from src.ingestion.audit_log import get_audit_log_path
    _show_project()

    csv_path = get_audit_log_path()
    if not csv_path.exists():
        console.print("[yellow]No log found")
        return
    if export:
        shutil.copy(csv_path, export)
        console.print(f"[green]Exported to {export}")
        return

    with open(csv_path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        console.print("[yellow]Empty log")
        return

    t = Table(title=f"Ingestion Log ({min(limit, len(rows))}/{len(rows)})", box=None)
    t.add_column("Time", width=16); t.add_column("File", width=30); t.add_column("Chunks", justify="right")
    t.add_column("Status"); t.add_column("Duration", justify="right")

    for r in list(reversed(rows))[:limit]:
        status = {"completed": "[green]OK", "failed": "[red]FAIL", "resumed": "[yellow]resumed"}.get(r['status'], r['status'])
        t.add_row(r['timestamp'][:16], Path(r['file_name']).name[:30], r['num_chunks'], status,
                  f"{float(r['duration_seconds']):.1f}s" if r['duration_seconds'] else "-")
    console.print(t)


@ingestion_app.command("checkpoints-list")
def checkpoints_list():
    from src.ingestion.checkpoint import list_checkpoints
    cps = list_checkpoints()
    if not cps:
        console.print("[yellow]No checkpoints")
        return
    for c in cps:
        console.print(f"  {c['doc_id'][-8:]} | {Path(c['file_path']).name} | {c['processed_batches']}/{c['total_chunks']}")


@ingestion_app.command("checkpoints-clean")
def checkpoints_clean(yes: bool = False, stale_only: bool = False):
    from src.ingestion.checkpoint import clean_all_checkpoints, clean_stale_checkpoints
    if stale_only:
        n = clean_stale_checkpoints()
    else:
        if not yes and not typer.confirm("Delete ALL checkpoints?"): return
        n = clean_all_checkpoints()
    console.print(f"[green]Deleted {n} checkpoint(s)" if n else "[yellow]None found")


# === Project Commands ===

@project_app.command("create")
def project_create(name: str, port: int = 9090, data_dir: str = None, docs_dir: str = None):
    pm = get_project_manager()
    try:
        cfg = pm.create_project(name=name, port=port, data_dir=data_dir, docs_dir=docs_dir)
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


@project_app.command("config")
def project_config(port: int = None, data_dir: str = None, docs_dir: str = None):
    pm = get_project_manager()
    active = pm.get_active_project()
    if not active:
        console.print("[yellow]No active project")
        raise typer.Exit(1)

    if any([port, data_dir, docs_dir]):
        try:
            active = pm.update_project(active.name, port=port, data_dir=data_dir, docs_dir=docs_dir)
            console.print(f"[green]Updated '{active.name}'[/green]")
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)

    paths = pm.get_project_paths(active.name)
    console.print(f"[bold]{active.name}[/bold] (:{active.port})")
    console.print(f"  MCP: {active.mcp_server_name}")
    console.print(f"  Data: {paths['data_dir']}")
    console.print(f"  Docs: {paths['docs_dir']}")


@project_app.command("delete")
def project_delete(name: str, data: bool = False, yes: bool = typer.Option(False, "-y")):
    pm = get_project_manager()
    if not pm.project_exists(name):
        console.print(f"[red]Not found: {name}[/red]")
        raise typer.Exit(1)
    if not yes and not typer.confirm(f"Delete '{name}'{'and data' if data else ''}?"): return
    pm.delete_project(name, delete_data=data)
    console.print(f"[green]Deleted '{name}'[/green]")


@project_app.command("info")
def project_info(name: str = None):
    pm = get_project_manager()
    cfg = pm.get_project(name) if name else pm.get_active_project()
    if not cfg:
        console.print("[yellow]No project")
        raise typer.Exit(1)

    paths = pm.get_project_paths(cfg.name)
    stats = pm.get_project_stats(cfg.name)
    is_active = pm.get_active_project_name() == cfg.name

    console.print(f"[bold]{cfg.name}[/bold]{'[green] (active)' if is_active else ''}")
    console.print(f"  Port: {cfg.port} | MCP: {cfg.mcp_server_name}")
    console.print(f"  Data: {paths['data_dir']}")
    console.print(f"  Docs: {paths['docs_dir']}")
    console.print(f"  Stats: {stats['documents']} docs, {stats['disk_usage_mb']} MB")


@app.command()
def init():
    pm = get_project_manager()
    if pm.list_projects():
        console.print("[yellow]Already initialized")
        project_list()
        return

    console.print("[bold cyan]RAG Setup[/bold cyan]\n")
    name = typer.prompt("Project name", default="default")
    port = int(typer.prompt("Port", default="9090"))

    try:
        cfg = pm.create_project(name=name, port=port)
        paths = pm.get_project_paths(name)
        console.print(f"\n[green]Created '{name}'![/green]")
        console.print(f"  Docs: {paths['docs_dir']}")
        console.print("\nNext: rag ingestion start")
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)


app.add_typer(project_app, name="project")
app.add_typer(ingestion_app, name="ingestion")
app.add_typer(config_app, name="config")
app.add_typer(mcp_app, name="mcp")

if __name__ == "__main__":
    try: app()
    except KeyboardInterrupt: raise typer.Exit(0)
