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
from rich.logging import RichHandler
from rich.table import Table
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

app = typer.Typer()

console = Console(force_terminal=True, legacy_windows=False)

# Single RichHandler for all logging — no other handlers anywhere
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%d-%b %H:%M:%S]",
    handlers=[RichHandler(console=console, show_path=False, markup=False, omit_repeated_times=False)],
    force=True,
)

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
    from src.storage.postgres import create_collection, list_documents
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
    from src.storage.postgres import create_collection, remove_document_by_id
    create_collection()
    if not yes and not typer.confirm(f"Remove document '{doc_id[-6:]}'?"): return
    n = remove_document_by_id(doc_id)
    console.print(f"[green]Removed ({n} chunks)" if n else f"[yellow]Not found: {doc_id}")


@app.command()
def ingest(
    env: str,
    recursive: bool = True,
    dry_run: bool = False,
    force: bool = False,
    folders: str = typer.Option(None, help="Pipe-separated folder names to include (overrides INCLUDE_FOLDERS)"),
    workers: int = typer.Option(3, "--workers", "-w", help="Number of parallel workers for ingestion")
):
    _load_env(env)
    from src.config import config, get_logger
    from src.ingestion.pipeline import ingest_document
    from src.utils import discover_files, is_supported_file, managed_resources
    from src.storage.postgres import create_collection, document_exists

    logger = get_logger(__name__)
    create_collection()
    doc_path = config("DOCUMENTS_DIR")

    # Resolve include_folders: CLI --folders overrides env var
    include_folders = None
    if folders:
        include_folders = [f.strip() for f in folders.split("|") if f.strip()]
    else:
        include_folders = config("INCLUDE_FOLDERS")

    if include_folders:
        logger.info(f"Folder filter active: {include_folders}")
        console.print(f"[cyan]Filtering to folders: {', '.join(include_folders)}[/cyan]")

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
        files = discover_files(doc_path, recursive=recursive, include_folders=include_folders)
        console.print(f"[cyan]Scanning {doc_path}...[/cyan]")

    # Safety: use a lock for thread‑safe checks against the DB and shared resources
    lock = threading.Lock()
    processed, skipped, failed = 0, 0, 0

    def _process_file(f: Path):
        """
        Process a single file.
        Returns a tuple (processed: bool, skipped: bool, failed: bool).
        """
        if dry_run:
            console.print(f"  + {f.name}")
            return True, False, False

        # Check existence in a thread‑safe manner
        if not force:
            with lock:
                if document_exists(f):
                    return False, True, False

        try:
            meta = ingest_document(f)
            console.print(f"[green]{f.name} ({meta.num_chunks} chunks)")
            return True, False, False
        except Exception as e:
            logger.error(f"{f.name}: {e}")
            console.print(f"[red]{f.name}: {e}")
            return False, False, True

    # ----------------------------------------------------------------------
    # Resource management
    # ----------------------------------------------------------------------
    # All resources (database connections, embedder cache, etc.) are created
    # inside the ``managed_resources`` context manager. Wrapping the parallel
    # ingestion loop ensures they are released even when we run multiple threads.
    with managed_resources():
        # Parallel execution using ThreadPoolExecutor
        with console.status(""), ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_file, f): f for f in files}
            for future in as_completed(futures):
                proc, skip, fail = future.result()
                if proc:
                    processed += 1
                if skip:
                    skipped += 1
                if fail:
                    failed += 1

        console.print("[cyan]ThreadPoolExecutor completed ...[/cyan]")

    logger.info("Resource cleanup finished.")
    console.print("[green]All resources cleaned up successfully.[/green]")

    console.print(f"\n[bold]Done:[/bold] {processed} processed, {skipped} skipped, {failed} failed")


@app.command()
def query(env: str, query_text: str, top_k: int = None, format: str = "json",
          groups: str = typer.Option(None, help="Comma-separated group names for RBAC filtering")):
    _load_env(env)
    from src.config import DEFAULT_TOP_K
    from src.query import query as query_fn
    import json

    group_list = [g.strip() for g in groups.split(",") if g.strip()] if groups else None
    r = query_fn(query_text, top_k or DEFAULT_TOP_K, groups=group_list)
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
    from src.storage.postgres import create_collection, get_stats, list_documents
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
    from src.storage.postgres import reset_collection
    if typer.confirm("Reset system?"): reset_collection(); console.print("[green]Reset complete")


@app.command()
def serve(env: str):
    _load_env(env)
    from src.config import config, MCP_HOST
    from src.storage.postgres import create_collection
    create_collection()
    from src.mcp.server import run_server
    port = config("MCP_PORT")
    console.print(f"[green]Starting server ({config('MCP_SERVER_NAME')}) on http://{MCP_HOST}:{port}")
    console.print(f"  MCP:       http://{MCP_HOST}:{port}/mcp")
    console.print(f"  API:       http://{MCP_HOST}:{port}/api")
    console.print(f"  Dashboard: http://{MCP_HOST}:{port}/")
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


# ============================================================
# RBAC management commands
# ============================================================

@app.command("add-group")
def add_group(env: str, name: str, description: str = ""):
    _load_env(env)
    from src.storage.postgres import create_collection, create_group
    create_collection()
    g = create_group(name, description)
    console.print(f"[green]Group created: {g['name']} (id={g['id']})")


@app.command("list-groups")
def list_groups_cmd(env: str):
    _load_env(env)
    from src.storage.postgres import create_collection, list_groups
    create_collection()
    groups = list_groups()
    if not groups:
        console.print("[yellow]No groups found")
        return
    t = Table(title="Groups", box=None)
    t.add_column("ID", style="dim", width=6)
    t.add_column("Name", style="cyan")
    t.add_column("Description", style="white")
    for g in groups:
        t.add_row(str(g["id"]), g["name"], g["description"])
    console.print(t)


@app.command("add-user")
def add_user(env: str, username: str, password: str = typer.Option(None),
             display_name: str = "", email: str = "",
             admin: bool = False, auth_type: str = "local"):
    _load_env(env)
    from src.auth.auth import hash_password
    from src.storage.postgres import create_collection, create_user
    create_collection()
    pw_hash = hash_password(password) if password else None
    if auth_type == "local" and not pw_hash:
        console.print("[red]Password required for local users")
        raise typer.Exit(1)
    u = create_user(username, pw_hash, display_name, email, admin, auth_type)
    console.print(f"[green]User created: {u['username']} (id={u['id']}, admin={u['is_admin']})")


@app.command("list-users")
def list_users_cmd(env: str):
    _load_env(env)
    from src.storage.postgres import create_collection, list_users
    create_collection()
    users = list_users()
    if not users:
        console.print("[yellow]No users found")
        return
    t = Table(title="Users", box=None)
    t.add_column("ID", style="dim", width=6)
    t.add_column("Username", style="cyan")
    t.add_column("Name", style="white")
    t.add_column("Admin", style="magenta", width=6)
    t.add_column("Active", style="green", width=6)
    t.add_column("Auth", style="blue", width=6)
    for u in users:
        t.add_row(str(u["id"]), u["username"], u["display_name"],
                  str(u["is_admin"]), str(u["is_active"]), u["auth_type"])
    console.print(t)


@app.command("assign-group")
def assign_group_cmd(env: str, username: str, group_name: str):
    _load_env(env)
    from src.storage.postgres import create_collection, get_user_by_username, assign_user_to_group, list_groups
    create_collection()
    user = get_user_by_username(username)
    if not user:
        console.print(f"[red]User not found: {username}")
        raise typer.Exit(1)
    groups = list_groups()
    group = next((g for g in groups if g["name"] == group_name), None)
    if not group:
        console.print(f"[red]Group not found: {group_name}")
        raise typer.Exit(1)
    assign_user_to_group(user["id"], group["id"])
    console.print(f"[green]Assigned {username} → {group_name}")


@app.command("assign-path")
def assign_path_cmd(env: str, path: str, group_name: str):
    _load_env(env)
    from src.storage.postgres import create_collection, add_path_permission, list_groups
    create_collection()
    groups = list_groups()
    group = next((g for g in groups if g["name"] == group_name), None)
    if not group:
        console.print(f"[red]Group not found: {group_name}")
        raise typer.Exit(1)
    add_path_permission(path, group["id"])
    console.print(f"[green]Path permission: {path} → {group_name}")


@app.command("list-permissions")
def list_permissions_cmd(env: str):
    _load_env(env)
    from src.storage.postgres import create_collection, list_path_permissions
    create_collection()
    perms = list_path_permissions()
    if not perms:
        console.print("[yellow]No path permissions found")
        return
    t = Table(title="Path Permissions", box=None)
    t.add_column("Path", style="cyan")
    t.add_column("Group", style="magenta")
    for p in perms:
        t.add_row(p["path"], p["group_name"])
    console.print(t)


@app.command("reset-password")
def reset_password_cmd(env: str, username: str, new_password: str = typer.Option(..., prompt=True, hide_input=True)):
    _load_env(env)
    from src.auth.auth import hash_password
    from src.storage.postgres import create_collection, get_user_by_username, update_user
    create_collection()
    user = get_user_by_username(username)
    if not user:
        console.print(f"[red]User not found: {username}")
        raise typer.Exit(1)
    update_user(user["id"], password_hash=hash_password(new_password), must_change_password=True)
    console.print(f"[green]Password reset for {username}. User will be prompted to change on next login.")


@app.command("fix-paths")
def fix_paths_cmd(env: str):
    """Normalize all paths to forward slashes, recompute doc_id and document_permissions."""
    _load_env(env)
    from src.storage.postgres import create_collection, get_pool
    create_collection()
    with get_pool().connection() as conn:
        # 1. Normalize file_path in chunks
        cur = conn.execute(r"UPDATE chunks SET file_path = REPLACE(file_path, '\', '/') WHERE file_path LIKE '%\\%'")
        chunks_fixed = cur.rowcount
        console.print(f"  Normalized {chunks_fixed} chunk file_paths")

        # 2. Normalize path in path_permissions
        cur = conn.execute(r"UPDATE path_permissions SET path = RTRIM(REPLACE(path, '\', '/'), '/') WHERE path LIKE '%\\%' OR path LIKE '%/'")
        perms_fixed = cur.rowcount
        console.print(f"  Normalized {perms_fixed} permission paths")

        # 3. Deduplicate path_permissions
        conn.execute("DELETE FROM path_permissions pp WHERE pp.id NOT IN (SELECT MIN(id) FROM path_permissions GROUP BY path, group_id)")

        # 4. Recompute doc_id = md5(file_path) so it's based on normalized path
        conn.execute("""
            CREATE TEMP TABLE doc_id_map AS
            SELECT DISTINCT doc_id AS old_doc_id, md5(file_path) AS new_doc_id
            FROM chunks WHERE doc_id != md5(file_path)
        """)
        cur = conn.execute("SELECT COUNT(*) FROM doc_id_map")
        docs_to_fix = cur.fetchone()[0]
        if docs_to_fix > 0:
            conn.execute("UPDATE chunks c SET doc_id = m.new_doc_id FROM doc_id_map m WHERE c.doc_id = m.old_doc_id")
            conn.execute("UPDATE document_permissions dp SET doc_id = m.new_doc_id FROM doc_id_map m WHERE dp.doc_id = m.old_doc_id")
            console.print(f"  Recomputed doc_id for {docs_to_fix} documents")
        else:
            console.print(f"  All doc_ids already consistent")
        conn.execute("DROP TABLE doc_id_map")

        # 5. Recompute document_permissions
        conn.execute("TRUNCATE document_permissions")
        cur = conn.execute("""
            INSERT INTO document_permissions (doc_id, group_id)
            SELECT DISTINCT c.doc_id, pp.group_id
            FROM (SELECT DISTINCT doc_id, file_path FROM chunks) c
            JOIN path_permissions pp
              ON c.file_path = pp.path
              OR c.file_path LIKE pp.path || '/%'
            ON CONFLICT (doc_id, group_id) DO NOTHING
        """)
        doc_perms = cur.rowcount
        conn.commit()
    console.print(f"[green]Done! {doc_perms} document permission entries computed")


@app.command("refresh-permissions")
def refresh_permissions_cmd(env: str):
    _load_env(env)
    from src.storage.postgres import create_collection, refresh_all_document_permissions
    create_collection()
    count = refresh_all_document_permissions()
    console.print(f"[green]Refreshed permissions for {count} documents")


@app.command()
def dashboard(env: str):
    """Alias for 'serve' — starts the combined server."""
    serve(env)


if __name__ == "__main__":
    try: app()
    except KeyboardInterrupt: raise typer.Exit(0)
