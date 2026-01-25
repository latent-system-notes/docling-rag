from pathlib import Path
import sys
import os

# Fix Windows console encoding for Arabic/Unicode characters
if sys.platform == "win32":
    # Set UTF-8 mode for Python I/O
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    # Set console code page to UTF-8
    os.system('chcp 65001 > nul 2>&1')

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from src.config import settings, get_logger
from src.storage.chroma_client import get_stats, initialize_collections, reset_collection, document_exists, list_documents, remove_document, remove_document_by_id
from src.utils import discover_files, is_file_modified
from src.project import get_project_manager, ProjectManager

logger = get_logger(__name__)

app = typer.Typer()
project_app = typer.Typer(help="Manage RAG projects")
ingestion_app = typer.Typer(help="Document ingestion commands")
config_app = typer.Typer(help="Configuration commands")
mcp_app = typer.Typer(help="MCP server management commands")
# Force UTF-8 output for Rich console (better Unicode support)
console = Console(force_terminal=True, legacy_windows=False)


@app.command()
def list_docs(
    full_path: bool = typer.Option(False, "--full-path", "-f", help="Show full file paths instead of just filenames"),
    limit: int = typer.Option(None, "--limit", "-l", help="Maximum number of documents to show"),
    offset: int = typer.Option(0, "--offset", "-o", help="Number of documents to skip (for pagination)"),
):
    """List all indexed documents with audit information.

    Shows:
    - File name (or full path with --full-path)
    - Document type
    - Language
    - Number of chunks
    - Ingestion timestamp

    Examples:
        rag list-docs                    # Show all documents
        rag list-docs --limit 20         # Show first 20 documents
        rag list-docs --limit 20 --offset 20  # Show next 20 documents (page 2)
    """
    from datetime import datetime
    from pathlib import Path
    from src.config import apply_project_settings

    # Apply project settings if active
    if apply_project_settings():
        pm = get_project_manager()
        active = pm.get_active_project()
        if active:
            console.print(f"[dim]Project: {active.name}[/dim]")

    def truncate_filename(filename: str, max_length: int = 55) -> str:
        """Truncate long filenames intelligently, keeping extension visible."""
        if len(filename) <= max_length:
            return filename

        # Get file extension
        path = Path(filename)
        ext = path.suffix  # e.g., ".pdf"
        name = path.stem   # filename without extension

        # Calculate how many chars we can keep from the name
        # Reserve space for "..." (3 chars) and extension
        available = max_length - len(ext) - 3

        if available > 10:  # Only truncate if we have reasonable space
            truncated_name = name[:available] + "..."
            return truncated_name + ext
        else:
            # If name is too short to truncate nicely, just cut it
            return filename[:max_length-3] + "..."

    initialize_collections()

    # Apply pagination
    documents = list_documents(limit=limit, offset=offset)

    # Get total count for display
    total_count = len(list_documents())

    if not documents:
        if offset > 0:
            console.print(f"[yellow]No documents found at offset {offset}. Total documents: {total_count}")
        else:
            console.print("[yellow]No documents indexed yet.")
        return

    # Create table title with pagination info
    if limit or offset:
        showing = len(documents)
        start = offset + 1
        end = offset + showing
        title = f"Indexed Documents (showing {start}-{end} of {total_count} total)"
    else:
        title = f"Indexed Documents ({total_count} total)"

    table = Table(title=title, box=None, show_header=True)
    table.add_column("ID", style="dim", width=6)
    table.add_column("File Name", style="cyan", no_wrap=True)
    table.add_column("Type", style="magenta", width=6)
    table.add_column("Lang", style="blue", width=6)
    table.add_column("Chunks", style="green", justify="right", width=8)
    table.add_column("Ingested At", style="yellow", width=19)

    for doc in documents:
        # Format timestamp
        ingested_str = doc['ingested_at']
        if ingested_str != 'unknown':
            try:
                dt = datetime.fromisoformat(ingested_str)
                ingested_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass

        # Show filename only (or full path if requested)
        if full_path:
            file_display = doc['file_path']
        else:
            filename = Path(doc['file_path']).name
            file_display = truncate_filename(filename)

        # Show last 6 digits of doc_id
        doc_id_short = doc['doc_id'][-6:] if len(doc['doc_id']) >= 6 else doc['doc_id']

        table.add_row(
            doc_id_short,
            file_display,
            doc['doc_type'],
            doc['language'],
            str(doc['num_chunks']),
            ingested_str
        )

    console.print(table)

    # Summary statistics
    shown_chunks = sum(doc['num_chunks'] for doc in documents)
    if limit or offset:
        console.print(f"\n[bold]Showing:[/bold] {len(documents)} documents, {shown_chunks} chunks")
        console.print(f"[bold]Total in database:[/bold] {total_count} documents")
    else:
        console.print(f"\n[bold]Total:[/bold] {total_count} documents, {shown_chunks} chunks")


@app.command()
def remove(
    doc_id: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove a document from the index by its ID.

    Use the last 6 digits shown in 'rag list-docs' or the full document ID.

    Examples:
        rag remove b77745                # Remove by last 6 digits
        rag remove cafe35ce5ec5228628e94420e75b7745 -y  # Remove by full ID
    """
    from src.config import apply_project_settings

    # Apply project settings if active
    if apply_project_settings():
        pm = get_project_manager()
        active = pm.get_active_project()
        if active:
            console.print(f"[dim]Project: {active.name}[/dim]")

    initialize_collections()

    # Confirm deletion unless --yes flag
    if not yes:
        if not typer.confirm(f"Remove document with ID ending in '{doc_id[-6:]}'?"):
            console.print("[yellow]Cancelled.")
            return

    try:
        num_removed = remove_document_by_id(doc_id)

        if num_removed > 0:
            console.print(f"[green]Removed document ({num_removed} chunks)")
        else:
            console.print(f"[yellow]Document not found with ID: {doc_id}")
    except Exception as e:
        console.print(f"[red]Error: {str(e)}")
        raise typer.Exit(1)


def _ingest_parallel(directory: Path, recursive: bool, dry_run: bool, force: bool, resume: bool, workers: int):
    """Handle parallel ingestion with Rich progress display."""
    from src.ingestion.parallel_pipeline import (
        parallel_ingest_documents,
        ParallelIngestionConfig,
        collect_files
    )
    from src.ingestion.lock import IngestionLockError

    console.print(f"[cyan]Parallel ingestion mode: {workers} workers[/cyan]")
    console.print(f"[cyan]Scanning {directory}...[/cyan]")

    # Collect files first for dry-run support
    config = ParallelIngestionConfig(
        num_workers=workers,
        recursive=recursive,
        force=force,
        resume=resume
    )

    files = collect_files(directory, config.extensions, recursive)

    if not files:
        console.print(f"[yellow]No supported files found in {directory}")
        return

    console.print(f"[cyan]Found {len(files)} documents[/cyan]")

    if dry_run:
        console.print("\n[yellow]Dry run - no changes will be made[/yellow]")
        console.print("\n[bold]Would process with parallel ingestion:[/bold]")
        for f in files[:20]:  # Show first 20
            # Sanitize filename for console output (handle non-ASCII characters)
            safe_name = f.name.encode('ascii', 'replace').decode('ascii')
            console.print(f"  + {safe_name}")
        if len(files) > 20:
            console.print(f"  ... and {len(files) - 20} more files")
        return

    # Run parallel ingestion
    console.print("")
    try:
        result = parallel_ingest_documents(directory, config)
    except IngestionLockError as e:
        console.print(f"\n[red]Error: {e}[/red]")
        console.print("\n[dim]To monitor progress: rag ingestion status")
        console.print("[dim]To stop current ingestion: rag ingestion stop[/dim]")
        raise typer.Exit(1)

    # Display results
    console.print(f"\n[bold]Parallel Ingestion Complete:[/bold]")
    if result.session_id:
        console.print(f"  [dim]Session:[/dim] {result.session_id}")
    console.print(f"  [green]Processed:[/green] {result.processed} files")
    if result.resumed > 0:
        console.print(f"    - New: {result.processed - result.resumed}")
        console.print(f"    - Resumed: {result.resumed}")
    if result.skipped > 0:
        console.print(f"  [dim]Skipped:[/dim] {result.skipped} files (already completed)")
    if result.failed > 0:
        console.print(f"  [red]Failed:[/red] {result.failed} files")
    console.print(f"  [cyan]Total chunks:[/cyan] {result.total_chunks}")
    console.print(f"  [dim]Duration:[/dim] {result.duration_seconds/60:.1f} minutes")

    if result.errors:
        console.print(f"\n[red]Errors:[/red]")
        for err in result.errors[:5]:  # Show first 5 errors
            console.print(f"  - {Path(err['file']).name}: {err['error'][:50]}")
        if len(result.errors) > 5:
            console.print(f"  ... and {len(result.errors) - 5} more errors")


@ingestion_app.command("start")
def ingest(
    path: str = typer.Argument(None, help="Path to file or folder (default: project's docs folder)"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan subdirectories"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be updated without doing it"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already exists"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from checkpoint if exists (default: enabled)"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers (default: 4)"),
    save_path: bool = typer.Option(False, "--save-path", "-s", help="Save this path as project's default docs folder"),
):
    """Ingest file(s) or folder using parallel processing.

    If no path is provided, uses the active project's configured docs folder.
    Use --save-path to remember a new path for future ingestions.

    For single files: Ingests the file directly
    For folders: Uses parallel workers for fast ingestion
    - Automatically skips already indexed files
    - Supports checkpoint-based resume for interrupted ingestion

    Uses multiple worker processes for CPU-intensive parsing,
    with a single writer thread for database operations.

    Examples:
        rag ingestion start                          # Use project's docs folder
        rag ingestion start paper.pdf                # Ingest single file
        rag ingestion start ./documents              # Ingest folder (4 workers)
        rag ingestion start ./documents --save-path  # Ingest and remember this path
        rag ingestion start ./docs --workers 8       # Use 8 workers
        rag ingestion start ./docs --dry-run         # Preview what would be ingested
        rag ingestion start ./docs --force           # Force re-ingest all
    """
    from src.ingestion.pipeline import ingest_document
    from src.config import apply_project_settings

    # Apply project settings if active
    pm = get_project_manager()
    active_project = None

    if apply_project_settings():
        active_project = pm.get_active_project()
        if active_project:
            console.print(f"[dim]Project: {active_project.name}[/dim]")

    # Resolve path: use provided path or fall back to project's docs_path
    if path is None:
        if active_project:
            # Use project's configured docs_path
            paths = pm.get_project_paths(active_project.name)
            file_path = paths["docs_path"]
            console.print(f"[dim]Using project docs folder: {file_path}[/dim]")
        else:
            console.print("[red]Error: No path provided and no active project.[/red]")
            console.print("Either specify a path or switch to a project first:")
            console.print("  rag ingestion start ./documents")
            console.print("  rag project switch <name>")
            raise typer.Exit(1)
    else:
        file_path = Path(path)

        # Optionally save this path as the project's docs folder
        if save_path and active_project:
            abs_path = file_path.absolute()
            pm.update_project(active_project.name, docs_path=str(abs_path))
            console.print(f"[dim]Saved as project docs folder: {abs_path}[/dim]")

    initialize_collections()

    # Handle single file
    if file_path.is_file():
        if not force and document_exists(file_path):
            console.print(f"[yellow]Document already exists. Use --force to re-ingest.")
            return

        try:
            metadata = ingest_document(file_path, resume=resume)
            console.print(f"[green][OK] Ingested: {file_path.name} ({metadata.num_chunks} chunks)")
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            console.print(f"[red][ERR] Failed to ingest {file_path.name}: {str(e)}")
            raise typer.Exit(1)
        return

    # Handle folder - always use parallel ingestion
    if not file_path.is_dir():
        console.print(f"[red]Error: {path} is not a file or directory")
        raise typer.Exit(1)

    _ingest_parallel(file_path, recursive, dry_run, force, resume, workers)


@app.command()
def query(
    query_text: str,
    top_k: int = settings.default_top_k,
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or text"),
):
    """Query the RAG system and retrieve relevant context chunks"""
    from src.query import query as query_fn
    from src.config import apply_project_settings
    from pathlib import Path
    import json

    # Apply project settings if active
    if apply_project_settings():
        pm = get_project_manager()
        active = pm.get_active_project()
        if active and format == "text":
            console.print(f"[dim]Project: {active.name}[/dim]")

    result = query_fn(query_text, top_k)

    if format == "json":
        # JSON format with all information
        output = {
            "query": result.query,
            "total_results": len(result.context),
            "results": []
        }

        for idx, res in enumerate(result.context, 1):
            chunk_data = {
                "rank": idx,
                "score": round(res.score, 4),
                "distance": round(res.distance, 4),
                "chunk": {
                    "text": res.chunk.text,
                    "page_num": res.chunk.page_num,
                    "doc_id": res.chunk.doc_id,
                },
                "document": {
                    "file_name": Path(res.chunk.metadata.get("file_path", "unknown")).name,
                    "file_path": res.chunk.metadata.get("file_path", "unknown"),
                    "doc_type": res.chunk.metadata.get("doc_type", "unknown"),
                    "language": res.chunk.metadata.get("language", "unknown"),
                    "ingested_at": res.chunk.metadata.get("ingested_at", "unknown"),
                }
            }
            output["results"].append(chunk_data)

        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        # Text format (legacy)
        console.print(f"[bold]Query:[/bold] {result.query}\n")

        if result.context:
            console.print(f"[bold]Found {len(result.context)} relevant chunks:[/bold]\n")
            for idx, res in enumerate(result.context, 1):
                file_name = Path(res.chunk.metadata.get("file_path", "unknown")).name
                page_info = f"page {res.chunk.page_num}" if res.chunk.page_num else "no page"

                console.print(f"[cyan]Result {idx}[/cyan] (Score: {res.score:.3f})")
                console.print(f"  File: {file_name} ({page_info})")
                console.print(f"  Text: {res.chunk.text[:200]}...")
                console.print()
        else:
            console.print("[yellow]No relevant context found[/yellow]")


@app.command()
def stats():
    """Show system statistics"""
    from src.config import apply_project_settings

    # Apply project settings if active
    if apply_project_settings():
        pm = get_project_manager()
        active = pm.get_active_project()
        if active:
            console.print(f"[dim]Project: {active.name}[/dim]")

    initialize_collections()

    statistics = get_stats()
    documents = list_documents()

    table = Table(title="System Statistics", box=None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    # Add document count first
    table.add_row("total_documents", str(len(documents)))

    for key, value in statistics.items():
        table.add_row(key, str(value))

    console.print(table)


@app.command()
def reset():
    """Reset the system (clear all data)"""
    from src.config import apply_project_settings

    # Apply project settings if active
    project_active = apply_project_settings()
    if project_active:
        pm = get_project_manager()
        active = pm.get_active_project()
        if active:
            console.print(f"[dim]Project: {active.name}[/dim]")

    if typer.confirm("Are you sure you want to reset the system?"):
        reset_collection()
        console.print("[green]System reset complete")


@mcp_app.command("serve")
def mcp_serve(
    mcp: bool = True,
    force: bool = typer.Option(False, "--force", "-f", help="Kill existing server if running"),
):
    """Start MCP server for the active project.

    If a server is already running for this project, the command will fail
    unless --force is used to kill the existing server first.

    Examples:
        rag mcp serve              # Start MCP server
        rag mcp serve --force      # Kill existing server and start new one
    """
    from src.mcp.server import run_server
    from src.mcp.status import get_mcp_status_manager
    from src.config import apply_project_settings

    # Apply project settings if active
    apply_project_settings()

    if mcp:
        # Show project info if active
        pm = get_project_manager()
        active = pm.get_active_project()
        if active:
            console.print(f"[dim]Project: {active.name}[/dim]")

        # Check if server is already running
        status_mgr = get_mcp_status_manager()
        if status_mgr.is_server_running():
            server_info = status_mgr.get_server_info()
            if force:
                # Kill existing server
                console.print(f"[yellow]Killing existing MCP server (PID {server_info.pid})...[/yellow]")
                try:
                    if sys.platform == "win32":
                        import subprocess
                        subprocess.run(
                            ["taskkill", "/PID", str(server_info.pid), "/F"],
                            capture_output=True,
                            timeout=5
                        )
                    else:
                        import os
                        import signal as sig
                        os.kill(server_info.pid, sig.SIGKILL)
                    status_mgr.mark_server_stopped()
                    console.print("[green]Existing server terminated.[/green]")
                    # Give it a moment to release the port
                    import time
                    time.sleep(1)
                except Exception as e:
                    console.print(f"[red]Failed to kill existing server: {e}[/red]")
                    raise typer.Exit(1)
            else:
                console.print(f"[red]Error: MCP server is already running (PID {server_info.pid})[/red]")
                console.print("[dim]Use --force to kill existing server and start a new one.[/dim]")
                console.print("[dim]Or use 'rag mcp stop' to stop the existing server.[/dim]")
                raise typer.Exit(1)

        console.print(f"[green]Starting MCP server on http://{settings.mcp_host}:{settings.mcp_port}")

        try:
            run_server()
        except KeyboardInterrupt:
            console.print("\n[yellow]MCP server stopped.[/yellow]")


@config_app.command("show")
def config_show():
    """Show current configuration"""
    table = Table(title="Configuration", box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    config_dict = settings.model_dump()
    for key, value in config_dict.items():
        table.add_row(key, str(value))

    console.print(table)


@config_app.command("models")
def models(
    download: bool = typer.Option(False, "--download", help="Download all models"),
    verify: bool = typer.Option(False, "--verify", help="Verify models exist"),
    info: bool = typer.Option(False, "--info", help="Show model information"),
):
    """Manage offline models"""
    if download:
        import os

        console.print("[yellow]Downloading models to ./models...")
        console.print("[dim]Temporarily disabling offline mode for download...[/dim]")

        # Temporarily disable offline mode for downloading
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_DATASETS_OFFLINE", None)

        from src.utils import download_embedding_model, download_docling_models

        # Download embedding model
        console.print("\n[cyan]1. Downloading embedding model...")
        download_embedding_model()
        console.print("[green]OK Embedding model downloaded")

        # Download Docling layout models
        console.print("\n[cyan]2. Downloading Docling layout models...")
        download_docling_models()
        console.print("[green]OK Docling layout models downloaded")

        # Re-enable offline mode
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

        console.print(f"\n[green]All models downloaded successfully!")
        console.print(f"[green]Models saved to: {settings.models_dir}")
        console.print("\n[bold]System is now ready for offline operation![/bold]")

    elif verify:
        from src.utils import verify_models_exist

        status = verify_models_exist()

        table = Table(title="Model Status", box=None)
        table.add_column("Model", style="cyan")
        table.add_column("Status", style="green")

        for model, exists in status.items():
            status_text = "Downloaded" if exists else "Not found"
            table.add_row(model, status_text)

        console.print(table)

        if not all(status.values()):
            console.print("[yellow]Run 'rag config models --download' to download missing models")

    elif info:
        from src.utils import get_model_paths

        paths = get_model_paths()

        table = Table(title="Model Information", box=None)
        table.add_column("Model", style="cyan")
        table.add_column("Path", style="green")

        for name, path in paths.items():
            table.add_row(name, str(path))

        console.print(table)
        console.print(f"\n[bold]Base directory:[/bold] {settings.models_dir}")


@config_app.command("cleanup")
def cleanup():
    """Cleanup cached resources and free memory.

    Clears:
    - Embedding model cache (~420MB)
    - ChromaDB client connections
    - BM25 index from memory

    Useful after batch operations or to free memory in long-running processes.
    """
    from src.utils import cleanup_all_resources

    console.print("[yellow]Cleaning up cached resources...")
    cleanup_all_resources()
    console.print("[green]Cleanup complete! Memory freed.")


@config_app.command("device")
def device():
    """Show device (CPU/GPU) configuration and status.

    Displays:
    - Current device setting
    - Available devices (CUDA, MPS, CPU)
    - GPU information if available

    To change device, set RAG_DEVICE environment variable:
        RAG_DEVICE=cuda   - Use NVIDIA GPU
        RAG_DEVICE=mps    - Use Apple Silicon GPU
        RAG_DEVICE=cpu    - Use CPU (default)
        RAG_DEVICE=auto   - Auto-detect best device
    """
    import torch

    table = Table(title="Device Configuration", box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    # Current setting
    table.add_row("Current Device Setting", settings.device)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        cuda_status = f"[green]Available[/green] ({torch.cuda.device_count()} GPU(s))"
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            table.add_row(f"  GPU {i}", f"{gpu_name} ({gpu_mem:.1f} GB)")
    else:
        cuda_status = "[dim]Not available[/dim]"
    table.add_row("CUDA (NVIDIA GPU)", cuda_status)

    # Check MPS availability (Apple Silicon)
    mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
    mps_status = "[green]Available[/green]" if mps_available else "[dim]Not available[/dim]"
    table.add_row("MPS (Apple Silicon)", mps_status)

    # CPU info
    table.add_row("CPU", "[green]Available[/green]")

    # Current torch default device
    try:
        current_device = str(torch.tensor([0]).device)
    except Exception:
        current_device = "cpu"
    table.add_row("PyTorch Default Device", current_device)

    console.print(table)

    # Show recommendation
    console.print("")
    if settings.device == "cpu" and cuda_available:
        console.print("[yellow]Tip:[/yellow] GPU available! Enable with: [bold]RAG_DEVICE=cuda[/bold]")
        console.print("[dim]Add to .env file or set environment variable before running.[/dim]")
    elif settings.device == "cpu" and mps_available:
        console.print("[yellow]Tip:[/yellow] Apple GPU available! Enable with: [bold]RAG_DEVICE=mps[/bold]")

    # Show how to install CUDA PyTorch if needed
    if not cuda_available and settings.device == "cuda":
        console.print("")
        console.print("[red]CUDA requested but not available![/red]")
        console.print("Install PyTorch with CUDA support:")
        console.print("[dim]pip install torch --index-url https://download.pytorch.org/whl/cu121[/dim]")


@ingestion_app.command("checkpoints-list")
def checkpoints_list():
    """List all active ingestion checkpoints.

    Shows checkpoints for documents that have incomplete ingestion.
    Resume ingestion by re-running the ingest command on the file.
    """
    from src.ingestion.checkpoint import list_checkpoints
    from datetime import datetime

    checkpoints = list_checkpoints()

    if not checkpoints:
        console.print("[yellow]No active checkpoints found.")
        return

    table = Table(title=f"Active Checkpoints ({len(checkpoints)} total)", box=None, show_header=True)
    table.add_column("Doc ID", style="dim", width=8)
    table.add_column("File Path", style="cyan")
    table.add_column("Total", style="green", justify="right", width=8)
    table.add_column("Done", style="green", justify="right", width=8)
    table.add_column("Progress", style="yellow", width=10)
    table.add_column("Last Update", style="blue", width=19)

    for cp in checkpoints:
        doc_id_short = cp['doc_id'][-8:] if len(cp['doc_id']) >= 8 else cp['doc_id']
        file_path = Path(cp['file_path']).name  # Show just filename

        # Calculate progress percentage
        total = cp['total_chunks']
        # processed_batches is count of batches, we need to estimate chunks
        # This is approximate since last batch might be partial
        done_batches = cp['processed_batches']
        # Assuming batch_size=100 (default)
        progress_pct = (done_batches * 100 / ((total + 99) // 100)) if total > 0 else 0
        progress_str = f"{progress_pct:.0f}%"

        # Format timestamp
        timestamp_str = cp['timestamp']
        if timestamp_str:
            try:
                dt = datetime.fromisoformat(timestamp_str)
                timestamp_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass

        table.add_row(
            doc_id_short,
            file_path,
            str(total),
            str(done_batches),
            progress_str,
            timestamp_str
        )

    console.print(table)
    console.print(f"\n[bold]Tip:[/bold] Resume ingestion by running: rag ingestion start <file_path>")


@ingestion_app.command("checkpoints-clean")
def checkpoints_clean(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    stale_only: bool = typer.Option(False, "--stale-only", help="Only delete stale checkpoints (older than retention period)"),
):
    """Delete ingestion checkpoints.

    By default, deletes ALL checkpoints. Use --stale-only to only delete
    checkpoints older than the retention period (default: 7 days).

    Examples:
        rag ingestion checkpoints-clean                  # Delete all checkpoints (with confirmation)
        rag ingestion checkpoints-clean -y               # Delete all checkpoints (no confirmation)
        rag ingestion checkpoints-clean --stale-only     # Delete only stale checkpoints
    """
    from src.ingestion.checkpoint import clean_all_checkpoints, clean_stale_checkpoints

    if stale_only:
        count = clean_stale_checkpoints()
        if count > 0:
            console.print(f"[green]Deleted {count} stale checkpoint(s)")
        else:
            console.print("[yellow]No stale checkpoints found")
        return

    # Delete all checkpoints
    if not yes:
        if not typer.confirm("Delete ALL checkpoints? This will discard all resume progress."):
            console.print("[yellow]Cancelled.")
            return

    count = clean_all_checkpoints()
    if count > 0:
        console.print(f"[green]Deleted {count} checkpoint(s)")
    else:
        console.print("[yellow]No checkpoints found")


@ingestion_app.command("log")
def ingestion_log(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of recent entries to show"),
    export: str = typer.Option(None, "--export", "-e", help="Export full log to this path")
):
    """View ingestion audit log.

    Shows a history of all document ingestions including:
    - Timestamp when ingestion started
    - File name and type
    - Number of pages (for PDFs)
    - Number of chunks created
    - Status (completed, resumed, or failed)
    - Duration in seconds

    The log is automatically updated during ingestion and stored as a CSV file
    that can be opened in Excel or any spreadsheet application.

    Examples:
        rag ingestion log                    # Show last 20 entries
        rag ingestion log --limit 50         # Show last 50 entries
        rag ingestion log --export log.csv   # Export full log
    """
    import csv
    import shutil
    from src.ingestion.audit_log import get_audit_log_path
    from src.config import apply_project_settings

    # Apply project settings if active
    if apply_project_settings():
        pm = get_project_manager()
        active = pm.get_active_project()
        if active:
            console.print(f"[dim]Project: {active.name}[/dim]")

    csv_path = get_audit_log_path()

    if not csv_path.exists():
        console.print("[yellow]No ingestion log found. Ingest some documents first.")
        return

    if export:
        # Export to specified path
        export_path = Path(export)
        shutil.copy(csv_path, export_path)
        console.print(f"[green]Exported log to: {export_path}")
        return

    # Read and display last N entries
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        console.print("[yellow]Log is empty.")
        return

    # Show most recent entries
    recent_rows = rows[-limit:] if len(rows) > limit else rows
    recent_rows.reverse()  # Most recent first

    table = Table(title=f"Recent Ingestion Log ({len(recent_rows)} of {len(rows)} total)", box=None)
    table.add_column("Time", style="yellow", width=16)
    table.add_column("Session", style="dim", width=8)
    table.add_column("File", style="cyan", width=30)
    table.add_column("Type", style="blue", width=6)
    table.add_column("Pages", style="green", justify="right", width=6)
    table.add_column("Chunks", style="green", justify="right", width=7)
    table.add_column("Status", style="magenta", width=10)
    table.add_column("Duration", style="dim", justify="right", width=8)

    for row in recent_rows:
        # Format timestamp (show only date and time, not seconds)
        timestamp = row['timestamp'][:16]  # Truncate to YYYY-MM-DD HH:MM

        # Truncate filename if too long
        filename = Path(row['file_name']).name
        if len(filename) > 30:
            filename = filename[:27] + "..."

        # Format status with color
        status = row['status']
        if status == "completed":
            status_display = "[green]completed[/green]"
        elif status == "failed":
            status_display = "[red]failed[/red]"
        elif status == "resumed":
            status_display = "[yellow]resumed[/yellow]"
        else:
            status_display = status

        # Format duration
        duration = f"{float(row['duration_seconds']):.1f}s" if row['duration_seconds'] else "-"

        # Get session_id (may not exist in older logs)
        session_id = row.get('session_id', '-') or '-'

        table.add_row(
            timestamp,
            session_id,
            filename,
            row['doc_type'],
            row['num_pages'] or "-",
            row['num_chunks'],
            status_display,
            duration
        )

    console.print(table)
    console.print(f"\n[dim]Full log location: {csv_path}[/dim]")

    # Calculate statistics
    failed_count = sum(1 for r in rows if r['status'] == 'failed')
    completed_count = sum(1 for r in rows if r['status'] in ['completed', 'resumed'])

    console.print(f"[dim]Total ingestions: {len(rows)} | Completed: {completed_count} | Failed: {failed_count}[/dim]")


@ingestion_app.command("status")
def ingestion_status(
    live: bool = typer.Option(True, "--live/--once", help="Live updating display (default) or single snapshot"),
    refresh: float = typer.Option(2.0, "--refresh", "-r", help="Refresh interval in seconds for live mode"),
    history: bool = typer.Option(False, "--history", "-h", help="Show recent session history"),
):
    """Monitor running ingestion process.

    Shows real-time status of parallel ingestion including:
    - Session information (files, progress, rate)
    - Per-worker status (current file, duration)
    - Writer status

    Examples:
        rag ingestion status                    # Live dashboard (updates every 2s)
        rag ingestion status --once             # Single snapshot
        rag ingestion status --refresh 1        # Faster refresh rate
        rag ingestion status --history          # Show recent session history
    """
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from src.ingestion.status import get_status_manager
    from src.config import apply_project_settings
    import time

    # Apply project settings to use correct status database
    if apply_project_settings():
        pm = get_project_manager()
        active = pm.get_active_project()
        if active:
            console.print(f"[dim]Project: {active.name}[/dim]")

    status_mgr = get_status_manager()

    def format_duration(seconds: float) -> str:
        """Format seconds into human-readable duration."""
        if seconds is None:
            return "-"
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds/3600:.1f}h"

    def build_dashboard() -> Panel:
        """Build the status dashboard."""
        session = status_mgr.get_active_session()

        if not session:
            return Panel(
                "[yellow]No active ingestion session[/yellow]\n\n"
                "[dim]Run 'rag ingestion start <directory>' to start ingestion[/dim]",
                title="RAG Ingestion Dashboard",
                border_style="dim"
            )

        workers = status_mgr.get_workers(session.session_id)

        # Build session info
        elapsed = format_duration(session.elapsed_seconds)
        eta = format_duration(session.eta_seconds) if session.eta_seconds else "calculating..."
        progress_pct = (session.processed_files + session.failed_files) / max(session.total_files - session.skipped_files, 1) * 100

        session_info = Table(box=None, show_header=False, padding=(0, 2))
        session_info.add_column("Key", style="dim")
        session_info.add_column("Value", style="bold")
        session_info.add_row("Session ID", session.session_id)
        session_info.add_row("Source", str(Path(session.source_path).name))
        session_info.add_row("Elapsed", elapsed)
        session_info.add_row("Progress", f"{session.processed_files}/{session.total_files - session.skipped_files} ({progress_pct:.0f}%)")
        session_info.add_row("Rate", f"{session.rate:.2f} docs/sec")
        session_info.add_row("ETA", eta)
        session_info.add_row("Chunks", str(session.total_chunks))
        if session.failed_files > 0:
            session_info.add_row("Failed", f"[red]{session.failed_files}[/red]")
        if session.skipped_files > 0:
            session_info.add_row("Skipped", f"[dim]{session.skipped_files}[/dim]")

        # Build workers table
        workers_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        workers_table.add_column("Worker", style="cyan", width=10)
        workers_table.add_column("Status", width=10)
        workers_table.add_column("Current File", style="white", width=40)
        workers_table.add_column("Duration", justify="right", width=10)
        workers_table.add_column("Done", justify="right", width=6)

        for w in workers:
            # Format status with color
            if w.status == "parsing":
                status_text = "[green]parsing[/green]"
            elif w.status == "writing":
                status_text = "[blue]writing[/blue]"
            elif w.status == "idle":
                status_text = "[dim]idle[/dim]"
            elif w.status == "stopped" or w.status == "finished":
                status_text = "[dim]done[/dim]"
            elif w.status == "error":
                status_text = "[red]error[/red]"
            else:
                status_text = w.status

            # Worker name
            if w.worker_type == "writer":
                worker_name = "Writer"
            else:
                worker_name = f"Worker-{w.worker_id}"

            # Current file (truncate if too long)
            current_file = w.current_file or "-"
            if len(current_file) > 38:
                current_file = current_file[:35] + "..."

            # Duration on current file
            duration = format_duration(w.file_duration_seconds)

            workers_table.add_row(
                worker_name,
                status_text,
                current_file,
                duration,
                str(w.files_processed)
            )

        # Combine into layout
        content = Table.grid(padding=(1, 0))
        content.add_row(session_info)
        content.add_row("")
        content.add_row(workers_table)

        return Panel(
            content,
            title=f"[bold]RAG Ingestion Dashboard[/bold] - Session {session.session_id}",
            subtitle=f"[dim]Last updated: {time.strftime('%H:%M:%S')}[/dim]",
            border_style="green" if session.status == "running" else "dim"
        )

    def show_history():
        """Show recent session history."""
        sessions = status_mgr.get_recent_sessions(limit=10)

        if not sessions:
            console.print("[yellow]No session history found.[/yellow]")
            return

        table = Table(title="Recent Ingestion Sessions", box=box.SIMPLE)
        table.add_column("Session", style="cyan", width=10)
        table.add_column("Source", width=30)
        table.add_column("Started", width=20)
        table.add_column("Files", justify="right", width=8)
        table.add_column("Chunks", justify="right", width=8)
        table.add_column("Status", width=12)
        table.add_column("Duration", justify="right", width=10)

        for s in sessions:
            # Format status
            if s.status == "running":
                status_text = "[green]running[/green]"
            elif s.status == "completed":
                status_text = "[blue]completed[/blue]"
            elif s.status == "stopped":
                status_text = "[yellow]stopped[/yellow]"
            elif s.status == "stale":
                status_text = "[dim]stale[/dim]"
            else:
                status_text = s.status

            table.add_row(
                s.session_id,
                str(Path(s.source_path).name),
                s.started_at.strftime("%Y-%m-%d %H:%M:%S"),
                str(s.processed_files),
                str(s.total_chunks),
                status_text,
                format_duration(s.elapsed_seconds)
            )

        console.print(table)

    # Handle history mode
    if history:
        show_history()
        return

    # Single snapshot mode
    if not live:
        console.print(build_dashboard())
        return

    # Live mode
    console.print("[dim]Press Ctrl+C to exit...[/dim]\n")

    try:
        with Live(build_dashboard(), refresh_per_second=1/refresh, console=console) as live_display:
            while True:
                time.sleep(refresh)
                live_display.update(build_dashboard())

                # Check if session is still running
                session = status_mgr.get_active_session()
                if not session:
                    live_display.update(build_dashboard())
                    break

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped monitoring.[/dim]")


@ingestion_app.command("stop")
def ingestion_stop(
    worker_id: int = typer.Option(None, "--worker", "-w", help="Stop specific worker by ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    force: bool = typer.Option(False, "--force", "-f", help="Force terminate worker processes immediately"),
):
    """Stop running ingestion process.

    Sends a graceful stop signal to workers. Workers will finish their
    current file before stopping.

    Use --force to immediately terminate worker processes (useful when
    workers are stuck on large files).

    Examples:
        rag ingestion stop                      # Stop all workers (with confirmation)
        rag ingestion stop -y                   # Stop all workers (no confirmation)
        rag ingestion stop --force              # Force terminate all workers
        rag ingestion stop --worker 2           # Stop only worker 2
    """
    import signal as sig
    from src.ingestion.status import get_status_manager
    from src.config import apply_project_settings

    # Apply project settings to use correct status database
    apply_project_settings()

    status_mgr = get_status_manager()

    # Find active session (check both running and stopping states)
    session = status_mgr.get_active_session()

    # Also check for recently stopped sessions if force is requested
    if not session and force:
        recent = status_mgr.get_recent_sessions(limit=1)
        if recent and recent[0].status in ("stopped", "stopping"):
            session = recent[0]

    if not session:
        console.print("[yellow]No active ingestion session found.[/yellow]")
        return

    if worker_id is not None:
        # Stop specific worker
        if not yes:
            if not typer.confirm(f"Stop worker {worker_id} in session {session.session_id}?"):
                console.print("[yellow]Cancelled.[/yellow]")
                return

        status_mgr.send_stop_signal(session.session_id, worker_id)
        console.print(f"[green]Stop signal sent to worker {worker_id}[/green]")
        console.print("[dim]Worker will stop after completing current file.[/dim]")

    else:
        # Stop all workers
        if not yes and not force:
            if not typer.confirm(f"Stop all workers in session {session.session_id}?"):
                console.print("[yellow]Cancelled.[/yellow]")
                return

        console.print(f"[dim]Sending stop signal to session {session.session_id} (db: {status_mgr.db_path})[/dim]")
        status_mgr.send_stop_signal(session.session_id)

        if force:
            # Helper function to kill a process
            def kill_process(pid: int, label: str) -> bool:
                try:
                    if sys.platform == "win32":
                        import subprocess
                        result = subprocess.run(
                            ["taskkill", "/PID", str(pid), "/F"],
                            capture_output=True,
                            timeout=5
                        )
                        if result.returncode == 0:
                            console.print(f"  [green]Terminated {label} (PID {pid})[/green]")
                            return True
                        else:
                            console.print(f"  [dim]{label} (PID {pid}) not found (already stopped?)[/dim]")
                            return False
                    else:
                        import os
                        os.kill(pid, sig.SIGKILL)
                        console.print(f"  [green]Terminated {label} (PID {pid})[/green]")
                        return True
                except ProcessLookupError:
                    console.print(f"  [dim]{label} (PID {pid}) not found (already stopped?)[/dim]")
                    return False
                except Exception as e:
                    console.print(f"  [red]Failed to terminate {label} (PID {pid}): {e}[/red]")
                    return False

            # Force terminate worker processes
            worker_pids = status_mgr.get_worker_pids(session.session_id)
            main_pid = status_mgr.get_main_pid(session.session_id)

            total_to_kill = len(worker_pids) + (1 if main_pid else 0)
            if total_to_kill > 0:
                console.print(f"[yellow]Force terminating {total_to_kill} process(es)...[/yellow]")
                terminated = 0

                # Kill workers first
                for pid in worker_pids:
                    if kill_process(pid, f"Worker"):
                        terminated += 1

                # Kill main process last
                if main_pid:
                    if kill_process(main_pid, "Main process"):
                        terminated += 1

                console.print(f"[green]Terminated {terminated} of {total_to_kill} process(es)[/green]")
            else:
                console.print("[dim]No process PIDs found to terminate.[/dim]")

        status_mgr.complete_session(session.session_id, "stopped")
        console.print("[green]Stop signal sent to all workers[/green]")
        if not force:
            console.print("[dim]Workers will stop after completing their current files.[/dim]")
            console.print("[dim]Use --force to terminate immediately if workers are stuck.[/dim]")


# === MCP Server Management Commands ===

@mcp_app.command("status")
def mcp_status(
    live: bool = typer.Option(True, "--live/--once", help="Live updating display (default) or single snapshot"),
    refresh: float = typer.Option(2.0, "--refresh", "-r", help="Refresh interval in seconds for live mode"),
):
    """Show MCP server status and metrics.

    Displays:
    - Server status (running/stopped)
    - Process ID and uptime
    - Query metrics (count, response times, error rate)
    - Recent queries

    Examples:
        rag mcp status                    # Live dashboard (updates every 2s)
        rag mcp status --once             # Single snapshot
        rag mcp status --refresh 1        # Faster refresh rate
    """
    from rich.live import Live
    from rich.panel import Panel
    from rich.text import Text
    from src.mcp.status import get_mcp_status_manager
    from src.config import apply_project_settings
    import time as time_module

    # Apply project settings to use correct status database
    if apply_project_settings():
        pm = get_project_manager()
        active = pm.get_active_project()
        if active:
            console.print(f"[dim]Project: {active.name}[/dim]")

    status_mgr = get_mcp_status_manager()

    def format_duration(seconds: float) -> str:
        """Format seconds into human-readable duration."""
        if seconds is None or seconds <= 0:
            return "-"
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes}m {secs}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"

    def build_dashboard() -> Panel:
        """Build the MCP status dashboard."""
        server_info = status_mgr.get_server_info()

        if not server_info:
            return Panel(
                "[yellow]MCP server has not been started yet[/yellow]\n\n"
                "[dim]Run 'rag mcp serve' to start the MCP server[/dim]",
                title="MCP Server Status",
                border_style="dim"
            )

        # Check if actually running
        is_running = status_mgr.is_server_running()

        # Build server info section
        server_table = Table(box=None, show_header=False, padding=(0, 2))
        server_table.add_column("Key", style="dim")
        server_table.add_column("Value", style="bold")

        if is_running:
            status_text = "[green]running[/green]"
        elif server_info.status == "stopped":
            status_text = "[yellow]stopped[/yellow]"
        else:
            status_text = f"[red]{server_info.status}[/red]"

        server_table.add_row("Status", status_text)
        server_table.add_row("PID", str(server_info.pid))
        server_table.add_row("Host", server_info.host)
        server_table.add_row("Port", str(server_info.port))
        server_table.add_row("Uptime", format_duration(server_info.uptime_seconds) if is_running else "-")

        # Get metrics summary (last hour)
        summary = status_mgr.get_metrics_summary(since_minutes=60)

        # Build metrics section
        metrics_table = Table(box=None, show_header=False, padding=(0, 2))
        metrics_table.add_column("Key", style="dim")
        metrics_table.add_column("Value", style="bold")

        metrics_table.add_row("Queries (1h)", str(summary.total_queries))
        metrics_table.add_row("Queries/min", f"{summary.queries_per_minute:.1f}")
        metrics_table.add_row("Avg Response", f"{summary.avg_response_time_ms:.0f} ms")
        if summary.total_queries > 0:
            metrics_table.add_row("Min Response", f"{summary.min_response_time_ms:.0f} ms")
            metrics_table.add_row("Max Response", f"{summary.max_response_time_ms:.0f} ms")
        error_pct = summary.error_rate * 100
        if error_pct > 0:
            metrics_table.add_row("Error Rate", f"[red]{error_pct:.1f}%[/red]")
        else:
            metrics_table.add_row("Error Rate", f"[green]{error_pct:.1f}%[/green]")

        # Get recent queries
        recent_queries = status_mgr.get_recent_queries(limit=5)

        queries_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        queries_table.add_column("Time", style="dim", width=8)
        queries_table.add_column("Query", style="white", width=40)
        queries_table.add_column("Duration", justify="right", width=10)
        queries_table.add_column("Results", justify="right", width=8)

        for q in recent_queries:
            time_str = q.timestamp.strftime("%H:%M:%S")
            query_text = q.query_text[:38] + "..." if len(q.query_text) > 40 else q.query_text
            duration_str = f"{q.response_time_ms:.0f}ms"

            if q.error:
                queries_table.add_row(
                    time_str,
                    f"[red]{query_text}[/red]",
                    duration_str,
                    "[red]ERR[/red]"
                )
            else:
                queries_table.add_row(
                    time_str,
                    query_text,
                    duration_str,
                    str(q.result_count)
                )

        # Combine into layout
        content = Table.grid(padding=(1, 0))
        content.add_row(Text("Server:", style="bold cyan"))
        content.add_row(server_table)
        content.add_row("")
        content.add_row(Text("Metrics (last hour):", style="bold cyan"))
        content.add_row(metrics_table)

        if recent_queries:
            content.add_row("")
            content.add_row(Text("Recent Queries:", style="bold cyan"))
            content.add_row(queries_table)

        border_color = "green" if is_running else "yellow"

        return Panel(
            content,
            title="[bold]MCP Server Status[/bold]",
            subtitle=f"[dim]Last updated: {time_module.strftime('%H:%M:%S')}[/dim]",
            border_style=border_color
        )

    # Single snapshot mode
    if not live:
        console.print(build_dashboard())
        return

    # Live mode
    console.print("[dim]Press Ctrl+C to exit...[/dim]\n")

    try:
        with Live(build_dashboard(), refresh_per_second=1/refresh, console=console) as live_display:
            while True:
                time_module.sleep(refresh)
                live_display.update(build_dashboard())

    except KeyboardInterrupt:
        console.print("\n[dim]Stopped monitoring.[/dim]")


@mcp_app.command("stop")
def mcp_stop(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    force: bool = typer.Option(False, "--force", "-f", help="Force kill the server process"),
):
    """Stop running MCP server.

    Sends a termination signal to the MCP server process.
    Use --force to immediately kill the process (SIGKILL on Unix, taskkill /F on Windows).

    Examples:
        rag mcp stop                      # Stop with confirmation
        rag mcp stop -y                   # Stop without confirmation
        rag mcp stop --force              # Force kill the server
    """
    import os
    import signal as sig
    from src.mcp.status import get_mcp_status_manager
    from src.config import apply_project_settings

    # Apply project settings to use correct status database
    apply_project_settings()

    status_mgr = get_mcp_status_manager()

    # Get server info
    server_info = status_mgr.get_server_info()

    if not server_info:
        console.print("[yellow]No MCP server found.[/yellow]")
        return

    # Check if server is actually running (validates process exists)
    is_running = status_mgr.is_server_running()

    if not is_running:
        # If force mode, try to kill by PID anyway (process might be zombie/stuck)
        if force and server_info.pid:
            console.print(f"[yellow]Server status shows not running, but attempting to kill PID {server_info.pid}...[/yellow]")
        else:
            console.print(f"[yellow]MCP server is not running (status: {server_info.status})[/yellow]")
            return

    # Confirm (skip if force mode)
    if not yes and not force:
        if not typer.confirm(f"Stop MCP server (PID {server_info.pid})?"):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    # Kill the server process
    try:
        if sys.platform == "win32":
            # On Windows, use taskkill with /F (force)
            import subprocess
            result = subprocess.run(
                ["taskkill", "/PID", str(server_info.pid), "/F"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                console.print(f"[green]Killed MCP server (PID {server_info.pid})[/green]")
            elif "not found" in result.stderr.lower() or result.returncode == 128:
                console.print(f"[yellow]Process {server_info.pid} not found (already stopped?)[/yellow]")
            else:
                console.print(f"[yellow]taskkill returned: {result.stderr or result.stdout}[/yellow]")
        else:
            # On Unix, use SIGKILL for force, SIGTERM otherwise
            kill_signal = sig.SIGKILL if force else sig.SIGTERM
            os.kill(server_info.pid, kill_signal)
            signal_name = "SIGKILL" if force else "SIGTERM"
            console.print(f"[green]Sent {signal_name} to MCP server (PID {server_info.pid})[/green]")

        # Mark as stopped in database
        status_mgr.mark_server_stopped()

        if not force:
            console.print("[dim]Server should shut down gracefully.[/dim]")

    except ProcessLookupError:
        # Process already dead
        status_mgr.mark_server_stopped()
        console.print("[yellow]Server process not found (already stopped?)[/yellow]")

    except PermissionError:
        console.print("[red]Permission denied. Try running as administrator.[/red]")

    except subprocess.TimeoutExpired:
        console.print("[red]Timeout waiting for taskkill. Process may still be running.[/red]")

    except Exception as e:
        console.print(f"[red]Failed to stop server: {e}[/red]")


@mcp_app.command("metrics")
def mcp_metrics(
    since: int = typer.Option(60, "--since", "-s", help="Time period in minutes"),
    export: str = typer.Option(None, "--export", "-e", help="Export metrics to JSON file"),
):
    """Show detailed MCP server metrics.

    Examples:
        rag mcp metrics                   # Show metrics for last hour
        rag mcp metrics --since 1440      # Show metrics for last 24 hours
        rag mcp metrics --export report.json  # Export to JSON
    """
    import json
    from src.mcp.status import get_mcp_status_manager
    from src.config import apply_project_settings

    # Apply project settings
    apply_project_settings()

    status_mgr = get_mcp_status_manager()

    # Get server info
    server_info = status_mgr.get_server_info()

    # Get metrics
    summary = status_mgr.get_metrics_summary(since_minutes=since)
    recent_queries = status_mgr.get_recent_queries(limit=20)

    if export:
        # Export to JSON
        data = {
            "server": {
                "pid": server_info.pid if server_info else None,
                "host": server_info.host if server_info else None,
                "port": server_info.port if server_info else None,
                "status": server_info.status if server_info else "unknown",
                "uptime_seconds": server_info.uptime_seconds if server_info else 0,
            },
            "metrics": {
                "period_minutes": summary.period_minutes,
                "total_queries": summary.total_queries,
                "successful_queries": summary.successful_queries,
                "failed_queries": summary.failed_queries,
                "avg_response_time_ms": summary.avg_response_time_ms,
                "min_response_time_ms": summary.min_response_time_ms,
                "max_response_time_ms": summary.max_response_time_ms,
                "error_rate": summary.error_rate,
                "queries_per_minute": summary.queries_per_minute,
            },
            "recent_queries": [
                {
                    "timestamp": q.timestamp.isoformat(),
                    "query_text": q.query_text,
                    "top_k": q.top_k,
                    "response_time_ms": q.response_time_ms,
                    "result_count": q.result_count,
                    "error": q.error,
                    "success": q.success,
                }
                for q in recent_queries
            ]
        }

        export_path = Path(export)
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        console.print(f"[green]Exported metrics to: {export_path}[/green]")
        return

    # Display metrics
    console.print()
    console.print(f"[bold cyan]MCP Server Metrics[/bold cyan] (last {since} minutes)")
    console.print()

    # Server info
    if server_info:
        is_running = status_mgr.is_server_running()
        status_str = "[green]running[/green]" if is_running else f"[yellow]{server_info.status}[/yellow]"
        console.print(f"Server: {status_str} (PID {server_info.pid}, port {server_info.port})")
    else:
        console.print("Server: [dim]not started[/dim]")

    console.print()

    # Metrics summary
    table = Table(title="Query Metrics", box=box.SIMPLE)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white", justify="right")

    table.add_row("Total Queries", str(summary.total_queries))
    table.add_row("Successful", str(summary.successful_queries))
    table.add_row("Failed", str(summary.failed_queries))
    table.add_row("Queries/min", f"{summary.queries_per_minute:.2f}")
    table.add_row("Avg Response", f"{summary.avg_response_time_ms:.1f} ms")
    table.add_row("Min Response", f"{summary.min_response_time_ms:.1f} ms")
    table.add_row("Max Response", f"{summary.max_response_time_ms:.1f} ms")
    table.add_row("Error Rate", f"{summary.error_rate * 100:.2f}%")

    console.print(table)

    # Recent queries
    if recent_queries:
        console.print()
        queries_table = Table(title="Recent Queries (last 20)", box=box.SIMPLE)
        queries_table.add_column("Time", style="dim", width=19)
        queries_table.add_column("Query", style="white", width=40)
        queries_table.add_column("top_k", justify="right", width=6)
        queries_table.add_column("Time (ms)", justify="right", width=10)
        queries_table.add_column("Results", justify="right", width=8)
        queries_table.add_column("Status", width=8)

        for q in recent_queries:
            time_str = q.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            query_text = q.query_text[:38] + "..." if len(q.query_text) > 40 else q.query_text
            status_str = "[green]OK[/green]" if q.success else f"[red]ERR[/red]"

            queries_table.add_row(
                time_str,
                query_text,
                str(q.top_k),
                f"{q.response_time_ms:.0f}",
                str(q.result_count),
                status_str
            )

        console.print(queries_table)


# Register MCP subcommand group
app.add_typer(mcp_app, name="mcp")


# === Project Management Commands ===

@project_app.command("create")
def project_create(
    name: str = typer.Argument(..., help="Project name"),
    port: int = typer.Option(9090, "--port", "-p", help="MCP server port"),
    description: str = typer.Option("", "--description", "-d", help="Project description"),
    device: str = typer.Option("cpu", "--device", help="Compute device (cpu, cuda, mps, auto)"),
    database: str = typer.Option(None, "--database", "--db", help="Custom database path (absolute or relative)"),
    documents: str = typer.Option(None, "--documents", "--docs", help="Custom documents folder path"),
    # Document Processing
    ocr: bool = typer.Option(False, "--ocr", help="Enable OCR for image-based PDFs"),
    ocr_engine: str = typer.Option("auto", "--ocr-engine", help="OCR engine (auto, rapidocr, easyocr, tesseract)"),
    ocr_languages: str = typer.Option("eng+ara", "--ocr-languages", help="OCR languages (e.g., eng+ara)"),
    asr: bool = typer.Option(True, "--asr/--no-asr", help="Enable audio transcription"),
    # Embedding & Chunking
    embedding_model: str = typer.Option(None, "--embedding-model", help="Embedding model name"),
    chunking_method: str = typer.Option("hybrid", "--chunking-method", help="Chunking method (hybrid, semantic, fixed)"),
    max_tokens: int = typer.Option(512, "--max-tokens", help="Max tokens per chunk"),
    # Retrieval
    top_k: int = typer.Option(5, "--top-k", help="Default number of results"),
    # MCP Server
    mcp_name: str = typer.Option(None, "--mcp-name", help="MCP server name"),
    mcp_transport: str = typer.Option("streamable-http", "--mcp-transport", help="MCP transport protocol"),
    mcp_host: str = typer.Option("0.0.0.0", "--mcp-host", help="MCP bind host"),
    mcp_cleanup: bool = typer.Option(True, "--mcp-cleanup/--no-mcp-cleanup", help="Enable cleanup on shutdown"),
    # Logging
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
):
    """Create a new RAG project.

    By default, database and documents are stored in ~/.rag/projects/<name>/
    Use --database and --documents to specify custom locations.

    Examples:
        rag project create my-docs
        rag project create safety-docs --port 9091 --description "Safety documentation"
        rag project create ml-papers --device cuda --ocr
        rag project create work --database D:/rag-data --documents D:/work-docs
        rag project create research --embedding-model all-MiniLM-L6-v2 --max-tokens 256
    """
    pm = get_project_manager()

    try:
        config = pm.create_project(
            name=name,
            port=port,
            description=description,
            device=device,
            db_path=database,
            docs_path=documents,
            # Document Processing
            enable_ocr=ocr,
            ocr_engine=ocr_engine,
            ocr_languages=ocr_languages,
            enable_asr=asr,
            # Embedding & Chunking
            embedding_model=embedding_model,
            chunking_method=chunking_method,
            max_tokens=max_tokens,
            # Retrieval
            default_top_k=top_k,
            # MCP Server
            mcp_server_name=mcp_name,
            mcp_transport=mcp_transport,
            mcp_host=mcp_host,
            mcp_enable_cleanup=mcp_cleanup,
            # Logging
            log_level=log_level,
            switch_to=True
        )

        console.print(f"[green]Project '{name}' created successfully![/green]")
        console.print()

        paths = pm.get_project_paths(name)

        # Basic settings
        table = Table(title="Configuration", box=box.SIMPLE)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Name", config.name)
        table.add_row("Port", str(config.port))
        table.add_row("Device", config.device)
        table.add_row("Database", str(paths["db_path"]))
        table.add_row("Documents", str(paths["docs_path"]))
        table.add_row("OCR", "Enabled" if config.enable_ocr else "Disabled")
        table.add_row("Embedding", config.embedding_model)
        table.add_row("Chunking", f"{config.chunking_method} ({config.max_tokens} tokens)")
        if config.description:
            table.add_row("Description", config.description)

        console.print(table)
        console.print()
        console.print("[dim]Project is now active.[/dim]")
        console.print("[dim]Run 'rag project config' to see all settings.[/dim]")
        console.print("[dim]Run 'rag ingestion start' to add documents.[/dim]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@project_app.command("list")
def project_list():
    """List all RAG projects.

    Shows all projects with their port, device, and status.
    """
    pm = get_project_manager()
    projects = pm.list_projects()
    active_name = pm.get_active_project_name()

    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        console.print()
        console.print("Create a project with: [cyan]rag project create <name>[/cyan]")
        return

    table = Table(title="RAG Projects", box=box.SIMPLE)
    table.add_column("", width=2)
    table.add_column("Name", style="cyan")
    table.add_column("Port", style="white")
    table.add_column("Device", style="white")
    table.add_column("Description", style="dim")

    for project in projects:
        is_active = project.name == active_name
        marker = "[green]*[/green]" if is_active else ""
        name_style = "bold green" if is_active else "cyan"

        table.add_row(
            marker,
            f"[{name_style}]{project.name}[/{name_style}]",
            str(project.port),
            project.device,
            project.description[:40] + "..." if len(project.description) > 40 else project.description
        )

    console.print(table)
    console.print()
    console.print("[dim]* = active project[/dim]")


@project_app.command("switch")
def project_switch(
    name: str = typer.Argument(..., help="Project name to switch to"),
):
    """Switch to a different project.

    Example:
        rag project switch safety-docs
    """
    pm = get_project_manager()

    try:
        config = pm.switch_project(name)
        console.print(f"[green]Switched to project '{name}'[/green]")
        console.print(f"[dim]Port: {config.port} | Device: {config.device}[/dim]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@project_app.command("config")
def project_config(
    port: int = typer.Option(None, "--port", "-p", help="Set MCP server port"),
    device: str = typer.Option(None, "--device", help="Set compute device"),
    description: str = typer.Option(None, "--description", "-d", help="Set description"),
    database: str = typer.Option(None, "--database", "--db", help="Set database path (absolute or relative)"),
    documents: str = typer.Option(None, "--documents", "--docs", help="Set documents folder path"),
    # Document Processing
    ocr: bool = typer.Option(None, "--ocr/--no-ocr", help="Enable/disable OCR"),
    ocr_engine: str = typer.Option(None, "--ocr-engine", help="OCR engine (auto, rapidocr, easyocr, tesseract)"),
    ocr_languages: str = typer.Option(None, "--ocr-languages", help="OCR languages (e.g., eng+ara)"),
    asr: bool = typer.Option(None, "--asr/--no-asr", help="Enable/disable audio transcription"),
    # Embedding & Chunking
    embedding_model: str = typer.Option(None, "--embedding-model", help="Embedding model name"),
    chunking_method: str = typer.Option(None, "--chunking-method", help="Chunking method (hybrid, semantic, fixed)"),
    max_tokens: int = typer.Option(None, "--max-tokens", help="Max tokens per chunk"),
    # Retrieval
    top_k: int = typer.Option(None, "--top-k", help="Default number of results"),
    # MCP Server
    mcp_name: str = typer.Option(None, "--mcp-name", help="MCP server name"),
    mcp_transport: str = typer.Option(None, "--mcp-transport", help="MCP transport protocol"),
    mcp_host: str = typer.Option(None, "--mcp-host", help="MCP bind host"),
    mcp_cleanup: bool = typer.Option(None, "--mcp-cleanup/--no-mcp-cleanup", help="Enable/disable cleanup on shutdown"),
    # Logging
    log_level: str = typer.Option(None, "--log-level", help="Logging level (DEBUG, INFO, WARNING, ERROR)"),
):
    """View or update current project configuration.

    Examples:
        rag project config                         # View current config
        rag project config --port 9091             # Change port
        rag project config --device cuda           # Change device
        rag project config --database D:/rag-data  # Change database location
        rag project config --embedding-model all-MiniLM-L6-v2  # Change embedding model
        rag project config --max-tokens 256        # Change chunk size
    """
    pm = get_project_manager()
    active = pm.get_active_project()

    if not active:
        console.print("[yellow]No active project.[/yellow]")
        console.print("Create one with: [cyan]rag project create <name>[/cyan]")
        raise typer.Exit(1)

    # Check if any updates requested
    has_updates = any(x is not None for x in [
        port, device, description, database, documents,
        ocr, ocr_engine, ocr_languages, asr,
        embedding_model, chunking_method, max_tokens,
        top_k, mcp_name, mcp_transport, mcp_host, mcp_cleanup,
        log_level
    ])

    if has_updates:
        try:
            active = pm.update_project(
                name=active.name,
                port=port,
                device=device,
                description=description,
                db_path=database,
                docs_path=documents,
                # Document Processing
                enable_ocr=ocr,
                ocr_engine=ocr_engine,
                ocr_languages=ocr_languages,
                enable_asr=asr,
                # Embedding & Chunking
                embedding_model=embedding_model,
                chunking_method=chunking_method,
                max_tokens=max_tokens,
                # Retrieval
                default_top_k=top_k,
                # MCP Server
                mcp_server_name=mcp_name,
                mcp_transport=mcp_transport,
                mcp_host=mcp_host,
                mcp_enable_cleanup=mcp_cleanup,
                # Logging
                log_level=log_level,
            )
            console.print(f"[green]Project '{active.name}' updated.[/green]")
            console.print()
        except ValueError as e:
            console.print(f"[red]Error: {e}[/red]")
            raise typer.Exit(1)

    # Display current config - organized by category
    paths = pm.get_project_paths(active.name)

    # Basic settings
    console.print(f"[bold cyan]Project: {active.name}[/bold cyan]")
    console.print()

    basic_table = Table(title="Basic Settings", box=box.SIMPLE)
    basic_table.add_column("Setting", style="cyan")
    basic_table.add_column("Value", style="white")
    basic_table.add_row("Port", str(active.port))
    basic_table.add_row("Device", active.device)
    basic_table.add_row("Description", active.description or "(none)")
    basic_table.add_row("Created", active.created_at[:19])
    basic_table.add_row("Log Level", active.log_level)
    console.print(basic_table)

    # Paths
    paths_table = Table(title="Paths", box=box.SIMPLE)
    paths_table.add_column("Setting", style="cyan")
    paths_table.add_column("Value", style="white")
    paths_table.add_row("Database", str(paths["db_path"]))
    paths_table.add_row("Documents", str(paths["docs_path"]))
    console.print(paths_table)

    # Document Processing
    doc_table = Table(title="Document Processing", box=box.SIMPLE)
    doc_table.add_column("Setting", style="cyan")
    doc_table.add_column("Value", style="white")
    doc_table.add_row("OCR", "Enabled" if active.enable_ocr else "Disabled")
    doc_table.add_row("OCR Engine", active.ocr_engine)
    doc_table.add_row("OCR Languages", active.ocr_languages)
    doc_table.add_row("ASR (Audio)", "Enabled" if active.enable_asr else "Disabled")
    console.print(doc_table)

    # Embedding & Chunking
    embed_table = Table(title="Embedding & Chunking", box=box.SIMPLE)
    embed_table.add_column("Setting", style="cyan")
    embed_table.add_column("Value", style="white")
    embed_table.add_row("Embedding Model", active.embedding_model)
    embed_table.add_row("Chunking Method", active.chunking_method)
    embed_table.add_row("Max Tokens", str(active.max_tokens))
    embed_table.add_row("Default Top K", str(active.default_top_k))
    console.print(embed_table)

    # MCP Server
    mcp_table = Table(title="MCP Server", box=box.SIMPLE)
    mcp_table.add_column("Setting", style="cyan")
    mcp_table.add_column("Value", style="white")
    mcp_table.add_row("Server Name", active.mcp_server_name)
    mcp_table.add_row("Transport", active.mcp_transport)
    mcp_table.add_row("Host", active.mcp_host)
    mcp_table.add_row("Cleanup", "Enabled" if active.mcp_enable_cleanup else "Disabled")
    console.print(mcp_table)

    # Show stats
    stats = pm.get_project_stats(active.name)
    console.print()
    console.print(f"[dim]Documents: {stats['documents']} | Disk usage: {stats['disk_usage_mb']} MB[/dim]")


@project_app.command("delete")
def project_delete(
    name: str = typer.Argument(..., help="Project name to delete"),
    data: bool = typer.Option(False, "--data", help="Also delete project data (databases, documents)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Delete a project.

    Examples:
        rag project delete old-project           # Delete config only
        rag project delete old-project --data    # Delete config and all data
    """
    pm = get_project_manager()

    if not pm.project_exists(name):
        console.print(f"[red]Project '{name}' not found.[/red]")
        raise typer.Exit(1)

    if data:
        msg = f"Delete project '{name}' and ALL its data (databases, documents)?"
    else:
        msg = f"Delete project '{name}' configuration? (data will be preserved)"

    if not yes:
        if not typer.confirm(msg):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    pm.delete_project(name, delete_data=data)
    console.print(f"[green]Project '{name}' deleted.[/green]")


@project_app.command("info")
def project_info(
    name: str = typer.Argument(None, help="Project name (default: active project)"),
):
    """Show detailed project information.

    Example:
        rag project info
        rag project info safety-docs
    """
    pm = get_project_manager()

    if name is None:
        config = pm.get_active_project()
        if not config:
            console.print("[yellow]No active project.[/yellow]")
            raise typer.Exit(1)
        name = config.name
    else:
        config = pm.get_project(name)
        if not config:
            console.print(f"[red]Project '{name}' not found.[/red]")
            raise typer.Exit(1)

    paths = pm.get_project_paths(name)
    stats = pm.get_project_stats(name)
    is_active = pm.get_active_project_name() == name

    console.print()
    console.print(f"[bold cyan]Project: {name}[/bold cyan]", end="")
    if is_active:
        console.print(" [green](active)[/green]")
    else:
        console.print()

    if config.description:
        console.print(f"[dim]{config.description}[/dim]")
    console.print()

    # Basic settings
    basic_table = Table(title="Basic Settings", box=box.SIMPLE, show_header=False)
    basic_table.add_column("", style="dim", width=14)
    basic_table.add_column("", style="white")
    basic_table.add_row("Port", str(config.port))
    basic_table.add_row("Device", config.device)
    basic_table.add_row("Log Level", config.log_level)
    basic_table.add_row("Created", config.created_at[:19])
    console.print(basic_table)

    # Paths
    path_table = Table(title="Paths", box=box.SIMPLE, show_header=False)
    path_table.add_column("", style="dim", width=14)
    path_table.add_column("", style="white")
    path_table.add_row("Project Dir", str(paths["project_dir"]))
    path_table.add_row("Database", str(paths["db_path"]))
    path_table.add_row("Documents", str(paths["docs_path"]))
    console.print(path_table)

    # Document Processing
    doc_table = Table(title="Document Processing", box=box.SIMPLE, show_header=False)
    doc_table.add_column("", style="dim", width=14)
    doc_table.add_column("", style="white")
    doc_table.add_row("OCR", "Enabled" if config.enable_ocr else "Disabled")
    doc_table.add_row("OCR Engine", config.ocr_engine)
    doc_table.add_row("OCR Languages", config.ocr_languages)
    doc_table.add_row("ASR (Audio)", "Enabled" if config.enable_asr else "Disabled")
    console.print(doc_table)

    # Embedding & Chunking
    embed_table = Table(title="Embedding & Chunking", box=box.SIMPLE, show_header=False)
    embed_table.add_column("", style="dim", width=14)
    embed_table.add_column("", style="white")
    embed_table.add_row("Embedding", config.embedding_model)
    embed_table.add_row("Chunking", config.chunking_method)
    embed_table.add_row("Max Tokens", str(config.max_tokens))
    embed_table.add_row("Default Top K", str(config.default_top_k))
    console.print(embed_table)

    # MCP Server
    mcp_table = Table(title="MCP Server", box=box.SIMPLE, show_header=False)
    mcp_table.add_column("", style="dim", width=14)
    mcp_table.add_column("", style="white")
    mcp_table.add_row("Server Name", config.mcp_server_name)
    mcp_table.add_row("Transport", config.mcp_transport)
    mcp_table.add_row("Host", config.mcp_host)
    mcp_table.add_row("Cleanup", "Enabled" if config.mcp_enable_cleanup else "Disabled")
    console.print(mcp_table)

    # Stats
    console.print()
    console.print(f"[bold]Stats:[/bold] {stats['documents']} documents | {stats['disk_usage_mb']} MB disk usage")


@app.command()
def init():
    """Initialize RAG system with first-time setup wizard.

    Creates your first project with guided configuration.
    """
    pm = get_project_manager()

    console.print()
    console.print("[bold cyan]Welcome to RAG Setup![/bold cyan]")
    console.print()

    # Check if already initialized
    projects = pm.list_projects()
    if projects:
        console.print(f"[yellow]RAG is already initialized with {len(projects)} project(s).[/yellow]")
        console.print()
        project_list()
        return

    console.print("Let's create your first project.")
    console.print()

    # Get project name
    name = typer.prompt("Project name", default="default")

    # Validate name
    if not all(c.isalnum() or c in "-_" for c in name):
        console.print("[red]Invalid name. Use only letters, numbers, hyphens, underscores.[/red]")
        raise typer.Exit(1)

    # Get port
    port = typer.prompt("MCP server port", default="9090")
    try:
        port = int(port)
    except ValueError:
        console.print("[red]Invalid port number.[/red]")
        raise typer.Exit(1)

    # Get device
    console.print()
    console.print("Select compute device:")
    console.print("  [cyan]cpu[/cyan]  - CPU only (default, works everywhere)")
    console.print("  [cyan]cuda[/cyan] - NVIDIA GPU (faster, requires CUDA)")
    console.print("  [cyan]mps[/cyan]  - Apple Silicon GPU")
    console.print("  [cyan]auto[/cyan] - Auto-detect best available")
    device = typer.prompt("Device", default="cpu")

    # Get OCR preference
    enable_ocr = typer.confirm("Enable OCR for image-based PDFs?", default=False)

    # Get description
    description = typer.prompt("Description (optional)", default="")

    console.print()

    # Create project
    try:
        config = pm.create_project(
            name=name,
            port=port,
            device=device,
            enable_ocr=enable_ocr,
            description=description,
            switch_to=True
        )

        console.print("[green]Setup complete![/green]")
        console.print()

        paths = pm.get_project_paths(name)

        console.print("[bold]Your project:[/bold]")
        console.print(f"  Name: [cyan]{config.name}[/cyan]")
        console.print(f"  Port: {config.port}")
        console.print(f"  Device: {config.device}")
        console.print(f"  Database: {paths['db_path']}")
        console.print(f"  Documents: {paths['docs_path']}")
        console.print()
        console.print("[bold]Next steps:[/bold]")
        console.print(f"  1. Add documents to: [cyan]{paths['docs_path']}[/cyan]")
        console.print("  2. Run: [cyan]rag ingestion start[/cyan]")
        console.print("  3. Query: [cyan]rag query \"your question\"[/cyan]")
        console.print("  4. Start server: [cyan]rag mcp serve[/cyan]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


# Register all subcommand groups
app.add_typer(project_app, name="project")
app.add_typer(ingestion_app, name="ingestion")
app.add_typer(config_app, name="config")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        raise typer.Exit(0)
