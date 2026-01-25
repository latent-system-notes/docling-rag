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

logger = get_logger(__name__)

app = typer.Typer()
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
    result = parallel_ingest_documents(directory, config)

    # Display results
    console.print(f"\n[bold]Parallel Ingestion Complete:[/bold]")
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


@app.command()
def ingest(
    path: str,
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan subdirectories"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be updated without doing it"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already exists"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from checkpoint if exists (default: enabled)"),
    workers: int = typer.Option(4, "--workers", "-w", help="Number of parallel workers (default: 4)"),
):
    """Ingest file(s) or folder using parallel processing.

    For single files: Ingests the file directly
    For folders: Uses parallel workers for fast ingestion
    - Automatically skips already indexed files
    - Supports checkpoint-based resume for interrupted ingestion

    Uses multiple worker processes for CPU-intensive parsing,
    with a single writer thread for database operations.

    Examples:
        rag ingest paper.pdf                # Ingest single file
        rag ingest ./documents              # Ingest folder (4 workers)
        rag ingest ./docs --workers 8       # Use 8 workers
        rag ingest ./docs -w 2              # Use 2 workers
        rag ingest ./docs --dry-run         # Preview what would be ingested
        rag ingest ./docs --no-recursive    # Ingest without subdirs
        rag ingest ./docs --force           # Force re-ingest all
    """
    from src.ingestion.pipeline import ingest_document

    file_path = Path(path)

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
    from pathlib import Path
    import json

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
    if typer.confirm("Are you sure you want to reset the system?"):
        reset_collection()
        console.print("[green]System reset complete")


@app.command()
def serve(mcp: bool = True):
    """Start MCP server"""
    from src.mcp.server import run_server

    if mcp:
        console.print(f"[green]Starting MCP server on http://{settings.mcp_host}:{settings.mcp_port}")

        try:
            run_server()
        except KeyboardInterrupt:
            console.print("\n[yellow]MCP server stopped.[/yellow]")


@app.command()
def config_show():
    """Show current configuration"""
    table = Table(title="Configuration", box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    config_dict = settings.model_dump()
    for key, value in config_dict.items():
        table.add_row(key, str(value))

    console.print(table)


@app.command()
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
            console.print("[yellow]Run 'rag models --download' to download missing models")

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


@app.command()
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


@app.command()
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


@app.command()
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
    console.print(f"\n[bold]Tip:[/bold] Resume ingestion by running: rag ingest <file_path>")


@app.command()
def checkpoints_clean(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
    stale_only: bool = typer.Option(False, "--stale-only", help="Only delete stale checkpoints (older than retention period)"),
):
    """Delete ingestion checkpoints.

    By default, deletes ALL checkpoints. Use --stale-only to only delete
    checkpoints older than the retention period (default: 7 days).

    Examples:
        rag checkpoints-clean                  # Delete all checkpoints (with confirmation)
        rag checkpoints-clean -y               # Delete all checkpoints (no confirmation)
        rag checkpoints-clean --stale-only     # Delete only stale checkpoints
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


@app.command()
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
        rag ingestion-log                    # Show last 20 entries
        rag ingestion-log --limit 50         # Show last 50 entries
        rag ingestion-log --export log.csv   # Export full log
    """
    import csv
    import shutil
    from src.ingestion.audit_log import get_audit_log_path

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

        table.add_row(
            timestamp,
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


@app.command()
def status(
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
        rag status                    # Live dashboard (updates every 2s)
        rag status --once             # Single snapshot
        rag status --refresh 1        # Faster refresh rate
        rag status --history          # Show recent session history
    """
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    from src.ingestion.status import get_status_manager
    import time

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
                "[dim]Run 'rag ingest <directory>' to start ingestion[/dim]",
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


@app.command()
def stop(
    worker_id: int = typer.Option(None, "--worker", "-w", help="Stop specific worker by ID"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Stop running ingestion process.

    Sends a graceful stop signal to workers. Workers will finish their
    current file before stopping.

    Examples:
        rag stop                      # Stop all workers (with confirmation)
        rag stop -y                   # Stop all workers (no confirmation)
        rag stop --worker 2           # Stop only worker 2
    """
    from src.ingestion.status import get_status_manager

    status_mgr = get_status_manager()

    # Find active session
    session = status_mgr.get_active_session()

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
        if not yes:
            if not typer.confirm(f"Stop all workers in session {session.session_id}?"):
                console.print("[yellow]Cancelled.[/yellow]")
                return

        status_mgr.send_stop_signal(session.session_id)
        status_mgr.complete_session(session.session_id, "stopped")
        console.print("[green]Stop signal sent to all workers[/green]")
        console.print("[dim]Workers will stop after completing their current files.[/dim]")


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        raise typer.Exit(0)
