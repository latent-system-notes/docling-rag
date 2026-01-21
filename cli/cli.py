from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from src.config import settings, get_logger
from src.storage.chroma_client import get_stats, initialize_collections, reset_collection, document_exists, list_documents, remove_document, remove_document_by_id
from src.utils import discover_files, is_file_modified

logger = get_logger(__name__)

app = typer.Typer()
console = Console()


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


@app.command()
def ingest(
    path: str,
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan subdirectories"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be updated without doing it"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already exists"),
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from checkpoint if exists (default: enabled)"),
):
    """Ingest file(s) or folder by ingesting new/modified files and removing deleted ones.

    For single files: Just ingests the file
    For folders: Automatically detects changes and syncs
    - New files (not in index)
    - Modified files (changed since ingestion)
    - Deleted files (in index but not on disk)

    Examples:
        rag ingest paper.pdf                # Ingest single file
        rag ingest ./documents              # Ingest folder
        rag ingest ./docs --dry-run         # Preview changes
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

    # Handle folder
    if not file_path.is_dir():
        console.print(f"[red]Error: {path} is not a file or directory")
        raise typer.Exit(1)

    console.print(f"[cyan]Analyzing {file_path}...")

    # Get all current documents from DB
    indexed_docs = list_documents()
    indexed_paths = {Path(doc['file_path']): doc for doc in indexed_docs}

    # Discover files on disk
    disk_files = set(discover_files(file_path, recursive=recursive))

    if not disk_files:
        console.print(f"[yellow]No supported files found in {file_path}")
        return

    # If --force, treat all files as needing re-ingestion
    if force:
        new_files = []
        modified_files = list(disk_files)  # Re-ingest everything
        deleted_files = []
    else:
        # Find new files (on disk but not in DB)
        new_files = [f for f in disk_files if f not in indexed_paths]

        # Find modified files (on disk, in DB, but modified)
        modified_files = []
        for f in disk_files:
            if f in indexed_paths:
                doc = indexed_paths[f]
                if is_file_modified(f, doc['ingested_at']):
                    modified_files.append(f)

        # Find deleted files (in DB but not on disk)
        deleted_files = [p for p in indexed_paths.keys() if p not in disk_files]

    # Summary
    console.print(f"\n[bold]Ingestion Analysis:[/bold]")
    console.print(f"  [green]New files:[/green] {len(new_files)}")
    console.print(f"  [yellow]Modified files:[/yellow] {len(modified_files)}")
    console.print(f"  [red]Deleted files:[/red] {len(deleted_files)}")

    if not new_files and not modified_files and not deleted_files:
        console.print(f"\n[green][OK] Everything is up to date!")
        return

    if dry_run:
        console.print("\n[yellow]Dry run - no changes will be made[/yellow]")

        if new_files:
            console.print("\n[bold]Would ingest:[/bold]")
            for f in new_files:
                console.print(f"  + {f.name}")

        if modified_files:
            console.print("\n[bold]Would update:[/bold]")
            for f in modified_files:
                console.print(f"  ~ {f.name}")

        if deleted_files:
            console.print("\n[bold]Would remove:[/bold]")
            for f in deleted_files:
                console.print(f"  - {f.name}")

        return

    # Execute ingestion
    console.print("\n[cyan]Ingesting...")

    total_files = len(deleted_files) + len(modified_files) + len(new_files)
    current = 0

    # 1. Remove deleted files
    for file_path in deleted_files:
        current += 1
        console.print(f"  [{current}/{total_files}] Removing {file_path.name}...")
        num_removed = remove_document(file_path)
        console.print(f"  [red][ERR][/red] Removed {file_path.name} ({num_removed} chunks)")

    # 2. Re-ingest modified files (remove old version first)
    for file_path in modified_files:
        current += 1
        console.print(f"  [{current}/{total_files}] Updating {file_path.name}...")
        remove_document(file_path)  # Remove old version
        try:
            metadata = ingest_document(file_path, resume=resume)
            console.print(f"  [yellow]↻[/yellow] Updated {file_path.name} ({metadata.num_chunks} chunks)")
        except Exception as e:
            logger.error(f"Failed to update {file_path}: {e}")
            console.print(f"  [red][ERR][/red] Failed to update {file_path.name}: {str(e)}")

    # 3. Ingest new files
    for file_path in new_files:
        current += 1
        console.print(f"  [{current}/{total_files}] Processing {file_path.name}...")
        try:
            metadata = ingest_document(file_path, resume=resume)
            console.print(f"  [green][OK][/green] Added {file_path.name} ({metadata.num_chunks} chunks)")
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            console.print(f"  [red][ERR][/red] Failed to add {file_path.name}: {str(e)}")

    # Final summary
    console.print(f"\n[bold]Ingestion Complete:[/bold]")
    console.print(f"  [green]Added:[/green] {len(new_files)} files")
    console.print(f"  [yellow]Updated:[/yellow] {len(modified_files)} files")
    console.print(f"  [red]Removed:[/red] {len(deleted_files)} files")


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

        print(json.dumps(output, ensure_ascii=True, indent=2))
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
        console.print("[green]Starting MCP server...")
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
        console.print("[yellow]Downloading embedding model to ./models...")
        from src.utils import download_embedding_model

        download_embedding_model()
        console.print("[green]Embedding model downloaded successfully!")
        console.print(f"[green]Models saved to: {settings.models_dir}")

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


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        raise typer.Exit(0)
