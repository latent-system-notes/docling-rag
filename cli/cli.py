from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from rich import box

from src.config import settings, get_logger
from src.storage.chroma_client import get_stats, initialize_collections, reset_collection, document_exists, list_documents, remove_document
from src.utils import discover_files, is_file_modified
from src.models import AnswerMode

logger = get_logger(__name__)

app = typer.Typer()
console = Console()


@app.command()
def ingest(
    path: str,
    force: bool = typer.Option(False, "--force", "-f", help="Re-ingest even if already exists"),
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan subdirectories"),
):
    """Ingest document(s) into the RAG system.

    Automatically detects whether the path is a file or directory.
    For directories, only supported file types are ingested (PDF, DOCX, etc.).
    Files already in the database are skipped unless --force is used.

    Examples:
        rag ingest paper.pdf                    # Single file
        rag ingest ./documents                  # Folder (auto-detected)
        rag ingest ./docs --no-recursive        # Folder, no subdirs
        rag ingest ./docs --force               # Re-ingest everything
    """
    from src.ingestion.pipeline import ingest_document
    from src.ingestion.progress import IngestionProgress

    file_path = Path(path)

    # Initialize collections
    initialize_collections()

    # Auto-detect folder vs file
    if file_path.is_dir():
        # Folder ingestion
        console.print(f"[cyan]Scanning {file_path}...")

        # Discover files with filtering
        all_files = discover_files(file_path, recursive=recursive)

        if not all_files:
            console.print(f"[yellow]No supported files found in {file_path}")
            return

        # Filter out already-ingested files (unless --force)
        if not force:
            new_files = []
            skipped = []
            for f in all_files:
                if document_exists(f):
                    skipped.append(f)
                else:
                    new_files.append(f)

            if skipped:
                console.print(f"[yellow]Skipping {len(skipped)} already-ingested files (use --force to re-ingest)")

            files_to_ingest = new_files
        else:
            files_to_ingest = all_files

        if not files_to_ingest:
            console.print(f"[green]All files already ingested. Use --force to re-ingest.")
            return

        console.print(f"[green]Found {len(files_to_ingest)} files to ingest\n")

        # Process with detailed progress
        results = []
        failed = []
        with IngestionProgress(len(files_to_ingest)) as progress:
            for file in files_to_ingest:
                try:
                    metadata = ingest_document(file)
                    results.append(metadata)
                    progress.update(file, len(results))
                    console.print(f"  [green]✓[/green] {file.name} ({metadata.num_chunks} chunks)")
                except Exception as e:
                    failed.append(file)
                    logger.error(f"Failed to ingest {file}: {e}")
                    console.print(f"  [red]✗[/red] {file.name} - {str(e)}")

        # Summary
        console.print(f"\n[bold]Summary:[/bold]")
        console.print(f"  [green]✓ Ingested:[/green] {len(results)} files")
        if failed:
            console.print(f"  [red]✗ Failed:[/red] {len(failed)} files")
        if not force and skipped:
            console.print(f"  [yellow]→ Already in DB:[/yellow] {len(skipped)} files")

    elif file_path.is_file():
        # Single file ingestion
        if not force and document_exists(file_path):
            console.print(f"[yellow]Document already exists. Use --force to re-ingest.")
            raise typer.Exit(0)

        try:
            metadata = ingest_document(file_path)
            console.print(f"[green]✓ Ingested: {file_path.name} ({metadata.num_chunks} chunks)")
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            console.print(f"[red]✗ Failed to ingest {file_path.name}: {str(e)}")
            raise typer.Exit(1)

    else:
        console.print(f"[red]Error: {path} is not a file or directory")
        raise typer.Exit(1)


@app.command()
def list_docs(
    full_path: bool = typer.Option(False, "--full-path", "-f", help="Show full file paths instead of just filenames")
):
    """List all indexed documents with audit information.

    Shows:
    - File name (or full path with --full-path)
    - Document type
    - Language
    - Number of chunks
    - Ingestion timestamp
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

    documents = list_documents()

    if not documents:
        console.print("[yellow]No documents indexed yet.")
        return

    table = Table(title=f"Indexed Documents ({len(documents)} total)", box=None, show_header=True)
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

        table.add_row(
            file_display,
            doc['doc_type'],
            doc['language'],
            str(doc['num_chunks']),
            ingested_str
        )

    console.print(table)

    # Summary statistics
    total_chunks = sum(doc['num_chunks'] for doc in documents)
    console.print(f"\n[bold]Total:[/bold] {len(documents)} documents, {total_chunks} chunks")


@app.command()
def remove(
    path: str,
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Remove a document from the index.

    Examples:
        rag remove paper.pdf
        rag remove ./documents/report.pdf -y
    """
    file_path = Path(path)

    initialize_collections()

    # Confirm deletion unless --yes flag
    if not yes:
        if not typer.confirm(f"Remove {file_path} from index?"):
            console.print("[yellow]Cancelled.")
            return

    num_removed = remove_document(file_path)

    if num_removed > 0:
        console.print(f"[green]✓ Removed {file_path} ({num_removed} chunks)")
    else:
        console.print(f"[yellow]Document not found: {file_path}")


@app.command()
def sync(
    path: str,
    recursive: bool = typer.Option(True, "--recursive/--no-recursive", help="Scan subdirectories"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be updated without doing it"),
):
    """Sync a folder by ingesting new/modified files and removing deleted ones.

    Automatically detects:
    - New files (not in index)
    - Modified files (changed since ingestion)
    - Deleted files (in index but not on disk)

    Examples:
        rag sync ./documents              # Sync folder
        rag sync ./docs --dry-run         # Preview changes
        rag sync ./docs --no-recursive    # Sync without subdirs
    """
    from src.ingestion.pipeline import ingest_document

    folder_path = Path(path)

    if not folder_path.is_dir():
        console.print(f"[red]Error: {path} is not a directory")
        raise typer.Exit(1)

    initialize_collections()

    console.print(f"[cyan]Analyzing {folder_path}...")

    # Get all current documents from DB
    indexed_docs = list_documents()
    indexed_paths = {Path(doc['file_path']): doc for doc in indexed_docs}

    # Discover files on disk
    disk_files = set(discover_files(folder_path, recursive=recursive))

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
    console.print(f"\n[bold]Sync Analysis:[/bold]")
    console.print(f"  [green]New files:[/green] {len(new_files)}")
    console.print(f"  [yellow]Modified files:[/yellow] {len(modified_files)}")
    console.print(f"  [red]Deleted files:[/red] {len(deleted_files)}")

    if not new_files and not modified_files and not deleted_files:
        console.print(f"\n[green]✓ Everything is up to date!")
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

    # Execute sync
    console.print("\n[cyan]Syncing...")

    # 1. Remove deleted files
    for file_path in deleted_files:
        num_removed = remove_document(file_path)
        console.print(f"  [red]✗[/red] Removed {file_path.name} ({num_removed} chunks)")

    # 2. Re-ingest modified files (remove old version first)
    for file_path in modified_files:
        remove_document(file_path)  # Remove old version
        try:
            metadata = ingest_document(file_path)
            console.print(f"  [yellow]↻[/yellow] Updated {file_path.name} ({metadata.num_chunks} chunks)")
        except Exception as e:
            logger.error(f"Failed to update {file_path}: {e}")
            console.print(f"  [red]✗[/red] Failed to update {file_path.name}: {str(e)}")

    # 3. Ingest new files
    for file_path in new_files:
        try:
            metadata = ingest_document(file_path)
            console.print(f"  [green]✓[/green] Added {file_path.name} ({metadata.num_chunks} chunks)")
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            console.print(f"  [red]✗[/red] Failed to add {file_path.name}: {str(e)}")

    # Final summary
    console.print(f"\n[bold]Sync Complete:[/bold]")
    console.print(f"  [green]Added:[/green] {len(new_files)} files")
    console.print(f"  [yellow]Updated:[/yellow] {len(modified_files)} files")
    console.print(f"  [red]Removed:[/red] {len(deleted_files)} files")


@app.command()
def query_cmd(
    query_text: str,
    mode: AnswerMode = settings.answer_mode,
    top_k: int = settings.default_top_k,
):
    """Query the RAG system"""
    from src.query import query

    result = query(query_text, top_k, mode)

    console.print(f"[bold]Query:[/bold] {result.query}")
    console.print(f"[bold]Mode:[/bold] {result.mode}\n")

    if result.context:
        console.print("[bold]Context:[/bold]")
        for idx, res in enumerate(result.context, 1):
            # Safely encode text for console display, replacing problematic characters
            try:
                preview = res.chunk.text[:200]
            except UnicodeError:
                preview = res.chunk.text[:200].encode('ascii', 'replace').decode('ascii')

            try:
                console.print(f"  [{idx}] (Score: {res.score:.3f}) {preview}...")
            except UnicodeEncodeError:
                # Fallback: show score and source only
                console.print(f"  [{idx}] (Score: {res.score:.3f}) [Text contains special characters - see source: page {res.chunk.page_num}]")

    if result.answer:
        try:
            console.print(f"\n[bold]Answer:[/bold]\n{result.answer}")
        except UnicodeEncodeError:
            # Fallback for answer with encoding issues
            safe_answer = result.answer.encode('ascii', 'replace').decode('ascii')
            console.print(f"\n[bold]Answer:[/bold]\n{safe_answer}")


@app.command()
def search_cmd(query_text: str, top_k: int = settings.default_top_k):
    """Search without generation"""
    from src.retrieval.search import search

    results = search(query_text, top_k)

    table = Table(title="Search Results", box=None)
    table.add_column("Rank", style="cyan")
    table.add_column("Score", style="magenta")
    table.add_column("Text", style="green")

    for idx, result in enumerate(results, 1):
        table.add_row(str(idx), f"{result.score:.3f}", result.chunk.text[:100] + "...")

    console.print(table)


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
        console.print("[yellow]Downloading all models to ./models...")
        from src.utils import download_all_models

        download_all_models()
        console.print("[green]All models downloaded successfully!")
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


if __name__ == "__main__":
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        raise typer.Exit(0)
