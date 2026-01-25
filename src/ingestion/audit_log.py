"""Ingestion audit log - CSV tracking of document ingestion history."""

import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import settings, get_logger

logger = get_logger(__name__)

CSV_HEADERS = [
    "timestamp",
    "session_id",
    "file_name",
    "file_path",
    "doc_id",
    "doc_type",
    "language",
    "num_pages",
    "num_chunks",
    "status",
    "duration_seconds"
]


def get_audit_log_path() -> Path:
    """Get path to ingestion audit log CSV."""
    log_dir = settings.chroma_persist_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "ingestion_summary.csv"


def initialize_audit_log():
    """Create CSV file with headers if it doesn't exist, or migrate if headers changed."""
    csv_path = get_audit_log_path()

    if not csv_path.exists():
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writeheader()
        logger.info(f"Created ingestion audit log: {csv_path}")
    else:
        # Check if existing file needs migration (missing session_id column)
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                existing_headers = next(reader, None)

            if existing_headers and 'session_id' not in existing_headers:
                # Migrate: read all data with old headers, write with new headers
                logger.info("Migrating audit log to include session_id column...")
                rows = []
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Add empty session_id for old entries
                        row['session_id'] = ''
                        rows.append(row)

                # Rewrite with new headers
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                logger.info(f"Migrated {len(rows)} entries to new format")
        except Exception as e:
            logger.warning(f"Could not check/migrate audit log: {e}")


def log_ingestion(
    file_path: Path,
    doc_id: str,
    doc_type: str,
    language: str,
    num_pages: Optional[int],
    num_chunks: int,
    status: str,
    start_time: datetime,
    end_time: datetime,
    session_id: Optional[str] = None
):
    """Append ingestion record to CSV.

    Args:
        file_path: Path to ingested document
        doc_id: Document ID
        doc_type: File type
        language: Detected language
        num_pages: Number of pages (None if not applicable)
        num_chunks: Number of chunks created
        status: "completed", "failed", or "resumed"
        start_time: When ingestion started
        end_time: When ingestion finished
        session_id: Ingestion session ID for tracking
    """
    initialize_audit_log()
    csv_path = get_audit_log_path()

    duration = (end_time - start_time).total_seconds()

    row = {
        "timestamp": start_time.isoformat(),
        "session_id": session_id or "",
        "file_name": file_path.name,
        "file_path": str(file_path.absolute()),
        "doc_id": doc_id[-8:] if len(doc_id) >= 8 else doc_id,  # Last 8 chars for brevity
        "doc_type": doc_type,
        "language": language,
        "num_pages": num_pages if num_pages else "",
        "num_chunks": num_chunks,
        "status": status,
        "duration_seconds": f"{duration:.2f}"
    }

    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
            writer.writerow(row)
        logger.debug(f"Logged ingestion to CSV: {file_path.name}")
    except Exception as e:
        logger.error(f"Failed to write to audit log: {e}")
