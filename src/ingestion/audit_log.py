import csv
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import settings

CSV_HEADERS = ["timestamp", "session_id", "file_name", "file_path", "doc_id", "doc_type", "language", "num_pages", "num_chunks", "status", "duration_seconds"]

def get_audit_log_path() -> Path:
    log_dir = settings.chroma_persist_dir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "ingestion_summary.csv"

def initialize_audit_log():
    csv_path = get_audit_log_path()
    if not csv_path.exists():
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writeheader()
    else:
        try:
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                existing_headers = next(csv.reader(f), None)
            if existing_headers and 'session_id' not in existing_headers:
                rows = []
                with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                    for row in csv.DictReader(f):
                        row['session_id'] = ''
                        rows.append(row)
                with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
        except Exception:
            pass

def log_ingestion(file_path: Path, doc_id: str, doc_type: str, language: str, num_pages: Optional[int],
                  num_chunks: int, status: str, start_time: datetime, end_time: datetime, session_id: Optional[str] = None):
    initialize_audit_log()
    csv_path = get_audit_log_path()
    duration = (end_time - start_time).total_seconds()
    row = {"timestamp": start_time.isoformat(), "session_id": session_id or "", "file_name": file_path.name,
           "file_path": str(file_path.absolute()), "doc_id": doc_id[-8:] if len(doc_id) >= 8 else doc_id,
           "doc_type": doc_type, "language": language, "num_pages": num_pages if num_pages else "",
           "num_chunks": num_chunks, "status": status, "duration_seconds": f"{duration:.2f}"}
    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            csv.DictWriter(f, fieldnames=CSV_HEADERS).writerow(row)
    except Exception:
        pass
