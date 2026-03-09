import os
import threading
import time
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from ...config import config, get_logger
from ..deps import require_admin

router = APIRouter(prefix="/api/ingestion", tags=["ingestion"], dependencies=[Depends(require_admin)])
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# In-memory job state (single active job at a time)
# ---------------------------------------------------------------------------

_job_lock = threading.Lock()
_current_job: dict | None = None


def _empty_job(started_by: str, folders: str | None, force: bool, workers: int) -> dict:
    return {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "status": "running",         # running | completed | failed | cancelled
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "started_by": started_by,
        "folders": folders,
        "force": force,
        "workers": workers,
        "total_files": 0,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "current_file": None,
        "logs": [],                   # list of {"ts": ..., "level": ..., "msg": ...}
        "cancel_flag": threading.Event(),
    }


def _add_log(job: dict, level: str, msg: str):
    job["logs"].append({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "level": level,
        "msg": msg,
    })


# ---------------------------------------------------------------------------
# Log handler that captures into the job
# ---------------------------------------------------------------------------

class _JobLogHandler(logging.Handler):
    """Captures log records into the active job's log buffer."""

    def __init__(self, job: dict):
        super().__init__(level=logging.INFO)
        self._job = job

    def emit(self, record):
        try:
            _add_log(self._job, record.levelname, self.format(record))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Background ingestion worker
# ---------------------------------------------------------------------------

def _run_ingestion(job: dict):
    global _current_job

    handler = _JobLogHandler(job)
    handler.setFormatter(logging.Formatter("%(name)s - %(message)s"))

    # Attach handler to root logger to capture all ingestion logs
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    try:
        from ...ingestion.pipeline import ingest_document
        from ...utils import discover_files, is_supported_file, managed_resources
        from ...storage.postgres import document_exists

        doc_path = config("DOCUMENTS_DIR")

        # Resolve folder filter
        include_folders = None
        if job["folders"]:
            include_folders = [f.strip() for f in job["folders"].split("|") if f.strip()]
        else:
            include_folders = config("INCLUDE_FOLDERS")

        if include_folders:
            _add_log(job, "INFO", f"Folder filter active: {', '.join(include_folders)}")

        if not Path(doc_path).exists():
            _add_log(job, "ERROR", f"Documents directory not found: {doc_path}")
            job["status"] = "failed"
            job["finished_at"] = datetime.now().isoformat()
            return

        # Discover files first to get a total count
        _add_log(job, "INFO", f"Scanning {doc_path} ...")
        files = list(discover_files(Path(doc_path), recursive=True, include_folders=include_folders))
        job["total_files"] = len(files)
        _add_log(job, "INFO", f"Found {len(files)} file(s) to process")

        if not files:
            job["status"] = "completed"
            job["finished_at"] = datetime.now().isoformat()
            return

        lock = threading.Lock()

        def _process_file(f: Path):
            if job["cancel_flag"].is_set():
                return False, False, True  # treat as failed/cancelled

            job["current_file"] = f.name

            if not job["force"]:
                with lock:
                    if document_exists(f):
                        return False, True, False

            try:
                meta = ingest_document(f)
                _add_log(job, "INFO", f"{f.name} ({meta.num_chunks} chunks)")
                return True, False, False
            except Exception as e:
                _add_log(job, "ERROR", f"{f.name}: {e}")
                return False, False, True

        from concurrent.futures import ThreadPoolExecutor, as_completed

        with managed_resources():
            with ThreadPoolExecutor(max_workers=job["workers"]) as executor:
                futures = {executor.submit(_process_file, f): f for f in files}
                for future in as_completed(futures):
                    if job["cancel_flag"].is_set():
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    proc, skip, fail = future.result()
                    if proc:
                        job["processed"] += 1
                    if skip:
                        job["skipped"] += 1
                    if fail:
                        job["failed"] += 1

        if job["cancel_flag"].is_set():
            job["status"] = "cancelled"
            _add_log(job, "WARNING", "Ingestion cancelled by user")
        else:
            job["status"] = "completed"
            _add_log(job, "INFO",
                     f"Done: {job['processed']} processed, {job['skipped']} skipped, {job['failed']} failed")

    except Exception as e:
        job["status"] = "failed"
        _add_log(job, "ERROR", f"Ingestion failed: {e}")
        logger.exception("Ingestion job failed")
    finally:
        job["finished_at"] = datetime.now().isoformat()
        job["current_file"] = None
        root_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    folders: str | None = None
    force: bool = False
    workers: int = 3


@router.post("/start")
async def start_ingestion(body: StartRequest, user: dict = Depends(require_admin)):
    global _current_job

    with _job_lock:
        if _current_job and _current_job["status"] == "running":
            raise HTTPException(status_code=409, detail="An ingestion job is already running")

        job = _empty_job(
            started_by=user.get("sub", "admin"),
            folders=body.folders,
            force=body.force,
            workers=body.workers,
        )
        _current_job = job

    thread = threading.Thread(target=_run_ingestion, args=(job,), daemon=True)
    thread.start()

    return {"ok": True, "job_id": job["id"], "message": "Ingestion started"}


@router.get("/status")
async def get_status():
    if not _current_job:
        return {"active": False}

    job = _current_job
    return {
        "active": job["status"] == "running",
        "id": job["id"],
        "status": job["status"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "started_by": job["started_by"],
        "total_files": job["total_files"],
        "processed": job["processed"],
        "skipped": job["skipped"],
        "failed": job["failed"],
        "current_file": job["current_file"],
    }


@router.get("/logs")
async def get_logs(offset: int = Query(0, ge=0)):
    if not _current_job:
        return {"logs": [], "total": 0}

    logs = _current_job["logs"]
    return {
        "logs": logs[offset:],
        "total": len(logs),
    }


@router.post("/cancel")
async def cancel_ingestion(user: dict = Depends(require_admin)):
    if not _current_job or _current_job["status"] != "running":
        raise HTTPException(status_code=400, detail="No running ingestion job to cancel")

    _current_job["cancel_flag"].set()
    return {"ok": True, "message": "Cancel signal sent"}
