import os
import itertools
import threading
import time
import logging
import concurrent.futures
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


MAX_JOB_LOGS = 5000  # cap in-memory log entries to avoid unbounded growth


def _empty_job(started_by: str, folders: str | None, force: bool, workers: int, ocr_mode: str = "smart") -> dict:
    return {
        "id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "status": "running",         # running | completed | failed | cancelled
        "started_at": datetime.now().isoformat(),
        "finished_at": None,
        "started_by": started_by,
        "folders": folders,
        "force": force,
        "workers": workers,
        "ocr_mode": ocr_mode,
        "total_files": 0,            # incremented as files are discovered (streaming)
        "scan_complete": False,       # True once the file generator is exhausted
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "current_file": None,
        "logs": [],                   # list of {"ts": ..., "level": ..., "msg": ...}
        "logs_dropped": 0,           # count of log entries dropped due to cap
        "log_lock": threading.Lock(), # protects logs list from concurrent access
        "cancel_flag": threading.Event(),
    }


def _add_log(job: dict, level: str, msg: str):
    with job["log_lock"]:
        if len(job["logs"]) >= MAX_JOB_LOGS:
            job["logs_dropped"] += 1
            return
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
            job["folders_resolved"] = include_folders
            _add_log(job, "INFO", f"Folder filter active: {', '.join(include_folders)}")
            _add_log(job, "INFO", f"Only files inside these subfolders of {doc_path} will be ingested")

        if not Path(doc_path).exists():
            _add_log(job, "ERROR", f"Documents directory not found: {doc_path}")
            job["status"] = "failed"
            job["finished_at"] = datetime.now().isoformat()
            return

        _add_log(job, "INFO", f"Discovering and ingesting files from {doc_path} ...")

        def _process_file(f: Path):
            if job["cancel_flag"].is_set():
                return False, False, True  # treat as failed/cancelled

            job["current_file"] = f.name

            if not job["force"]:
                # No lock needed — document_exists is a read-only DB call,
                # concurrent reads are safe via the connection pool
                if document_exists(f):
                    _add_log(job, "SKIP", f"[SKIPPED] {f.name} — already ingested")
                    return False, True, False

            try:
                meta = ingest_document(f, ocr_mode=job["ocr_mode"])
                _add_log(job, "INFO", f"[OK] {f.name} ({meta.num_chunks} chunks)")
                return True, False, False
            except Exception as e:
                _add_log(job, "ERROR", f"[FAILED] {f.name} — {e}")
                return False, False, True

        from concurrent.futures import ThreadPoolExecutor

        # Single-pass streaming: discover and ingest in one pass, no upfront count
        with managed_resources():
            with ThreadPoolExecutor(max_workers=job["workers"]) as executor:
                batch_size = job["workers"] * 2
                file_gen = discover_files(Path(doc_path), recursive=True, include_folders=include_folders)
                active = {}
                gen_exhausted = False

                def _submit_from_gen(count):
                    """Pull up to `count` files from generator and submit them."""
                    nonlocal gen_exhausted
                    submitted = 0
                    for f in itertools.islice(file_gen, count):
                        job["total_files"] += 1
                        active[executor.submit(_process_file, f)] = f
                        submitted += 1
                    if submitted < count:
                        gen_exhausted = True

                # Seed initial batch
                _submit_from_gen(batch_size)

                while active:
                    if job["cancel_flag"].is_set():
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    done, _ = concurrent.futures.wait(
                        active, return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    for future in done:
                        del active[future]
                        proc, skip, fail = future.result()
                        if proc:
                            job["processed"] += 1
                        if skip:
                            job["skipped"] += 1
                        if fail:
                            job["failed"] += 1

                    # Refill from generator
                    if not gen_exhausted:
                        _submit_from_gen(len(done))

                job["scan_complete"] = True

        if job["total_files"] == 0:
            _add_log(job, "INFO", "No files found to process")

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
    ocr_mode: str = "smart"  # "smart" or "simple"


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
            ocr_mode=body.ocr_mode,
        )
        _current_job = job

    thread = threading.Thread(target=_run_ingestion, args=(job,), daemon=True)
    thread.start()

    return {"ok": True, "job_id": job["id"], "message": "Ingestion started"}


@router.get("/status")
async def get_status():
    # Grab reference under lock to avoid TOCTOU race
    with _job_lock:
        job = _current_job
    if not job:
        return {"active": False}

    return {
        "active": job["status"] == "running",
        "id": job["id"],
        "status": job["status"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "started_by": job["started_by"],
        "folders": job.get("folders") or None,
        "folders_resolved": job.get("folders_resolved") or None,
        "force": job.get("force", False),
        "ocr_mode": job.get("ocr_mode", "smart"),
        "total_files": job["total_files"],
        "scan_complete": job.get("scan_complete", False),
        "processed": job["processed"],
        "skipped": job["skipped"],
        "failed": job["failed"],
        "current_file": job["current_file"],
    }


@router.get("/logs")
async def get_logs(offset: int = Query(0, ge=0)):
    with _job_lock:
        job = _current_job
    if not job:
        return {"logs": [], "total": 0}

    with job["log_lock"]:
        logs_snapshot = job["logs"][offset:]
        total = len(job["logs"])
        dropped = job.get("logs_dropped", 0)

    return {
        "logs": logs_snapshot,
        "total": total,
        "dropped": dropped,
    }


@router.post("/cancel")
async def cancel_ingestion(user: dict = Depends(require_admin)):
    with _job_lock:
        job = _current_job
    if not job or job["status"] != "running":
        raise HTTPException(status_code=400, detail="No running ingestion job to cancel")

    job["cancel_flag"].set()
    return {"ok": True, "message": "Cancel signal sent"}
