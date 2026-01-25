"""
Ingestion status manager for monitoring and controlling parallel ingestion.

Uses SQLite for concurrent-safe status updates from multiple workers.
"""
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

from ..config import settings, get_logger

logger = get_logger(__name__)

# Status file location
STATUS_DB_PATH = settings.chroma_persist_dir.parent / "ingestion_status.db"


@dataclass
class WorkerStatus:
    """Status of a single worker."""
    worker_id: int
    worker_type: str  # "parser" or "writer"
    status: str  # "idle", "parsing", "writing", "stopped", "error"
    current_file: Optional[str] = None
    started_at: Optional[datetime] = None
    file_started_at: Optional[datetime] = None
    files_processed: int = 0
    files_failed: int = 0
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None

    @property
    def file_duration_seconds(self) -> Optional[float]:
        """How long the worker has been on current file."""
        if self.file_started_at:
            return (datetime.now() - self.file_started_at).total_seconds()
        return None

    @property
    def total_duration_seconds(self) -> Optional[float]:
        """How long the worker has been running."""
        if self.started_at:
            return (datetime.now() - self.started_at).total_seconds()
        return None


@dataclass
class SessionStatus:
    """Status of the current ingestion session."""
    session_id: str
    source_path: str
    started_at: datetime
    num_workers: int
    total_files: int
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    status: str = "running"  # "running", "completed", "stopped", "error"
    error_message: Optional[str] = None

    @property
    def elapsed_seconds(self) -> float:
        """Total elapsed time."""
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def rate(self) -> float:
        """Documents per second."""
        if self.elapsed_seconds > 0:
            return self.processed_files / self.elapsed_seconds
        return 0.0

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining."""
        remaining = self.total_files - self.processed_files - self.failed_files - self.skipped_files
        if self.rate > 0 and remaining > 0:
            return remaining / self.rate
        return None


class StatusManager:
    """Manages ingestion status using SQLite for concurrent access."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or STATUS_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS session (
                    session_id TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    num_workers INTEGER NOT NULL,
                    total_files INTEGER NOT NULL,
                    processed_files INTEGER DEFAULT 0,
                    failed_files INTEGER DEFAULT 0,
                    skipped_files INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running',
                    error_message TEXT
                );

                CREATE TABLE IF NOT EXISTS workers (
                    session_id TEXT NOT NULL,
                    worker_id INTEGER NOT NULL,
                    worker_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_file TEXT,
                    started_at TEXT,
                    file_started_at TEXT,
                    files_processed INTEGER DEFAULT 0,
                    files_failed INTEGER DEFAULT 0,
                    last_heartbeat TEXT,
                    error_message TEXT,
                    PRIMARY KEY (session_id, worker_id, worker_type)
                );

                CREATE TABLE IF NOT EXISTS signals (
                    session_id TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    target_worker INTEGER,
                    created_at TEXT NOT NULL,
                    processed INTEGER DEFAULT 0,
                    PRIMARY KEY (session_id, signal_type, target_worker)
                );

                CREATE INDEX IF NOT EXISTS idx_workers_session ON workers(session_id);
                CREATE INDEX IF NOT EXISTS idx_signals_session ON signals(session_id);
            """)

    @contextmanager
    def _get_connection(self):
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_session(
        self,
        source_path: str,
        num_workers: int,
        total_files: int,
        skipped_files: int = 0
    ) -> str:
        """Create a new ingestion session."""
        # Clean up any existing sessions first
        self.cleanup_stale_sessions()

        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO session (
                    session_id, source_path, started_at, num_workers,
                    total_files, skipped_files, status
                ) VALUES (?, ?, ?, ?, ?, ?, 'running')
            """, (session_id, source_path, now, num_workers, total_files, skipped_files))

        logger.info(f"Created ingestion session: {session_id}")
        return session_id

    def register_worker(
        self,
        session_id: str,
        worker_id: int,
        worker_type: str = "parser"
    ):
        """Register a worker for the session."""
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workers (
                    session_id, worker_id, worker_type, status,
                    started_at, last_heartbeat
                ) VALUES (?, ?, ?, 'idle', ?, ?)
            """, (session_id, worker_id, worker_type, now, now))

    def update_worker_status(
        self,
        session_id: str,
        worker_id: int,
        worker_type: str = "parser",
        status: str = None,
        current_file: str = None,
        file_started: bool = False,
        file_completed: bool = False,
        file_failed: bool = False,
        error_message: str = None
    ):
        """Update a worker's status."""
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            # Build update query dynamically
            updates = ["last_heartbeat = ?"]
            params = [now]

            if status:
                updates.append("status = ?")
                params.append(status)

            if current_file is not None:
                updates.append("current_file = ?")
                params.append(current_file if current_file else None)

            if file_started:
                updates.append("file_started_at = ?")
                params.append(now)

            if file_completed:
                updates.append("files_processed = files_processed + 1")
                updates.append("file_started_at = NULL")
                updates.append("current_file = NULL")

            if file_failed:
                updates.append("files_failed = files_failed + 1")
                updates.append("file_started_at = NULL")
                updates.append("current_file = NULL")

            if error_message:
                updates.append("error_message = ?")
                params.append(error_message)

            params.extend([session_id, worker_id, worker_type])

            conn.execute(f"""
                UPDATE workers
                SET {', '.join(updates)}
                WHERE session_id = ? AND worker_id = ? AND worker_type = ?
            """, params)

    def update_session_stats(
        self,
        session_id: str,
        processed_delta: int = 0,
        failed_delta: int = 0,
        chunks_delta: int = 0
    ):
        """Update session statistics."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE session
                SET processed_files = processed_files + ?,
                    failed_files = failed_files + ?,
                    total_chunks = total_chunks + ?
                WHERE session_id = ?
            """, (processed_delta, failed_delta, chunks_delta, session_id))

    def complete_session(self, session_id: str, status: str = "completed", error: str = None):
        """Mark session as completed."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE session SET status = ?, error_message = ?
                WHERE session_id = ?
            """, (status, error, session_id))

            # Mark all workers as stopped
            conn.execute("""
                UPDATE workers SET status = 'stopped'
                WHERE session_id = ?
            """, (session_id,))

    def get_active_session(self) -> Optional[SessionStatus]:
        """Get the currently active ingestion session."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM session
                WHERE status = 'running'
                ORDER BY started_at DESC
                LIMIT 1
            """).fetchone()

            if row:
                return SessionStatus(
                    session_id=row['session_id'],
                    source_path=row['source_path'],
                    started_at=datetime.fromisoformat(row['started_at']),
                    num_workers=row['num_workers'],
                    total_files=row['total_files'],
                    processed_files=row['processed_files'],
                    failed_files=row['failed_files'],
                    skipped_files=row['skipped_files'],
                    total_chunks=row['total_chunks'],
                    status=row['status'],
                    error_message=row['error_message']
                )
        return None

    def get_session(self, session_id: str) -> Optional[SessionStatus]:
        """Get a specific session by ID."""
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT * FROM session WHERE session_id = ?
            """, (session_id,)).fetchone()

            if row:
                return SessionStatus(
                    session_id=row['session_id'],
                    source_path=row['source_path'],
                    started_at=datetime.fromisoformat(row['started_at']),
                    num_workers=row['num_workers'],
                    total_files=row['total_files'],
                    processed_files=row['processed_files'],
                    failed_files=row['failed_files'],
                    skipped_files=row['skipped_files'],
                    total_chunks=row['total_chunks'],
                    status=row['status'],
                    error_message=row['error_message']
                )
        return None

    def get_workers(self, session_id: str) -> List[WorkerStatus]:
        """Get all workers for a session."""
        workers = []

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM workers
                WHERE session_id = ?
                ORDER BY worker_type, worker_id
            """, (session_id,)).fetchall()

            for row in rows:
                workers.append(WorkerStatus(
                    worker_id=row['worker_id'],
                    worker_type=row['worker_type'],
                    status=row['status'],
                    current_file=row['current_file'],
                    started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    file_started_at=datetime.fromisoformat(row['file_started_at']) if row['file_started_at'] else None,
                    files_processed=row['files_processed'],
                    files_failed=row['files_failed'],
                    last_heartbeat=datetime.fromisoformat(row['last_heartbeat']) if row['last_heartbeat'] else None,
                    error_message=row['error_message']
                ))

        return workers

    def send_stop_signal(self, session_id: str, worker_id: int = None):
        """Send a stop signal to workers."""
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            # Use -1 for "all workers" signal
            target = worker_id if worker_id is not None else -1
            conn.execute("""
                INSERT OR REPLACE INTO signals (
                    session_id, signal_type, target_worker, created_at, processed
                ) VALUES (?, 'stop', ?, ?, 0)
            """, (session_id, target, now))

        if worker_id is not None:
            logger.info(f"Sent stop signal to worker {worker_id}")
        else:
            logger.info("Sent stop signal to all workers")

    def check_stop_signal(self, session_id: str, worker_id: int) -> bool:
        """Check if a stop signal exists for this worker."""
        with self._get_connection() as conn:
            # Check for specific worker signal or all-workers signal (-1)
            row = conn.execute("""
                SELECT 1 FROM signals
                WHERE session_id = ?
                AND signal_type = 'stop'
                AND (target_worker = ? OR target_worker = -1)
                AND processed = 0
            """, (session_id, worker_id)).fetchone()

            return row is not None

    def mark_signal_processed(self, session_id: str, worker_id: int):
        """Mark a stop signal as processed."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE signals SET processed = 1
                WHERE session_id = ?
                AND signal_type = 'stop'
                AND (target_worker = ? OR target_worker = -1)
            """, (session_id, worker_id))

    def cleanup_stale_sessions(self, max_age_hours: int = 24):
        """Clean up old sessions."""
        with self._get_connection() as conn:
            # Mark old running sessions as stale
            conn.execute("""
                UPDATE session SET status = 'stale'
                WHERE status = 'running'
                AND datetime(started_at) < datetime('now', ?)
            """, (f'-{max_age_hours} hours',))

            # Delete very old sessions
            conn.execute("""
                DELETE FROM workers
                WHERE session_id IN (
                    SELECT session_id FROM session
                    WHERE datetime(started_at) < datetime('now', '-7 days')
                )
            """)
            conn.execute("""
                DELETE FROM signals
                WHERE session_id IN (
                    SELECT session_id FROM session
                    WHERE datetime(started_at) < datetime('now', '-7 days')
                )
            """)
            conn.execute("""
                DELETE FROM session
                WHERE datetime(started_at) < datetime('now', '-7 days')
            """)

    def get_recent_sessions(self, limit: int = 10) -> List[SessionStatus]:
        """Get recent sessions."""
        sessions = []

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT * FROM session
                ORDER BY started_at DESC
                LIMIT ?
            """, (limit,)).fetchall()

            for row in rows:
                sessions.append(SessionStatus(
                    session_id=row['session_id'],
                    source_path=row['source_path'],
                    started_at=datetime.fromisoformat(row['started_at']),
                    num_workers=row['num_workers'],
                    total_files=row['total_files'],
                    processed_files=row['processed_files'],
                    failed_files=row['failed_files'],
                    skipped_files=row['skipped_files'],
                    total_chunks=row['total_chunks'],
                    status=row['status'],
                    error_message=row['error_message']
                ))

        return sessions


# Global status manager instance
_status_manager: Optional[StatusManager] = None


def get_status_manager() -> StatusManager:
    """Get or create the global status manager."""
    global _status_manager
    if _status_manager is None:
        _status_manager = StatusManager()
    return _status_manager
