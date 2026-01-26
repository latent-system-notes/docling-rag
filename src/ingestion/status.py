import os
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..config import settings, get_logger

logger = get_logger(__name__)

def get_status_db_path() -> Path:
    return settings.chroma_persist_dir.parent / "ingestion_status.db"

@dataclass
class WorkerStatus:
    worker_id: int
    worker_type: str
    status: str
    current_file: Optional[str] = None
    started_at: Optional[datetime] = None
    file_started_at: Optional[datetime] = None
    files_processed: int = 0
    files_failed: int = 0
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None

    @property
    def file_duration_seconds(self) -> Optional[float]:
        return (datetime.now() - self.file_started_at).total_seconds() if self.file_started_at else None

    @property
    def total_duration_seconds(self) -> Optional[float]:
        return (datetime.now() - self.started_at).total_seconds() if self.started_at else None

@dataclass
class SessionStatus:
    session_id: str
    source_path: str
    started_at: datetime
    num_workers: int
    total_files: int
    processed_files: int = 0
    failed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    status: str = "running"
    error_message: Optional[str] = None

    @property
    def elapsed_seconds(self) -> float:
        return (datetime.now() - self.started_at).total_seconds()

    @property
    def rate(self) -> float:
        return self.processed_files / self.elapsed_seconds if self.elapsed_seconds > 0 else 0.0

    @property
    def eta_seconds(self) -> Optional[float]:
        remaining = self.total_files - self.processed_files - self.failed_files - self.skipped_files
        return remaining / self.rate if self.rate > 0 and remaining > 0 else None

class StatusManager:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or get_status_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS session (session_id TEXT PRIMARY KEY, source_path TEXT NOT NULL,
                    started_at TEXT NOT NULL, num_workers INTEGER NOT NULL, total_files INTEGER NOT NULL,
                    processed_files INTEGER DEFAULT 0, failed_files INTEGER DEFAULT 0, skipped_files INTEGER DEFAULT 0,
                    total_chunks INTEGER DEFAULT 0, status TEXT DEFAULT 'running', error_message TEXT, main_pid INTEGER);
                CREATE TABLE IF NOT EXISTS workers (session_id TEXT NOT NULL, worker_id INTEGER NOT NULL,
                    worker_type TEXT NOT NULL, status TEXT NOT NULL, pid INTEGER, current_file TEXT, started_at TEXT,
                    file_started_at TEXT, files_processed INTEGER DEFAULT 0, files_failed INTEGER DEFAULT 0,
                    last_heartbeat TEXT, error_message TEXT, PRIMARY KEY (session_id, worker_id, worker_type));
                CREATE TABLE IF NOT EXISTS signals (session_id TEXT NOT NULL, signal_type TEXT NOT NULL,
                    target_worker INTEGER, created_at TEXT NOT NULL, processed INTEGER DEFAULT 0,
                    PRIMARY KEY (session_id, signal_type, target_worker));
                CREATE INDEX IF NOT EXISTS idx_workers_session ON workers(session_id);
                CREATE INDEX IF NOT EXISTS idx_signals_session ON signals(session_id);
            """)
            try:
                conn.execute("SELECT pid FROM workers LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE workers ADD COLUMN pid INTEGER")
            try:
                conn.execute("SELECT main_pid FROM session LIMIT 1")
            except sqlite3.OperationalError:
                conn.execute("ALTER TABLE session ADD COLUMN main_pid INTEGER")

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_session(self, source_path: str, num_workers: int, total_files: int, skipped_files: int = 0, main_pid: int = None) -> str:
        self.cleanup_stale_sessions()
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        if main_pid is None:
            main_pid = os.getpid()
        with self._get_connection() as conn:
            conn.execute("INSERT INTO session (session_id, source_path, started_at, num_workers, total_files, skipped_files, status, main_pid) VALUES (?, ?, ?, ?, ?, ?, 'running', ?)",
                (session_id, source_path, now, num_workers, total_files, skipped_files, main_pid))
        return session_id

    def register_worker(self, session_id: str, worker_id: int, worker_type: str = "parser", pid: int = None):
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO workers (session_id, worker_id, worker_type, status, pid, started_at, last_heartbeat) VALUES (?, ?, ?, 'idle', ?, ?, ?)",
                (session_id, worker_id, worker_type, pid, now, now))

    def update_worker_status(self, session_id: str, worker_id: int, worker_type: str = "parser", status: str = None,
                            current_file: str = None, file_started: bool = False, file_completed: bool = False,
                            file_failed: bool = False, error_message: str = None):
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
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
                updates.extend(["files_processed = files_processed + 1", "file_started_at = NULL", "current_file = NULL"])
            if file_failed:
                updates.extend(["files_failed = files_failed + 1", "file_started_at = NULL", "current_file = NULL"])
            if error_message:
                updates.append("error_message = ?")
                params.append(error_message)
            params.extend([session_id, worker_id, worker_type])
            conn.execute(f"UPDATE workers SET {', '.join(updates)} WHERE session_id = ? AND worker_id = ? AND worker_type = ?", params)

    def update_session_stats(self, session_id: str, processed_delta: int = 0, failed_delta: int = 0, chunks_delta: int = 0):
        with self._get_connection() as conn:
            conn.execute("UPDATE session SET processed_files = processed_files + ?, failed_files = failed_files + ?, total_chunks = total_chunks + ? WHERE session_id = ?",
                (processed_delta, failed_delta, chunks_delta, session_id))

    def complete_session(self, session_id: str, status: str = "completed", error: str = None):
        with self._get_connection() as conn:
            conn.execute("UPDATE session SET status = ?, error_message = ? WHERE session_id = ?", (status, error, session_id))
            conn.execute("UPDATE workers SET status = 'stopped' WHERE session_id = ?", (session_id,))

    def get_active_session(self) -> Optional[SessionStatus]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM session WHERE status = 'running' ORDER BY started_at DESC LIMIT 1").fetchone()
            if row:
                return SessionStatus(session_id=row['session_id'], source_path=row['source_path'],
                    started_at=datetime.fromisoformat(row['started_at']), num_workers=row['num_workers'],
                    total_files=row['total_files'], processed_files=row['processed_files'], failed_files=row['failed_files'],
                    skipped_files=row['skipped_files'], total_chunks=row['total_chunks'], status=row['status'], error_message=row['error_message'])
        return None

    def get_session(self, session_id: str) -> Optional[SessionStatus]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM session WHERE session_id = ?", (session_id,)).fetchone()
            if row:
                return SessionStatus(session_id=row['session_id'], source_path=row['source_path'],
                    started_at=datetime.fromisoformat(row['started_at']), num_workers=row['num_workers'],
                    total_files=row['total_files'], processed_files=row['processed_files'], failed_files=row['failed_files'],
                    skipped_files=row['skipped_files'], total_chunks=row['total_chunks'], status=row['status'], error_message=row['error_message'])
        return None

    def get_workers(self, session_id: str) -> List[WorkerStatus]:
        workers = []
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM workers WHERE session_id = ? ORDER BY worker_type, worker_id", (session_id,)).fetchall()
            for row in rows:
                workers.append(WorkerStatus(worker_id=row['worker_id'], worker_type=row['worker_type'], status=row['status'],
                    current_file=row['current_file'], started_at=datetime.fromisoformat(row['started_at']) if row['started_at'] else None,
                    file_started_at=datetime.fromisoformat(row['file_started_at']) if row['file_started_at'] else None,
                    files_processed=row['files_processed'], files_failed=row['files_failed'],
                    last_heartbeat=datetime.fromisoformat(row['last_heartbeat']) if row['last_heartbeat'] else None, error_message=row['error_message']))
        return workers

    def get_worker_pids(self, session_id: str) -> List[int]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT pid FROM workers WHERE session_id = ? AND pid IS NOT NULL", (session_id,)).fetchall()
            return [row['pid'] for row in rows if row['pid']]

    def get_main_pid(self, session_id: str) -> Optional[int]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT main_pid FROM session WHERE session_id = ?", (session_id,)).fetchone()
            return row['main_pid'] if row and row['main_pid'] else None

    def send_stop_signal(self, session_id: str, worker_id: int = None):
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            target = worker_id if worker_id is not None else -1
            conn.execute("INSERT OR REPLACE INTO signals (session_id, signal_type, target_worker, created_at, processed) VALUES (?, 'stop', ?, ?, 0)",
                (session_id, target, now))

    def check_stop_signal(self, session_id: str, worker_id: int) -> bool:
        with self._get_connection() as conn:
            row = conn.execute("SELECT 1 FROM signals WHERE session_id = ? AND signal_type = 'stop' AND (target_worker = ? OR target_worker = -1) AND processed = 0",
                (session_id, worker_id)).fetchone()
            return row is not None

    def mark_signal_processed(self, session_id: str, worker_id: int):
        with self._get_connection() as conn:
            conn.execute("UPDATE signals SET processed = 1 WHERE session_id = ? AND signal_type = 'stop' AND (target_worker = ? OR target_worker = -1)",
                (session_id, worker_id))

    def cleanup_stale_sessions(self, max_age_hours: int = 24):
        with self._get_connection() as conn:
            conn.execute("UPDATE session SET status = 'stale' WHERE status = 'running' AND datetime(started_at) < datetime('now', ?)", (f'-{max_age_hours} hours',))
            conn.execute("DELETE FROM workers WHERE session_id IN (SELECT session_id FROM session WHERE datetime(started_at) < datetime('now', '-7 days'))")
            conn.execute("DELETE FROM signals WHERE session_id IN (SELECT session_id FROM session WHERE datetime(started_at) < datetime('now', '-7 days'))")
            conn.execute("DELETE FROM session WHERE datetime(started_at) < datetime('now', '-7 days')")

    def get_recent_sessions(self, limit: int = 10) -> List[SessionStatus]:
        sessions = []
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM session ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
            for row in rows:
                sessions.append(SessionStatus(session_id=row['session_id'], source_path=row['source_path'],
                    started_at=datetime.fromisoformat(row['started_at']), num_workers=row['num_workers'],
                    total_files=row['total_files'], processed_files=row['processed_files'], failed_files=row['failed_files'],
                    skipped_files=row['skipped_files'], total_chunks=row['total_chunks'], status=row['status'], error_message=row['error_message']))
        return sessions

_status_manager: Optional[StatusManager] = None
_status_manager_path: Optional[Path] = None

def get_status_manager() -> StatusManager:
    global _status_manager, _status_manager_path
    current_path = get_status_db_path()
    if _status_manager is None or _status_manager_path != current_path:
        _status_manager = StatusManager(current_path)
        _status_manager_path = current_path
    return _status_manager
