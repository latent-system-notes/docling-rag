import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from ..config import settings, get_logger

logger = get_logger(__name__)

def get_mcp_status_db_path() -> Path:
    return settings.chroma_persist_dir.parent / "mcp_status.db"

@dataclass
class ServerInfo:
    pid: int
    host: str
    port: int
    started_at: datetime
    status: str
    @property
    def uptime_seconds(self) -> float:
        return (datetime.now() - self.started_at).total_seconds() if self.status == "running" else 0.0

@dataclass
class QueryMetric:
    id: int
    timestamp: datetime
    query_text: str
    top_k: int
    response_time_ms: float
    result_count: int
    error: Optional[str]
    success: bool

@dataclass
class MetricsSummary:
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    error_rate: float
    queries_per_minute: float
    period_minutes: int

class MCPStatusManager:
    def __init__(self, db_path: Path = None):
        self.db_path = db_path or get_mcp_status_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS server_info (id INTEGER PRIMARY KEY CHECK (id = 1), pid INTEGER NOT NULL,
                    host TEXT NOT NULL, port INTEGER NOT NULL, started_at TEXT NOT NULL, status TEXT DEFAULT 'running');
                CREATE TABLE IF NOT EXISTS query_metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL,
                    query_text TEXT NOT NULL, top_k INTEGER NOT NULL, response_time_ms REAL NOT NULL,
                    result_count INTEGER NOT NULL, error TEXT, success INTEGER DEFAULT 1);
                CREATE INDEX IF NOT EXISTS idx_query_metrics_timestamp ON query_metrics(timestamp);
            """)

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

    def register_server(self, pid: int, host: str, port: int) -> None:
        with self._get_connection() as conn:
            conn.execute("INSERT OR REPLACE INTO server_info (id, pid, host, port, started_at, status) VALUES (1, ?, ?, ?, ?, 'running')",
                (pid, host, port, datetime.now().isoformat()))

    def mark_server_stopped(self) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE server_info SET status = 'stopped' WHERE id = 1")

    def mark_server_crashed(self) -> None:
        with self._get_connection() as conn:
            conn.execute("UPDATE server_info SET status = 'crashed' WHERE id = 1")

    def get_server_info(self) -> Optional[ServerInfo]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT pid, host, port, started_at, status FROM server_info WHERE id = 1").fetchone()
            if row:
                return ServerInfo(pid=row['pid'], host=row['host'], port=row['port'],
                    started_at=datetime.fromisoformat(row['started_at']), status=row['status'])
        return None

    def is_server_running(self) -> bool:
        info = self.get_server_info()
        if not info or info.status != "running":
            return False
        if self._is_process_running(info.pid):
            return True
        self.mark_server_crashed()
        return False

    def _is_process_running(self, pid: int) -> bool:
        try:
            if os.name == 'nt':
                import ctypes
                handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
                if handle:
                    ctypes.windll.kernel32.CloseHandle(handle)
                    return True
                return False
            else:
                os.kill(pid, 0)
                return True
        except (OSError, PermissionError):
            return False

    def log_query(self, query_text: str, top_k: int, response_time_ms: float, result_count: int, error: Optional[str] = None) -> None:
        with self._get_connection() as conn:
            conn.execute("INSERT INTO query_metrics (timestamp, query_text, top_k, response_time_ms, result_count, error, success) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (datetime.now().isoformat(), query_text, top_k, response_time_ms, result_count, error, 1 if error is None else 0))

    def get_recent_queries(self, limit: int = 10) -> List[QueryMetric]:
        with self._get_connection() as conn:
            rows = conn.execute("SELECT id, timestamp, query_text, top_k, response_time_ms, result_count, error, success FROM query_metrics ORDER BY timestamp DESC LIMIT ?", (limit,)).fetchall()
            return [QueryMetric(id=r['id'], timestamp=datetime.fromisoformat(r['timestamp']), query_text=r['query_text'],
                top_k=r['top_k'], response_time_ms=r['response_time_ms'], result_count=r['result_count'],
                error=r['error'], success=bool(r['success'])) for r in rows]

    def get_metrics_summary(self, since_minutes: int = 60) -> MetricsSummary:
        with self._get_connection() as conn:
            row = conn.execute("""SELECT COUNT(*) as total, SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed, AVG(response_time_ms) as avg_time,
                MIN(response_time_ms) as min_time, MAX(response_time_ms) as max_time FROM query_metrics
                WHERE timestamp >= datetime('now', ?)""", (f'-{since_minutes} minutes',)).fetchone()
            total = row['total'] or 0
            successful = row['successful'] or 0
            failed = row['failed'] or 0
            return MetricsSummary(total_queries=total, successful_queries=successful, failed_queries=failed,
                avg_response_time_ms=row['avg_time'] or 0.0, min_response_time_ms=row['min_time'] or 0.0,
                max_response_time_ms=row['max_time'] or 0.0, error_rate=failed / total if total > 0 else 0.0,
                queries_per_minute=total / since_minutes if since_minutes > 0 else 0.0, period_minutes=since_minutes)

    def cleanup_old_metrics(self, retention_days: int = 7) -> int:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM query_metrics WHERE timestamp < datetime('now', ?)", (f'-{retention_days} days',))
            return cursor.rowcount

_mcp_status_manager: Optional[MCPStatusManager] = None
_mcp_status_manager_path: Optional[Path] = None

def get_mcp_status_manager() -> MCPStatusManager:
    global _mcp_status_manager, _mcp_status_manager_path
    current_path = get_mcp_status_db_path()
    if _mcp_status_manager is None or _mcp_status_manager_path != current_path:
        _mcp_status_manager = MCPStatusManager(current_path)
        _mcp_status_manager_path = current_path
    return _mcp_status_manager
