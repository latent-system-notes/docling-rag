"""
MCP server status manager for monitoring and metrics tracking.

Uses SQLite for concurrent-safe status updates and query metrics.
"""
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from contextlib import contextmanager

from ..config import settings, get_logger

logger = get_logger(__name__)


def get_mcp_status_db_path() -> Path:
    """Get the MCP status database path for the current project.

    Must be called AFTER apply_project_settings() to get correct path.
    """
    return settings.chroma_persist_dir.parent / "mcp_status.db"


@dataclass
class ServerInfo:
    """Information about the running MCP server."""
    pid: int
    host: str
    port: int
    started_at: datetime
    status: str  # running, stopped, crashed

    @property
    def uptime_seconds(self) -> float:
        """Calculate uptime in seconds."""
        if self.status == "running":
            return (datetime.now() - self.started_at).total_seconds()
        return 0.0


@dataclass
class QueryMetric:
    """A single query metric record."""
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
    """Summary of query metrics over a time period."""
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
    """Manages MCP server status using SQLite for persistent storage."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or get_mcp_status_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                -- Server info table (singleton - only one row)
                CREATE TABLE IF NOT EXISTS server_info (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    pid INTEGER NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    status TEXT DEFAULT 'running'
                );

                -- Query metrics table
                CREATE TABLE IF NOT EXISTS query_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    query_text TEXT NOT NULL,
                    top_k INTEGER NOT NULL,
                    response_time_ms REAL NOT NULL,
                    result_count INTEGER NOT NULL,
                    error TEXT,
                    success INTEGER DEFAULT 1
                );

                -- Index for efficient time-based queries
                CREATE INDEX IF NOT EXISTS idx_query_metrics_timestamp
                ON query_metrics(timestamp);

                -- Index for success filtering
                CREATE INDEX IF NOT EXISTS idx_query_metrics_success
                ON query_metrics(success);
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

    # =========================================================================
    # Server Lifecycle Methods
    # =========================================================================

    def register_server(self, pid: int, host: str, port: int) -> None:
        """Register the server as running.

        Args:
            pid: Process ID of the server
            host: Host address the server is bound to
            port: Port the server is listening on
        """
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            # Use INSERT OR REPLACE to handle singleton pattern
            conn.execute("""
                INSERT OR REPLACE INTO server_info (id, pid, host, port, started_at, status)
                VALUES (1, ?, ?, ?, ?, 'running')
            """, (pid, host, port, now))

        logger.info(f"Registered MCP server (PID={pid}, host={host}, port={port})")

    def mark_server_stopped(self) -> None:
        """Mark the server as stopped."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE server_info SET status = 'stopped' WHERE id = 1
            """)

        logger.info("Marked MCP server as stopped")

    def mark_server_crashed(self) -> None:
        """Mark the server as crashed."""
        with self._get_connection() as conn:
            conn.execute("""
                UPDATE server_info SET status = 'crashed' WHERE id = 1
            """)

        logger.warning("Marked MCP server as crashed")

    def get_server_info(self) -> Optional[ServerInfo]:
        """Get current server information.

        Returns:
            ServerInfo if server has been registered, None otherwise
        """
        with self._get_connection() as conn:
            row = conn.execute("""
                SELECT pid, host, port, started_at, status
                FROM server_info WHERE id = 1
            """).fetchone()

            if row:
                return ServerInfo(
                    pid=row['pid'],
                    host=row['host'],
                    port=row['port'],
                    started_at=datetime.fromisoformat(row['started_at']),
                    status=row['status']
                )
        return None

    def is_server_running(self) -> bool:
        """Check if the server is currently running.

        Also validates that the process is actually alive.

        Returns:
            True if server is registered as running and process exists
        """
        info = self.get_server_info()
        if not info or info.status != "running":
            return False

        # Verify the process is actually running
        if self._is_process_running(info.pid):
            return True
        else:
            # Process is not running, update status
            self.mark_server_crashed()
            return False

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with the given PID is running.

        Works correctly on both Windows and Unix.
        """
        import sys

        if sys.platform == "win32":
            # On Windows, use tasklist to check if process exists
            import subprocess
            try:
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # If process exists, output contains the PID
                return str(pid) in result.stdout
            except Exception:
                return False
        else:
            # On Unix, os.kill with signal 0 checks if process exists
            try:
                os.kill(pid, 0)
                return True
            except (OSError, ProcessLookupError):
                return False

    # =========================================================================
    # Query Metrics Methods
    # =========================================================================

    def log_query(
        self,
        query_text: str,
        top_k: int,
        response_time_ms: float,
        result_count: int,
        error: Optional[str] = None
    ) -> None:
        """Log a query metric.

        Args:
            query_text: The query text
            top_k: Number of results requested
            response_time_ms: Response time in milliseconds
            result_count: Number of results returned
            error: Error message if query failed, None otherwise
        """
        now = datetime.now().isoformat()
        success = 1 if error is None else 0

        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO query_metrics
                (timestamp, query_text, top_k, response_time_ms, result_count, error, success)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (now, query_text, top_k, response_time_ms, result_count, error, success))

    def get_recent_queries(self, limit: int = 10) -> List[QueryMetric]:
        """Get recent query metrics.

        Args:
            limit: Maximum number of queries to return

        Returns:
            List of QueryMetric objects, most recent first
        """
        queries = []

        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT id, timestamp, query_text, top_k, response_time_ms,
                       result_count, error, success
                FROM query_metrics
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,)).fetchall()

            for row in rows:
                queries.append(QueryMetric(
                    id=row['id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    query_text=row['query_text'],
                    top_k=row['top_k'],
                    response_time_ms=row['response_time_ms'],
                    result_count=row['result_count'],
                    error=row['error'],
                    success=bool(row['success'])
                ))

        return queries

    def get_metrics_summary(self, since_minutes: int = 60) -> MetricsSummary:
        """Get a summary of query metrics over a time period.

        Args:
            since_minutes: Look back this many minutes (default: 60)

        Returns:
            MetricsSummary with aggregated statistics
        """
        with self._get_connection() as conn:
            # Get aggregate statistics
            row = conn.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                    SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failed,
                    AVG(response_time_ms) as avg_time,
                    MIN(response_time_ms) as min_time,
                    MAX(response_time_ms) as max_time
                FROM query_metrics
                WHERE timestamp >= datetime('now', ?)
            """, (f'-{since_minutes} minutes',)).fetchone()

            total = row['total'] or 0
            successful = row['successful'] or 0
            failed = row['failed'] or 0
            avg_time = row['avg_time'] or 0.0
            min_time = row['min_time'] or 0.0
            max_time = row['max_time'] or 0.0

            error_rate = failed / total if total > 0 else 0.0
            queries_per_minute = total / since_minutes if since_minutes > 0 else 0.0

            return MetricsSummary(
                total_queries=total,
                successful_queries=successful,
                failed_queries=failed,
                avg_response_time_ms=avg_time,
                min_response_time_ms=min_time,
                max_response_time_ms=max_time,
                error_rate=error_rate,
                queries_per_minute=queries_per_minute,
                period_minutes=since_minutes
            )

    def cleanup_old_metrics(self, retention_days: int = 7) -> int:
        """Delete metrics older than the retention period.

        Args:
            retention_days: Delete metrics older than this many days

        Returns:
            Number of deleted records
        """
        with self._get_connection() as conn:
            cursor = conn.execute("""
                DELETE FROM query_metrics
                WHERE timestamp < datetime('now', ?)
            """, (f'-{retention_days} days',))

            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old query metrics")
            return deleted


# Global status manager instance (cached per path)
_mcp_status_manager: Optional[MCPStatusManager] = None
_mcp_status_manager_path: Optional[Path] = None


def get_mcp_status_manager() -> MCPStatusManager:
    """Get or create the MCP status manager for the current project.

    The manager is cached, but recreated if the project path changes.
    """
    global _mcp_status_manager, _mcp_status_manager_path

    current_path = get_mcp_status_db_path()

    # Recreate if path changed (project switched)
    if _mcp_status_manager is None or _mcp_status_manager_path != current_path:
        _mcp_status_manager = MCPStatusManager(current_path)
        _mcp_status_manager_path = current_path

    return _mcp_status_manager
