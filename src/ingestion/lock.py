"""Project-level locking for ingestion to prevent concurrent sessions."""
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from ..config import get_logger

logger = get_logger(__name__)


class IngestionLockError(Exception):
    """Raised when attempting to start ingestion while another session is running."""
    pass


class IngestionLock:
    """File-based lock to prevent concurrent ingestion on same project.

    Uses a lock file with PID and timestamp to:
    - Prevent multiple ingestion sessions from running simultaneously
    - Detect and clean up stale locks from crashed processes
    - Provide information about the blocking session
    """

    def __init__(self, project_dir: Path):
        """Initialize lock for a project directory.

        Args:
            project_dir: The project's data directory (parent of chroma_persist_dir)
        """
        self.lock_file = project_dir / ".ingestion.lock"
        self.locked = False
        self._pid = os.getpid()

    def acquire(self, timeout: int = 5) -> bool:
        """Try to acquire the lock.

        Args:
            timeout: Not used currently, reserved for future use

        Returns:
            True if lock acquired, False if another process holds the lock
        """
        # Check for existing lock
        if self.lock_file.exists():
            lock_info = self._read_lock()
            if lock_info:
                # Check if the process holding the lock is still alive
                if self._is_process_alive(lock_info.get('pid')):
                    logger.debug(f"Lock held by PID {lock_info.get('pid')}")
                    return False  # Active lock exists
                else:
                    # Stale lock - process died without cleanup
                    logger.warning(f"Removing stale lock from dead process {lock_info.get('pid')}")
                    self._remove_stale_lock()
            else:
                # Corrupted lock file, remove it
                logger.warning("Removing corrupted lock file")
                self._remove_stale_lock()

        # Create lock file with our PID
        try:
            self._write_lock()
            self.locked = True
            logger.info(f"Acquired ingestion lock (PID {self._pid})")
            return True
        except Exception as e:
            logger.error(f"Failed to create lock file: {e}")
            return False

    def release(self):
        """Release the lock."""
        if self.locked and self.lock_file.exists():
            try:
                # Verify we own the lock before releasing
                lock_info = self._read_lock()
                if lock_info and lock_info.get('pid') == self._pid:
                    self.lock_file.unlink()
                    logger.info("Released ingestion lock")
                else:
                    logger.warning("Lock file exists but owned by another process")
            except Exception as e:
                logger.error(f"Failed to release lock: {e}")
            finally:
                self.locked = False

    def get_lock_info(self) -> Optional[Dict[str, Any]]:
        """Get information about the current lock holder.

        Returns:
            Dict with lock info or None if no lock exists
        """
        if self.lock_file.exists():
            return self._read_lock()
        return None

    @contextmanager
    def hold(self):
        """Context manager for holding the lock.

        Raises:
            IngestionLockError: If lock cannot be acquired
        """
        if not self.acquire():
            lock_info = self.get_lock_info()
            if lock_info:
                raise IngestionLockError(
                    f"Another ingestion is already running (PID {lock_info.get('pid')}, "
                    f"started {lock_info.get('started_at', 'unknown')})"
                )
            else:
                raise IngestionLockError("Another ingestion process is running.")
        try:
            yield
        finally:
            self.release()

    def _write_lock(self):
        """Write lock file with current process info."""
        # Ensure parent directory exists
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)

        lock_data = {
            'pid': self._pid,
            'started_at': datetime.now().isoformat(),
            'hostname': os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'unknown'))
        }

        with open(self.lock_file, 'w') as f:
            json.dump(lock_data, f)

    def _read_lock(self) -> Optional[Dict[str, Any]]:
        """Read lock file contents.

        Returns:
            Dict with lock data or None if file is corrupted/empty
        """
        try:
            with open(self.lock_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.debug(f"Failed to read lock file: {e}")
            return None

    def _remove_stale_lock(self):
        """Remove a stale lock file."""
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception as e:
            logger.error(f"Failed to remove stale lock: {e}")

    def _is_process_alive(self, pid: Optional[int]) -> bool:
        """Check if a process is still running.

        Args:
            pid: Process ID to check

        Returns:
            True if process exists, False otherwise
        """
        if pid is None:
            return False

        try:
            # Windows-compatible process check
            if os.name == 'nt':
                # On Windows, use ctypes to check process
                import ctypes
                kernel32 = ctypes.windll.kernel32
                PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            else:
                # On Unix, send signal 0 to check if process exists
                os.kill(pid, 0)
                return True
        except (OSError, PermissionError):
            return False
        except Exception as e:
            logger.debug(f"Error checking process {pid}: {e}")
            return False
