import json
import os
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

from ..config import get_logger

logger = get_logger(__name__)

class IngestionLockError(Exception):
    pass

class IngestionLock:
    def __init__(self, project_dir: Path):
        self.lock_file = project_dir / ".ingestion.lock"
        self.locked = False
        self._pid = os.getpid()

    def acquire(self, timeout: int = 5) -> bool:
        if self.lock_file.exists():
            lock_info = self._read_lock()
            if lock_info:
                if self._is_process_alive(lock_info.get('pid')):
                    return False
                else:
                    self._remove_stale_lock()
            else:
                self._remove_stale_lock()
        try:
            self._write_lock()
            self.locked = True
            return True
        except Exception:
            return False

    def release(self):
        if self.locked and self.lock_file.exists():
            try:
                lock_info = self._read_lock()
                if lock_info and lock_info.get('pid') == self._pid:
                    self.lock_file.unlink()
            except Exception:
                pass
            finally:
                self.locked = False

    def get_lock_info(self) -> Optional[Dict[str, Any]]:
        return self._read_lock() if self.lock_file.exists() else None

    @contextmanager
    def hold(self):
        if not self.acquire():
            lock_info = self.get_lock_info()
            if lock_info:
                raise IngestionLockError(f"Another ingestion is already running (PID {lock_info.get('pid')}, started {lock_info.get('started_at', 'unknown')})")
            else:
                raise IngestionLockError("Another ingestion process is running.")
        try:
            yield
        finally:
            self.release()

    def _write_lock(self):
        self.lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_data = {'pid': self._pid, 'started_at': datetime.now().isoformat(),
                     'hostname': os.environ.get('COMPUTERNAME', os.environ.get('HOSTNAME', 'unknown'))}
        with open(self.lock_file, 'w') as f:
            json.dump(lock_data, f)

    def _read_lock(self) -> Optional[Dict[str, Any]]:
        try:
            with open(self.lock_file, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None

    def _remove_stale_lock(self):
        try:
            if self.lock_file.exists():
                self.lock_file.unlink()
        except Exception:
            pass

    def _is_process_alive(self, pid: Optional[int]) -> bool:
        if pid is None:
            return False
        try:
            if os.name == 'nt':
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, pid)
                if handle:
                    kernel32.CloseHandle(handle)
                    return True
                return False
            else:
                os.kill(pid, 0)
                return True
        except (OSError, PermissionError):
            return False
        except Exception:
            return False
