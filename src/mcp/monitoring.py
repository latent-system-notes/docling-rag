import functools
import time
from typing import Callable, Any

from ..config import settings, get_logger
from .status import get_mcp_status_manager

logger = get_logger(__name__)

def track_query(func: Callable) -> Callable:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        if not settings.mcp_metrics_enabled:
            return await func(*args, **kwargs)

        query_text = args[0] if args else kwargs.get('query_text', '')
        top_k = args[1] if len(args) > 1 else kwargs.get('top_k', 5)
        start_time = time.perf_counter()
        error_msg = None
        result_count = 0

        try:
            result = await func(*args, **kwargs)
            if hasattr(result, 'context'):
                result_count = len(result.context) if result.context else 0
            return result
        except Exception as e:
            error_msg = str(e)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            try:
                get_mcp_status_manager().log_query(query_text=str(query_text)[:500], top_k=int(top_k),
                    response_time_ms=elapsed_ms, result_count=result_count, error=error_msg)
            except Exception:
                pass
    return wrapper
