"""
MCP server monitoring utilities.

Provides decorators and utilities for tracking query metrics.
"""
import functools
import time
from typing import Callable, Any

from ..config import settings, get_logger
from .status import get_mcp_status_manager

logger = get_logger(__name__)


def track_query(func: Callable) -> Callable:
    """Decorator to track query metrics for MCP tools.

    Automatically logs:
    - Query text (first positional argument)
    - top_k (from keyword arguments or second positional)
    - Response time in milliseconds
    - Result count (from response.context if QueryResult)
    - Errors if any exception occurs

    Works with both sync and async functions.

    Usage:
        @track_query
        async def query_rag(query_text: str, top_k: int = 5) -> QueryResult:
            ...
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs) -> Any:
        # Check if metrics are enabled
        if not getattr(settings, 'mcp_metrics_enabled', True):
            return await func(*args, **kwargs)

        # Extract query parameters
        query_text = args[0] if args else kwargs.get('query_text', '')
        top_k = args[1] if len(args) > 1 else kwargs.get('top_k', 5)

        start_time = time.perf_counter()
        error_msg = None
        result_count = 0

        try:
            result = await func(*args, **kwargs)

            # Extract result count from QueryResult
            if hasattr(result, 'context'):
                result_count = len(result.context) if result.context else 0

            return result

        except Exception as e:
            error_msg = str(e)
            raise

        finally:
            # Calculate response time
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Log the query metric
            try:
                manager = get_mcp_status_manager()
                manager.log_query(
                    query_text=str(query_text)[:500],  # Truncate long queries
                    top_k=int(top_k),
                    response_time_ms=elapsed_ms,
                    result_count=result_count,
                    error=error_msg
                )
            except Exception as log_error:
                # Don't let logging failures affect the main function
                logger.warning(f"Failed to log query metric: {log_error}")

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs) -> Any:
        # Check if metrics are enabled
        if not getattr(settings, 'mcp_metrics_enabled', True):
            return func(*args, **kwargs)

        # Extract query parameters
        query_text = args[0] if args else kwargs.get('query_text', '')
        top_k = args[1] if len(args) > 1 else kwargs.get('top_k', 5)

        start_time = time.perf_counter()
        error_msg = None
        result_count = 0

        try:
            result = func(*args, **kwargs)

            # Extract result count from QueryResult
            if hasattr(result, 'context'):
                result_count = len(result.context) if result.context else 0

            return result

        except Exception as e:
            error_msg = str(e)
            raise

        finally:
            # Calculate response time
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Log the query metric
            try:
                manager = get_mcp_status_manager()
                manager.log_query(
                    query_text=str(query_text)[:500],  # Truncate long queries
                    top_k=int(top_k),
                    response_time_ms=elapsed_ms,
                    result_count=result_count,
                    error=error_msg
                )
            except Exception as log_error:
                # Don't let logging failures affect the main function
                logger.warning(f"Failed to log query metric: {log_error}")

    # Return appropriate wrapper based on whether function is async
    import asyncio
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


class QueryTimer:
    """Context manager for timing query execution.

    Usage:
        with QueryTimer() as timer:
            # Execute query
            result = await query_rag(...)

        print(f"Query took {timer.elapsed_ms} ms")
    """

    def __init__(self):
        self.start_time: float = 0
        self.end_time: float = 0
        self.elapsed_ms: float = 0

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_ms = (self.end_time - self.start_time) * 1000
        return False  # Don't suppress exceptions
