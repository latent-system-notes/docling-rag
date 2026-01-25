"""
HTTP health check and metrics endpoints for MCP server.

Provides /health and /metrics endpoints for monitoring.
"""
from starlette.requests import Request
from starlette.responses import JSONResponse

from ..config import get_logger
from .status import get_mcp_status_manager

logger = get_logger(__name__)


async def health_check(request: Request) -> JSONResponse:
    """Health check endpoint.

    GET /health

    Returns:
        200 OK with status info if server is healthy
        503 Service Unavailable if server has issues

    Response format:
        {
            "status": "healthy" | "unhealthy",
            "pid": 12345,
            "uptime_seconds": 9252.5
        }
    """
    try:
        manager = get_mcp_status_manager()
        server_info = manager.get_server_info()

        if server_info and server_info.status == "running":
            return JSONResponse(
                content={
                    "status": "healthy",
                    "pid": server_info.pid,
                    "uptime_seconds": round(server_info.uptime_seconds, 1)
                },
                status_code=200
            )
        else:
            return JSONResponse(
                content={
                    "status": "unhealthy",
                    "message": "Server not running"
                },
                status_code=503
            )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "error": str(e)
            },
            status_code=503
        )


async def metrics_endpoint(request: Request) -> JSONResponse:
    """Metrics endpoint with server and query statistics.

    GET /metrics

    Optional query parameters:
        - since: Time period in minutes (default: 60)

    Returns:
        200 OK with metrics data

    Response format:
        {
            "server": {
                "pid": 12345,
                "host": "0.0.0.0",
                "port": 8080,
                "status": "running",
                "uptime_seconds": 9252.5
            },
            "metrics": {
                "period_minutes": 60,
                "total_queries": 247,
                "successful_queries": 245,
                "failed_queries": 2,
                "avg_response_time_ms": 156.3,
                "min_response_time_ms": 42.1,
                "max_response_time_ms": 892.5,
                "error_rate": 0.008,
                "queries_per_minute": 4.12
            }
        }
    """
    try:
        # Get time period from query params
        since_minutes = int(request.query_params.get("since", 60))
        since_minutes = max(1, min(since_minutes, 10080))  # 1 min to 1 week

        manager = get_mcp_status_manager()

        # Get server info
        server_info = manager.get_server_info()
        server_data = None

        if server_info:
            server_data = {
                "pid": server_info.pid,
                "host": server_info.host,
                "port": server_info.port,
                "status": server_info.status,
                "uptime_seconds": round(server_info.uptime_seconds, 1)
            }

        # Get metrics summary
        summary = manager.get_metrics_summary(since_minutes=since_minutes)

        metrics_data = {
            "period_minutes": summary.period_minutes,
            "total_queries": summary.total_queries,
            "successful_queries": summary.successful_queries,
            "failed_queries": summary.failed_queries,
            "avg_response_time_ms": round(summary.avg_response_time_ms, 2),
            "min_response_time_ms": round(summary.min_response_time_ms, 2),
            "max_response_time_ms": round(summary.max_response_time_ms, 2),
            "error_rate": round(summary.error_rate, 4),
            "queries_per_minute": round(summary.queries_per_minute, 2)
        }

        return JSONResponse(
            content={
                "server": server_data,
                "metrics": metrics_data
            },
            status_code=200
        )

    except Exception as e:
        logger.error(f"Metrics endpoint failed: {e}")
        return JSONResponse(
            content={
                "error": str(e)
            },
            status_code=500
        )


def register_health_routes(app):
    """Register health and metrics routes with a Starlette/FastAPI app.

    Args:
        app: Starlette or FastAPI application instance
    """
    from starlette.routing import Route

    routes = [
        Route("/health", health_check, methods=["GET"]),
        Route("/metrics", metrics_endpoint, methods=["GET"]),
    ]

    # Add routes to the app
    for route in routes:
        app.routes.append(route)

    logger.info("Registered health endpoints: /health, /metrics")
