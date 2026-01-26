from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from ..config import get_logger
from .status import get_mcp_status_manager

logger = get_logger(__name__)

async def health_check(request: Request) -> JSONResponse:
    try:
        manager = get_mcp_status_manager()
        server_info = manager.get_server_info()
        if server_info and server_info.status == "running":
            return JSONResponse({"status": "healthy", "pid": server_info.pid, "uptime_seconds": round(server_info.uptime_seconds, 1)}, status_code=200)
        return JSONResponse({"status": "unhealthy", "message": "Server not running"}, status_code=503)
    except Exception as e:
        return JSONResponse({"status": "unhealthy", "error": str(e)}, status_code=503)

async def metrics_endpoint(request: Request) -> JSONResponse:
    try:
        since_minutes = max(1, min(int(request.query_params.get("since", 60)), 10080))
        manager = get_mcp_status_manager()
        server_info = manager.get_server_info()
        server_data = {"pid": server_info.pid, "host": server_info.host, "port": server_info.port,
            "status": server_info.status, "uptime_seconds": round(server_info.uptime_seconds, 1)} if server_info else None
        summary = manager.get_metrics_summary(since_minutes=since_minutes)
        return JSONResponse({"server": server_data, "metrics": {
            "period_minutes": summary.period_minutes, "total_queries": summary.total_queries,
            "successful_queries": summary.successful_queries, "failed_queries": summary.failed_queries,
            "avg_response_time_ms": round(summary.avg_response_time_ms, 2),
            "min_response_time_ms": round(summary.min_response_time_ms, 2),
            "max_response_time_ms": round(summary.max_response_time_ms, 2),
            "error_rate": round(summary.error_rate, 4), "queries_per_minute": round(summary.queries_per_minute, 2)}}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

def register_health_routes(app):
    for route in [Route("/health", health_check, methods=["GET"]), Route("/metrics", metrics_endpoint, methods=["GET"])]:
        app.routes.append(route)
