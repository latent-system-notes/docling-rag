from fastmcp import FastMCP

from ..config import settings
from ..query import query
from ..storage.chroma_client import get_stats
from ..models import QueryResult

mcp = FastMCP(settings.mcp_server_name)


@mcp.tool()
async def query_rag(
    query_text: str,
    top_k: int = 5,
) -> QueryResult:
    """Query RAG system and return relevant context chunks for LLM reasoning"""
    return query(query_text, top_k)


@mcp.tool()
async def get_statistics() -> dict:
    """Get RAG system statistics"""
    return get_stats()


def run_server():
    mcp.run()


if __name__ == "__main__":
    run_server()
