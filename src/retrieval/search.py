from ..config import settings, get_logger
from ..utils import embed
from ..storage.chroma_client import search_vectors
from ..models import Chunk, SearchResult

logger = get_logger(__name__)


def search(
    query: str,
    top_k: int | None = None,
    filter_expr: str | None = None,
) -> list[SearchResult]:
    """Search for relevant chunks using vector similarity.

    Args:
        query: Search query text
        top_k: Number of results to return (defaults to config setting)
        filter_expr: Optional metadata filter

    Returns:
        List of search results sorted by relevance
    """
    top_k = top_k or settings.default_top_k

    query_vector = embed(query)
    raw_results = search_vectors(query_vector, top_k, filter_expr=filter_expr)

    if not raw_results:
        logger.info("No results found")
        return []

    results = [
        SearchResult(
            chunk=Chunk(
                id=hit["id"],
                text=hit.get("text", ""),
                doc_id=hit.get("doc_id", ""),
                page_num=hit.get("page_num"),
                metadata=hit.get("metadata", {}),
            ),
            score=float(hit.get("score", 0.0)),
            distance=float(hit.get("distance", 0.0)),
        )
        for hit in raw_results
    ]

    logger.info(f"Found {len(results)} results for query")
    return results
