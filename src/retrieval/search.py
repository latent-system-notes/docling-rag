from ..config import settings, get_logger
from ..utils import embed
from ..storage.chroma_client import search_vectors, get_chroma_client
from ..storage.bm25_index import get_bm25_index
from ..models import Chunk, SearchResult

logger = get_logger(__name__)


def _reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60
) -> dict[str, float]:
    """Combine multiple rankings using Reciprocal Rank Fusion (RRF).

    RRF is an effective way to combine rankings from different retrieval methods.
    Formula: RRF_score(doc) = sum(1 / (k + rank))

    Args:
        rankings: List of rankings, each ranking is a list of (doc_id, score) tuples
        k: Constant parameter (typically 60)

    Returns:
        Dictionary mapping doc_id to RRF score
    """
    rrf_scores = {}

    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            rrf_scores[doc_id] += 1.0 / (k + rank)

    return rrf_scores


def search(
    query: str,
    top_k: int | None = None,
    filter_expr: str | None = None,
    use_hybrid: bool = True,
) -> list[SearchResult]:
    """Search for relevant chunks using hybrid search (BM25 + vector similarity).

    Combines keyword-based BM25 search with semantic vector search using
    Reciprocal Rank Fusion (RRF) for optimal results.

    Args:
        query: Search query text
        top_k: Number of results to return (defaults to config setting)
        filter_expr: Optional metadata filter
        use_hybrid: Use hybrid search (BM25 + vector). If False, use vector only.

    Returns:
        List of search results sorted by relevance (RRF score)
    """
    top_k = top_k or settings.default_top_k

    # Load BM25 index
    bm25_index = get_bm25_index()
    if not bm25_index.index:
        logger.info("BM25 index not loaded, attempting to load...")
        if not bm25_index.load():
            logger.warning("BM25 index not available, falling back to vector-only search")
            use_hybrid = False

    if use_hybrid and bm25_index.index:
        # Hybrid search: BM25 + Vector
        logger.info("Using hybrid search (BM25 + vector)")

        # 1. BM25 search - get more candidates for fusion
        bm25_results = bm25_index.search(query, top_k=top_k * 3)

        # 2. Vector search
        query_vector = embed(query)
        vector_results = search_vectors(query_vector, top_k * 3, filter_expr=filter_expr)

        # 3. Reciprocal Rank Fusion
        bm25_ranking = [(doc_id, score) for doc_id, score in bm25_results]
        vector_ranking = [(hit["id"], hit["score"]) for hit in vector_results]

        rrf_scores = _reciprocal_rank_fusion([bm25_ranking, vector_ranking])

        # 4. Get top-k by RRF score
        top_doc_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # 5. Fetch full chunk data from ChromaDB
        client = get_chroma_client()
        collection = client.get_collection(settings.chroma_collection_name)

        results = []
        for doc_id, rrf_score in top_doc_ids:
            # Get chunk data
            chunk_data = collection.get(ids=[doc_id])
            if not chunk_data['ids']:
                continue

            metadata = chunk_data['metadatas'][0]
            text = chunk_data['documents'][0]

            results.append(
                SearchResult(
                    chunk=Chunk(
                        id=doc_id,
                        text=text,
                        doc_id=metadata.get("doc_id", ""),
                        page_num=metadata.get("page_num"),
                        metadata={
                            "file_path": metadata.get("file_path", ""),
                            "doc_type": metadata.get("doc_type", ""),
                            "language": metadata.get("language", ""),
                            "ingested_at": metadata.get("ingested_at", ""),
                            **metadata.get("metadata", {})
                        },
                    ),
                    score=float(rrf_score),
                    distance=1.0 - float(rrf_score),  # Approximate distance from RRF score
                )
            )

        logger.info(f"Found {len(results)} results using hybrid search")
        return results

    else:
        # Vector-only search (fallback)
        logger.info("Using vector-only search")

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
                    metadata={
                        "file_path": hit.get("file_path", ""),
                        "doc_type": hit.get("doc_type", ""),
                        "language": hit.get("language", ""),
                        "ingested_at": hit.get("ingested_at", ""),
                        # Include any additional nested metadata
                        **hit.get("metadata", {})
                    },
                ),
                score=float(hit.get("score", 0.0)),
                distance=float(hit.get("distance", 0.0)),
            )
            for hit in raw_results
        ]

        logger.info(f"Found {len(results)} results for query")
        return results
