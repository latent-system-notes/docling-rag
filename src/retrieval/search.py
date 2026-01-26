from ..config import settings, get_logger
from ..utils import embed
from ..storage.chroma_client import search_vectors, get_chroma_client
from ..storage.bm25_index import get_bm25_index
from ..models import Chunk, SearchResult

logger = get_logger(__name__)

def _reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], k: int = 60) -> dict[str, float]:
    rrf_scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return rrf_scores

def search(query: str, top_k: int | None = None, filter_expr: str | None = None, use_hybrid: bool = True) -> list[SearchResult]:
    top_k = top_k or settings.default_top_k
    bm25_index = get_bm25_index()
    bm25_ready = bm25_index.num_docs > 0

    if not bm25_ready:
        bm25_index.load()
        bm25_ready = bm25_index.num_docs > 0
        if not bm25_ready:
            use_hybrid = False

    if use_hybrid and bm25_ready:
        bm25_results = bm25_index.search(query, top_k=top_k * 3)
        query_vector = embed(query)
        vector_results = search_vectors(query_vector, top_k * 3, filter_expr=filter_expr)

        bm25_ranking = [(doc_id, score) for doc_id, score in bm25_results]
        vector_ranking = [(hit["id"], hit["score"]) for hit in vector_results]
        rrf_scores = _reciprocal_rank_fusion([bm25_ranking, vector_ranking])

        top_doc_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        client = get_chroma_client()
        collection = client.get_collection(settings.chroma_collection_name)

        results = []
        for doc_id, rrf_score in top_doc_ids:
            chunk_data = collection.get(ids=[doc_id])
            if not chunk_data['ids']:
                continue
            metadata = chunk_data['metadatas'][0]
            text = chunk_data['documents'][0]
            results.append(SearchResult(
                chunk=Chunk(id=doc_id, text=text, doc_id=metadata.get("doc_id", ""), page_num=metadata.get("page_num"),
                    metadata={"file_path": metadata.get("file_path", ""), "doc_type": metadata.get("doc_type", ""),
                              "language": metadata.get("language", ""), "ingested_at": metadata.get("ingested_at", "")}),
                score=float(rrf_score), distance=1.0 - float(rrf_score)))
        return results

    query_vector = embed(query)
    raw_results = search_vectors(query_vector, top_k, filter_expr=filter_expr)
    if not raw_results:
        return []

    return [SearchResult(
        chunk=Chunk(id=hit["id"], text=hit.get("text", ""), doc_id=hit.get("doc_id", ""), page_num=hit.get("page_num"),
            metadata={"file_path": hit.get("file_path", ""), "doc_type": hit.get("doc_type", ""),
                      "language": hit.get("language", ""), "ingested_at": hit.get("ingested_at", "")}),
        score=float(hit.get("score", 0.0)), distance=float(hit.get("distance", 0.0))) for hit in raw_results]
