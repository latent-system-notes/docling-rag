from ..config import DEFAULT_TOP_K
from ..utils import embed
from ..storage.postgres import search_vectors, search_fulltext, get_chunks_by_ids
from ..models import Chunk, SearchResult


def _reciprocal_rank_fusion(rankings: list[list[tuple[str, float]]], k: int = 60) -> dict[str, float]:
    rrf_scores = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return rrf_scores


def _make_result(doc_id: str, text: str, metadata: dict, score: float) -> SearchResult:
    return SearchResult(
        chunk=Chunk(id=doc_id, text=text, doc_id=metadata.get("doc_id", ""), page_num=metadata.get("page_num"),
            metadata={k: metadata.get(k, "") for k in ("file_path", "doc_type", "language", "ingested_at")}),
        score=float(score), distance=1.0 - float(score))


def search(query: str, top_k: int = DEFAULT_TOP_K, groups: list[str] | None = None) -> list[SearchResult]:
    # Compute embedding once — reused for both hybrid and vector-only paths
    query_embedding = embed(query)

    ft_results = search_fulltext(query, top_k=top_k * 3, groups=groups)

    if ft_results:
        vector_results = search_vectors(query_embedding, top_k * 3, groups=groups)
        rrf_scores = _reciprocal_rank_fusion([ft_results, [(h["id"], h["score"]) for h in vector_results]])
        top_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        batch_ids = [doc_id for doc_id, _ in top_ids]
        if not batch_ids:
            return []
        data = get_chunks_by_ids(batch_ids)
        id_to_idx = {cid: i for i, cid in enumerate(data['ids'])}
        results = []
        for doc_id, rrf_score in top_ids:
            if doc_id in id_to_idx:
                i = id_to_idx[doc_id]
                results.append(_make_result(doc_id, data['documents'][i], data['metadatas'][i], rrf_score))
        del data
        return results

    raw = search_vectors(query_embedding, top_k, groups=groups)
    return [_make_result(h["id"], h.get("text", ""), h, h.get("score", 0.0)) for h in raw]
