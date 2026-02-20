from ..config import DEFAULT_TOP_K, COLLECTION_NAME
from ..utils import embed
from ..storage.chroma_client import search_vectors, get_chroma_client
from ..storage.bm25_index import get_bm25_index
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


def search(query: str, top_k: int = DEFAULT_TOP_K) -> list[SearchResult]:
    bm25 = get_bm25_index()
    if bm25.num_docs == 0:
        bm25._load_statistics()

    if bm25.num_docs > 0:
        bm25_results = bm25.search(query, top_k=top_k * 3)
        vector_results = search_vectors(embed(query), top_k * 3)
        rrf_scores = _reciprocal_rank_fusion([bm25_results, [(h["id"], h["score"]) for h in vector_results]])
        top_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        collection = get_chroma_client().get_collection(COLLECTION_NAME)
        results = []
        for doc_id, rrf_score in top_ids:
            data = collection.get(ids=[doc_id])
            if data['ids']:
                results.append(_make_result(doc_id, data['documents'][0], data['metadatas'][0], rrf_score))
        return results

    raw = search_vectors(embed(query), top_k)
    return [_make_result(h["id"], h.get("text", ""), h, h.get("score", 0.0)) for h in raw]
