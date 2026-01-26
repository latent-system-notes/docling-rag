from .config import settings
from .retrieval.search import search
from .models import QueryResult

def query(query_text: str, top_k: int | None = None) -> QueryResult:
    top_k = top_k or settings.default_top_k
    results = search(query_text, top_k=top_k)
    return QueryResult(query=query_text, context=results)
