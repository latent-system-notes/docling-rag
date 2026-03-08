from .config import DEFAULT_TOP_K
from .retrieval.search import search
from .models import QueryResult

def query(query_text: str, top_k: int = DEFAULT_TOP_K, groups: list[str] | None = None) -> QueryResult:
    return QueryResult(query=query_text, context=search(query_text, top_k, groups=groups))
