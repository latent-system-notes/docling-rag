"""High-level query orchestration for the RAG system.

This is the main entry point for asking questions. It coordinates:
1. Search: Find relevant chunks from the vector database
2. Return: Provide context for external LLM reasoning
"""
from .config import settings, get_logger
from .retrieval.search import search
from .models import QueryResult

logger = get_logger(__name__)


def query(
    query_text: str,
    top_k: int | None = None,
) -> QueryResult:
    """Ask a question and retrieve relevant context using RAG.

    The RAG (Retrieval-Augmented Generation) process:
    1. Convert query to vector embedding
    2. Find similar chunks in vector database
    3. Return chunks for external LLM reasoning (e.g., Claude)

    Args:
        query_text: The question to ask
        top_k: How many chunks to retrieve (default: 5)

    Returns:
        QueryResult with query and context chunks
    """
    top_k = top_k or settings.default_top_k

    # Search for relevant chunks
    results = search(query_text, top_k=top_k)

    # Return just the chunks for external LLM reasoning
    return QueryResult(query=query_text, context=results)
