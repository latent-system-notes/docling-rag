"""High-level query orchestration for the RAG system.

This is the main entry point for asking questions. It coordinates:
1. Search: Find relevant chunks from the vector database
2. Generation: Use LLM to create an answer (optional)
3. Return: Provide context and/or answer based on mode
"""
from .config import settings, get_logger
from .generation.granite import generate_answer
from .retrieval.search import search
from .models import AnswerMode, QueryResult

logger = get_logger(__name__)


def query(
    query_text: str,
    top_k: int | None = None,
    mode: AnswerMode | None = None,
) -> QueryResult:
    """Ask a question and get an answer using RAG.

    The RAG (Retrieval-Augmented Generation) process:
    1. Convert query to vector embedding
    2. Find similar chunks in vector database
    3. (Optional) Feed chunks to LLM for answer generation

    Three answer modes:
    - "context_only": Just return relevant chunks (let external LLM like Claude reason)
    - "granite": Return only Granite's generated answer
    - "both": Return both chunks AND Granite's answer

    Args:
        query_text: The question to ask
        top_k: How many chunks to retrieve (default: 5)
        mode: Answer mode (default: from config)

    Returns:
        QueryResult with query, context chunks, and/or answer
    """
    mode = mode or settings.answer_mode
    top_k = top_k or settings.default_top_k

    # Step 1: Search for relevant chunks
    results = search(query_text, top_k=top_k)

    # Step 2: Generate answer based on mode
    match mode:
        case "context_only":
            # Return just the chunks for external LLM reasoning
            return QueryResult(query=query_text, context=results, answer=None, mode=mode)

        case "granite":
            # Return only Granite's answer (hide the chunks)
            answer = generate_answer(query_text, results) if results else None
            return QueryResult(query=query_text, context=[], answer=answer, mode=mode)

        case "both":
            # Return both chunks and Granite's answer
            answer = generate_answer(query_text, results) if results else None
            return QueryResult(query=query_text, context=results, answer=answer, mode=mode)

        case _:
            raise ValueError(f"Invalid answer mode: {mode}")
