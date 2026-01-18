from functools import cache

from ..config import settings, get_logger
from ..models import GenerationError, SearchResult

logger = get_logger(__name__)


# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context.
Use the context to provide accurate, concise answers. If the answer is not in the context, say so."""


def format_context(results: list[SearchResult]) -> str:
    """Format search results into a context string for the prompt.

    Args:
        results: List of search results to format

    Returns:
        Formatted context string with document references
    """
    chunks = []
    for idx, result in enumerate(results, 1):
        doc_ref = f"[Doc {result.chunk.doc_id[:8]}"
        if result.chunk.page_num:
            doc_ref += f", Page {result.chunk.page_num}"
        doc_ref += "]"

        chunks.append(f"[{idx}] {doc_ref}\n{result.chunk.text}")

    return "\n\n".join(chunks)


def build_rag_prompt(query: str, context: list[SearchResult]) -> str:
    """Build a RAG prompt with context and query.

    Args:
        query: User query
        context: Search results to include as context

    Returns:
        Formatted prompt string
    """
    context_text = format_context(context)

    prompt = f"""Context:
{context_text}

Question: {query}

Answer:"""

    return prompt


# ============================================================================
# Model Loading and Generation
# ============================================================================


@cache
def get_granite_pipeline():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    from ..utils import get_model_paths

    paths = get_model_paths()
    tok_path = paths["granite_tokenizer"]
    model_path = paths["granite_model"]

    if not tok_path.exists() or not model_path.exists():
        raise GenerationError(
            f"Granite model not found at {model_path}. "
            "Run 'rag models --download' first."
        )

    logger.info(f"Loading Granite model from {model_path}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(tok_path), local_files_only=True)

        model_kwargs = {
            "device_map": "auto" if settings.device != "cpu" else None,
            "local_files_only": True,
        }

        if settings.granite_quantization == "4bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
        elif settings.granite_quantization == "8bit":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForCausalLM.from_pretrained(str(model_path), **model_kwargs)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            torch_dtype=torch.bfloat16 if settings.device != "cpu" else torch.float32,
        )

        logger.info("Granite model loaded successfully")
        return pipe

    except Exception as e:
        raise GenerationError(f"Failed to load Granite model: {e}") from e


def generate_answer(query: str, context: list[SearchResult]) -> str:
    try:
        pipe = get_granite_pipeline()
        prompt = build_rag_prompt(query, context)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        response = pipe(messages)
        answer = response[0]["generated_text"][-1]["content"]

        return answer.strip()

    except Exception as e:
        raise GenerationError(f"Failed to generate answer: {e}") from e
