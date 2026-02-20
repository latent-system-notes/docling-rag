import hashlib
from datetime import datetime
from pathlib import Path

import httpx
from docling_core.types.doc import DoclingDocument

from ..config import config, get_logger
from ..models import DocumentLoadError, DocumentMetadata
from ..utils import detect_language

logger = get_logger(__name__)


def _get_page_count(source: str | Path) -> int | None:
    if Path(source).suffix.lower() != ".pdf":
        return None
    try:
        import pypdf
        with open(source, "rb") as f:
            return len(pypdf.PdfReader(f).pages)
    except Exception:
        return None


def _convert_via_api(source: str | Path) -> DoclingDocument:
    url = config("DOCLING_SERVE_URL").rstrip("/") + "/v1/convert/file"
    api_key = config("DOCLING_SERVE_API_KEY")
    headers = {"X-API-Key": api_key} if api_key else {}
    source_path = Path(source)
    with open(source_path, "rb") as f:
        resp = httpx.post(url, files=[("files", (source_path.name, f, "application/octet-stream"))],
                          data={"to_formats": "json"}, headers=headers, timeout=config("DOCLING_SERVE_TIMEOUT"))
    try:
        resp.raise_for_status()
        return DoclingDocument.model_validate(resp.json()["document"]["json_content"])
    finally:
        resp.close()


def load_document(source: str | Path) -> tuple[DoclingDocument, int | None]:
    try:
        page_count = _get_page_count(source)
        logger.info(f"Converting via docling-serve: {source}")
        return _convert_via_api(source), page_count
    except DocumentLoadError:
        raise
    except httpx.HTTPStatusError as e:
        raise DocumentLoadError(f"docling-serve returned {e.response.status_code} for {source}: {e.response.text}") from e
    except httpx.ConnectError as e:
        raise DocumentLoadError(f"Cannot reach docling-serve at {config('DOCLING_SERVE_URL')}: {e}") from e
    except Exception as e:
        raise DocumentLoadError(f"Failed to load document: {e}") from e


def extract_metadata(doc: DoclingDocument, file_path: str | Path, num_chunks: int,
                     num_pages: int | None = None) -> DocumentMetadata:
    file_path = Path(file_path)
    doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
    doc_type = file_path.suffix.lstrip(".").lower() or "unknown"

    language = "unknown"
    if hasattr(doc, "properties") and doc.properties:
        language = doc.properties.get("language", "unknown")
    if language == "unknown":
        language = detect_language(doc.export_to_markdown()[:500]) or "en"
    if language == "unknown":
        language = "en"

    return DocumentMetadata(doc_id=doc_id, doc_type=doc_type, language=language, file_path=str(file_path),
                            num_chunks=num_chunks, num_pages=num_pages, ingested_at=datetime.now())
