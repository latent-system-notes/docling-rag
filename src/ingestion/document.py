import copy
import hashlib
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import httpx

# Reduce noisy httpx INFO logs (e.g., during status polling)
logging.getLogger("httpx").setLevel(logging.WARNING)
from docling_core.types.doc import DoclingDocument


from ..config import config, get_logger
from ..models import DocumentLoadError, DocumentMetadata
from ..utils import detect_language

logger = get_logger(__name__)

DOCLING_MIME_TYPES = {
    # DOCUMENTS
    ".pdf": "application/pdf",
    ".doc": "application/msword",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".csv": "text/csv",

    # HTML
    ".html": "text/html",
    ".htm": "text/html",
    ".md": "text/markdown",
    ".txt": "text/plain",

    # Image
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",

    # Audio
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",

    # JSON
    ".json": "application/json",
    ".xml": "application/xml",
}


def _get_mime(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    return DOCLING_MIME_TYPES.get(ext, "application/octet-stream")

def _post_convert(url: str, options: dict, files: dict, extra_headers: dict | None = None,
                   timeout: int = 60) -> httpx.Response:
    headers = {"Host": "docling.s10.mil.dir"}
    if extra_headers:
        headers.update(extra_headers)
    resp = httpx.post(url, headers=headers, files=files, data=options, verify=False, timeout=timeout)
    resp.raise_for_status()
    return resp

def _get_page_count(source: str | Path) -> int | None:
    if Path(source).suffix.lower() != ".pdf":
        return None
    try:
        import pypdf
        with open(source, "rb") as f:
            return len(pypdf.PdfReader(f).pages)
    except Exception:
        return None

CUSTOM_PROMPT = (
    "Analyze this image through. Describe all visible elements including:"
    "text content, charts, diagrams, tables, graphs, logos, and any visual data. "
    "If there are numbers or labels, include them. Be detailed and precise."
)

def tidy(s: str) -> str:
    return re.sub(r'\s+', ' ', re.sub(r'[\r\n]+|<!-- image -->', ' ', s)).strip()

def _convert_via_api(source: str | Path) -> DoclingDocument:
    base_url = config("DOCLING_SERVE_URL").rstrip("/")
    api_key = config("DOCLING_SERVE_API_KEY")
    headers = {"X-API-Key": api_key} if api_key else {}
    source_path = Path(source)
    timeout = config("DOCLING_SERVE_TIMEOUT") or 600
    poll_interval = 5

    options = {
            "do_picture_description": True,
            "do_picture_classification": True,
            "pdf_backend": "dlparse_v4",
            "do_ocr": True,
            "force_ocr": False,
            "bitmap_area_treshold": 0.75,
            "to_formats": ["json"],
            "image_export_mode": "embedded",
        "picture_description_local": json.dumps({
                "repo_id": "HuggingFaceTB/SmolVLM-256M-Instruct",
                "prompt": CUSTOM_PROMPT
        }),
            "ocr_engine": "auto"
    }

    files = {
        "files": (source_path.name, open(source_path, "rb"), _get_mime(source_path)),
    }

    if Path(source).suffix.lower() == ".pdf":
        # Prepare a minimal payload for the first three pages (markdown placeholder)
        pdf_range = copy.deepcopy(options)
        pdf_range.update(
            {
                "page_range": [1, 2],
                "to_formats": ["md"],
                "image_export_mode": "placeholder",
            }
        )

        def _get_md(do_ocr: bool) -> str:
            payload = copy.deepcopy(pdf_range)
            payload["do_ocr"] = do_ocr
            payload["force_ocr"] = do_ocr
            payload["do_picture_description"] = do_ocr
            payload["do_picture_classification"] = do_ocr
            resp = _post_convert(f"{base_url}/v1/convert/file", payload, files)
            return tidy(resp.json()["document"]["md_content"])

        without_ocr = _get_md(False)
        with_ocr = _get_md(True)

        # If OCR adds >50% content, enable full OCR for the async conversion
        if len(without_ocr) == 0:
            options["force_ocr"] = True
        elif (len(with_ocr) - len(without_ocr)) / len(without_ocr) > 0.5:
            options["force_ocr"] = True

    logger.info(f"Using full OCR {options["force_ocr"]}")
    resp = httpx.post(
        f"{base_url}/v1/convert/file/async",
        headers={"Host": "docling.s10.mil.dir"},
        data=options,
        files=files,
        verify=False,
        timeout=60
    )

    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    logger.info(f"Async task submitted: {task_id}")

    # Step 2: Poll for completion
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        poll_resp = httpx.get(
            f"{base_url}/v1/status/poll/{task_id}",
            params={"wait": poll_interval},
            headers=headers,
            timeout=poll_interval + 10,
        )
        poll_resp.raise_for_status()
        status = poll_resp.json()["task_status"]
        if status == "success":
            break
        elif status == "failure":
            raise DocumentLoadError(f"Async conversion failed for {source}")
    else:
        # Ensure newline before final timeout error
        raise DocumentLoadError(f"Async conversion timed out after {timeout}s for {source}")

    # Step 3: Fetch result
    result_resp = httpx.get(
        f"{base_url}/v1/result/{task_id}",
        headers=headers,
        timeout=60,
    )

    result_resp.raise_for_status()
    content = result_resp.json()["document"]
    return DoclingDocument.model_validate(content["json_content"])


def load_document(source: str | Path) -> tuple[DoclingDocument, int | None]:
    try:
        page_count = _get_page_count(source)
        logger.info(f"Converting via docling-serve")
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
