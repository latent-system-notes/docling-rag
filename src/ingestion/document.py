import copy
import hashlib
import io
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

def _async_convert(base_url: str, options: dict, files: dict, headers: dict | None = None,
                    timeout: int = 600, poll_interval: int = 5) -> dict:
    """Submit async conversion, poll until done, return the result document dict."""
    merged = {"Host": "docling.s10.mil.dir"}
    if headers:
        merged.update(headers)

    # Step 1: Submit
    resp = httpx.post(f"{base_url}/v1/convert/file/async", headers=merged, files=files,
                      data=options, verify=False, timeout=60)
    resp.raise_for_status()
    task_id = resp.json()["task_id"]
    logger.info(f"Async task submitted: {task_id}")

    # Step 2: Poll
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        poll_resp = httpx.get(f"{base_url}/v1/status/poll/{task_id}",
                              params={"wait": poll_interval}, headers=merged,
                              timeout=poll_interval + 10, verify=False)
        poll_resp.raise_for_status()
        status = poll_resp.json()["task_status"]
        if status == "success":
            break
        elif status == "failure":
            raise DocumentLoadError(f"Async conversion failed (task {task_id})")
    else:
        raise DocumentLoadError(f"Async conversion timed out after {timeout}s (task {task_id})")

    # Step 3: Fetch result
    result_resp = httpx.get(f"{base_url}/v1/result/{task_id}", headers=merged,
                            timeout=60, verify=False)
    result_resp.raise_for_status()
    return result_resp.json()["document"]


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
    "Analyze this image thoroughly. Describe all visible elements including: "
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
    mime = _get_mime(source_path)
    file_bytes = source_path.read_bytes()

    def _make_files():
        """Create a fresh file-like object for each HTTP request."""
        return {"files": (source_path.name, io.BytesIO(file_bytes), mime)}

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

    logger.info(f"Converting: {source_path.absolute()}")

    if Path(source).suffix.lower() == ".pdf":
        logger.info("PDF detected, running OCR probe on first 2 pages...")
        probe_opts = copy.deepcopy(options)
        probe_opts.update({
            "page_range": [1, 2],
            "to_formats": ["md"],
            "image_export_mode": "placeholder",
        })

        def _get_md(do_ocr: bool) -> str:
            payload = copy.deepcopy(probe_opts)
            payload["do_ocr"] = do_ocr
            payload["force_ocr"] = do_ocr
            payload["do_picture_description"] = do_ocr
            payload["do_picture_classification"] = do_ocr
            doc = _async_convert(base_url, payload, _make_files(), headers, timeout=120)
            return tidy(doc["md_content"])

        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as pool:
                fut_without = pool.submit(_get_md, False)
                fut_with = pool.submit(_get_md, True)
                without_ocr = fut_without.result()
                with_ocr = fut_with.result()

            logger.info(f"OCR probe: without={len(without_ocr)} chars, with={len(with_ocr)} chars")

            if len(without_ocr) == 0:
                options["force_ocr"] = True
            elif (len(with_ocr) - len(without_ocr)) / len(without_ocr) > 0.5:
                options["force_ocr"] = True
        except Exception as e:
            logger.warning(f"OCR probe failed, defaulting to force_ocr=False: {e}")

    logger.info(f"Using force_ocr={options['force_ocr']}")
    content = _async_convert(base_url, options, _make_files(), headers, timeout=timeout)
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
