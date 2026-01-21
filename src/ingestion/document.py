"""Document loading and metadata extraction.

This module handles the first steps of the RAG pipeline:
1. Load documents using Docling (supports PDF, DOCX, images, etc.)
2. Extract metadata (language, file type, document ID)
"""
import hashlib
from datetime import datetime
from pathlib import Path
from typing import BinaryIO

from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from ..config import settings, get_logger, enforce_logging_format
from ..models import DocumentLoadError, DocumentMetadata
from ..utils import detect_language

logger = get_logger(__name__)


# ============================================================================
# OCR Configuration
# ============================================================================


def _get_ocr_options():
    """Get OCR options based on configured engine.

    Returns appropriate OcrOptions subclass based on settings.ocr_engine.
    Each subclass has a 'kind' ClassVar that identifies the engine.

    Available engines:
    - "auto": Let Docling choose the best available engine (OcrAutoOptions)
    - "rapidocr": Force RapidOCR engine (RapidOcrOptions)
    - "easyocr": Force EasyOCR engine (EasyOcrOptions)
    - "tesseract": Force Tesseract CLI (TesseractCliOcrOptions)
    - "tesserocr": Force Tesseract library (TesseractOcrOptions)
    - "ocrmac": Force OCRMac engine - macOS only (OcrMacOptions)
    """
    engine = settings.ocr_engine.lower()
    langs = settings.ocr_languages.split("+") if settings.ocr_languages else []

    if engine == "auto":
        # Use OcrAutoOptions for automatic engine selection
        from docling.datamodel.pipeline_options import OcrAutoOptions
        return OcrAutoOptions(
            lang=langs if langs else ['en'],
            force_full_page_ocr=True,
        )
    elif engine == "rapidocr":
        from docling.datamodel.pipeline_options import RapidOcrOptions
        # RapidOCR uses full language names (english, arabic, chinese, etc.)
        lang_map = {'eng': 'english', 'ara': 'arabic', 'chi': 'chinese', 'jpn': 'japanese'}
        rapidocr_langs = [lang_map.get(lang, lang) for lang in langs]
        return RapidOcrOptions(
            lang=rapidocr_langs if rapidocr_langs else ['english'],
            force_full_page_ocr=True,
        )
    elif engine == "easyocr":
        from docling.datamodel.pipeline_options import EasyOcrOptions
        return EasyOcrOptions(
            lang=langs if langs else ['en'],
            force_full_page_ocr=True,
        )
    elif engine == "tesseract":
        from docling.datamodel.pipeline_options import TesseractCliOcrOptions
        return TesseractCliOcrOptions(
            lang=langs if langs else ['eng'],
            force_full_page_ocr=True,
        )
    elif engine == "tesserocr":
        from docling.datamodel.pipeline_options import TesseractOcrOptions
        return TesseractOcrOptions(
            lang=langs if langs else ['eng'],
            force_full_page_ocr=True,
        )
    elif engine == "ocrmac":
        from docling.datamodel.pipeline_options import OcrMacOptions
        return OcrMacOptions(
            lang=langs if langs else ['en'],
            force_full_page_ocr=True,
        )
    else:
        # Default to auto if unknown engine specified
        logger.warning(f"Unknown OCR engine '{engine}', falling back to 'auto'")
        from docling.datamodel.pipeline_options import OcrAutoOptions
        return OcrAutoOptions(
            lang=langs if langs else ['en'],
            force_full_page_ocr=True,
        )


# ============================================================================
# Document Loading
# ============================================================================


def load_document(
    source: str | Path | bytes | BinaryIO,
    doc_format: str | None = None,
) -> tuple[DoclingDocument, int | None]:
    """Load a document using IBM Docling.

    Docling is a powerful document processor that handles:
    - PDFs (with OCR for scanned documents)
    - Office documents (DOCX, PPTX, XLSX)
    - Images (PNG, JPG with OCR)
    - Audio (with ASR/transcription)
    - HTML and Markdown

    Args:
        source: File path, bytes, or file handle
        doc_format: Optional format hint (auto-detected if None)

    Returns:
        Tuple of (DoclingDocument, page_count)
        page_count is None for non-paginated documents

    Raises:
        DocumentLoadError: If document cannot be loaded
    """
    try:
        # Enforce our logging format on RapidOCR (it adds its own handlers)
        enforce_logging_format()

        # Get page count for progress indication (PDF only)
        page_count = None
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.suffix.lower() == '.pdf':
                try:
                    import pypdf
                    with open(source_path, 'rb') as f:
                        pdf_reader = pypdf.PdfReader(f)
                        page_count = len(pdf_reader.pages)
                    if page_count:
                        logger.info(f"Processing PDF with {page_count} pages (OCR enabled: {settings.enable_ocr})")
                except Exception:
                    # If page count detection fails, continue without it
                    pass

        # Force CPU-only processing (no GPU usage)
        accelerator_options = AcceleratorOptions(
            num_threads=4,  # Use multiple CPU threads for performance
            device=AcceleratorDevice.CPU  # Explicitly force CPU device
        )

        # Configure OCR options based on settings
        ocr_options = _get_ocr_options()

        # Configure PDF pipeline options with CPU-only accelerator and OCR
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options
        pipeline_options.ocr_options = ocr_options
        pipeline_options.do_ocr = settings.enable_ocr

        # Configure document converter with OCR settings and CPU-only mode
        converter = DocumentConverter(
            allowed_formats=None,  # Accept all formats
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            },
        )

        # Show progress indication for OCR processing
        if settings.enable_ocr and page_count:
            logger.info(f"Starting OCR processing (this may take several minutes for large documents)...")

        # Convert the document
        # Note: Docling processes page-by-page internally, but doesn't expose progress callbacks
        # The processing time is roughly proportional to page count for OCR
        result = converter.convert(source)

        if page_count:
            logger.info(f"Document processing complete ({page_count} pages processed)")

        return result.document, page_count

    except Exception as e:
        raise DocumentLoadError(f"Failed to load document: {e}") from e


# ============================================================================
# Metadata Extraction
# ============================================================================


def extract_metadata(
    doc: DoclingDocument,
    file_path: str | Path,
    num_chunks: int,
    num_pages: int | None = None
) -> DocumentMetadata:
    """Extract metadata from a document.

    Metadata helps with:
    - Tracking documents in the vector database
    - Filtering searches by language or file type
    - Audit trails (when was this ingested?)

    Args:
        doc: Parsed Docling document
        file_path: Original file path
        num_chunks: Number of chunks created from this document
        num_pages: Number of pages (None if not applicable)

    Returns:
        DocumentMetadata object with doc_id, language, etc.
    """
    file_path = Path(file_path) if isinstance(file_path, str) else file_path

    # Create a unique document ID from the file path (MD5 hash)
    doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()

    # Extract file type from extension (.pdf, .docx, etc.)
    doc_type = file_path.suffix.lstrip(".").lower() if file_path.suffix else "unknown"

    # Try to detect language (used for filtering and better search)
    language = "unknown"

    # First, check if Docling already detected the language
    if hasattr(doc, "properties") and doc.properties:
        language = doc.properties.get("language", "unknown")

    # If unknown, detect from text sample
    if language == "unknown":
        text_sample = doc.export_to_markdown()[:500]  # Use first 500 chars
        if text_sample:
            language = detect_language(text_sample)
            if language == "unknown":
                language = "en"  # Default to English if detection fails

    return DocumentMetadata(
        doc_id=doc_id,
        doc_type=doc_type,
        language=language,
        file_path=str(file_path),
        num_chunks=num_chunks,
        num_pages=num_pages,
        ingested_at=datetime.now(),
    )
