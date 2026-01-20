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
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from ..config import settings, get_logger, enforce_logging_format
from ..models import DocumentLoadError, DocumentMetadata
from ..utils import detect_language

logger = get_logger(__name__)


# ============================================================================
# Document Loading
# ============================================================================


def load_document(
    source: str | Path | bytes | BinaryIO,
    doc_format: str | None = None,
) -> DoclingDocument:
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
        Parsed DoclingDocument with text, structure, and metadata

    Raises:
        DocumentLoadError: If document cannot be loaded
    """
    try:
        # Enforce our logging format on RapidOCR (it adds its own handlers)
        enforce_logging_format()

        # Force CPU-only processing (no GPU usage)
        accelerator_options = AcceleratorOptions(
            num_threads=4,  # Use multiple CPU threads for performance
            device=AcceleratorDevice.CPU  # Explicitly force CPU device
        )

        # Configure RapidOCR options with Arabic and English support
        ocr_options = RapidOcrOptions(
            lang=['english', 'arabic'],  # Support both English and Arabic
        )

        # Configure PDF pipeline options with CPU-only accelerator and RapidOCR
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

        # Convert and return the document
        result = converter.convert(source)
        return result.document

    except Exception as e:
        raise DocumentLoadError(f"Failed to load document: {e}") from e


# ============================================================================
# Metadata Extraction
# ============================================================================


def extract_metadata(
    doc: DoclingDocument,
    file_path: str | Path,
    num_chunks: int
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
        ingested_at=datetime.now(),
    )
