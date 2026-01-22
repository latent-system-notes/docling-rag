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
from docling.datamodel.pipeline_options import PdfPipelineOptions, LayoutModelConfig
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DoclingDocument

from ..config import settings, get_logger, enforce_logging_format
from ..models import DocumentLoadError, DocumentMetadata
from ..utils import detect_language, get_model_paths

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

    logger.info(f"OCR CONFIG: Configuring OCR engine: {engine}")
    logger.info(f"OCR CONFIG: OCR languages: {langs if langs else ['default']}")

    if engine == "auto":
        # Use OcrAutoOptions for automatic engine selection
        from docling.datamodel.pipeline_options import OcrAutoOptions
        logger.info("OCR CONFIG: Using OcrAutoOptions (auto-select best engine)")
        return OcrAutoOptions(
            lang=langs if langs else ['en'],
            force_full_page_ocr=True,
        )
    elif engine == "rapidocr":
        from docling.datamodel.pipeline_options import RapidOcrOptions
        # RapidOCR uses full language names (english, arabic, chinese, etc.)
        lang_map = {'eng': 'english', 'ara': 'arabic', 'chi': 'chinese', 'jpn': 'japanese'}
        rapidocr_langs = [lang_map.get(lang, lang) for lang in langs]
        logger.info(f"OCR CONFIG: Using RapidOcrOptions with languages: {rapidocr_langs if rapidocr_langs else ['english']}")
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
        logger.warning(f"OCR CONFIG: Unknown OCR engine '{engine}', falling back to 'auto'")
        from docling.datamodel.pipeline_options import OcrAutoOptions
        logger.info("OCR CONFIG: Using OcrAutoOptions (fallback)")
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
        logger.info("=" * 60)
        logger.info("DOCUMENT LOADING: Starting document processing")
        logger.info("=" * 60)

        # Enforce our logging format on RapidOCR (it adds its own handlers)
        enforce_logging_format()

        # Get page count for progress indication (PDF only)
        page_count = None
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            logger.info(f"DOCUMENT LOADING: Source file: {source_path}")
            logger.info(f"DOCUMENT LOADING: File type: {source_path.suffix}")

            if source_path.suffix.lower() == '.pdf':
                try:
                    import pypdf
                    with open(source_path, 'rb') as f:
                        pdf_reader = pypdf.PdfReader(f)
                        page_count = len(pdf_reader.pages)
                    if page_count:
                        logger.info(f"DOCUMENT LOADING: PDF with {page_count} pages (OCR enabled: {settings.enable_ocr})")
                except Exception as e:
                    # If page count detection fails, continue without it
                    logger.warning(f"DOCUMENT LOADING: Could not detect page count: {e}")
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

        # CRITICAL: Disable remote services to enforce offline mode
        pipeline_options.enable_remote_services = False
        logger.info("OFFLINE MODE: Remote services disabled (enable_remote_services=False)")

        # Configure local Docling model paths for offline mode
        local_docling_path = get_model_paths()["docling_layout"]
        logger.info(f"OFFLINE MODE: Docling base cache path: {local_docling_path}")

        # Set HF_HOME environment variable to use local cache
        # This ensures all HuggingFace model downloads use our local cache
        import os
        os.environ["HF_HOME"] = str(local_docling_path.parent)
        logger.info(f"OFFLINE MODE: Set HF_HOME={os.environ['HF_HOME']}")
        logger.info("OFFLINE MODE: All HuggingFace models will use local cache")

        # ========================================================================
        # 1. Configure Layout Model (docling-layout-heron)
        # ========================================================================
        # The HuggingFace cache stores models as: models/.cache/hub/models--docling-project--docling-layout-heron
        # We need to point to the actual snapshot directory
        layout_repo_path = local_docling_path / "models--docling-project--docling-layout-heron"
        layout_snapshot_path = layout_repo_path / "snapshots"

        logger.info(f"OFFLINE MODE: Checking for model at: {layout_repo_path}")
        logger.info(f"OFFLINE MODE: Snapshot directory: {layout_snapshot_path}")

        # Find the latest snapshot directory (should only be one)
        if layout_snapshot_path.exists():
            logger.info(f"OFFLINE MODE: Snapshot directory exists, scanning for snapshots...")
            snapshots = list(layout_snapshot_path.iterdir())
            logger.info(f"OFFLINE MODE: Found {len(snapshots)} snapshot(s)")

            if snapshots:
                layout_model_path = snapshots[0]  # Use the first (and likely only) snapshot
                logger.info(f"OFFLINE MODE: Using snapshot: {layout_model_path.name}")
                logger.info(f"OFFLINE MODE: Full model path: {layout_model_path}")

                # Verify model files exist
                model_files = list(layout_model_path.glob("*"))
                logger.info(f"OFFLINE MODE: Model snapshot contains {len(model_files)} files")

                # Configure the model specification with local path
                pipeline_options.layout_options.model_spec = LayoutModelConfig(
                    name="docling_layout_heron",
                    repo_id="docling-project/docling-layout-heron",
                    revision="main",
                    model_path=str(layout_model_path),  # Use local snapshot path
                    supported_devices=[AcceleratorDevice.CPU]
                )
                logger.info("OFFLINE MODE: Layout model configured with local path")
            else:
                logger.error("OFFLINE MODE: No snapshots found in snapshot directory!")
                logger.error(f"OFFLINE MODE: Run 'rag models --download' to download required models")
        else:
            logger.error(f"OFFLINE MODE: Snapshot path does not exist: {layout_snapshot_path}")
            logger.error(f"OFFLINE MODE: Model repo path exists: {layout_repo_path.exists()}")
            if layout_repo_path.exists():
                contents = list(layout_repo_path.iterdir())
                logger.error(f"OFFLINE MODE: Model repo contains: {[p.name for p in contents]}")
            logger.error(f"OFFLINE MODE: Run 'rag models --download' to download required models")

        # ========================================================================
        # 2. Configure Table Structure Model (docling-models)
        # ========================================================================
        table_repo_path = local_docling_path / "models--docling-project--docling-models"
        table_snapshot_path = table_repo_path / "snapshots"

        logger.info(f"OFFLINE MODE: Checking for table model at: {table_repo_path}")
        logger.info(f"OFFLINE MODE: Table snapshot directory: {table_snapshot_path}")

        if table_snapshot_path.exists():
            logger.info(f"OFFLINE MODE: Table snapshot directory exists, scanning...")
            table_snapshots = list(table_snapshot_path.iterdir())
            logger.info(f"OFFLINE MODE: Found {len(table_snapshots)} table snapshot(s)")

            if table_snapshots:
                table_model_path = table_snapshots[0]
                logger.info(f"OFFLINE MODE: Using table snapshot: {table_model_path.name}")
                logger.info(f"OFFLINE MODE: Full table model path: {table_model_path}")

                # Verify table model files exist
                table_files = list(table_model_path.glob("*"))
                logger.info(f"OFFLINE MODE: Table model snapshot contains {len(table_files)} files")
                logger.info("OFFLINE MODE: Table model configured (will use HF_HOME cache)")
            else:
                logger.error("OFFLINE MODE: No table snapshots found!")
                logger.error(f"OFFLINE MODE: Run 'rag models --download' to download required models")
        else:
            logger.error(f"OFFLINE MODE: Table snapshot path does not exist: {table_snapshot_path}")
            logger.error(f"OFFLINE MODE: Run 'rag models --download' to download required models")

        # Configure document converter with OCR settings and CPU-only mode
        logger.info("DOCUMENT LOADING: Creating DocumentConverter with offline pipeline")
        logger.info(f"DOCUMENT LOADING: Pipeline options hash: {hash(str(pipeline_options))}")
        logger.info(f"DOCUMENT LOADING: OCR enabled: {pipeline_options.do_ocr}")
        logger.info(f"DOCUMENT LOADING: Remote services: {pipeline_options.enable_remote_services}")

        converter = DocumentConverter(
            allowed_formats=None,  # Accept all formats
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                ),
            },
        )
        logger.info("DOCUMENT LOADING: DocumentConverter created successfully")

        # Show progress indication for OCR processing
        if settings.enable_ocr and page_count:
            logger.info(f"DOCUMENT LOADING: Starting OCR processing for {page_count} pages (this may take several minutes)...")

        # Convert the document
        # Note: Docling processes page-by-page internally, but doesn't expose progress callbacks
        # The processing time is roughly proportional to page count for OCR
        logger.info("DOCUMENT LOADING: Starting document conversion...")

        try:
            result = converter.convert(source)
            logger.info("DOCUMENT LOADING: Document conversion completed successfully")
        except Exception as conv_error:
            logger.error(f"DOCUMENT LOADING: Conversion failed with error: {conv_error}")
            logger.error(f"DOCUMENT LOADING: Error type: {type(conv_error).__name__}")
            import traceback
            logger.error(f"DOCUMENT LOADING: Traceback:\n{traceback.format_exc()}")
            raise

        if page_count:
            logger.info(f"DOCUMENT LOADING: Processing complete ({page_count} pages processed)")

        logger.info("=" * 60)
        logger.info("DOCUMENT LOADING: Document processing finished successfully")
        logger.info("=" * 60)

        return result.document, page_count

    except Exception as e:
        logger.error("=" * 60)
        logger.error("DOCUMENT LOADING: FAILED")
        logger.error("=" * 60)
        logger.error(f"DOCUMENT LOADING: Error message: {e}")
        logger.error(f"DOCUMENT LOADING: Error type: {type(e).__name__}")

        # Check if it's a HuggingFace Hub error
        error_str = str(e).lower()
        if "hub" in error_str or "snapshot" in error_str or "revision" in error_str:
            logger.error("DOCUMENT LOADING: This appears to be a HuggingFace Hub model loading error")
            logger.error("DOCUMENT LOADING: Checking model availability...")

            model_paths = get_model_paths()
            docling_path = model_paths["docling_layout"]
            logger.error(f"DOCUMENT LOADING: Expected model base path: {docling_path}")
            logger.error(f"DOCUMENT LOADING: Path exists: {docling_path.exists()}")

            if docling_path.exists():
                try:
                    contents = list(docling_path.rglob("*"))
                    logger.error(f"DOCUMENT LOADING: Found {len(contents)} files/directories in model path")
                except Exception as scan_error:
                    logger.error(f"DOCUMENT LOADING: Could not scan directory: {scan_error}")

            logger.error("DOCUMENT LOADING: SOLUTION: Run 'rag models --download' to ensure all models are downloaded")

        import traceback
        logger.error(f"DOCUMENT LOADING: Full traceback:\n{traceback.format_exc()}")
        logger.error("=" * 60)

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
