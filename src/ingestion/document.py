import hashlib
import os
from datetime import datetime
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, LayoutModelConfig, ThreadedPdfPipelineOptions
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from docling.utils.profiling import ProfilingItem
from docling_core.types.doc import DoclingDocument
from pathlib import Path
from typing import BinaryIO, Optional

from ..config import device as app_device, get_logger
from ..models import DocumentLoadError, DocumentMetadata
from ..utils import detect_language, get_model_paths

logger = get_logger(__name__)

_converter_cache: Optional[DocumentConverter] = None

_DEVICE_MAP = {
    "cuda": AcceleratorDevice.CUDA,
    "mps": AcceleratorDevice.MPS,
    "cpu": AcceleratorDevice.CPU,
    "auto": AcceleratorDevice.AUTO,
}

def _get_ocr_options():
    from docling.datamodel.pipeline_options import RapidOcrOptions
    return RapidOcrOptions(lang=['english', 'arabic'], force_full_page_ocr=True)


def get_converter() -> DocumentConverter:
    global _converter_cache
    if _converter_cache is not None:
        return _converter_cache

    accel_device = _DEVICE_MAP.get(app_device, AcceleratorDevice.AUTO)

    accelerator_options = AcceleratorOptions(
        device=accel_device,
        num_threads=4,
        cuda_use_flash_attention2=accel_device == AcceleratorDevice.CUDA,
    )

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=accelerator_options,
        ocr_batch_size=4,
        layout_batch_size=64,
        table_batch_size=4,
        document_timeout=120,
    )
    pipeline_options.do_ocr = False
    pipeline_options.do_code_enrichment = False
    pipeline_options.do_formula_enrichment = False
    pipeline_options.generate_page_images = False
    pipeline_options.generate_picture_images = False
    pipeline_options.images_scale = 1.0

    settings.perf.page_batch_size = 64

    local_docling_path = get_model_paths()["docling_layout"]
    os.environ["HF_HOME"] = str(local_docling_path.parent)

    layout_repo_path = local_docling_path / "models--docling-project--docling-layout-heron"
    layout_snapshot_path = layout_repo_path / "snapshots"
    if layout_snapshot_path.exists():
        snapshots = list(layout_snapshot_path.iterdir())
        if snapshots:
            layout_model_path = snapshots[0]
            pipeline_options.layout_options.model_spec = LayoutModelConfig(
                name="docling_layout_heron", repo_id="docling-project/docling-layout-heron",
                revision="main", model_path=str(layout_model_path))

    converter = DocumentConverter(allowed_formats=None,
                                  format_options={
                                      InputFormat.PDF: PdfFormatOption(
                                          pipeline_cls=ThreadedStandardPdfPipeline,
                                          pipeline_options=pipeline_options)})
    converter.initialize_pipeline(InputFormat.PDF)

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        logger.info(f"ONNX Runtime providers: {providers}")
        if accel_device == AcceleratorDevice.CUDA and "CUDAExecutionProvider" not in providers:
            logger.warning("CUDA device requested but CUDAExecutionProvider not available — layout model will run on CPU")
    except ImportError:
        pass

    _converter_cache = converter
    return converter


def cleanup_converter() -> None:
    global _converter_cache
    if _converter_cache is not None:
        _converter_cache = None


def load_document(source: str | Path | bytes | BinaryIO, doc_format: str | None = None) -> tuple[
    DoclingDocument, int | None]:
    try:
        page_count = None
        if isinstance(source, (str, Path)):
            source_path = Path(source)
            if source_path.suffix.lower() == '.pdf':
                try:
                    import pypdf
                    with open(source_path, 'rb') as f:
                        page_count = len(pypdf.PdfReader(f).pages)
                except Exception:
                    pass

        converter = get_converter()
        logger.info(f"Processing {source}")
        result = converter.convert(source)
        return result.document, page_count
    except Exception as e:
        raise DocumentLoadError(f"Failed to load document: {e}") from e


def extract_metadata(doc: DoclingDocument, file_path: str | Path, num_chunks: int,
                     num_pages: int | None = None) -> DocumentMetadata:
    file_path = Path(file_path) if isinstance(file_path, str) else file_path
    doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()
    doc_type = file_path.suffix.lstrip(".").lower() if file_path.suffix else "unknown"
    language = "unknown"
    if hasattr(doc, "properties") and doc.properties:
        language = doc.properties.get("language", "unknown")
    if language == "unknown":
        text_sample = doc.export_to_markdown()[:500]
        if text_sample:
            language = detect_language(text_sample)
            if language == "unknown":
                language = "en"
    return DocumentMetadata(doc_id=doc_id, doc_type=doc_type, language=language, file_path=str(file_path),
                            num_chunks=num_chunks, num_pages=num_pages, ingested_at=datetime.now())
