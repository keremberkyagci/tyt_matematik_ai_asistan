"""
Docling pipeline — TYT matematik PDF.

PDF profili (fitz): 155/160 sayfa gomulu metin katmanli.
- Metin sayfalari  -> layout + backend text (OCR kapali, hizli, Turkce dogru)
- Seyrek/taranmis  -> EasyOCR yalnizca o sayfalarda
- Formul VLM      -> opsiyonel (DOCLING_FORMULA_ENRICHMENT=1, GPU onerilir)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import fitz
import torch
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.layout_model_specs import DOCLING_LAYOUT_HERON
from docling.datamodel.pipeline_options import (
    CodeFormulaVlmOptions,
    EasyOcrOptions,
    LayoutOptions,
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

MIN_TEXT_CHARS = 50
MIN_CHUNK_CHARS = 80

# PDF font haritalama hatalari (metin katmaninda sik gorulen)
MATH_CHAR_FIXES: dict[str, str] = {
    "Î": "∈",
}

_TYT_HEADER_NOISE = re.compile(r"^\s*(?:TYT[\s\t]*)+$", re.MULTILINE)
_FORMULA_PLACEHOLDER = re.compile(r"<!--\s*formula-not-decoded\s*-->", re.IGNORECASE)
_MULTI_BLANK = re.compile(r"\n{3,}")


class PipelineMode(str, Enum):
    TEXT = "text"
    OCR = "ocr"
    AUTO = "auto"


@dataclass(frozen=True)
class PageProfile:
    page_num: int
    char_count: int
    image_count: int

    @property
    def needs_ocr(self) -> bool:
        return self.char_count < MIN_TEXT_CHARS

    @property
    def is_empty(self) -> bool:
        return self.char_count == 0


@dataclass(frozen=True)
class PdfProfile:
    pages: tuple[PageProfile, ...]

    @property
    def total_pages(self) -> int:
        return len(self.pages)

    @property
    def text_page_ratio(self) -> float:
        if not self.pages:
            return 0.0
        text_pages = sum(1 for p in self.pages if not p.needs_ocr and not p.is_empty)
        return text_pages / len(self.pages)

    def recommended_mode(self) -> PipelineMode:
        if self.text_page_ratio >= 0.85:
            return PipelineMode.TEXT
        return PipelineMode.OCR

    def batch_needs_ocr(self, page_start: int, page_end: int) -> bool:
        for p in self.pages[page_start - 1 : page_end]:
            if p.needs_ocr and not p.is_empty:
                return True
        return False

    def batch_is_empty(self, page_start: int, page_end: int) -> bool:
        batch = self.pages[page_start - 1 : page_end]
        return bool(batch) and all(p.is_empty for p in batch)


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return float(raw)


def _resolve_use_gpu(use_gpu: Optional[bool]) -> bool:
    if use_gpu is not None:
        return use_gpu
    if _env_flag("DOCLING_USE_GPU", default=True):
        return torch.cuda.is_available()
    return False


def profile_pdf(pdf_path: Path) -> PdfProfile:
    doc = fitz.open(pdf_path)
    try:
        pages = []
        for i in range(doc.page_count):
            page = doc[i]
            text = (page.get_text("text") or "").strip()
            pages.append(
                PageProfile(
                    page_num=i + 1,
                    char_count=len(text),
                    image_count=len(page.get_images(full=True)),
                )
            )
        return PdfProfile(pages=tuple(pages))
    finally:
        doc.close()


def clean_markdown(markdown: str) -> str:
    """Baslik/footer gurultusu ve bilinen karakter hatalarini temizle."""
    lines: list[str] = []
    for line in markdown.splitlines():
        stripped = line.strip()
        # Dikey footer harfleri (K, O, N, U tek tek satirlarda)
        if len(stripped) == 1 and stripped.isalpha():
            continue
        lines.append(line)

    text = "\n".join(lines)
    text = _TYT_HEADER_NOISE.sub("", text)
    text = _FORMULA_PLACEHOLDER.sub("[formul]", text)
    for wrong, right in MATH_CHAR_FIXES.items():
        text = text.replace(wrong, right)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def _build_accelerator_options(use_gpu: bool) -> AcceleratorOptions:
    cuda_active = use_gpu and torch.cuda.is_available()
    device = AcceleratorDevice.CUDA if cuda_active else AcceleratorDevice.CPU
    num_threads = int(os.environ.get("DOCLING_NUM_THREADS", "8"))
    return AcceleratorOptions(num_threads=num_threads, device=device)


def _base_pipeline_options(
    *,
    do_ocr: bool,
    use_gpu: bool,
    do_formula_enrichment: bool,
    images_scale: float,
    force_full_page_ocr: bool,
    lang: list[str],
) -> PdfPipelineOptions:
    cuda_active = use_gpu and torch.cuda.is_available()

    ocr_options = EasyOcrOptions(
        lang=lang,
        use_gpu=True if cuda_active else False,
        force_full_page_ocr=force_full_page_ocr,
        confidence_threshold=0.5,
        recog_network="standard",
        download_enabled=True,
    )

    return PdfPipelineOptions(
        do_ocr=do_ocr,
        do_table_structure=True,
        do_formula_enrichment=do_formula_enrichment,
        force_backend_text=False,
        images_scale=images_scale,
        generate_page_images=do_formula_enrichment,
        document_timeout=_env_float("DOCLING_TIMEOUT", 180.0),
        accelerator_options=_build_accelerator_options(use_gpu),
        table_structure_options=TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE,
        ),
        layout_options=LayoutOptions(
            model_spec=DOCLING_LAYOUT_HERON,
            create_orphan_clusters=True,
            skip_cell_assignment=False,
        ),
        code_formula_options=CodeFormulaVlmOptions.from_preset("codeformulav2"),
        ocr_options=ocr_options,
    )


def build_converter(
    *,
    do_ocr: bool = True,
    force_full_page_ocr: bool = False,
    use_gpu: Optional[bool] = None,
    do_formula_enrichment: Optional[bool] = None,
) -> DocumentConverter:
    """
    GPU (CUDA) acik Docling converter.

    Turkce EasyOCR, TableFormer ACCURATE, Heron layout.
    Metin katmanli PDF icin do_ocr=False kullanin.
    """
    gpu = _resolve_use_gpu(use_gpu)
    if _env_flag("DOCLING_USE_GPU", default=True) and not torch.cuda.is_available():
        print("Uyari: DOCLING_USE_GPU=1 ama CUDA yok, CPU kullanilacak.")

    if do_formula_enrichment is None:
        do_formula_enrichment = _env_flag("DOCLING_FORMULA_ENRICHMENT", default=False)

    pipeline_options = _base_pipeline_options(
        do_ocr=do_ocr,
        use_gpu=gpu,
        do_formula_enrichment=do_formula_enrichment,
        images_scale=_env_float("DOCLING_IMAGES_SCALE", 2.0 if do_formula_enrichment else 1.0),
        force_full_page_ocr=force_full_page_ocr,
        lang=["tr"],
    )
    return _build_converter(pipeline_options)


def create_text_converter(
    *,
    use_gpu: Optional[bool] = None,
    do_formula_enrichment: Optional[bool] = None,
) -> DocumentConverter:
    """Gomulu metin katmanli sayfalar icin — OCR yok, Turkce korunur."""
    return build_converter(
        do_ocr=False,
        use_gpu=use_gpu,
        do_formula_enrichment=do_formula_enrichment,
    )


def create_ocr_converter(
    *,
    use_gpu: Optional[bool] = None,
    force_full_page_ocr: bool = True,
) -> DocumentConverter:
    """Metin katmani olmayan seyrek/taranmis sayfalar icin."""
    return build_converter(
        do_ocr=True,
        force_full_page_ocr=force_full_page_ocr,
        use_gpu=use_gpu,
    )


def create_converter(
    pdf_path: Optional[Path] = None,
    mode: PipelineMode | str = PipelineMode.AUTO,
) -> DocumentConverter:
    """PDF profiline gore uygun converter dondurur."""
    resolved = PipelineMode(mode)
    if resolved == PipelineMode.AUTO:
        if pdf_path is None:
            resolved = PipelineMode.TEXT
        else:
            resolved = profile_pdf(pdf_path).recommended_mode()

    if resolved == PipelineMode.OCR:
        return create_ocr_converter()
    return create_text_converter()


def get_converter_for_batch(
    pdf_profile: PdfProfile,
    page_start: int,
    page_end: int,
    *,
    text_converter: DocumentConverter | None,
    ocr_converter: DocumentConverter | None,
) -> tuple[DocumentConverter, DocumentConverter | None, DocumentConverter | None]:
    if pdf_profile.batch_needs_ocr(page_start, page_end):
        if ocr_converter is None:
            ocr_converter = create_ocr_converter()
        return ocr_converter, text_converter, ocr_converter
    if text_converter is None:
        text_converter = create_text_converter()
    return text_converter, text_converter, ocr_converter


def _build_converter(pipeline_options: PdfPipelineOptions) -> DocumentConverter:
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                backend=DoclingParseDocumentBackend,
                pipeline_options=pipeline_options,
            ),
        }
    )