"""
TYT Matematik PDF -> Docling (adaptif pipeline) -> ChromaDB

Kullanim:
    python upload_book.py --reset-chroma
    python upload_book.py --page-batch 10 --embed-batch 50

PDF once profillenir; metin katmanli sayfalar OCR'siz, seyrek sayfalar OCR ile islenir.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

import fitz
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from docling_ocr import (
    MIN_CHUNK_CHARS,
    PipelineMode,
    clean_markdown,
    get_converter_for_batch,
    profile_pdf,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = _PROJECT_ROOT / "datasets" / "tyt_matematik.pdf"
PERSIST_DIR = _PROJECT_ROOT / "datasets" / "chroma_db"
EMBEDDING_MODEL = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
COLLECTION_NAME = "tyt_matematik"

HEADERS_TO_SPLIT = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def verify_gpu() -> None:
    print("=" * 60)
    print("GPU dogrulamasi")
    print("=" * 60)

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        print("nvidia-smi:")
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5:
                idx, name, mem_used, mem_total, util = parts[:5]
                print(f"  GPU {idx}: {name} | VRAM {mem_used}/{mem_total} MiB | util {util}%")
            else:
                print(f"  {line}")
    except FileNotFoundError:
        print("  nvidia-smi bulunamadi.")
    except subprocess.CalledProcessError as exc:
        print(f"  nvidia-smi hata: {exc.returncode}")

    cuda_ok = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_ok}")
    if cuda_ok:
        print(f"  Cihaz: {torch.cuda.get_device_name(0)}")
    else:
        print("  CPU modu aktif.")
    print("=" * 60)


def log_gpu_snapshot(label: str) -> None:
    if not torch.cuda.is_available():
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        mem, util = [p.strip() for p in result.stdout.strip().split(",")[:2]]
        print(f"  [{label}] VRAM {mem} MiB | GPU util {util}%")
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass


def get_page_count(pdf_path: Path) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def reset_chroma_dir(persist_dir: Path) -> None:
    import shutil

    if persist_dir.exists():
        shutil.rmtree(persist_dir)
        print(f"Mevcut ChromaDB silindi: {persist_dir}")
    persist_dir.mkdir(parents=True, exist_ok=True)


def print_pdf_profile(pdf_path: Path) -> None:
    profile = profile_pdf(pdf_path)
    text_pages = sum(1 for p in profile.pages if not p.needs_ocr and not p.is_empty)
    ocr_pages = sum(1 for p in profile.pages if p.needs_ocr and not p.is_empty)
    empty_pages = sum(1 for p in profile.pages if p.is_empty)
    mode = profile.recommended_mode()

    print(f"PDF profili: {profile.total_pages} sayfa")
    print(f"  Metin katmanli : {text_pages}")
    print(f"  OCR gerekli    : {ocr_pages}")
    print(f"  Bos            : {empty_pages}")
    print(f"  Onerilen mod   : {mode.value} ({profile.text_page_ratio:.0%} metin)")


def markdown_to_chunks(
    markdown: str,
    *,
    pdf_name: str,
    page_start: int,
    page_end: int,
    pipeline_mode: str,
) -> list[Document]:
    cleaned = clean_markdown(markdown)
    if len(cleaned) < MIN_CHUNK_CHARS:
        return []

    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    if cleaned.startswith("#") or "\n## " in cleaned or "\n### " in cleaned:
        splits = markdown_splitter.split_text(cleaned)
        chunks = text_splitter.split_documents(splits)
    else:
        chunks = text_splitter.split_documents([Document(page_content=cleaned, metadata={})])

    meta = {
        "source": pdf_name,
        "page_start": page_start,
        "page_end": page_end,
        "page_range": f"{page_start}-{page_end}",
        "pipeline": pipeline_mode,
    }

    result: list[Document] = []
    for chunk in chunks:
        content = (chunk.page_content or "").strip()
        if len(content) < MIN_CHUNK_CHARS:
            continue
        chunk.metadata.update(meta)
        result.append(chunk)
    return result


def add_chunks_to_chroma(
    chunks: list[Document],
    embeddings: HuggingFaceEmbeddings,
    persist_dir: Path,
    embed_batch_size: int,
    db: Chroma | None,
) -> Chroma | None:
    if not chunks:
        return db

    total_batches = (len(chunks) + embed_batch_size - 1) // embed_batch_size
    for i in range(0, len(chunks), embed_batch_size):
        batch = chunks[i : i + embed_batch_size]
        batch_num = i // embed_batch_size + 1
        print(f"    Embedding -> Chroma {batch_num}/{total_batches} ({len(batch)} parca)")

        if db is None:
            db = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(persist_dir),
                collection_name=COLLECTION_NAME,
            )
        else:
            db.add_documents(batch)
        gc.collect()

    return db


def create_embeddings() -> HuggingFaceEmbeddings:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Embedding modeli: {EMBEDDING_MODEL} ({device})")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
    )


def process_pdf_in_batches(
    pdf_path: Path,
    persist_dir: Path,
    *,
    page_batch_size: int = 10,
    embed_batch_size: int = 50,
    reset_chroma: bool = False,
) -> int:
    """PDF'i sayfa batch'leri halinde DoclingLoader ile isler, ChromaDB'ye yazar."""
    if reset_chroma:
        reset_chroma_dir(persist_dir)
    else:
        persist_dir.mkdir(parents=True, exist_ok=True)

    pdf_profile = profile_pdf(pdf_path)
    total_pages = pdf_profile.total_pages
    total_page_batches = (total_pages + page_batch_size - 1) // page_batch_size

    embeddings = create_embeddings()
    db: Chroma | None = None
    total_chunks = 0
    skipped_batches = 0
    text_converter = None
    ocr_converter = None

    for batch_idx, page_start in enumerate(range(1, total_pages + 1, page_batch_size), start=1):
        page_end = min(page_start + page_batch_size - 1, total_pages)

        if pdf_profile.batch_is_empty(page_start, page_end):
            print(f"[Batch {batch_idx}/{total_page_batches}] Sayfa {page_start}-{page_end} atlandi (bos)")
            skipped_batches += 1
            continue

        needs_ocr = pdf_profile.batch_needs_ocr(page_start, page_end)
        mode_label = PipelineMode.OCR.value if needs_ocr else PipelineMode.TEXT.value
        print(f"[Batch {batch_idx}/{total_page_batches}] Sayfa {page_start}-{page_end} [{mode_label}]")

        converter, text_converter, ocr_converter = get_converter_for_batch(
            pdf_profile,
            page_start,
            page_end,
            text_converter=text_converter,
            ocr_converter=ocr_converter,
        )

        loader = DoclingLoader(
            file_path=str(pdf_path),
            converter=converter,
            export_type=ExportType.MARKDOWN,
            convert_kwargs={"page_range": (page_start, page_end)},
            md_export_kwargs={"image_placeholder": ""},
        )

        for doc in loader.lazy_load():
            markdown = clean_markdown(doc.page_content)
            chunks = markdown_to_chunks(
                markdown,
                pdf_name=pdf_path.name,
                page_start=page_start,
                page_end=page_end,
                pipeline_mode=mode_label,
            )
            print(f"  {len(markdown):,} karakter -> {len(chunks)} parca (temizlenmis)")
            db = add_chunks_to_chroma(chunks, embeddings, persist_dir, embed_batch_size, db)
            total_chunks += len(chunks)

        log_gpu_snapshot(f"sayfa {page_start}-{page_end}")
        gc.collect()

    print(
        f"\nTamamlandi: {total_chunks} parca kaydedildi "
        f"({skipped_batches} bos batch atlandi) -> {persist_dir}"
    )
    return total_chunks


def process_full_pdf(
    pdf_path: Path,
    persist_dir: Path,
    *,
    page_batch_size: int = 10,
    embed_batch_size: int = 50,
    reset_chroma: bool = False,
) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF bulunamadi: {pdf_path}")

    verify_gpu()
    print_pdf_profile(pdf_path)

    if reset_chroma:
        reset_chroma_dir(persist_dir)
    else:
        persist_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nPDF: {pdf_path.name}")
    print(f"Sayfa batch: {page_batch_size} | Embedding batch: {embed_batch_size}")
    print(f"ChromaDB: {persist_dir} (collection: {COLLECTION_NAME})\n")

    process_pdf_in_batches(
        pdf_path,
        persist_dir,
        page_batch_size=page_batch_size,
        embed_batch_size=embed_batch_size,
        reset_chroma=False,
    )
    verify_gpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TYT Matematik PDF - Docling - ChromaDB")
    parser.add_argument("--pdf", type=Path, default=PDF_PATH)
    parser.add_argument("--persist-dir", type=Path, default=PERSIST_DIR)
    parser.add_argument(
        "--page-batch",
        type=int,
        default=int(os.environ.get("PAGE_BATCH_SIZE", "10")),
    )
    parser.add_argument(
        "--embed-batch",
        type=int,
        default=int(os.environ.get("EMBED_BATCH_SIZE", "50")),
    )
    parser.add_argument(
        "--reset-chroma",
        action="store_true",
        default=_env_flag("CHROMA_RESET"),
        help="Mevcut ChromaDB'yi silip sifirdan olustur",
    )
    return parser.parse_args()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    args = parse_args()
    try:
        process_full_pdf(
            args.pdf,
            args.persist_dir,
            page_batch_size=args.page_batch,
            embed_batch_size=args.embed_batch,
            reset_chroma=args.reset_chroma,
        )
        return 0
    except KeyboardInterrupt:
        print("\nDurduruldu.")
        return 130
    except Exception as exc:
        print(f"Hata: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
