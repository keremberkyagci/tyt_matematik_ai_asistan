import os
from pathlib import Path
import gc

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def process_pdf_in_batches(pdf_path: Path, batch_size: int = 10):
    print(f"PDF batch işleme başlatılıyor: {pdf_path.name}")

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    all_chunks = []

    loader = DoclingLoader(
        file_path=str(pdf_path),
        export_type=ExportType.MARKDOWN
    )

    # Tüm dokümanları lazy olarak işle
    docs = loader.lazy_load()  # load() yerine lazy_load() kullan

    batch = []
    batch_num = 0

    for doc in docs:
        batch.append(doc)

        if len(batch) >= batch_size:
            batch_num += 1
            print(f"Batch {batch_num} işleniyor ({batch_size} sayfa)...")

            for d in batch:
                header_splits = markdown_splitter.split_text(d.page_content)
                all_chunks.extend(text_splitter.split_documents(header_splits))

            batch.clear()
            gc.collect()  # Belleği temizle

    # Kalan sayfaları işle
    if batch:
        batch_num += 1
        print(f"Son batch {batch_num} işleniyor ({len(batch)} sayfa)...")
        for d in batch:
            header_splits = markdown_splitter.split_text(d.page_content)
            all_chunks.extend(text_splitter.split_documents(header_splits))
        batch.clear()
        gc.collect()

    return all_chunks


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
pdf_path = _PROJECT_ROOT / "datasets" / "tyt_matematik.pdf"
persist_dir = _PROJECT_ROOT / "datasets" / "chroma_db"

if not pdf_path.exists():
    raise FileNotFoundError(f"PDF dosyası bulunamadı: {pdf_path}")

chunks = process_pdf_in_batches(pdf_path, batch_size=10)
print(f"Kitap {len(chunks)} kaliteli parçaya bölündü.")

print("Vektörleştirme başlatılıyor (Turkish BERT)...")
embeddings = HuggingFaceEmbeddings(model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

# Chroma'ya da batch batch ekle (bir anda hepsini değil)
print(f"Veritabanı kaydediliyor: {persist_dir}")
EMBED_BATCH = 50
db = None

for i in range(0, len(chunks), EMBED_BATCH):
    batch_chunks = chunks[i:i + EMBED_BATCH]
    print(f"Embedding batch {i // EMBED_BATCH + 1} / {len(chunks) // EMBED_BATCH + 1}...")

    if db is None:
        db = Chroma.from_documents(
            documents=batch_chunks,
            embedding=embeddings,
            persist_directory=str(persist_dir),
        )
    else:
        db.add_documents(batch_chunks)

    gc.collect()

print("İşlem tamam! ChromaDB güncellendi.")