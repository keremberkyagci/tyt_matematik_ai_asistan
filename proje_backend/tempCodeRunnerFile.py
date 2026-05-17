import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def load_pdf_pages(pdf_path: Path) -> list[Document]:
    """Önce PyMuPDF (hızlı, karmaşık şemalarda pdfplumber gibi takılmaz), yoksa PyPDF."""
    try:
        import fitz  # type: ignore[import-untyped]  # pymupdf
    except ImportError:
        fitz = None

    if fitz is not None:
        doc = fitz.open(str(pdf_path))
        try:
            pages: list[Document] = []
            n = len(doc)
            for i in range(n):
                if n > 20 and (i + 1) % 50 == 0:
                    print(f"  Sayfa {i + 1}/{n} okunuyor...")
                text = doc[i].get_text() or ""
                pages.append(Document(page_content=text, metadata={"page": i + 1}))
            return pages
        finally:
            doc.close()

    print("pymupdf yok; PyPDFLoader kullanılıyor. Hız/takılma için: pip install pymupdf")
    return PyPDFLoader(str(pdf_path)).load()


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


pdf_path = Path(__file__).resolve().parent / "tyt_matematik.pdf"
if not pdf_path.exists():
    raise FileNotFoundError(f"PDF bulunamadı: {pdf_path}")

if _env_flag("UNSTRUCTURED_OCR"):
    # Tesseract (+ Windows'ta Poppler) gerekir. Ağır bağımlılık: pip install -r requirements-ocr.txt
    from langchain_community.document_loaders import UnstructuredPDFLoader

    print("UnstructuredPDFLoader (strategy=ocr_only) — bu yol yavaş olabilir, Tesseract kullanır.")
    pages = UnstructuredPDFLoader(str(pdf_path), strategy="ocr_only").load()
else:
    print("PDF yükleniyor (PyMuPDF öncelikli)...")
    pages = load_pdf_pages(pdf_path)

    if not pages or all(not (d.page_content or "").strip() for d in pages):
        print("Metin katmanı boş görünüyor; PyPDFLoader ile tekrar deneniyor...")
        pages = PyPDFLoader(str(pdf_path)).load()

    empty_pages = [d.metadata.get("page") for d in pages if not (d.page_content or "").strip()]
    if empty_pages:
        print(
            f"Uyarı: {len(empty_pages)} sayfada metin yok. "
            "Tam OCR için: UNSTRUCTURED_OCR=1 ve pip install -r requirements-ocr.txt."
        )

print(f"Bulunan sayfa sayısı: {len(pages)}")
print("İlk sayfanın kısa örnek metni:")
print((pages[0].page_content or "")[:400] if pages else "Metin yok")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = text_splitter.split_documents(pages)
print(f"Kitap toplam {len(chunks)} mantıksal parçaya bölündü.")

print("Vektörleştirme başlatılıyor...")
embeddings = HuggingFaceEmbeddings(model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

print("İşlem tamam! Veriler başarıyla ChromaDB'ye kaydedildi.")
