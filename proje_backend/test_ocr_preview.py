"""Belirli sayfa araligini Docling ile donusturup markdown yazar."""
import sys
from pathlib import Path

from docling_ocr import clean_markdown, create_converter, profile_pdf

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = _PROJECT_ROOT / "datasets" / "tyt_matematik.pdf"
OUT_PATH = _PROJECT_ROOT / "datasets" / "tyt_matematik_preview_10p.md"

if len(sys.argv) >= 3:
    PAGE_START, PAGE_END = int(sys.argv[1]), int(sys.argv[2])
elif len(sys.argv) == 2:
    PAGE_START, PAGE_END = 1, int(sys.argv[1])
else:
    PAGE_START, PAGE_END = 1, 10


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    profile = profile_pdf(PDF_PATH)
    needs_ocr = profile.batch_needs_ocr(PAGE_START, PAGE_END)
    print(f"Donusturuluyor: sayfa {PAGE_START}-{PAGE_END} | OCR={needs_ocr}")

    converter = create_converter(PDF_PATH)
    result = converter.convert(source=str(PDF_PATH), page_range=(PAGE_START, PAGE_END))
    md = clean_markdown(result.document.export_to_markdown(image_placeholder=""))
    OUT_PATH.write_text(md, encoding="utf-8")
    print(f"Kaydedildi: {OUT_PATH} ({len(md)} karakter)")


if __name__ == "__main__":
    main()
