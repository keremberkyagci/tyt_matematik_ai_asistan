"""
tyt_matematik.pdf dosyasının taranmış (görüntü) mi yoksa metin katmanlı mı
olduğunu PyMuPDF (fitz) ile kontrol eder.
"""

from pathlib import Path

import fitz

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = _PROJECT_ROOT / "datasets" / "tyt_matematik.pdf"
SAMPLE_COUNT = 6
MIN_TEXT_CHARS = 50  # Bu kadar karakterten azsa "metin yok" sayılır


def pick_sample_pages(page_count: int, sample_count: int) -> list[int]:
    """İlk, son ve araya eşit aralıklı sayfalar seç."""
    if page_count <= sample_count:
        return list(range(page_count))

    indices = {0, page_count - 1}
    step = (page_count - 1) / (sample_count - 1)
    for i in range(sample_count):
        indices.add(round(i * step))

    return sorted(indices)[:sample_count]


def analyze_page(page: fitz.Page) -> dict:
    text = page.get_text("text").strip()
    images = page.get_images(full=True)
    blocks = page.get_text("dict")["blocks"]

    text_blocks = sum(1 for b in blocks if b.get("type") == 0)
    image_blocks = sum(1 for b in blocks if b.get("type") == 1)

    char_count = len(text)
    has_text_layer = char_count >= MIN_TEXT_CHARS

    if has_text_layer and images:
        verdict = "KARMA (metin + görüntü)"
    elif has_text_layer:
        verdict = "METİN KATMANLI"
    elif images:
        verdict = "TARANMIŞ / GÖRÜNTÜ (metin katmanı yok)"
    else:
        verdict = "BELİRSİZ (az metin, görüntü de yok)"

    return {
        "char_count": char_count,
        "image_count": len(images),
        "text_blocks": text_blocks,
        "image_blocks": image_blocks,
        "verdict": verdict,
        "text_preview": text[:400] + ("..." if len(text) > 400 else ""),
    }


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF bulunamadı: {PDF_PATH}")

    doc = fitz.open(PDF_PATH)
    page_count = doc.page_count
    sample_pages = pick_sample_pages(page_count, SAMPLE_COUNT)

    print(f"Dosya: {PDF_PATH.name}")
    print(f"Toplam sayfa: {page_count}")
    print(f"Örnek sayfalar (0 tabanlı): {sample_pages}")
    print("=" * 70)

    verdicts: list[str] = []

    for page_idx in sample_pages:
        page = doc[page_idx]
        info = analyze_page(page)
        verdicts.append(info["verdict"])

        print(f"\n--- Sayfa {page_idx + 1} / {page_count} ---")
        print(f"Karakter sayısı : {info['char_count']}")
        print(f"Gömülü görüntü  : {info['image_count']}")
        print(f"Metin blokları  : {info['text_blocks']}")
        print(f"Görüntü blokları: {info['image_blocks']}")
        print(f"Sonuç           : {info['verdict']}")
        print("-" * 40)
        if info["text_preview"]:
            print("Metin önizleme:")
            print(info["text_preview"])
        else:
            print("(Bu sayfadan metin çıkarılamadı)")

    doc.close()

    print("\n" + "=" * 70)
    text_layer_pages = sum(1 for v in verdicts if "METİN" in v or "KARMA" in v)
    print(f"Özet: {text_layer_pages}/{len(verdicts)} örnek sayfada anlamlı metin katmanı var.")

    if text_layer_pages == 0:
        print("Genel değerlendirme: PDF büyük olasılıkla TARANMIŞ (OCR gerekir).")
    elif text_layer_pages == len(verdicts):
        print("Genel değerlendirme: PDF METİN KATMANLI görünüyor.")
    else:
        print("Genel değerlendirme: PDF KARMA — bazı sayfalar metinli, bazıları görüntü tabanlı olabilir.")


if __name__ == "__main__":
    main()
