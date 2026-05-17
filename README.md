# 🎓 TYT Matematik AI Asistanı

Bu proje, TYT Matematik hazırlık sürecindeki öğrencilere yardımcı olmak amacıyla geliştirilmiş, **RAG (Retrieval-Augmented Generation)** tabanlı bir yapay zeka asistanıdır. Öğrenciler, sistem içerisindeki ders kitaplarından beslenen yapay zekaya sorularını sorabilir, konu anlatımı alabilir ve takıldıkları yerlerde anlık yanıtlar alabilirler.

## 🚀 Özellikler

-   **RAG Tabanlı Yanıt Sistemi:** Yapay zeka, sadece genel bilgisiyle değil, sisteme yüklenen TYT Matematik kitaplarındaki spesifik verilerle cevap verir.
-   **Akışlı Yanıt (Streaming):** Yapay zeka cevapları kelime kelime, gerçek zamanlı olarak ekrana yansır.
-   **LaTeX Desteği:** Matematiksel ifadeler ve formüller görsel olarak şık bir formatta sunulur.
-   **Hızlı Yanıt:** Backend tarafındaki cache mekanizması ve model ön yükleme (lifespan) sayesinde yüksek performans sunar.
-   **Modern Arayüz:** Flutter ile geliştirilmiş, kullanıcı dostu ve gece moduna uygun karanlık tema.

## 🛠️ Teknoloji Yığını

### Backend (Python)
-   **FastAPI:** Yüksek performanslı API sunucusu.
-   **LangChain:** RAG yapısını ve yapay zeka zincirlerini yönetmek için.
-   **ChromaDB:** Vektör veritabanı olarak doküman takibi.
-   **Gemini 3 Flash:** Google'ın hızlı ve verimli yapay zeka modeli.
-   **HuggingFace:** Türkçe metin anlamlandırma (Embedding) modelleri.

### Mobile (Flutter)
-   **Flutter Markdown & Math:** Formülleri ve zengin metinleri görüntülemek için.
-   **HTTP & Streams:** Backend ile asenkron ve akışlı veri iletişimi.

## 📦 Kurulum

### Backend Kurulumu
1. `proje_backend` dizinine gidin.
2. Gerekli kütüphaneleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```
3. `.env` dosyasını oluşturun ve `GOOGLE_API_KEY` değişkeninizi ekleyin.
4. Sunucuyu başlatın:
   ```bash
   python main.py
   ```

### Mobil Kurulum
1. `aiprojeflutter` dizinine gidin.
2. Bağımlılıkları çekin:
   ```bash
   flutter pub get
   ```
3. Uygulamayı çalıştırın:
   ```bash
   flutter run
   ```

## 📝 Kullanım Notları
-   Matematiksel formüller `$formül$` şeklinde yazıldığında otomatik olarak güzelleştirilir.
-   Yapay zeka, kaynak kitapta bulamadığı bir bilgi verdiğinde kullanıcıyı "Bu bilgi çalışma kitabında yer almamaktadır" şeklinde uyarır.

---
**Geliştirici:** [Kerem Berk Yağcı](https://github.com/keremberkyagci)
