import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Kendi yazdığımız RAG (Retrieval-Augmented Generation) fonksiyonlarını içe aktarıyoruz
from rag import ask, ask_stream, chroma_persist_dir

# .env dosyasındaki değişkenleri (API anahtarları vb.) yüklüyoruz
load_dotenv()

# --- Veri Modelleri (Pydantic) ---

class ChatRequest(BaseModel):
    """Kullanıcıdan gelen mesajın yapısını tanımlar."""
    message: str = Field(..., min_length=1, max_length=4000)

class ChatResponse(BaseModel):
    """Normal (akışsız) cevaplar için dönüş yapısı."""
    response: str

class HealthResponse(BaseModel):
    """Sistem durumunu kontrol etmek için kullanılan yapı."""
    status: str
    chroma_dir: str
    chroma_ready: bool

# --- Uygulama Yaşam Döngüsü ---

@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Uygulama başladığında ve kapandığında çalışan bölümdür.
    Burada ağır modelleri uygulama henüz istek almadan belleğe yüklüyoruz.
    """
    print("Modeller ve veritabanı ön yükleniyor, lütfen bekleyin...")
    try:
        from rag import get_embeddings, get_vectorstore, _get_llm
        # Modelleri çağırarak hafızaya yüklenmelerini tetikliyoruz (LRU Cache sayesinde orada kalacaklar)
        get_embeddings()    # Cümleleri sayısal vektöre çeviren model
        get_vectorstore()   # Kitap verilerinin olduğu veritabanı
        _get_llm()          # Yapay zeka (Gemini/GPT) motoru
        print("Ön yükleme başarıyla tamamlandı. Sistem hazır!")
    except Exception as e:
        print(f"Ön yükleme sırasında uyarı/hata: {e}")
    yield

# FastAPI uygulamasını oluşturuyoruz
app = FastAPI(title="AI Proje Backend", lifespan=lifespan)

# --- Güvenlik ve CORS Ayarları ---
# Flutter gibi farklı platformlardan gelen isteklere izin veriyoruz
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoint'ler (Yollar) ---

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Sistemin çalışıp çalışmadığını ve veritabanının yerini kontrol eder."""
    chroma_dir = str(chroma_persist_dir())
    ready = (chroma_persist_dir() / "chroma.sqlite3").exists()
    return HealthResponse(status="ok", chroma_dir=chroma_dir, chroma_ready=ready)

@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest) -> ChatResponse:
    """
    Klasik (tek parça) cevap döndüren endpoint.
    Yapay zeka cevabı bitirene kadar bekler ve sonucu tek seferde gönderir.
    """
    try:
        answer = ask(body.message)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return ChatResponse(response=answer)

@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """
    Akışlı (streaming) cevap döndüren endpoint.
    Yapay zeka cevap ürettikçe kelime kelime Flutter tarafına gönderir.
    Daha hızlı bir kullanıcı deneyimi sağlar.
    """
    return StreamingResponse(ask_stream(body.message), media_type="text/event-stream")

# --- Uygulamayı Başlatma ---
if __name__ == "__main__":
    import uvicorn
    # Çevresel değişkenlerden host ve port bilgilerini alıyoruz
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    # Sunucuyu başlatıyoruz
    uvicorn.run("main:app", host=host, port=port, reload=True)
