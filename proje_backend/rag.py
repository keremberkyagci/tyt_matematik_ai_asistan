import os
from functools import lru_cache
from pathlib import Path

# Bazı kütüphanelerin uyarı mesajlarını gizlemek için
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings

# Dosya yolları tanımlamaları
_BACKEND_DIR = Path(__file__).resolve().parent
_DEFAULT_CHROMA_DIR = _BACKEND_DIR.parent / "datasets" / "chroma_db"
# Türkçe metinleri anlamak için kullanılan açık kaynaklı embedding modeli
EMBEDDING_MODEL = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"

# --- Yapay Zekaya Verilen Komut (Prompt) ---
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Sen TYT Matematik konularında yardımcı bir öğretmen asistanısın. "
            "Önemli Kural: Matematiksel ifadeleri asla LaTeX ($...$) formatında yazma. "
            "Bunun yerine herkesin anlayabileceği düz metin formatını kullan. "
            "Örneğin: a^n, karekök(x), a/b, eşit değildir gibi ifadeler kullan. "
            "Öncelikle sana verilen bağlamdaki (kitap içeriği) bilgileri kullan. "
            "Eğer cevap verdiğin bilgi doğrudan kitapta varsa normal şekilde cevap ver. "
            "Eğer kitapta yoksa genel matematik bilgini kullan ama sonuna mutlaka "
            "'\n\n⚠️ Not: Bu bilgi çalışma kitabında yer almamaktadır.' ekle.",
        ),
        ("human", "Bağlam:\n{context}\n\nSoru: {question}"),
    ]
)

def chroma_persist_dir() -> Path:
    """Veritabanının diskteki konumunu belirler."""
    raw = os.environ.get("CHROMA_PERSIST_DIR", "").strip()
    return Path(raw).expanduser().resolve() if raw else _DEFAULT_CHROMA_DIR.resolve()

def _format_docs(docs: list[Document]) -> str:
    """Veritabanından gelen dokümanları metin haline getirir."""
    if not docs: return "İlgili kaynak bulunamadı."
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

@lru_cache(maxsize=1)
def get_embeddings():
    """Embedding modelini belleğe bir kez yükler ve orada tutar."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def get_vectorstore() -> Chroma:
    """Vektör veritabanını (kitap içeriğini) belleğe bir kez yükler."""
    persist_dir = chroma_persist_dir()
    if not persist_dir.exists():
        raise FileNotFoundError(f"Veritabanı bulunamadı: {persist_dir}")
    return Chroma(persist_directory=str(persist_dir), embedding_function=get_embeddings())

@lru_cache(maxsize=1)
def get_retriever():
    """Sorulan soruya en yakın 4 dökümanı bulacak olan arama motoru."""
    return get_vectorstore().as_retriever(search_kwargs={"k": 4})

@lru_cache(maxsize=1)
def _get_llm():
    """
    Yapay zeka modelini (Gemini veya OpenAI) yapılandırır.
    Önce Gemini anahtarını kontrol eder, yoksa OpenAI'a bakar.
    """
    g_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if g_key and "YOUR" not in g_key:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=g_key,
            timeout=60,
            temperature=0.7
        )

    o_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if o_key and o_key.startswith("sk-"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", api_key=o_key, timeout=30)

    return None

def ask(question: str) -> str:
    """Soruyu alır, veritabanında arama yapar ve yapay zekadan tam bir cevap döndürür."""
    text = question.strip()
    if not text: return "Lütfen bir soru yazın."

    try:
        llm = _get_llm()
        if llm:
            retriever = get_retriever()
            # LangChain Chain Yapısı:
            # 1. Dokümanları bul ve formatla (context)
            # 2. Soruyu direkt geçir (question)
            # 3. Prompt içine yerleştir
            # 4. LLM'e gönder
            # 5. Çıktıyı metin olarak al
            chain = (
                {"context": retriever | _format_docs, "question": RunnablePassthrough()}
                | PROMPT
                | llm
                | StrOutputParser()
            )
            return chain.invoke(text)
    except Exception as exc:
        print(f"AI Hatası detay: {exc}")

    return "Üzgünüm, şu an bir hata oluştu."

def ask_stream(question: str):
    """
    Soruyu alır ve cevabı akış (generator) olarak döndürür.
    Bu sayede Flutter kelime kelime ekrana yazdırabilir.
    """
    text = question.strip()
    if not text:
        yield "Lütfen bir soru yazın."
        return

    try:
        llm = _get_llm()
        if llm:
            retriever = get_retriever()
            chain = (
                {"context": retriever | _format_docs, "question": RunnablePassthrough()}
                | PROMPT
                | llm
                | StrOutputParser()
            )
            # chain.stream kullanarak parçalar halinde gönderiyoruz
            for chunk in chain.stream(text):
                yield chunk
            return
    except Exception as exc:
        print(f"AI Hatası detay: {exc}")

    yield "⚠️ Bir hata oluştu."
