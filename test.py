import os
import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore

os.environ["OPENAI_API_KEY"] = "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"

# --- 1. DOSYA YOLLARINI KESİNLEŞTİRME ---
# test.py dosyasının tam bulunduğu klasörü baz alıyoruz
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
db_dir = os.path.join(script_dir, "chroma_db")

print(f"Okunan veri klasörü: {data_dir}")

# --- 3. MODELİ GPT-3.5-TURBO OLARAK BELİRLİYORUZ ---
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
# ----------------------------------------------------

print(f"ChromaDB klasörü: {db_dir}")

# 2. Belgeleri kesin olarak doğru klasörden oku
documents = SimpleDirectoryReader(data_dir, recursive=True).load_data()

# 3. ChromaDB'yi sabitlediğimiz yolda başlat
db = chromadb.PersistentClient(path=db_dir)

# Her seferinde temiz bir başlangıç yapmak için eski koleksiyonu siliyoruz
# try:
#   db.delete_collection("my_documents")
# except Exception:
#   pass

chroma_collection = db.get_or_create_collection("my_documents")

# 4. Storage context ve index oluşturma
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Belgeler indeksleniyor...")
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# 5. Sorgu motoru (Türkçe ve kişiselasistan ayarıyla)
query_engine = index.as_query_engine(
    system_prompt=(
        "Sen kişisel dokümanlarıma hakim bir asistansın. "
        "Sana sorulan soruları sadece verilen belgelere dayanarak "
        "açık, anlaşılır ve Türkçe olarak yanıtlayacaksın."
    )
)

response = query_engine.query("Haksızlık yapanlar kimler?")
print("\nYanıt:")
print(response)