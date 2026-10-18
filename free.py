import os
import shutil
import chromadb
from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# 1. Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "data")
db_dir = os.path.join(script_dir, "chroma_db")


# 2. Local Models (Gemma 2B for speed)
#Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1, request_timeout=600.0)
Settings.llm = Ollama(model="gemma:7b", request_timeout=600.0)
Settings.embed_model = HuggingFaceEmbedding(model_name="qwen3-embedding")

# 3. ChromaDB Connection & Dimension Check
db = chromadb.PersistentClient(path=db_dir)

try:
    chroma_collection = db.get_or_create_collection("my_documents")
    _ = chroma_collection.count()
except Exception:
    print("⚠️ Database dimension mismatch, resetting...")
    del db
    if os.path.exists(db_dir):
        shutil.rmtree(db_dir)
    db = chromadb.PersistentClient(path=db_dir)
    chroma_collection = db.get_or_create_collection("my_documents")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 4. Storage Optimization
if chroma_collection.count() > 0:
    print("🔄 Loading existing documents from ChromaDB (0 Cost)!")
    index = VectorStoreIndex.from_vector_store(
        vector_store, storage_context=storage_context
    )
else:
    print("📥 Embedding documents with local model for the first time...")
    documents = SimpleDirectoryReader(data_dir, recursive=True).load_data()
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context
    )

# 5. Query Engine (Optimized with top_k=2 for speed)
query_engine = index.as_query_engine(
    similarity_top_k=2,
    system_prompt=(
        "Sen kişisel dokümanlarıma hakim bir asistansın. "
        "Sana sorulan soruları sadece verilen belgelere dayanarak "
        "açık, anlaşılır ve Türkçe olarak yanıtlayacaksın."
    ),
)

# 6. Query
query = "Eserlerdeki en acımasız karakteri odağına alarak kim olduğunu ve hangi eserde yer aldığını özetler aracılığıyla bul ve bana söyle. Eğer karakteri bulamazsanız sadece karakterin hangi eserde yer aldığını söyle. Derinlemesine bir karakter analizi yap. Karakterin hikaye boyunca zihnini kurcalayan, onu dönüştüren ana olayı ve bu olayın onda yarattığı baskın duygusal yükü psikolojik boyutuyla ele al. Karakterin yaşadığı bu yoğun duygunun eylemlerine ve diğer karakterlerle olan ilişkilerine nasıl yansıdığını en az 1 adet uzun ve edebi paragraf halinde Tamamen Türkçe olarak ifade et. Unutma her şey Türkçe olmalı bana Türkçe cevap vermelisin !"
print(f"\nQuery: {query}")

response = query_engine.query(query)


print(f"\nFinal Response:\n{response}")