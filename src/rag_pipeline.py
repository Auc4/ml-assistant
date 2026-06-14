import os
import requests
import chromadb
from chromadb.api.types import EmbeddingFunction

# Configuración de rutas y endpoints locales
CORPUS_DIR = "data/corpus"
CHROMA_DB_DIR = "data/chroma_db"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:7b-instruct-q4_K_M"
EMBED_MODEL_NAME = "nomic-embed-text"

COLLECTION_NAME = "time_machine_es"
FORCE_REINDEX = False
DEBUG_RETRIEVAL = False

# Inicializar el cliente persistente de ChromaDB
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)


class OllamaEmbeddingFunction(EmbeddingFunction):
    """Clase que hereda de la interfaz nativa de ChromaDB para embeddings locales"""

    def __call__(self, input):
        embeddings = []

        for text in input:
            try:
                response = requests.post(
                    OLLAMA_EMBED_URL,
                    json={
                        "model": EMBED_MODEL_NAME,
                        "prompt": text
                    },
                    timeout=60
                )

                response.raise_for_status()
                embeddings.append(response.json()["embedding"])

            except Exception as e:
                print(f"❌ Error generando embedding: {e}")

        return embeddings

    def name(self) -> str:
        return "OllamaEmbeddingFunction"


embedding_function = OllamaEmbeddingFunction()

if FORCE_REINDEX:
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print(f"🗑️ Colección anterior eliminada: {COLLECTION_NAME}")
    except Exception:
        pass

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)


def clean_gutenberg_text(text):
    """Elimina encabezado y licencia de Project Gutenberg."""
    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***"
    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***"

    start = text.find(start_marker)
    end = text.find(end_marker)

    if start != -1:
        text = text[start + len(start_marker):]

    if end != -1:
        text = text[:end]

    return text.strip()


def load_and_chunk_corpus():
    """Lee el libro y lo divide en chunks con overlap"""
    chunks = []
    chunk_size = 900
    overlap = 200

    if not os.path.exists(CORPUS_DIR) or not os.listdir(CORPUS_DIR):
        print(f"⚠️ Alerta: Coloca el archivo .txt de la novela en {CORPUS_DIR}")
        return []

    filepath = os.path.join(CORPUS_DIR, "la_maquina_del_tiempo.txt")

    if not os.path.exists(filepath):
        print(f"⚠️ No se encontró el archivo: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    text = clean_gutenberg_text(text)

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        chunks.append({
            "text": chunk,
            "id": f"chunk_{start}"
        })

        start += chunk_size - overlap

    return chunks


def index_docs():
    """Indexa los fragmentos en ChromaDB"""
    chunks = load_and_chunk_corpus()

    if not chunks:
        return

    print(f"📦 Indexando {len(chunks)} fragmentos de la novela en ChromaDB...")

    documents = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]

    collection.add(documents=documents, ids=ids)

    print("✅ Base de datos vectorial persistida localmente.")


def add_keyword_matches(query_text, retrieved_docs):
    """
    Refuerzo simple por palabra exacta.
    Útil para nombres propios como Morlocks, Eloi, Weena, etc.
    """
    keywords = ["morlocks", "eloi", "weena", "time traveller"]

    query_lower = query_text.lower()

    matched_keywords = [
        keyword for keyword in keywords
        if keyword in query_lower
    ]

    if not matched_keywords:
        return retrieved_docs

    all_docs = collection.get(include=["documents"])
    all_documents = all_docs.get("documents", [])

    for keyword in matched_keywords:
        keyword_docs = [
            doc for doc in all_documents
            if keyword in doc.lower()
        ]

        for doc in keyword_docs[:3]:
            if doc not in retrieved_docs:
                retrieved_docs.insert(0, doc)

    return retrieved_docs


def query_rag(query_text, use_rag=True, top_k=4):
    """Ejecuta la inferencia consultando con o sin RAG"""
    context = ""

    if use_rag:
        results = collection.query(
            query_texts=[query_text],
            n_results=top_k
        )

        if results["documents"] and results["documents"][0]:
            retrieved_docs = results["documents"][0]

            retrieved_docs = add_keyword_matches(query_text, retrieved_docs)

            if DEBUG_RETRIEVAL:
                    print(
                        f"\n📌 Recuperados "
                        f"{len(retrieved_docs)} fragmentos "
                        f"para la consulta."
                    )

                    for i, doc in enumerate(retrieved_docs[:2]):
                        preview = doc[:120].replace("\n", " ")

                    print(
                        f"Chunk {i+1}: "
                        f"{preview}..."
                    )

            context = "\n---\n".join(retrieved_docs)

    if use_rag:
        prompt_final = f"""You are an expert literary assistant. Use the following extracted context from the novel to answer the reader's question accurately.

If the answer is present in the context, answer directly.
If the answer cannot be deduced from the provided context, state clearly that you do not have that information in the text.

Context from the book:
{context}

Reader's Question: {query_text}

Grounded Answer:"""
    else:
        prompt_final = query_text

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt_final,
        "stream": False,
        "options": {
            "num_gpu": 0,
            "temperature": 0.1,
            "num_ctx": 4096,
            "num_predict": 250
        }
    }

    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json=payload,
            timeout=180
        )

        response.raise_for_status()
        return response.json().get("response", "Error en generación")

    except Exception as e:
        return f"❌ Error en la API de Ollama: {e}"


if __name__ == "__main__":
    if collection.count() == 0:
        index_docs()
    else:
        print(f"📚 Base vectorial cargada con {collection.count()} fragmentos.")

    print("\n🔍 Probando recuperación semántica (RAG)...")
    pregunta_test = "Who are the Morlocks and where do they live?"

    print(f"Pregunta: {pregunta_test}")
    print("-" * 50)

    print(query_rag(pregunta_test, use_rag=True, top_k=8))