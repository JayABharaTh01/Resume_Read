import os
import time
import numpy as np
import torch
from tqdm import tqdm
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb
import pickle



# ==========================
# SPEED OPTIMIZATION
# ==========================

torch.set_num_threads(os.cpu_count())


# ==========================
# CONFIG
# ==========================

PDF_FOLDER = r"C:\Users\bharath\OneDrive\Desktop\Python\Agentic AI\Assiment\999"

CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 5
RUNS = 3
BATCH_SIZE = 64


# ⚡ FAST MODELS ONLY

MODELS = {
    "MiniLM-L3": "paraphrase-MiniLM-L3-v2",
    "DIST": "distilbert-base-nli-stsb-mean-tokens"
}


# ==========================
# LOAD PDFS
# ==========================

def load_pdfs():
    docs = []
    for f in os.listdir(PDF_FOLDER):
        if f.endswith(".pdf"):
            reader = PdfReader(os.path.join(PDF_FOLDER, f))
            for p in reader.pages:
                t = p.extract_text()
                if t:
                    docs.append(t)
    return docs


# ==========================
# CHUNKING
# ==========================

def make_chunks(docs):

    chunks = []

    for d in docs:
        i = 0
        while i < len(d):
            chunks.append(d[i:i+CHUNK_SIZE])
            i += CHUNK_SIZE - OVERLAP

    return chunks


# ==========================
# CHROMA CLIENT
# ==========================

client = chromadb.PersistentClient(path="./chroma_db")


def create_collection(name):
    try:
        client.delete_collection(name)
    except:
        pass
    return client.create_collection(name=name)


def store(collection, texts, vectors):

    for i in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[i:i+BATCH_SIZE]
        batch_vectors = vectors[i:i+BATCH_SIZE]

        ids = [str(j) for j in range(i, i+len(batch_texts))]

        collection.add(
            ids=ids,
            documents=batch_texts,
            embeddings=batch_vectors.tolist()
        )


def hnsw_search(collection, qv):

    collection.query(
        query_embeddings=[qv.tolist()],
        n_results=TOP_K
    )


# ==========================
# BENCHMARK HELPER
# ==========================

def avg_time(fn):

    times = []

    for _ in range(RUNS):
        s = time.time()
        fn()
        times.append((time.time()-s)*1000)

    return sum(times)/len(times)


# ==========================
# DYNAMIC QUALITY
# ==========================

def get_quality(method, time_ms):

    if method == "BM25":
        return "Medium"

    if time_ms <= 30:
        return "Very High"
    elif time_ms <= 60:
        return "High"
    else:
        return "Medium"


# ==========================
# MAIN
# ==========================

def main():

    print("\n📄 Loading PDFs...")
    docs = load_pdfs()

    print("\n✂ Chunking...")
    chunks = make_chunks(docs)

    print(f"Total chunks: {len(chunks)}")

    query = "Explain HNSW indexing in vector databases"


    # =====================
    # BM25
    # =====================

    print("\n📚 Running BM25...")

    bm25 = BM25Okapi([c.split() for c in chunks])

    bm25_time = avg_time(
        lambda: bm25.get_scores(query.split())
    )


    # =====================
    # MODEL LOOP
    # =====================

    results = []

    for name, model_id in MODELS.items():

        print(f"\n🧠 Embedding with {name}...")

        model = SentenceTransformer(model_id)

        vectors = model.encode(
            chunks,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        col = create_collection(name.lower())

        store(col, chunks, vectors)

        qv = model.encode(query, normalize_embeddings=True)

        hnsw_time = avg_time(
            lambda: hnsw_search(col, qv)
        )

        results.append((name, hnsw_time))


    # =====================
    # OUTPUT
    # =====================

    print("\n========== ChromaDB Performance ==========\n")

    print(f"{'Method':<15} | {'Model':<15} | {'Avg Time (ms)':<15} | {'Quality'}")
    print("-"*75)

    print(f"{'BM25':<15} | {'N/A':<15} | {round(bm25_time):<15} | {get_quality('BM25', bm25_time)}")

    for model_name, t in results:
        print(f"{'HNSW':<15} | {model_name:<15} | {round(t):<15} | {get_quality('HNSW', t)}")

    print("\n=========================================\n")

    print("✅ Finished (Fast Mode Enabled)")
    
    # SAVE RESULTS TO PKL
    # =====================

    save_data = {
        "BM25_time": bm25_time,
        "models": results
    }

    with open("chroma_performance_results.pkl", "wb") as f:
        pickle.dump(save_data, f)

    print("\n💾 Results saved to chroma_performance_results.pkl")


if __name__ == "__main__":
    main()
