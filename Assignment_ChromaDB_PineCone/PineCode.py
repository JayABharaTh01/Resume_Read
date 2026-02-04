import os
import time
import torch
import pickle
from tqdm import tqdm
from pypdf import PdfReader
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric


# =========================================================
# SPEED OPTIMIZATION
# =========================================================

torch.set_num_threads(os.cpu_count())


# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")


# =========================================================
# CONFIG
# =========================================================

PDF_FOLDER = r"C:\Users\bharath\OneDrive\Desktop\Python\Agentic AI\Pinecone\Class Pinecone\Education\999"

CHUNK_SIZE = 500
OVERLAP = 100
TOP_K = 5
RUNS = 3
BATCH_SIZE = 64


MODELS = {
    "MiniLM-L3": "paraphrase-MiniLM-L3-v2",
    "DIST": "distilbert-base-nli-stsb-mean-tokens"
}


# =========================================================
# INIT PINECONE
# =========================================================

pc = Pinecone(api_key=api_key)


# =========================================================
# PDF LOADING
# =========================================================

def load_pdfs(folder_path):

    documents = {}
    doc_id = 1

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".pdf"):

            file_path = os.path.join(folder_path, file_name)
            reader = PdfReader(file_path)

            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "

            if text.strip():
                documents[f"doc_{doc_id}"] = text
                doc_id += 1

    return documents


# =========================================================
# CHUNKING
# =========================================================

def make_chunks(docs):

    chunks = []

    for text in docs:
        i = 0
        while i < len(text):
            chunks.append(text[i:i + CHUNK_SIZE])
            i += CHUNK_SIZE - OVERLAP

    return chunks


# =========================================================
# PINECONE INDEX HANDLING
# =========================================================

def create_index(name, dim):

    if pc.has_index(name):
        print(f"🗑 Deleting existing index: {name}")
        pc.delete_index(name)

        while pc.has_index(name):
            time.sleep(2)

    print(f"📦 Creating index: {name}")

    pc.create_index(
        name=name,
        dimension=dim,
        metric=Metric.COSINE,
        spec=ServerlessSpec(
            cloud=CloudProvider.AWS,
            region=AwsRegion.US_EAST_1
        )
    )

    host = pc.describe_index(name).host
    return pc.Index(host=host)


def store(index, texts, vectors):

    for i in range(0, len(texts), BATCH_SIZE):

        batch_texts = texts[i:i + BATCH_SIZE]
        batch_vectors = vectors[i:i + BATCH_SIZE]

        ids = [str(j) for j in range(i, i + len(batch_texts))]

        payload = [
            (ids[k], batch_vectors[k].tolist(), {"text": batch_texts[k]})
            for k in range(len(batch_texts))
        ]

        index.upsert(vectors=payload)


def pinecone_search(index, qv):

    index.query(
        vector=qv.tolist(),
        top_k=TOP_K,
        include_metadata=False
    )


# =========================================================
# BENCHMARK UTIL
# =========================================================

def avg_time(fn):

    times = []

    for _ in range(RUNS):
        start = time.time()
        fn()
        times.append((time.time() - start) * 1000)

    return sum(times) / len(times)


# =========================================================
# QUALITY SCORE (SIMPLE)
# =========================================================

def get_quality(method, time_ms):

    if method == "BM25":
        return "Medium"

    if time_ms <= 30:
        return "Very High"
    elif time_ms <= 60:
        return "High"
    else:
        return "Medium"


# =========================================================
# MAIN PIPELINE
# =========================================================

def main():

    print("\n📄 Loading PDFs...")
    docs = load_pdfs(PDF_FOLDER)

    print("\n✂ Chunking text...")
    chunks = make_chunks(docs)

    print(f"Total chunks created: {len(chunks)}")

    query = "Explain HNSW indexing in vector databases"


    # ==============================
    # BM25 BASELINE
    # ==============================

    print("\n📚 Running BM25 baseline...")

    bm25 = BM25Okapi([c.split() for c in chunks])

    bm25_time = avg_time(
        lambda: bm25.get_scores(query.split())
    )


    # ==============================
    # EMBEDDING + PINECONE LOOP
    # ==============================

    results = []

    for name, model_id in MODELS.items():

        print(f"\n🧠 Using model: {name}")

        model = SentenceTransformer(model_id)

        vectors = model.encode(
            chunks,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        dim = vectors.shape[1]

        index = create_index(name.lower(), dim)

        print("⬆ Uploading embeddings to Pinecone...")
        store(index, chunks, vectors)

        qv = model.encode(query, normalize_embeddings=True)

        pinecone_time = avg_time(
            lambda: pinecone_search(index, qv)
        )

        results.append((name, pinecone_time))


    # ==============================
    # RESULTS
    # ==============================

    print("\n========== PINECONE PERFORMANCE ==========\n")

    print(f"{'Method':<15} | {'Model':<15} | {'Avg Time (ms)':<15} | {'Quality'}")
    print("-" * 75)

    print(
        f"{'BM25':<15} | {'N/A':<15} | {round(bm25_time):<15} | {get_quality('BM25', bm25_time)}"
    )

    for model_name, t in results:
        print(
            f"{'Pinecone':<15} | {model_name:<15} | {round(t):<15} | {get_quality('Pinecone', t)}"
        )

    print("\n=========================================\n")


    # ==============================
    # SAVE BENCHMARK
    # ==============================

    save_data = {
        "BM25_time": bm25_time,
        "models": results
    }

    with open("pinecone_performance_results.pkl", "wb") as f:
        pickle.dump(save_data, f)

    print("💾 Results saved as pinecone_performance_results.pkl")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
