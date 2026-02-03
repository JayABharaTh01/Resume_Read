import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import chromadb
from chromadb import PersistentClient
from tabulate import tabulate
import math

# ----------------------------------------
# Load persistent Chroma client
# ----------------------------------------
client = PersistentClient(path="./chroma_db")
collection = client.get_collection(name="tickets")

# ----------------------------------------
# Load documents for BM25 and vector-based methods
# ----------------------------------------
all_documents = []
all_ids = []

results = collection.get()
for doc, doc_id in zip(results["documents"], results["ids"]):
    all_documents.append(doc)
    all_ids.append(doc_id)

# Replace BM25 implementation with match_tickets.py logic
def calculate_bm25(query, tokenized_corpus):
    N = len(tokenized_corpus)
    doc_lengths = [len(doc) for doc in tokenized_corpus]
    avgDL = sum(doc_lengths) / N
    k1 = 1.5
    b = 0.75

    # Build vocabulary
    vocab = set(word for doc in tokenized_corpus for word in doc)

    # Calculate term frequency (TF)
    tf = []
    for doc in tokenized_corpus:
        tf.append({word: doc.count(word) for word in vocab})

    # Calculate document frequency (DF)
    df = {word: sum(1 for doc in tokenized_corpus if word in doc) for word in vocab}

    # Calculate inverse document frequency (IDF)
    idf = {word: math.log((N - df[word] + 0.5) / (df[word] + 0.5) + 1) for word in vocab}

    # Calculate BM25 scores
    scores = []
    for doc in tokenized_corpus:
        score = 0
        for word in query.lower().split():
            if word in doc:
                TF = tf[tokenized_corpus.index(doc)][word]
                numerator = TF * (k1 + 1)
                denominator = TF + k1 * (1 - b + b * len(doc) / avgDL)
                score += idf[word] * (numerator / denominator)
        scores.append(score)

    return scores

# Tokenize documents for BM25
tokenized_corpus = [doc.split() for doc in all_documents]

# ----------------------------------------
# User question
# ----------------------------------------
question = "User cannot login and password reset is failing"
question_tokens = question.split()

# ----------------------------------------
# Benchmarking methods
# ----------------------------------------
benchmark_results = []

# BM25 Search
start_time = time.perf_counter()
bm25_scores = calculate_bm25(question, tokenized_corpus)
bm25_top_indices = np.argsort(bm25_scores)[::-1][:5]
bm25_results = [(all_ids[i], bm25_scores[i]) for i in bm25_top_indices]
end_time = time.perf_counter()
bm25_latency = (end_time - start_time) * 1000
benchmark_results.append(["BM25", "Lexical Search", f"{bm25_latency:.2f} ms"])

# ChromaDB Search
start_time = time.perf_counter()
chromadb_results = collection.query(query_texts=[question], n_results=5)
end_time = time.perf_counter()
chromadb_latency = (end_time - start_time) * 1000
benchmark_results.append(["ChromaDB Search", "Embedding + HNSW", f"{chromadb_latency:.2f} ms"])

# Reintroduce embedding extraction
embeddings = np.array([meta["embedding"] for meta in results["metadatas"] if "embedding" in meta])

# Ensure embeddings are not empty
if embeddings.size == 0:
    raise ValueError("No embeddings found in the metadata. Ensure the collection contains embeddings.")

# Generate embedding for the question (dummy embedding for demonstration)
question_embedding = np.random.rand(1, embeddings.shape[1])

# Re-add vector-based methods to benchmarking
# Cosine Similarity Search
start_time = time.perf_counter()
cosine_scores = cosine_similarity(question_embedding, embeddings)[0]
cosine_top_indices = np.argsort(cosine_scores)[::-1][:5]
cosine_results = [(all_ids[i], cosine_scores[i]) for i in cosine_top_indices]
end_time = time.perf_counter()
cosine_latency = (end_time - start_time) * 1000
benchmark_results.append(["Cosine Similarity", "Vector-based", f"{cosine_latency:.2f} ms"])

# Euclidean Distance Search
start_time = time.perf_counter()
euclidean_scores = euclidean_distances(question_embedding, embeddings)[0]
euclidean_top_indices = np.argsort(euclidean_scores)[:5]
euclidean_results = [(all_ids[i], euclidean_scores[i]) for i in euclidean_top_indices]
end_time = time.perf_counter()
euclidean_latency = (end_time - start_time) * 1000
benchmark_results.append(["Euclidean Distance", "Vector-based", f"{euclidean_latency:.2f} ms"])

# ----------------------------------------
# Display result comparison table
# ----------------------------------------
print("\n📊 Search Methods Average Response Time\n")
print(tabulate(
    benchmark_results,
    headers=["Method", "Search Type", "Average Response Time"],
    tablefmt="grid"
))

# ----------------------------------------
# Display sample results for each method
# ----------------------------------------
print("\n🔍 Sample Results\n")

print("BM25 Results:")
for rank, (doc_id, score) in enumerate(bm25_results, start=1):
    print(f"Rank {rank}: ID={doc_id}, Score={score:.4f}")

print("\nChromaDB Results:")
for rank in range(len(chromadb_results["ids"][0])):
    print(f"Rank {rank + 1}: ID={chromadb_results['ids'][0][rank]}, Distance={chromadb_results['distances'][0][rank]:.4f}")

print("\nCosine Similarity Results:")
for rank, (doc_id, score) in enumerate(cosine_results, start=1):
    print(f"Rank {rank}: ID={doc_id}, Score={score:.4f}")

print("\nEuclidean Distance Results:")
for rank, (doc_id, score) in enumerate(euclidean_results, start=1):
    print(f"Rank {rank}: ID={doc_id}, Distance={score:.4f}")