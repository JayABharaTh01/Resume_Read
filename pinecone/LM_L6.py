import os
import time
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# ---------------------------------------------------------
# Load ENV variables
# ---------------------------------------------------------
load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    raise ValueError("PINECONE_API_KEY not found in .env file")

# ---------------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embedding_dim = model.get_sentence_embedding_dimension()
print("Embedding dimension:", embedding_dim)

np.save("embedding_dim.npy", embedding_dim)

# ---------------------------------------------------------
# Initialize Pinecone
# ---------------------------------------------------------
pc = Pinecone(api_key=api_key)

index_name = "resume-search-index"

# ---------------------------------------------------------
# Delete Existing Index Safely
# ---------------------------------------------------------
if pc.has_index(index_name):
    print("Deleting existing index...")
    pc.delete_index(index_name)

    while pc.has_index(index_name):
        time.sleep(2)

# ---------------------------------------------------------
# Create Index
# ---------------------------------------------------------
pc.create_index(
    name=index_name,
    dimension=embedding_dim,
    metric=Metric.COSINE,
    spec=ServerlessSpec(
        cloud=CloudProvider.AWS,
        region=AwsRegion.US_EAST_1
    )
)

index = pc.Index(host=pc.describe_index(index_name).host)

print("Index created successfully!")

# ---------------------------------------------------------
# PDF Text Extraction
# ---------------------------------------------------------
def extract_text_from_pdfs(folder_path):
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

# ---------------------------------------------------------
# Load PDFs
# ---------------------------------------------------------
pdf_folder_path = r"C:\Users\bharath\OneDrive\Desktop\Python\Agentic AI\Pinecone\Class Pinecone\Education\999"

start_time = time.time()

documents = extract_text_from_pdfs(pdf_folder_path)

end_time = time.time()

print(f"Text extraction completed in {end_time - start_time:.2f} seconds")
print(f"Total documents loaded: {len(documents)}")

# ---------------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------------
doc_ids = list(documents.keys())
doc_texts = list(documents.values())

print("Generating embeddings...")

embeddings = model.encode(
    doc_texts,
    batch_size=16,
    show_progress_bar=True
).tolist()

vectors = list(zip(doc_ids, embeddings))

# ---------------------------------------------------------
# Batch Upsert into Pinecone
# ---------------------------------------------------------
print("Uploading vectors in batches...")

batch_size = 200

for i in range(0, len(vectors), batch_size):

    batch = vectors[i:i + batch_size]

    index.upsert(vectors=batch)

    print(f"Uploaded batch {i // batch_size + 1}")

print("All vectors uploaded!")

# ---------------------------------------------------------
# Wait until indexing completes
# ---------------------------------------------------------
def wait_until_indexing_complete(idx, expected_count, interval=5):

    while True:
        stats = idx.describe_index_stats()
        current = stats.total_vector_count

        print(f"Indexed: {current}/{expected_count}")

        if current >= expected_count:
            break

        time.sleep(interval)

wait_until_indexing_complete(index, len(documents))

# ---------------------------------------------------------
# Query Performance Test
# ---------------------------------------------------------
queries = {
    "Sentence 1": "data engineering resume azure data factory databricks",
    "Sentence 2": "data science machine learning langchain generative ai",
    "Sentence 3": "python developer cloud automation pipelines"
}

results_table = []

for label, query_text in queries.items():

    print(f"\nRunning {label} query...")

    # Embedding time
    embed_start = time.time()
    query_embedding = model.encode(query_text).tolist()
    embed_end = time.time()

    # Pinecone search time
    search_start = time.time()

    results = index.query(
        vector=query_embedding,
        top_k=5,
        include_values=False
    )

    search_end = time.time()

    embedding_time = embed_end - embed_start
    pinecone_time = search_end - search_start
    total_time = search_end - embed_start

    results_table.append({
        "Query": label,
        "Embedding": round(embedding_time, 4),
        "Pinecone": round(pinecone_time, 4),
        "Total": round(total_time, 4)
    })

# ---------------------------------------------------------
# Print Performance Results
# ---------------------------------------------------------
print("\n================ PERFORMANCE COMPARISON ================\n")

print(f"{'Query':<12} | {'Embedding (s)':<14} | {'Pinecone (s)':<14} | {'Total (s)':<10}")
print("-" * 65)

for row in results_table:
    print(
        f"{row['Query']:<12} | "
        f"{row['Embedding']:<14} | "
        f"{row['Pinecone']:<14} | "
        f"{row['Total']:<10}"
    )

print("\n========================================================")
