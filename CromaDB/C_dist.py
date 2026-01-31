import os
import time
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------------
model = SentenceTransformer("sentence-transformers/all-distilroberta-v1")
embedding_dim = model.get_sentence_embedding_dimension()

print("Embedding dimension:", embedding_dim)

# ---------------------------------------------------------
# Initialize ChromaDB (Modern Persistent Client)
# ---------------------------------------------------------
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"     # folder where vectors are stored
)

collection_name = "resume_search"

# Delete old collection if exists
try:
    chroma_client.delete_collection(collection_name)
    print("Old collection deleted")
except:
    pass

# Create new collection
collection = chroma_client.create_collection(
    name=collection_name
)

print("ChromaDB collection created!")

# ---------------------------------------------------------
# PDF Text Extraction Function
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
    batch_size=16,        # increase to 32/64 if RAM allows
    show_progress_bar=True
).tolist()

print("Embedding vector length:", len(embeddings[0]))

# ---------------------------------------------------------
# Insert into ChromaDB (Batch Safe)
# ---------------------------------------------------------
print("Storing vectors in ChromaDB...")

batch_size = 100   # you can increase to 200–300

for i in range(0, len(doc_ids), batch_size):

    batch_ids = doc_ids[i:i + batch_size]
    batch_texts = doc_texts[i:i + batch_size]
    batch_embeddings = embeddings[i:i + batch_size]

    collection.add(
        ids=batch_ids,
        documents=batch_texts,
        embeddings=batch_embeddings
    )

    print(f"Inserted batch {i // batch_size + 1}")

print("All vectors stored locally!")

# ---------------------------------------------------------
# Queries for performance test
# ---------------------------------------------------------
queries = {
    "Sentence 1": "data engineering resume azure data factory databricks",
    "Sentence 2": "data science machine learning langchain gen ai agentic ai",
    "Sentence 3": "python developer machine learning projects"
}

results_table = []

# ---------------------------------------------------------
# Run Benchmark
# ---------------------------------------------------------
for label, query_text in queries.items():

    print(f"\nRunning {label}...")

    # Embedding timing
    embed_start = time.time()
    query_embedding = model.encode(query_text).tolist()
    embed_end = time.time()

    # ChromaDB search timing
    search_start = time.time()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    search_end = time.time()

    embedding_time = embed_end - embed_start
    chroma_time = search_end - search_start
    total_time = search_end - embed_start

    results_table.append({
        "Query": label,
        "Embedding Time": round(embedding_time, 4),
        "ChromaDB Time": round(chroma_time, 4),
        "Total Time": round(total_time, 4)
    })

# ---------------------------------------------------------
# Print Results
# ---------------------------------------------------------
print("\n================ PERFORMANCE COMPARISON ================\n")

print(f"{'Query':<12} | {'Embedding (s)':<14} | {'ChromaDB (s)':<14} | {'Total (s)':<10}")
print("-" * 65)

for row in results_table:
    print(
        f"{row['Query']:<12} | "
        f"{row['Embedding Time']:<14} | "
        f"{row['ChromaDB Time']:<14} | "
        f"{row['Total Time']:<10}"
    )

print("\n=======================================================\n")
