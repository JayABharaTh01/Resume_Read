import chromadb
from chromadb import PersistentClient

# ----------------------------------------
# Load persistent Chroma client
# ----------------------------------------
client = PersistentClient(path="./chroma_data")

# ----------------------------------------
# Get existing collection
# ----------------------------------------
collection = client.get_collection(name="tickets")

# ----------------------------------------
# User question
# ----------------------------------------
question = "User cannot login and password reset is failing"

import time
starttime = time.perf_counter()
# ----------------------------------------
# Query top 5 similar documents
# ----------------------------------------
results = collection.query(
    query_texts=[question],
    n_results=2
)
endtime = time.perf_counter()
latency = (endtime - starttime) * 1000
print(f"Query Latency: {latency:.2f} ms")
# ----------------------------------------
# Display results
# ----------------------------------------
for i in range(len(results["ids"][0])):
    print(f"\nResult Rank: {i + 1}")
    print("Ticket ID:", results["ids"][0][i])
    print("Document:", results["documents"][0][i])
    print("Metadata:", results["metadatas"][0][i])
    print("Distance:", results["distances"][0][i])

