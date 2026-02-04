import pickle

# ==========================================
# LOAD PKL FILES
# ==========================================

with open("chroma_performance_results.pkl", "rb") as f:
    chroma_data = pickle.load(f)

with open("pinecone_performance_results.pkl", "rb") as f:
    pinecone_data = pickle.load(f)


# ==========================================
# EXTRACT VALUES
# ==========================================

bm25_time = round(chroma_data["BM25_time"])

chroma_models = chroma_data["models"]
pinecone_models = pinecone_data["models"]


def list_to_dict(lst):
    return {name: round(time) for name, time in lst}


chroma_dict = list_to_dict(chroma_models)
pinecone_dict = list_to_dict(pinecone_models)


# ==========================================
# QUALITY LOGIC
# ==========================================

def quality(time_ms):
    if time_ms <= 30:
        return "Very High"
    elif time_ms <= 60:
        return "High"
    else:
        return "Medium"


# ==========================================
# COMPUTE AVERAGES
# ==========================================

chroma_avg = round(sum(chroma_dict.values()) / len(chroma_dict))
pinecone_avg = round(sum(pinecone_dict.values()) / len(pinecone_dict))


# ==========================================
# FINAL RESULT STRUCTURE
# ==========================================

final_result = {
    "BM25": {
        "avg_latency_ms": bm25_time,
        "accuracy_level": "Medium"
    },
    "HNSW_ChromaDB": {
        "avg_latency_ms": chroma_avg,
        "accuracy_level": quality(chroma_avg)
    },
    "HNSW_Pinecone": {
        "avg_latency_ms": pinecone_avg,
        "accuracy_level": quality(pinecone_avg)
    },
    "model_wise": {
        "ChromaDB": chroma_dict,
        "Pinecone": pinecone_dict
    }
}


# ==========================================
# SAVE FINALRESULT.PKL
# ==========================================

with open("finalresult.pkl", "wb") as f:
    pickle.dump(final_result, f)

print("\n💾 Final comparison saved as finalresult.pkl")


# ==========================================
# PRINT SUMMARY
# ==========================================

print("\n========== Comparative Summary ==========\n")

print(f"{'Configuration':<20} | {'Avg Latency (ms)':<15} | {'Accuracy Level'}")
print("-" * 60)

print(f"{'BM25':<20} | {bm25_time:<15} | Medium")
print(f"{'HNSW (ChromaDB)':<20} | {chroma_avg:<15} | {quality(chroma_avg)}")
print(f"{'HNSW (Pinecone)':<20} | {pinecone_avg:<15} | {quality(pinecone_avg)}")

print("\n========================================\n")


# ==========================================
# OPTIONAL: DETAILED VIEW
# ==========================================

print("---- Detailed Model Comparison ----\n")

print(f"{'Model':<15} | {'Chroma(ms)':<12} | {'Pinecone(ms)'}")
print("-" * 45)

for model in chroma_dict:
    print(f"{model:<15} | {chroma_dict[model]:<12} | {pinecone_dict.get(model,'N/A')}")

print("\n-----------------------------------\n")
