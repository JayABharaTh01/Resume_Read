# import os
# import json
# import chromadb
# from chromadb import PersistentClient

# client=PersistentClient(path="./chroma_db")

# collection=client.get_or_create_collection(
#     name="tickets",
#     metadata={"hnsw:space": "cosine"})

# data_folder="/.data"

# documents=[]
# metadatas=[]
# ids=[]


# for filename in sorted(os.listdir(data_folder)):
#     if filename.endswith(".json"):
#         file_path = os.path.join(data_folder, filename)

#         with open(file_path, "r", encoding="utf-8") as f:
#             ticket_data = json.load(f)
            
#             doc_id=str(ticket_data.get("ticket_id", filename.replace(".json", "")))
            
#             documents_text=f"{ticket_data.get('title', '')} {ticket_data.get('description', '')}"
            
#             documents.append(documents_text)
#             ids.append(doc_id)
            
#             metadatas.append({
#                 "ticket_id": ticket_data.get("ticket_id"),  
#                 "title": ticket_data.get("title"),
#                 "status": ticket_data.get("status"),
#                 "priority": ticket_data.get("priority"),
#                 "category": ticket_data.get("category"),
#                 "department": ticket_data.get("department"),
#                 "assigned_to": ticket_data.get("assigned_to"),
#                 "tags":", ".join(ticket_data.get("tags", [])),
#             })
            
            
# if documents:
#     collection.add(
#         documents=documents,
#         metadatas=metadatas,
#         ids=ids
#     )
#     print(f"✅ Successfully added {len(documents)} tickets to the Chroma collection!")
# else:
#     print("⚠️ No JSON files found in the data folder.")


import os
import json
import chromadb
from chromadb import PersistentClient

# ----------------------------------------
# Initialize Chroma Persistent Client
# ----------------------------------------
client = PersistentClient(
    path="./chroma_data"
)

# ----------------------------------------
# Create or get a collection
# ----------------------------------------
collection = client.get_or_create_collection(
    name="tickets",
    metadata={"hnsw:space": "cosine"}
)

# ----------------------------------------
# Path to data folder
# ----------------------------------------
data_folder = "./data"

documents = []
metadatas = []
ids = []

# ----------------------------------------
# Read all JSON files and prepare data
# ----------------------------------------
for filename in sorted(os.listdir(data_folder)):
    if filename.endswith(".json"):
        file_path = os.path.join(data_folder, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            ticket_data = json.load(f)

        # Use ticket_id as document ID (fallback to filename)
        doc_id = str(ticket_data.get("ticket_id", filename.replace(".json", "")))

        # Create document text
        document_text = f"{ticket_data.get('title', '')} {ticket_data.get('description', '')}"

        documents.append(document_text)
        ids.append(doc_id)

        # Store metadata
        metadatas.append({
            "ticket_id": ticket_data.get("ticket_id"),
            "title": ticket_data.get("title"),
            "status": ticket_data.get("status"),
            "priority": ticket_data.get("priority"),
            "category": ticket_data.get("category"),
            "department": ticket_data.get("department"),
            "assigned_to": ticket_data.get("assigned_to"),
            "tags": ",".join(ticket_data.get("tags", []))
        })

# ----------------------------------------
# Add documents to Chroma
# ----------------------------------------
if documents:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )

    print(f"✅ Successfully added {len(documents)} tickets to the Chroma collection!")
else:
    print("⚠️ No JSON files found in the data folder.")