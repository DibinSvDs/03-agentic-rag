# ingest.py

import os
import chromadb
import ollama


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

# Folder containing our source documents.
DOCUMENTS_FOLDER = "documents"

# Folder where ChromaDB will permanently store the vectors.
VECTOR_DB_FOLDER = "vector_db"

# The local embedding model we are using.
EMBEDDING_MODEL = "nomic-embed-text"


# ---------------------------------------------------------
# 2. Create the ChromaDB client
# ---------------------------------------------------------

# PersistentClient means the vector database is saved
# to disk instead of disappearing when the program ends.
client = chromadb.PersistentClient(
    path=VECTOR_DB_FOLDER
)


# Create a collection to store our document chunks.
collection = client.get_or_create_collection(
    name="company_documents"
)


# ---------------------------------------------------------
# 3. Read documents
# ---------------------------------------------------------

documents = []

for filename in os.listdir(DOCUMENTS_FOLDER):

    # We only want text files.
    if not filename.endswith(".txt"):
        continue

    filepath = os.path.join(
        DOCUMENTS_FOLDER,
        filename
    )

    # Open the file and read its contents.
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()

    documents.append(
        {
            "filename": filename,
            "text": text
        }
    )


# ---------------------------------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------
# 4. Create chunks
# ---------------------------------------------------------

import re

chunks = []

for document in documents:

    text = document["text"]
    filename = document["filename"]

    # Find numbered sections such as:
    # "1. LEAVE AND TIME OFF"
    # "2. WORKING HOURS AND ATTENDANCE"
    #
    # Each section will become one chunk.
    sections = re.split(
        r"\n(?=\d+\. )",
        text
    )

    for i, section in enumerate(sections):

        # Remove unnecessary whitespace.
        section = section.strip()

        # Ignore empty sections.
        if not section:
            continue

        chunks.append(
            {
                "id": f"{filename}_{i}",
                "text": section,
                "source": filename
            }
        )
# 5. Create embeddings
# ---------------------------------------------------------

for chunk in chunks:

    # Send the chunk to Ollama's embedding model.
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=chunk["text"]
    )

    embedding = response["embedding"]

    # Store the chunk and its embedding in ChromaDB.
    collection.upsert(
        ids=[chunk["id"]],
        documents=[chunk["text"]],
        embeddings=[embedding],
        metadatas=[
            {
                "source": chunk["source"]
            }
        ]
    )


print(f"Added {len(chunks)} chunks to ChromaDB.")