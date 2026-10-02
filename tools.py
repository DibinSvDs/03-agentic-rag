# tools.py

import chromadb
import ollama


# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------

VECTOR_DB_FOLDER = "vector_db"

EMBEDDING_MODEL = "nomic-embed-text"


# ---------------------------------------------------------
# 2. Connect to ChromaDB
# ---------------------------------------------------------

# Connect to the ChromaDB database that we created
# using ingest.py.
client = chromadb.PersistentClient(
    path=VECTOR_DB_FOLDER
)


# Get the collection containing our document chunks.
collection = client.get_collection(
    name="company_documents"
)


# ---------------------------------------------------------
# 3. Retrieval tool
# ---------------------------------------------------------

def search_company_documents(question):
    """
    Search the company knowledge base for information
    relevant to the user's question.
    """

    # Convert the user's question into an embedding.
    response = ollama.embeddings(
        model=EMBEDDING_MODEL,
        prompt=question
    )

    question_embedding = response["embedding"]


    # Search ChromaDB for the most similar chunks.
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3
    )


        # Get the retrieved document chunks.
    retrieved_documents = results["documents"][0]

    # Combine the chunks into one string.
    context = "\n\n".join(
        retrieved_documents
    )

    # Show exactly what ChromaDB retrieved.
    print("\n[Retrieved context]")
    print(context)

    return context

# Tool definition for Qwen
tools = [
    {
        "type": "function",
        "function": {
            "name": "search_company_documents",
            "description": (
                "Search the company knowledge base for information "
                "about company policies and internal procedures."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to search for in the company documents."
                    }
                },
                "required": ["question"]
            }
        }
    }
]

