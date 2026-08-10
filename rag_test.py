"""
RAG (Retrieval-Augmented Generation) version of your agent.

New concept here: instead of only searching the live web, the agent can now
search a LOCAL DOCUMENT you provide (afut_university_info.txt) and answer
questions using its actual content.

How it works, step by step:
1. On startup, we read the document, split it into small chunks
2. Each chunk gets converted into an "embedding" (a list of numbers that
   represents its meaning) using a local embedding model
3. All chunks + embeddings get stored in ChromaDB (a local vector database)
4. When the model calls search_documents(query), we embed the QUESTION
   the same way, and ask ChromaDB: "which stored chunks are most similar
   in meaning to this question?"
5. Those chunks get returned to the model as context to answer from
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# --- Step 1: Load and chunk the document ---

DOCUMENT_PATH = "afut_university_info.txt"

with open(DOCUMENT_PATH, "r", encoding="utf-8") as f:
    full_text = f.read()

# Simple chunking: split by paragraphs (blank lines), which works well
# for a document that's already organized into clear sections.
raw_chunks = [chunk.strip() for chunk in full_text.split("\n\n") if chunk.strip()]

print(f"Loaded document, split into {len(raw_chunks)} chunks.")

# --- Step 2 & 3: Embed chunks and store them in ChromaDB ---

# This uses a small, free, local embedding model (downloads automatically
# the first time you run this — no API key needed).
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

chroma_client = chromadb.Client()  # in-memory — resets each time you restart

# Delete the collection if it already exists (avoids duplicate errors on restart)
try:
    chroma_client.delete_collection("university_docs")
except Exception:
    pass

collection = chroma_client.create_collection(
    name="university_docs",
    embedding_function=embedding_fn
)

collection.add(
    documents=raw_chunks,
    ids=[f"chunk_{i}" for i in range(len(raw_chunks))]
)

print(f"Stored {len(raw_chunks)} chunks in the vector database.")


def search_documents(query: str, n_results: int = 3) -> str:
    """Find the most relevant chunks of the document for a given question."""
    results = collection.query(query_texts=[query], n_results=n_results)
    chunks = results["documents"][0]
    if not chunks:
        return "No relevant information found in the document."
    return "\n\n---\n\n".join(chunks)


# --- Quick standalone test, before we wire this into the full agent ---

if __name__ == "__main__":
    test_questions = [
        "How much does the BS Computer Science program cost per semester?",
        "Who is the Vice Chancellor?",
        "When was the AI degree program launched?",
    ]

    for q in test_questions:
        print(f"\n\n=== Question: {q} ===")
        retrieved = search_documents(q)
        print("--- Retrieved chunks ---")
        print(retrieved)