import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Define where ChromaDB will save its data locally on your Mac
CHROMA_DATA_DIR = "/tmp/chroma_db"

# Use a pure-Python native embedding model (No Ollama server dependency)
# This model is tiny (~45MB) and runs directly inside your Python process safely
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def chunk_text(raw_text: str) -> list:
    """Splits a large string of text into smaller, overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

def store_in_vector_db(chunks: list):
    """
    Takes a list of text chunks, generates their embeddings natively,
    and stores them locally inside ChromaDB.
    """
    vector_store = Chroma(
        collection_name="wikipedia_rag",
        embedding_function=embedding_model,
        persist_directory=CHROMA_DATA_DIR
    )
    vector_store.add_texts(texts=chunks)

def retrieve_relevant_chunks(query: str, k: int = 3) -> list:
    """
    Searches ChromaDB for the top 'k' most relevant text chunks.
    """
    vector_store = Chroma(
        collection_name="wikipedia_rag",
        embedding_function=embedding_model,
        persist_directory=CHROMA_DATA_DIR
    )
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]