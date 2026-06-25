# app/services/rag_engine.py
import os
import tempfile
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ✨ FIX: Use the container's globally writeable /tmp partition to bypass permission checks
PERSIST_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
vector_store = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

# ── Session cleanup ───────────────────────────────────────────────────────────

def delete_session_documents(session_id: str):
    print(f"\n[DEBUG] ---> Deleting all vector chunks for Session: {session_id}")
    try:
        vector_store._collection.delete(where={"session_id": session_id})
        print(f"[DEBUG] ---> Cleaned up database chunks for: {session_id}")
    except Exception as e:
        print(f"[DEBUG] ---> Delete operation failed or database empty: {e}")

# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_document(text: str, session_id: str, article_title: str,
                    images: List[Dict[str, str]] = None):
    """
    Ingests an article into the vector store with per-chunk content_type metadata.

    - Paragraph + table text → content_type: "text" (split into 600-char chunks)
    - Each image caption     → content_type: "image" (one doc per image, with image_url)

    All existing docs for this session are wiped first.
    """
    print(f"\n[DEBUG] ---> Starting ingestion for Session: {session_id} | Article: '{article_title}'")

    # 1. Wipe previous session data
    delete_session_documents(session_id)

    all_docs: List[Document] = []

    # 2. Chunk the combined text (paragraphs + tables)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    text_docs = text_splitter.create_documents([text])
    for doc in text_docs:
        doc.metadata = {
            "session_id": session_id,
            "article_title": article_title,
            "content_type": "text",
        }
    all_docs.extend(text_docs)
    print(f"[DEBUG] ---> Created {len(text_docs)} text/table chunk(s)")

    # 3. Embed each image caption as its own document
    if images:
        for img in images:
            caption = img.get("caption", "").strip()
            url = img.get("url", "").strip()
            if caption and url:
                doc = Document(
                    page_content=caption,
                    metadata={
                        "session_id": session_id,
                        "article_title": article_title,
                        "content_type": "image",
                        "image_url": url,
                    },
                )
                all_docs.append(doc)
        image_count = len([d for d in all_docs if d.metadata.get("content_type") == "image"])
        print(f"[DEBUG] ---> Created {image_count} image caption chunk(s)")

    # 4. Commit everything to ChromaDB
    vector_store.add_documents(all_docs)
    print(f"[DEBUG] ---> Successfully stored {len(all_docs)} total chunk(s) for session: {session_id}\n")

# ── Retrieval ─────────────────────────────────────────────────────────────────

def retrieve_context(query: str, session_id: str, article_title: str) -> List[str]:
    """
    Retrieve the top-k text/table chunks for the query.
    Image chunks are explicitly excluded so they never pollute the LLM context.
    """
    print(f"\n[DEBUG] ---> Querying text context for Session: {session_id} | Article: '{article_title}'")
    print(f"[DEBUG] ---> User Query: '{query}'")

    relevant_docs = vector_store.similarity_search(
        query,
        k=5,
        filter={
            "$and": [
                {"session_id": session_id},
                {"article_title": article_title},
                {"content_type": "text"},   # only text/table chunks
            ]
        },
    )

    print(f"[DEBUG] ---> Found {len(relevant_docs)} text chunk(s)")
    for i, doc in enumerate(relevant_docs):
        print(f"    └─ Chunk {i+1}: {doc.page_content[:50]}...")

    return [doc.page_content for doc in relevant_docs]


def retrieve_images(query: str, session_id: str, article_title: str,
                    top_k: int = 3, score_threshold: float = 1.2) -> List[Dict[str, str]]:
    """
    Find image captions semantically similar to the query.
    Returns a list of {url, caption} dicts whose L2 distance < score_threshold.

    A lower L2 distance = more similar. Typical range: 0 (identical) – 2+ (unrelated).
    score_threshold=1.2 keeps only genuinely relevant matches.
    """
    print(f"\n[DEBUG] ---> Querying image captions for Session: {session_id} | Query: '{query}'")

    results_with_scores = vector_store.similarity_search_with_score(
        query,
        k=top_k,
        filter={
            "$and": [
                {"session_id": session_id},
                {"article_title": article_title},
                {"content_type": "image"},
            ]
        },
    )

    matched: List[Dict[str, str]] = []
    for doc, score in results_with_scores:
        print(f"[DEBUG] ---> Image candidate: score={score:.3f} | caption='{doc.page_content[:60]}...'")
        if score <= score_threshold:
            matched.append({
                "url": doc.metadata.get("image_url", ""),
                "caption": doc.page_content,
            })

    print(f"[DEBUG] ---> {len(matched)} image(s) passed the relevance threshold")
    return matched