# app/services/manager.py
from typing import Optional, List, Dict
from app.services.scraper import scrape_wikipedia
from app.services.rag_engine import ingest_document, retrieve_context, retrieve_images
from app.services.llm_service import generate_answer

# In-memory mapping of session_id → {"title": str, "images": list}
active_articles: Dict[str, Dict] = {}

def get_active_article_title(session_id: str) -> Optional[str]:
    # Check in-memory first
    entry = active_articles.get(session_id)
    if entry:
        return entry.get("title")

    # Fallback: recover from DB metadata (e.g. after backend restart)
    try:
        from app.services.rag_engine import vector_store
        docs = vector_store.similarity_search("", k=1, filter={"session_id": session_id})
        if docs:
            title = docs[0].metadata.get("article_title")
            if title:
                active_articles[session_id] = {"title": title, "images": []}
                return title
    except Exception as e:
        print(f"[DEBUG] ---> Error recovering session metadata from DB: {e}")
    return None

def ingest_wikipedia_by_query(url: str, session_id: str) -> str:
    # 1. Scrape: paragraphs + tables as text, plus image list
    data = scrape_wikipedia(url)
    title = data["title"]
    images = data.get("images", [])

    # 2. Track in-memory
    active_articles[session_id] = {"title": title, "images": images}

    # 3. Ingest text chunks + image caption embeddings
    ingest_document(data["text"], session_id, title, images=images)

    # 4. Return the article title so the frontend can display it
    return title

def answer_user_query(query: str, session_id: str) -> Dict:
    """
    Returns a dict:
        {"answer": str, "images": list[{url, caption}]}
    """
    # 1. Get the current active article title
    title = get_active_article_title(session_id)
    if not title:
        return {
            "answer": "No active Wikipedia article found for this session. Please search and ingest an article first!",
            "images": [],
        }

    # 2. Retrieve text/table context chunks for the LLM
    chunks = retrieve_context(query, session_id, title)
    if not chunks:
        return {
            "answer": f"No context retrieved from the database for the article '{title}'. Please ingest it again.",
            "images": [],
        }

    # 3. Retrieve candidate images (uses loose semantic search)
    candidate_images = retrieve_images(query, session_id, title)

    # 4. Generate the answer and strictly filter images via LLM JSON response
    response_data = generate_answer(query, chunks, title, candidate_images)
    answer = response_data.get("answer", "")
    relevant_urls = set(response_data.get("relevant_image_urls", []))

    # 5. Filter the original candidate images down to only those the LLM explicitly kept
    final_images = [img for img in candidate_images if img["url"] in relevant_urls]

    return {"answer": answer, "images": final_images}