# app/services/manager.py
from typing import Optional
from app.services.scraper import scrape_wikipedia
from app.services.rag_engine import ingest_document, retrieve_context
from app.services.llm_service import generate_answer

# In-memory mapping of session_id to current article title
active_articles = {}

def get_active_article_title(session_id: str) -> Optional[str]:
    # Check in-memory first
    if session_id in active_articles:
        return active_articles[session_id]
    
    # Fallback to checking the database metadata (e.g. if backend restarted)
    try:
        from app.services.rag_engine import vector_store
        docs = vector_store.similarity_search("", k=1, filter={"session_id": session_id})
        if docs:
            title = docs[0].metadata.get("article_title")
            if title:
                active_articles[session_id] = title
                return title
    except Exception as e:
        print(f"[DEBUG] ---> Error recovering session metadata from DB: {e}")
    return None

def ingest_wikipedia_by_query(url: str, session_id: str):
    # 1. Scrape the page text and true article title
    data = scrape_wikipedia(url)
    title = data["title"]
    
    # 2. Track the active article title for this session
    active_articles[session_id] = title
    
    # 3. Fragment, filter, and ingest using both session_id and title
    ingest_document(data["text"], session_id, title)
    
    # 4. Return the article title so the frontend can display it
    return title

def answer_user_query(query: str, session_id: str):
    # 1. Get the current active article title for this session
    title = get_active_article_title(session_id)
    if not title:
        return "No active Wikipedia article found for this session. Please search and ingest an article first!"
        
    # 2. Pull context ONLY from this specific user session's partition and active article
    context = retrieve_context(query, session_id, title)
    if not context:
        return f"No context retrieved from the database for the article '{title}'. Please ingest it again."
        
    # 3. Feed the query and isolated context to your LLM generator
    answer = generate_answer(query, context)
    return answer