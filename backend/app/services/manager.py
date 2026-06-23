# app/services/manager.py
from app.services.scraper import scrape_wikipedia
from app.services.rag_engine import ingest_document, retrieve_context
from app.services.llm_service import generate_answer # Or whatever your actual LLM function is named

def ingest_wikipedia_by_query(url: str, session_id: str):
    # 1. Scrape the page text and true article title
    data = scrape_wikipedia(url)
    
    # 2. Fragment, filter, and ingest using the session_id tracking token
    ingest_document(data["text"], session_id)
    
    # 3. Return the article title so the frontend can display it
    return data["title"]

def answer_user_query(query: str, session_id: str):
    # 1. Pull context ONLY from this specific user session's partition
    context = retrieve_context(query, session_id)
    
    if not context:
        return "No loaded Wikipedia context found for this session. Please search an article first!"
        
    # 2. Feed the query and isolated context to your LLM generator
    answer = generate_answer(query, context)
    return answer