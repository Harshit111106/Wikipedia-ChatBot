import requests
from app.services.scraper import scrape_wikipedia
from app.services.rag_engine import chunk_text, store_in_vector_db, retrieve_relevant_chunks
from app.services.llm_service import generate_answer

def search_wikipedia_topic(query: str) -> tuple:
    """Hits the Wikipedia OpenSearch API to handle typos and find the best matching topic."""
    # We switch to the opensearch endpoint which handles fuzzy matching/typos perfectly
    search_url = "https://en.wikipedia.org/w/api.php"
    
    headers = {
        "User-Agent": "WikiRAGEngine/1.0 (contact: engineering_project@example.com)"
    }
    
    params = {
        "action": "opensearch",
        "search": query,
        "limit": 1,          # Just give us the top corrected match
        "namespace": 0,      # Main articles only
        "format": "json"
    }
    
    response = requests.get(search_url, params=params, headers=headers).json()
    
    # OpenSearch returns a nested list layout: [original_query, [titles], [descriptions], [urls]]
    titles = response[1]
    urls = response[3]
    
    if not titles or not urls:
        raise Exception(f"No Wikipedia articles found matching '{query}'. Try a different topic!")
        
    top_title = titles[0]
    resolved_url = urls[0]
    
    return resolved_url, top_title
    
    # Added headers=headers to the request configuration
    response = requests.get(search_url, params=params, headers=headers).json()
    search_results = response.get("query", {}).get("search", [])
    
    if not search_results:
        raise Exception(f"No Wikipedia articles found matching '{query}'. Try a different topic!")
        
    top_title = search_results[0]["title"]
    formatted_title = top_title.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{formatted_title}", top_title
    
    response = requests.get(search_url, params=params).json()
    search_results = response.get("query", {}).get("search", [])
    
    if not search_results:
        raise Exception(f"No Wikipedia articles found matching '{query}'. Try a different topic!")
        
    top_title = search_results[0]["title"]
    formatted_title = top_title.replace(" ", "_")
    return f"https://en.wikipedia.org/wiki/{formatted_title}", top_title

def ingest_wikipedia_by_query(query: str) -> dict:
    """Finds the best Wikipedia page for a query, scrapes it, and embeds it."""
    url, actual_title = search_wikipedia_topic(query)
    
    raw_text = scrape_wikipedia(url)
    chunks = chunk_text(raw_text)
    store_in_vector_db(chunks)
    
    return {
        "status": "success",
        "message": f"Successfully found and ingested article: '{actual_title}' ({len(chunks)} chunks loaded)."
    }

def answer_user_query(question: str) -> dict:
    """Retrieves relevant chunks from ChromaDB and passes them to Llama 3.3 via Groq."""
    # Maps perfectly to your original function name with the optimized k=6 window
    context_chunks = retrieve_relevant_chunks(question, k=6)
    
    # Generate response using the LLM service
    answer = generate_answer(question, context_chunks)
    
    return {
        "answer": answer,
        "sources_used": len(context_chunks)
    }