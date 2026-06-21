from app.services.scraper import scrape_wikipedia_page
from app.services.rag_engine import chunk_text, store_in_vector_db, retrieve_relevant_chunks
from app.services.llm_service import generate_answer

def ingest_wikipedia_url(url: str) -> dict:
    """
    Orchestrates the entire data ingestion pipeline.
    """
    print(f"\n[Ingestion] Starting process for URL: {url}")
    
    # 1. Scrape the content
    raw_text = scrape_wikipedia_page(url)
    print("[Ingestion] Successfully extracted raw text from Wikipedia.")
    
    # 2. Slice text into readable chunks
    chunks = chunk_text(raw_text)
    print(f"[Ingestion] Split document into {len(chunks)} chunks.")
    
    # 3. Vectorize and save locally into ChromaDB
    store_in_vector_db(chunks)
    print("[Ingestion] Chunks successfully embedded and saved to ChromaDB.")
    
    return {
        "status": "success",
        "message": f"Successfully ingested {len(chunks)} text chunks into the system."
    }

def answer_user_query(query: str) -> dict:
    """
    Orchestrates the complete retrieval-augmented generation pipeline.
    """
    print(f"\n[Query] Processing user question: '{query}'")
    
    # 1. Retrieve the top 3 most relevant snippets from ChromaDB
    context_chunks = retrieve_relevant_chunks(query, k=6)
    print(f"[Query] Retrieved {len(context_chunks)} matching context snippets from local database.")
    
    # 2. Feed the snippets and question to Groq's Llama 3.3 70B
    answer = generate_answer(query, context_chunks)
    print("[Query] Generation complete. Sending response back.")
    
    return {
        "query": query,
        "answer": answer,
        "sources_used": len(context_chunks)
    }