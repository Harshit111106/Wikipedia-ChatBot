# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.manager import answer_user_query, ingest_wikipedia_by_query

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Crucial: Models MUST explicitly require session_id strings
class ScrapeRequest(BaseModel):
    query: str
    session_id: str

class QueryRequest(BaseModel):
    question: str
    session_id: str

class CleanupRequest(BaseModel):
    session_id: str

@app.post("/api/ingest")
def ingest_endpoint(payload: ScrapeRequest):
    try:
        print(f"\n[API ROUTE] Received Ingest URL/Query with Session: {payload.session_id}")
        title = ingest_wikipedia_by_query(payload.query, payload.session_id)
        return {"status": "success", "title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query")
def query_endpoint(payload: QueryRequest):
    try:
        print(f"\n[API ROUTE] Received User Chat Query with Session: {payload.session_id}")
        answer = answer_user_query(payload.question, payload.session_id)
        return {"answer": answer, "sources_used": 4}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/session_status/{session_id}")
def session_status_endpoint(session_id: str):
    try:
        print(f"\n[API ROUTE] Checking status for Session: {session_id}")
        from app.services.manager import get_active_article_title
        title = get_active_article_title(session_id)
        return {"title": title}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cleanup")
def cleanup_endpoint(payload: CleanupRequest):
    try:
        print(f"\n[API ROUTE] Received Cleanup request for Session: {payload.session_id}")
        # Remove from active_articles mapping
        from app.services.manager import active_articles
        active_articles.pop(payload.session_id, None)
        # Wipe database documents for this session
        from app.services.rag_engine import delete_session_documents
        delete_session_documents(payload.session_id)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)