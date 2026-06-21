from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.services.manager import ingest_wikipedia_url, answer_user_query

# 1. Initialize the FastAPI application
app = FastAPI(
    title="Wikipedia RAG Engine API",
    description="Production-ready API endpoints for document ingestion and QA retrieval",
    version="1.0.0"
)

# 2. Enable CORS (Cross-Origin Resource Sharing)
# This allows your future frontend app to talk to this backend securely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permits requests from any origin (perfect for development)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],
)

# 3. Define Pydantic Data Models for Request Validation
class IngestRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    question: str

# 4. Root Endpoint (Health Check)
@app.get("/")
def read_root():
    return {"status": "online", "message": "Wikipedia RAG API is running smoothly."}

# 5. Ingestion Endpoint
@app.post("/api/ingest")
def ingest_url(payload: IngestRequest):
    """Takes a Wikipedia URL, scrapes it, chunks it, and saves it to ChromaDB."""
    try:
        result = ingest_wikipedia_url(payload.url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")

# 6. Query / QA Endpoint
@app.post("/api/query")
def query_rag(payload: QueryRequest):
    """Searches local ChromaDB for context and asks Groq to generate the final answer."""
    try:
        result = answer_user_query(payload.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")