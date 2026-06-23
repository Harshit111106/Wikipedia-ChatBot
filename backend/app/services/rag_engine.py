# app/services/rag_engine.py
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def ingest_document(text: str, session_id: str):
    print(f"\n[DEBUG] ---> Starting ingestion for Session: {session_id}")
    
    # 1. Clear out any older chunks belonging to THIS specific session
    try:
        vector_store._collection.delete(where={"session_id": session_id})
        print(f"[DEBUG] ---> Cleaned up previous session chunks for: {session_id}")
    except Exception as e:
        print(f"[DEBUG] ---> No old session chunks to delete or wipe failed: {e}")
    
    # 2. Split your text data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs = text_splitter.create_documents([text])
    
    # 3. Explicitly attach the session tracking token to metadata
    for doc in docs:
        doc.metadata = {"session_id": session_id}
        
    # 4. Commit to database matrix
    vector_store.add_documents(docs)
    print(f"[DEBUG] ---> Successfully vectorized and saved {len(docs)} chunks for session: {session_id}\n")

def retrieve_context(query: str, session_id: str):
    print(f"\n[DEBUG] ---> Querying Vector Database for Session: {session_id}")
    print(f"[DEBUG] ---> User Query: '{query}'")
    
    # Enforce strict metadata filtering via explicit dictionary parameter mapping
    relevant_docs = vector_store.similarity_search(
        query, 
        k=4, 
        filter={"session_id": session_id}
    )
    
    print(f"[DEBUG] ---> Found {len(relevant_docs)} matching chunks strictly locked to this session.")
    for i, doc in enumerate(relevant_docs):
        print(f"    └─ Chunk {i+1} Metadata: {doc.metadata} | Snippet: {doc.page_content[:40]}...")
        
    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    return context