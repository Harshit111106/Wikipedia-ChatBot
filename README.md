# Wikipedia Knowledge Retrieval-Augmented Generation (RAG) Engine

A production-ready, full-stack, decoupled RAG application that allows users to dynamically search and ingest any Wikipedia topic into a local vector database and perform accurate, context-bound question answering using an advanced Language Model.

---

## 🛠️ Tech Stack & Architecture

The application is engineered with a strict separation of concerns, dividing the system into an isolated backend processing server and a lightweight frontend dashboard interface.

### Backend (Python & FastAPI)
* **API Framework:** FastAPI with Uvicorn (Asynchronous server gateway)
* **Search Integration:** Wikipedia MediaWiki OpenSearch API (Handles query fuzzy matching and auto-corrections)
* **Orchestration Layer:** LangChain (Document loading, text splitting, and pipeline routing)
* **Vector Database:** ChromaDB (Local vector storage matrix)
* **Embedding Model:** HuggingFace `all-MiniLM-L6-v2` (Executed entirely locally)
* **Generative AI Brain:** Llama 3.3 70B via Groq API (High-speed, cloud-hosted inference)

### Frontend (HTML5 / Tailwind CSS / Vanilla JavaScript)
* Fully decoupled SPA (Single Page Application) interacting with the backend purely via asynchronous HTTP REST requests (`fetch` API).
* Responsive UI styled dynamically using utility-first classes via the Tailwind CSS engine.

---

## 📁 Project Directory Structure

```text
internship_project/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI Application Core & Routes
│   │   └── services/
│   │       ├── __init__.py
│   │       ├── llm_service.py   # Groq & Llama Generation Config
│   │       ├── manager.py       # Pipeline Ingestion & Query Coordinator
│   │       ├── rag_engine.py    # Chunking, Embeddings & Vector DB Operations
│   │       └── scraper.py       # BeautifulSoup Wikipedia Web Scraper
│   ├── .env                     # Private API Keys & Environment Variables
│   └── requirements.txt         # Python Project Dependencies
├── frontend/
│   └── index.html               # Unified UI Dashboard & Fetch Interface
└── README.md                    # Project Documentation
```

---

## ⚙️ Installation & Setup Guide

### 1. Backend Setup
Navigate into the backend directory, initialize an isolated virtual environment, and install the frozen project dependencies:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a file named `.env` inside the `backend/` directory and supply your Groq API credentials:
```env
GROQ_API_KEY=your_actual_groq_api_key_here
```

Launch the FastAPI development application server:
```bash
uvicorn app.main:app --reload
```
> **Note:** The server will boot up locally at `http://127.0.0.1:8000`. You can explore and test individual live endpoints using the built-in interactive Swagger documentation panel at `http://127.0.0.1:8000/docs`.

### 2. Frontend Setup
Because the frontend interface is engineered using pure web standards, no complex build steps, package installations, or compilation managers are required.
* Open your system's file explorer, navigate to the `frontend/` directory, and double-click **`index.html`** to open the operational dashboard inside any web browser.

---

## 🚀 Operational Workflow

1. **Fuzzy Search & Ingestion (Search ➔ Scrape ➔ Chunk ➔ Embed ➔ Store):**
   * The user types any keyword, phrase, or poorly-spelled topic (e.g., `"viyat koli"`) into the UI input panel.
   * The backend fires an internal request to the **Wikipedia OpenSearch API**. The API automatically corrects typos and maps the input string to the closest high-ranking matching article path.
   * The backend extracts raw text content layers from the resolved page URL, stripping away noisy HTML elements using `BeautifulSoup`.
   * The document text is broken down into structured semantic fragments using a `RecursiveCharacterTextSplitter` configured with a chunk size of 1000 characters and a 200-character overlap.
   * Fragments are mapped to 384-dimensional vector spaces natively using `HuggingFaceEmbeddings` and indexed safely within a local `ChromaDB` collection.

2. **Contextual Retrieval-Augmented Querying (Retrieve ➔ Prompt ➔ Generate):**
   * The user submits a topical question via the chat workspace.
   * The question is embedded and passed through a vector similarity algorithm inside ChromaDB to isolate the top **6** highest-ranking contextual snippets.
   * The text fragments are injected alongside the user's question into a strict system prompt template.
   * The payload is shipped securely to the Llama 3.3 70B model via Groq, which synthesizes a clean, factually verified response.

---

## 🔒 Engineered Features & Safeguards

* **Fuzzy Autocorrect Ingestion:** Upgraded user onboarding from static URL copying to a resilient, search-driven ingestion engine that intelligently handles spelling errors and missing characters natively.
* **Anti-Hallucination Guardrails:** The core LLM prompt enforces strict alignment rules. If retrieved vector contexts do not satisfy the baseline information requirements needed to answer a query, the model is instructed to explicitly state that it cannot verify the fact rather than generating ungrounded assertions.
* **Optimized Retrieval Window:** Tuned the similarity retrieval matrix to fetch $k=6$ context nodes. This ensures that high-level introductory biographical outlines and dense historical data matrices are captured together during vector execution.
* **Cross-Origin Security:** Full CORS (Cross-Origin Resource Sharing) middleware configurations are integrated natively into the FastAPI pipeline to enable seamless, authenticated cross-origin browser interactions during standard development routines.