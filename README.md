# ◈ Gallery AI — Porsche Knowledge Engine

> A sophisticated Retrieval-Augmented Generation (RAG) chatbot designed to interrogate specific datasets with strict knowledge boundaries. Built with a **FastAPI** backend and a custom-styled **Streamlit** frontend.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 PDF Ingestion | Upload any PDF to instantly expand the knowledge base |
| 🌐 URL Ingestion | Paste a web link and scrape its content into the vector store |
| 🔀 MMR Reranking | Maximal Marginal Relevance ensures diverse, non-redundant retrieval |
| 🧠 Query Rewriting | Conversation memory rewrites follow-up questions for better retrieval |
| 🚧 Strict Knowledge Boundary | System only answers from provided data — no hallucinations |
| 🧾 Evidence Citations | Answers return confidence, source snippets, and page references when available |
| 🔐 Admin Protection | Optional `X-Admin-Key` protects ingestion and source-management endpoints |
| 🗑️ Source Management | List and delete individual sources from the knowledge base |
| ⚡ Streaming Support | Optional streaming endpoint (`/query_stream`) for real-time text generation |

---

## 📸 Interface & Response Logic

Place your screenshot at `assets/screenshot.png` after cloning the repo.

```
assets/
└── screenshot.png   ← Add your chatbot screenshot here
```

The UI features a dark-themed, gallery-style design with:
- A **sidebar** for ingesting PDFs and URLs, and reviewing active sources
- A **main chat area** for natural-language queries against the knowledge base
- **Meta pills** on each response showing the classified topic and source documents

### Strict Response Logic

The system enforces two response boundaries:

**✅ Knowledge-Based Answer**
When the retrieved context is sufficient, the engine generates a precise, source-cited answer. A topic badge (e.g., `Porsche History`, `Product Info`) and source pills are displayed beneath the response.

**❌ Out-of-Scope / Insufficient Coverage**
When a question falls outside the ingested knowledge base (e.g., asking about the weather when only Porsche data is loaded), the system identifies low confidence and responds:

> *"I'm sorry, I cannot accurately answer this question based on the current knowledge base. Please try uploading more relevant materials."*

This prevents the LLM from hallucinating and keeps answers grounded in your data.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────┐
│            Streamlit Frontend           │  ← appui.py
│  Sidebar: PDF/URL Ingest | Sources      │
│  Main: Chat UI + Meta Pills             │
└────────────────┬────────────────────────┘
                 │ HTTP (REST)
┌────────────────▼────────────────────────┐
│            FastAPI Backend              │  ← main.py
│  POST /query         (sync answer)      │
│  POST /query_stream  (streaming)        │
│  POST /admin/ingest-pdf                 │
│  POST /admin/ingest-url                 │
│  GET  /admin/sources                    │
│  DELETE /admin/source                   │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│              RAG Engine                 │  ← rag_engine.py
│                                         │
│  1. Query Rewriting   (GPT-4o mini)     │
│  2. MMR Retrieval     (ChromaDB)        │
│  3. Confidence Check  (GPT-4o mini)     │
│  4. Answer Generation (GPT-4o mini)     │
│  5. Topic Classification                │
│  6. Memory Sync       (ChatHistory)     │
└────────────────┬────────────────────────┘
                 │
┌────────────────▼────────────────────────┐
│              ChromaDB                   │
│  Embeddings: text-embedding-3-small     │
│  Persisted at: ./data/chroma_db         │
└─────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit + Custom CSS (Space Grotesk / Space Mono) |
| **Backend** | FastAPI + Uvicorn |
| **LLM** | GPT-4o mini (OpenAI) |
| **Embeddings** | text-embedding-3-small (OpenAI) |
| **Vector Database** | ChromaDB (persisted locally) |
| **RAG Framework** | LangChain |
| **PDF Loader** | PyPDFLoader (LangChain Community) |
| **Web Scraper** | WebBaseLoader (LangChain Community) |
| **Text Splitting** | RecursiveCharacterTextSplitter (1000 tokens, 150 overlap) |

---

## 📁 Project Structure

```
gallery-ai-rag/
│
├── appui.py              # Streamlit frontend (UI + API calls)
├── main.py               # FastAPI backend (routes + middleware)
├── rag_engine.py         # Core RAG logic (retrieval, generation, memory)
├── models.py             # Pydantic request/response models
│
├── .env                  # Environment variables (not committed)
├── .gitignore            # Git ignore rules
├── requirements.txt      # Python dependencies
│
├── assets/
│   └── screenshot.png    # UI screenshot for README
│
└── data/                 # Auto-created at runtime (not committed)
    ├── chroma_db/        # Persisted ChromaDB vector store
    └── uploads/          # Uploaded PDF files
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API key with access to GPT-4o mini and text-embedding-3-small

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/gallery-ai-rag.git
cd gallery-ai-rag
```

---

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

---

### 3. Set Up Environment Variables

Copy `.env.example` to `.env`, then fill in your real OpenAI key:

```env
OPENAI_API_KEY=your_openai_api_key_here
PERSIST_DIRECTORY=./data/chroma_db
UPLOAD_DIRECTORY=./data/uploads
API_PORT=8001
API_BASE_URL=http://127.0.0.1:8001
```

> ⚠️ **Never commit your `.env` file.** Make sure `.env` is listed in your `.gitignore`.

---

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key packages required (`requirements.txt`):**

```
fastapi
uvicorn
streamlit
requests
python-dotenv
langchain
langchain-openai
langchain-chroma
langchain-community
pypdf
chromadb
beautifulsoup4
```

---

### 5. Run the Application

You need **two terminals** running simultaneously.

**Terminal 1 — Start the FastAPI backend:**

```bash
python main.py
```

The API will be available at: `http://127.0.0.1:8001`
Interactive API docs (Swagger UI): `http://127.0.0.1:8001/docs`

**Terminal 2 — Start the Streamlit frontend:**

```bash
streamlit run appui.py
```

The app will open automatically at: `http://localhost:8501`

---

## 📖 How to Use

1. **Open the app** in your browser at `http://localhost:8501`
2. **Ingest a document** via the sidebar:
   - Upload a PDF and click **Ingest PDF →**
   - Or paste a URL and click **Ingest URL →**
3. **Wait for confirmation** — the sidebar shows the number of chunks indexed
4. **Refresh Sources** to verify your document is in the knowledge base
5. **Ask questions** in the chat input — the engine retrieves relevant chunks and generates a grounded answer
6. **Review meta pills** beneath each response to see the classified topic and source files used

---

## 🔌 API Reference

All endpoints are served from `http://127.0.0.1:8001` by default.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/query` | Synchronous Q&A (returns JSON) |
| `POST` | `/query_stream` | Streaming Q&A (returns text/plain stream) |
| `POST` | `/admin/ingest-pdf` | Upload and ingest a PDF file |
| `POST` | `/admin/ingest-url` | Scrape and ingest a URL |
| `GET` | `/admin/sources` | List all ingested sources |
| `DELETE` | `/admin/source` | Delete a source and all its chunks |

If `ADMIN_API_KEY` is set, all `/admin/*` endpoints require:

```http
X-Admin-Key: your_admin_key
```

**Example `/query` request:**

```json
POST /query
{
  "question": "What was the first Porsche model ever produced?"
}
```

**Example `/query` response:**

```json
{
  "answer": "The first Porsche model was the 356, introduced in 1948...",
  "topic": "Porsche History",
  "sources": ["porsche_history.pdf"],
  "confidence": 0.86,
  "citations": [
    {
      "source": "porsche_history.pdf",
      "page": 3,
      "snippet": "The Porsche 356 was introduced in 1948...",
      "score": 0.86
    }
  ],
  "refined_question": "first Porsche model ever produced",
  "needs_more_context": false
}
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | Your OpenAI API key |
| `PERSIST_DIRECTORY` | `./data/chroma_db` | Where ChromaDB stores vectors |
| `UPLOAD_DIRECTORY` | `./data/uploads` | Where uploaded PDFs are saved |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8001` | FastAPI bind port |
| `API_BASE_URL` | `http://127.0.0.1:8001` | Streamlit backend URL |
| `CORS_ALLOW_ORIGINS` | `http://localhost:8501,http://127.0.0.1:8501` | Allowed browser origins |
| `ADMIN_API_KEY` | *(empty)* | Optional admin key for ingestion/source endpoints |
| `MAX_UPLOAD_BYTES` | `10485760` | PDF upload size limit |
| `REQUEST_TIMEOUT_SECONDS` | `30` | Streamlit API request timeout |
| `MIN_RETRIEVAL_CONFIDENCE` | `0.45` | Low-confidence refusal threshold |

---

## ✅ Verification

Run the local test suite:

```bash
python -m unittest discover -s tests -v
```

MMR retrieval parameters (editable in `rag_engine.py`):

| Parameter | Value | Effect |
|---|---|---|
| `k` | 5 | Number of chunks returned to the LLM |
| `fetch_k` | 20 | Candidate pool size before MMR reranking |
| `lambda_mult` | 0.6 | Balance between relevance (1.0) and diversity (0.0) |

---

## 🚢 Deploying to GitHub

Run these commands from your project root:

```bash
# 1. Initialise Git
git init

# 2. Stage all files
git add .

# 3. Commit
git commit -m "Initial commit: Gallery AI RAG system"

# 4. Create a new repo on GitHub (leave 'Initialize with README' unchecked)
# Then link and push:
git remote add origin https://github.com/your-username/gallery-ai-rag.git
git branch -M main
git push -u origin main
```

> 💡 **Pro tip:** Ensure your `.gitignore` excludes `data/`, `.env`, and `__pycache__/` before pushing.

**Recommended `.gitignore`:**

```gitignore
.env
data/
__pycache__/
*.pyc
.venv/
venv/
*.egg-info/
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

---

## 📄 License

This project was developed as part of an internship program.

© 2026 Daniel — Raffles University Internship Project at **DigiMagic**
