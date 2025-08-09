# LLM RAG (FastAPI + FAISS) Demo

Three-step flow with a tiny FastAPI backend and a static frontend:

- Upload files (`.pdf`, `.md`, `.txt`) → `/upload`
- Load/Process: chunk + embed + index (FAISS) → `/load`
- Ask questions with retrieval → `/ask`

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run API + static frontend on http://localhost:5173
python -m backend.main
```

Open `http://localhost:5173` and follow the steps.

## API

- POST `/upload`: multipart/form-data
  - `files[]`: 1..N files
  - optional `session_id`: string
  - returns: `{ session_id, files: [{filename, size}] }`

- POST `/load`: JSON
  - `{ session_id, chunk_size, chunk_overlap }`
  - returns: `{ indexed_chunks, status: "ok" }`

- POST `/ask`: JSON
  - `{ session_id, question, top_k, include_sources }`
  - returns: `{ answer, sources?, latency_ms }`

Notes:
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector store: FAISS IP with cosine-normalized embeddings
- PDF extraction: `pypdf`

