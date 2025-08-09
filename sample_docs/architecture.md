# Architecture (High Level)

1) Upload (POST /upload)
- Accepts 1..N files (pdf, md, txt). Saves them under `data/uploads/<session_id>/`.
- Returns `{ session_id, files: [{filename, size}] }`.

2) Load/Process (POST /load)
- Reads uploaded files, extracts text (pypdf for PDFs), splits into overlapping chunks.
- Builds a TF‑IDF index (scikit‑learn) and persists vectorizer + sparse matrix to `data/indices/` per session.
- Returns `{ indexed_chunks, status: "ok" }`.

3) Ask (POST /ask)
- Transforms the question with the session's TF‑IDF vectorizer and scores chunks by cosine similarity.
- Returns top‑k sources and an answer:
  - If `OLLAMA_MODEL` is set, query local Ollama with the top chunks as context.
  - Otherwise, return a retrieval‑only answer by concatenating the top snippets.

Notes:
- Static UI (`frontend/`) is served by the same FastAPI app at `/` to avoid CORS.
- Environment variables:
  - `OLLAMA_MODEL` (e.g., `llama3.1:8b`) to enable RAG+LLM.
  - `OLLAMA_ENDPOINT` (default `http://localhost:11434`).
  - Optional `CORS_ORIGINS` if serving UI separately.
