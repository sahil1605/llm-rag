from __future__ import annotations

import os
import time
import uuid
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
import traceback
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import httpx


APP_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = APP_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "indices"

for d in (DATA_DIR, UPLOAD_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)


class AskRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = 4
    include_sources: bool = True


class LoadRequest(BaseModel):
    session_id: str
    chunk_size: int = 800
    chunk_overlap: int = 120


def _new_session_id() -> str:
    return str(uuid.uuid4())


def _session_paths(session_id: str) -> Dict[str, Path]:
    return {
        "uploads": UPLOAD_DIR / session_id,
        "index": INDEX_DIR / f"{session_id}.index",
        "meta": INDEX_DIR / f"{session_id}.meta.json",
    }


def _vectorizer_path(session_id: str) -> Path:
    return _session_paths(session_id)["index"].with_suffix(".vec.pkl")


def _matrix_path(session_id: str) -> Path:
    return _session_paths(session_id)["index"].with_suffix(".mtx.pkl")


def read_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return path.read_text(errors="ignore")
    if suffix == ".pdf":
        # Minimal PDF text extraction using pypdf
        try:
            from pypdf import PdfReader  # lazy import
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"PDF support requires pypdf: {exc}")
        text_parts: List[str] = []
        with path.open("rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                try:
                    text_parts.append(page.extract_text() or "")
                except Exception:
                    text_parts.append("")
        return "\n".join(text_parts)
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    text = text.strip()
    if chunk_size <= 0:
        chunk_size = 800
    if chunk_overlap < 0:
        chunk_overlap = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += step
    return chunks


def save_index(session_id: str, vectorizer: TfidfVectorizer, matrix, chunks: List[Dict[str, Any]]) -> None:
    paths = _session_paths(session_id)
    import orjson
    # Persist vectorizer and sparse matrix
    joblib.dump(vectorizer, _vectorizer_path(session_id))
    joblib.dump(matrix, _matrix_path(session_id))
    paths["meta"].write_bytes(orjson.dumps({"chunks": chunks}))


def load_index(session_id: str):
    paths = _session_paths(session_id)
    if not _vectorizer_path(session_id).exists() or not _matrix_path(session_id).exists() or not paths["meta"].exists():
        raise HTTPException(status_code=404, detail="Index not found for session")
    import orjson
    vectorizer: TfidfVectorizer = joblib.load(_vectorizer_path(session_id))
    matrix = joblib.load(_matrix_path(session_id))
    meta = orjson.loads(paths["meta"].read_bytes())
    chunks = meta.get("chunks", [])
    return vectorizer, matrix, chunks


app = FastAPI(title="LLM RAG Backend", version="0.1.0")

# Optionally enable CORS for dev if you serve frontend elsewhere
if os.environ.get("CORS_ORIGINS"):
    origins = [o.strip() for o in os.environ["CORS_ORIGINS"].split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.post("/upload")
async def upload_files(
    files: Optional[List[UploadFile]] = File(default=None, description="files list", alias="files"),
    files_array: Optional[List[UploadFile]] = File(default=None, description="files[] list", alias="files[]"),
    session_id: Optional[str] = Form(default=None),
):
    # Accept both 'files' and 'files[]' field names
    incoming: List[UploadFile] = []
    if files:
        incoming.extend(files)
    if files_array:
        incoming.extend(files_array)
    if not incoming:
        raise HTTPException(status_code=400, detail="No files provided")

    sid = session_id or _new_session_id()
    paths = _session_paths(sid)
    paths["uploads"].mkdir(parents=True, exist_ok=True)

    saved: List[Dict[str, Any]] = []
    for f in incoming:
        # Persist to disk; FastAPI streams to temp, we re-save for session
        destination = paths["uploads"] / f.filename
        with destination.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        size = destination.stat().st_size
        saved.append({"filename": f.filename, "size": size})

    return JSONResponse({"session_id": sid, "files": saved})


@app.post("/load")
def load_and_index(req: LoadRequest):
    sid = req.session_id
    paths = _session_paths(sid)
    upload_dir = paths["uploads"]
    if not upload_dir.exists():
        raise HTTPException(status_code=404, detail="No uploaded files for this session")

    # Read files and build chunks
    all_chunks: List[Dict[str, Any]] = []
    for path in upload_dir.iterdir():
        if not path.is_file():
            continue
        try:
            text = read_text_from_file(path)
        except HTTPException:
            # Skip unsupported files
            continue
        chunks = chunk_text(text, req.chunk_size, req.chunk_overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "text": chunk,
                    "source": str(path),
                    "title": path.name,
                    "loc": {"chunk": i},
                }
            )

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No extractable text from uploaded files")

    # Fit TF-IDF on chunks and persist
    texts = [c["text"] for c in all_chunks]
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), dtype=np.float32)
    matrix = vectorizer.fit_transform(texts)  # L2 normalized by default
    save_index(sid, vectorizer, matrix, all_chunks)

    return {"indexed_chunks": len(all_chunks), "status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    try:
        sid = req.session_id
        t0 = time.time()
        vectorizer, matrix, chunks = load_index(sid)

        # Transform query and compute cosine similarity scores
        q = vectorizer.transform([req.question])  # shape (1, n_features)
        # scores = X dot q.T; sparse-safe
        scores_arr = (matrix @ q.T).toarray().ravel()

        # Top-k indices
        if req.top_k <= 0:
            top_k = 4
        else:
            top_k = min(req.top_k, len(scores_arr))
        if top_k == 0:
            return JSONResponse({"answer": "", "sources": [], "latency_ms": 0})
        idxs = np.argpartition(-scores_arr, top_k - 1)[:top_k]
        # Sort top-k by score descending
        idxs = idxs[np.argsort(-scores_arr[idxs])]

        sources: List[Dict[str, Any]] = []
        for i in [int(x) for x in idxs.tolist()]:
            score = float(scores_arr[i])
            if i < 0 or i >= len(chunks):
                continue
            chunk = chunks[i]
            sources.append(
                {
                    "id": int(i),
                    "title": chunk.get("title", ""),
                    "snippet": chunk.get("text", "")[:400],
                    "source": chunk.get("source", ""),
                    "score": score,
                }
            )

        # Build context from retrieved sources
        context_blocks = []
        for s in sources:
            title = s.get("title", "")
            snippet = s.get("snippet", "")
            context_blocks.append(f"# {title}\n{snippet}")
        context_text = "\n\n".join(context_blocks) or "No context available."

        # If OLLAMA_MODEL is set, ask local Ollama; otherwise fallback to concatenated snippets
        ollama_model = os.environ.get("OLLAMA_MODEL") or os.environ.get("OLLAMA")
        if ollama_model:
            try:
                answer = query_ollama(ollama_model, req.question, context_text)
            except Exception:
                # Fallback to snippets on failure
                answer = "\n\n".join([s.get("snippet", "") for s in sources])
        else:
            # Fallback: simple retrieval-only answer
            answer = "\n\n".join([s.get("snippet", "") for s in sources])

        latency_ms = int((time.time() - t0) * 1000)
        result: Dict[str, Any] = {"answer": answer, "latency_ms": latency_ms}
        if req.include_sources:
            result["sources"] = sources
        return JSONResponse(result)
    except Exception:
        return PlainTextResponse(traceback.format_exc(), status_code=500)


def query_ollama(model_name: str, question: str, context: str) -> str:
    endpoint = os.environ.get("OLLAMA_ENDPOINT", "http://localhost:11434")
    url = f"{endpoint.rstrip('/')}/api/generate"
    prompt = (
        "You are a helpful assistant. Answer the user's question using only the provided context.\n"
        "If the answer is not in the context, say you don't know succinctly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    with httpx.Client(timeout=60) as client:
        # Use stream=false to get a single JSON with 'response'
        resp = client.post(url, json={"model": model_name, "prompt": prompt, "stream": False})
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "")


# Mount static frontend LAST so API routes take precedence
frontend_dir = APP_ROOT / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "5173"))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)

