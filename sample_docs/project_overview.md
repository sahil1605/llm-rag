# LLM RAG Demo: Product Overview

This demo showcases a Retrieval-Augmented Generation (RAG) pipeline you can run locally.

- Upload files (.pdf/.md/.txt)
- Load/Process to chunk + index your content (TF‑IDF)
- Ask questions; the app retrieves relevant chunks and, if configured, asks a local LLM (Ollama)

Key properties:
- Same-origin UI + API on http://localhost:5173
- Per-session isolation for uploads and indices
- Pluggable LLM via OLLAMA_MODEL env, with graceful fallback to retrieval-only answers

What this is good for:
- Q&A on small to medium knowledge bases
- Summaries, comparisons, and extraction tasks using your own documents

What this is not:
- A full production RAG stack; this is a minimal, learnable blueprint
