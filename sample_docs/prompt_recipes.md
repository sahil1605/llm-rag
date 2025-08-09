# Prompt Recipes (Ask Better Questions)

General rule: reference the documents explicitly when useful, ask for structure, and request citations.

## Summarize
- "Give a 5-bullet summary of the architecture described in these documents. Cite the titles in parentheses."
- "Create a concise executive summary (<=120 words) of what this demo can do."

## Locate & Extract
- "Which endpoint receives file uploads? Answer with JSON: {endpoint, method, fields}."
- "List all request fields for /load with defaults as a table."
- "Quote any lines that mention `OLLAMA_MODEL` and explain its effect in 2 sentences."

## Compare & Contrast
- "Compare the retrieval-only behavior vs. RAG+LLM behavior in 5 bullets."
- "What are the tradeoffs of TF‑IDF vs. dense embeddings for this demo?"

## Procedures & Checklists
- "Produce a step-by-step checklist to run the app from scratch on macOS."
- "Create a runbook to debug 'Method Not Allowed' on /upload."

## Structured Outputs
- "Return a JSON object with keys: steps, endpoints, env_vars. Populate from the docs only."
- "Generate a CSV with columns: endpoint, method, body_type, success_response."

## Safety & Grounding
- "If the answer is not present in the provided sources, say 'I don't know'. Cite the closest source if helpful."

## Test-style Questions (good for retrieval)
- "What does /ask return when include_sources=true?"
- "Where are uploaded files saved on disk? Provide the full path pattern."
- "What file types can I upload?"
