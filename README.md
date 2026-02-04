# RAG Practice

A lightweight Retrieval-Augmented Generation (RAG) demo that indexes local Markdown/Text documents and answers questions grounded strictly in those sources.

## Overview
- Indexes `.md` and `.txt` files in `docs/` using sentence-transformer embeddings.
- Stores vectors in an in-memory ChromaDB collection.
- Builds a context prompt from top matches and answers via a Pydantic-AI Agent.
- Designed for fast local experiments and easy customization.

## Tech Stack
- Python 3.13
- Embeddings: sentence-transformers (`all-MiniLM-L6-v2`)
- Vector store: ChromaDB
- LLM Agent: pydantic-ai-slim with `google-gla:gemini-2.5-pro`
- Utilities: LangChain text splitters, Logfire telemetry, python-dotenv

## Project Structure
- `main.py` — end-to-end RAG pipeline: ingest, chunk, embed, search, answer
- `docs/` — your knowledge base; add `.md` or `.txt` files here
- `pyproject.toml` — dependencies and project metadata
- `uv.lock` — dependency lockfile for `uv`
- `eval.py` — placeholder for evaluation scripts
- `utils.py` — optional helpers (currently unused)

## Setup
Prerequisites:
- Python 3.13 (see `.python-version`)
- One of: `uv` (recommended) or `pip`

Using `uv`:
```bash
# Install uv if needed
pip install uv

# Create and activate virtual environment (optional)
uv venv
source .venv/bin/activate

# Install dependencies
uv sync
```

Using `pip`:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r <(python -c "import tomllib,sys;print('\n'.join(tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']))")
```

## Configuration
- LLM credentials: set `GOOGLE_API_KEY` in your environment for Gemini.
- Hugging Face mirror: `main.py` sets `HF_ENDPOINT=https://hf-mirror.com` to speed model downloads in China; change or remove if not needed.
- Embedding model: adjust `EMBEDDING_MODEL_NAME` in `main.py`.
- Chunking: tweak `chunk_size` and `chunk_overlap` in the splitter.
- Persistence: switch to `chromadb.PersistentClient` if you want a persistent DB.

Example environment setup:
```bash
export GOOGLE_API_KEY="your_gemini_api_key"
export HF_ENDPOINT="https://hf-mirror.com"  # optional
```

## Usage
1. Add your knowledge files to `docs/` (Markdown or plain text).
2. Run the demo:
```bash
python main.py
```
3. Default question is “What is my plan?”; edit `question` in `main.py` or call `run()` programmatically.

Programmatic example:
```python
from main import run
answer = run("What is my plan?")
print(answer.answer)
print(answer.source_snippet)
```

## How It Works
- Ingestion: loads files from `docs/` and splits them via `RecursiveCharacterTextSplitter`.
- Embedding: uses `SentenceTransformerEmbeddingFunction` with `all-MiniLM-L6-v2`.
- Indexing: adds chunks to a ChromaDB collection with metadata.
- Retrieval: queries by text; returns top matching chunks.
- Generation: builds a context-only prompt and answers via a Pydantic-AI Agent constrained to the provided context.

## Customization Ideas
- Swap the embedding model for domain-specific vectors.
- Persist the vector store and add CRUD for documents.
- Add reranking or hybrid search.
- Expose a web API or CLI for interactive queries.

## Troubleshooting
- Empty results: ensure `docs/` contains `.md` or `.txt` files.
- First-run slowness: embedding model downloads; keep a stable network.
- Credential errors: verify `GOOGLE_API_KEY` and model name availability.
- ChromaDB issues: if you need persistence, use `chromadb.PersistentClient`.

## License
MIT (or your preferred license).
