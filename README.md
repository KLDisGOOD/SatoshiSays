## SatoshiSays (LangChain + Streamlit + OSS LLM + Blockchair)

### Features
- Natural-language blockchain questions (Bitcoin, Ethereum, Dogecoin, etc.)
- RAG over Blockchair API docs using FAISS
- LLM (OpenRouter) plans endpoint + parameters
- Executes Blockchair requests and shows JSON
- Summarized, friendly answer; optional charts

### Prerequisites
- Python 3.10+
- Jina API key (for embeddings)
- (Optional) Blockchair API key

### Setup
```bash
# From the project root (folder containing projects/)
python -m venv .venv
source .venv/bin/activate  # Windows Git Bash
# On Windows cmd: .venv\Scripts\activate

pip install -r requirements.txt

# Configure environment: copy the example and edit values
cp .env.example .env
# Then edit .env to add your JINA_API_KEY and (optionally) BLOCKCHAIR_API_KEY

# (Optional) customize embeddings
export JINA_EMBED_MODEL="jina-embeddings-v3"
export JINA_EMBED_TASK="retrieval.passage"

# Build the vector store from Blockchair docs (requires JINA_API_KEY in .env)
python -m projects.ingest_docs

# Run SatoshiSays
streamlit run projects/app.py
```

### How it works
1. `projects/ingest_docs.py` scrapes the main Blockchair API docs page, chunks it, embeds via OpenRouter embeddings, and saves a FAISS index in `vectorstore/`.
2. `projects/rag_pipeline.py` retrieves top-k docs and asks the LLM to produce a structured `ApiCallPlan` (endpoint type, chain, params).
3. `projects/blockchair_client.py` executes the HTTP request(s) to Blockchair with basic pagination and rate-limit handling.
4. The app combines docs context + JSON and has the LLM write a clear answer.

### Notes
- The SQL/Infinitable endpoint varies by resource across Blockchair. The app uses a generic `/tables?q=...` placeholder; consult docs for exact shape.
- If FAISS loading warns about "dangerous deserialization", it's expected for local use. Ensure you trust your own index files.
- For Streamlit Cloud, put secrets in `.streamlit/secrets.toml` with keys `OPENROUTER_API_KEY` and optionally `BLOCKCHAIR_API_KEY`.

### Troubleshooting
- "Failed to load vectorstore": Re-run the ingest step and ensure your JINA_API_KEY is set.
- 429 Too Many Requests from Blockchair: The client retries with backoff, but you may need an API key or wait.
***

### GitHub
Add this `.gitignore` at repo root:
```
# Python
__pycache__/
*.pyc
.pytest_cache/

# Environments
.venv/
.env
.env.*

# Streamlit
.streamlit/

# IDE
.vscode/
.idea/

# Data/Artifacts
vectorstore/
*.faiss
*.pkl

# OS
.DS_Store
Thumbs.db
```