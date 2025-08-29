## AI-Powered FAQ Chatbot

An end-to-end FAQ chatbot that indexes your FAQ data and answers user queries using local embeddings with SentenceTransformers and a lightweight built-in vector store. Optionally uses OpenAI for embeddings and fallback answers.

### Features
- Upload FAQ data (CSV with `question,answer` or TXT with `Q:`/`A:` blocks)
- Create embeddings (SentenceTransformers by default; optional OpenAI)
- Store and persist vectors locally (JSON + NumPy cosine)
- Query by semantic similarity to retrieve best matching FAQ
- Fallback to OpenAI Chat if no sufficiently close FAQ is found
- Streamlit UI

### Tech Stack
- Backend: Python
- Vector Store: Built-in local (JSON + NumPy cosine)
- Embeddings: SentenceTransformers or OpenAI
- UI: Streamlit

### Project Structure
```
app/
  __init__.py
  config.py
  embeddings.py
  vector_store.py
  faq_loader.py
  llm.py
streamlit_app.py
requirements.txt
README.md
.env.example
```

### Setup
1) Create and activate a virtual environment (recommended)
```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows
```

2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Configure environment (optional but recommended for OpenAI use)
`.env`  fill in values.

### Configuration (.env)
Create a local `.env` file to configure the app. The app automatically loads it via `python-dotenv`.

Example variables:
```env
# OpenAI (optional)
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini

# Storage
CHROMA_DIR=data/chroma

# Embeddings
EMBEDDINGS_PROVIDER=sentence-transformers
ST_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Query
TOP_K=5
DISTANCE_THRESHOLD=0.35
```

Notes:
- Do not commit `.env` to source control. It is ignored via `.gitignore`.
- If `OPENAI_API_KEY` is not set, the LLM fallback will be disabled; local search still works.

### Running the App
```bash
python -m streamlit run streamlit_app.py
```

### Data Format
- CSV expected columns: `question,answer`
- TXT supports blocks like:
```
Q: What is your refund policy?
A: You can request a refund within 30 days.

Q: Do you offer support?
A: Yes, via email.
```

### Notes
- Default embedding model: `sentence-transformers/all-MiniLM-L6-v2`
- Data persists under `data/chroma/` by default. You can clear it from the sidebar.
- OpenAI is optional. If not configured, only local embeddings are used and fallback will be disabled.




