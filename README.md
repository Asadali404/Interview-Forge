# ⚒️ Interview Forge

Interview Forge is a free-stack Streamlit MVP that turns a candidate CV and optional job description into an evidence-based interview preparation plan.

## Architecture

The application follows the supplied architecture:

1. **CV** → Candidate Profile
2. **Job Description + Company (optional)** → Job + Company Profile
3. **Information Check** → sufficient / missing information → follow-up questions
4. **Job Match Analysis**
5. **Interview Risk Engine**
   - SWOT
   - 30-second risk areas
   - 60-second risk areas
6. **Question Engine**
   - HR
   - Behavioral
   - Technical
   - Resume-based
   - Job-specific
   - Company-specific
7. **Question Priority**
   - High
   - Medium
   - Low
8. **Answer Blueprint**
   - Why asked
   - What to show
   - CV evidence
9. **Tough Question Coach**
10. **Interview Battle Plan**
11. **Forge Score**
   - Job Fit
   - Technical Readiness
   - Behavioral Readiness
12. **Top Improvement Areas**
13. **Refine & Practice Loop**

## RAG implementation

The MVP uses:

**PDF/TXT → text extraction → tokenization → overlapping token chunks → SentenceTransformer embeddings → FAISS IndexFlatIP → similarity retrieval → Groq LLM**

Embedding model:

`sentence-transformers/all-MiniLM-L6-v2`

The FAISS index is session-local. This keeps the project free and avoids a paid external vector database. A production version can later move the index to a persistent database/object store.

## LLM

The default model is:

`openai/gpt-oss-20b`

The model can be changed with the `GROQ_MODEL` secret.

## Free deployment

Recommended deployment target: **Streamlit Community Cloud**.

You need:

- GitHub account
- Streamlit Community Cloud
- Groq API key
- This repository

The app itself does not require a paid database, paid embedding API, or GPU.

> Important: Groq's free tier has usage/rate limits. "Free" means no paid service is required for the MVP; API availability is still subject to Groq's current free-tier limits.

## Local run

Python 3.12 is a good choice for current Streamlit Community Cloud compatibility.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Set your Groq key.

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="YOUR_KEY"
```

Run:

```bash
streamlit run app.py
```

## Streamlit Cloud deployment

1. Create a GitHub repository, for example `interview-forge`.
2. Upload:
   - `app.py`
   - `rag.py`
   - `interview_engine.py`
   - `requirements.txt`
   - `README.md`
   - `.gitignore`
   - `.streamlit/config.toml`
3. Open Streamlit Community Cloud.
4. Create a new app.
5. Select your GitHub repository.
6. Set the main file to `app.py`.
7. In **Advanced settings → Secrets**, add:

```toml
GROQ_API_KEY = "your_groq_api_key"
GROQ_MODEL = "openai/gpt-oss-20b"
```

8. Deploy.

Never commit the API key to GitHub.

## Important free-tier design decisions

### Why FAISS instead of Pinecone/Weaviate?

FAISS runs locally inside the Streamlit process, so no vector database account or payment is required.

### Why local embeddings?

The embedding model runs inside the Streamlit server. This removes the need for an embedding API key.

### Why not save uploaded CVs permanently?

For privacy and simplicity, the MVP processes documents in memory. The FAISS index is rebuilt for the current session.

### What happens after a restart?

The user uploads the documents again and the RAG index is rebuilt. This is intentional for the free MVP.

## Suggested GitHub structure

```text
interview-forge/
├── app.py
├── rag.py
├── interview_engine.py
├── requirements.txt
├── README.md
├── .gitignore
└── .streamlit/
    └── config.toml
```

## Future production upgrades

- Persistent FAISS indexes per user
- User authentication
- PostgreSQL/Supabase metadata storage
- Persistent object storage for documents
- Streaming interview mode
- Voice interview mode
- Speech-to-text
- Answer scoring dashboard
- Company research agent
- Job-board ingestion
- Multi-document RAG
- Evaluation dataset and automated regression tests
