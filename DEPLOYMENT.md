# 🚨 Emergency Alert Systems Chat Assistant

AI-powered chat assistant for emergency alert systems, public safety communications, and FCC regulations.

## 🚀 Deployment on Streamlit Cloud

### Prerequisites
- GitHub account
- OpenAI API key
- SerpAPI key (optional, for external search)

### Steps to Deploy

1. **Fork/Push this repository to your GitHub**
   - Repository: `https://github.com/MurtazaSFakhry/CPSC491Fall2025-MyVersion`

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**
   - Sign in with your GitHub account

3. **Create a new app**
   - Click "New app"
   - Select your repository: `CPSC491Fall2025-MyVersion`
   - Branch: `main`
   - Main file path: `VectordB/streamlit_app.py`

4. **Configure Secrets**
   - Click on "Advanced settings" → "Secrets"
   - Add the following secrets:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   SERPAPI_KEY = "your-serpapi-key-here"
   # Enable Pinecone (recommended for cloud persistence)
   USE_PINECONE = true
   PINECONE_API_KEY = "your-pinecone-api-key-here"
   PINECONE_INDEX = "fcc-chatbot-index"
   ```

5. **Deploy**
   - Click "Deploy!"
   - Wait for the app to build and launch

### ⚠️ Important Note about Storage (Chroma vs Pinecone)

By default this app can run in two modes:

- Pinecone (recommended): uses a hosted vector DB (set `USE_PINECONE=true`).
- Chroma (fallback/local): uses `chroma_fcc_storage` bundled in the repo.

If you choose Chroma on Streamlit Cloud, you have two options:

**Option 1: Include ChromaDB in Git (Recommended for small DBs < 100MB)**
- Remove `chroma_fcc_storage/` from `.gitignore`
- Commit and push the database files
- Note: GitHub has a 100MB file size limit

**Option 2: Use Cloud Storage**
- Store ChromaDB in AWS S3, Google Cloud Storage, or similar
- Modify `streamlit_app.py` to download/sync the database on startup

### 📊 Current Setup
- **Embeddings**: 5,301 documents
- **Model**: OpenAI text-embedding-3-small
- **Database**: ChromaDB (Persistent)

## 🖥️ Local Development

### Run locally:
```bash
cd VectordB
streamlit run streamlit_app.py
```

### Environment Variables (`.env`):
```
OPENAI_API_KEY=your-key-here
SERPAPI_KEY=your-key-here
```

## 📝 Features
- 🔍 Semantic search across 5,301+ emergency alert documents
- 🤖 GPT-4o-mini powered responses
- 📚 Source citations with links
- 🌐 External web search integration
- 💬 Chat history preservation
- ⚡ Real-time streaming responses

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **LLM**: OpenAI GPT-4o-mini
- **Vector DB**: ChromaDB
- **Embeddings**: OpenAI text-embedding-3-small
- **Search**: SerpAPI

---

Made with ❤️ for emergency communications research
