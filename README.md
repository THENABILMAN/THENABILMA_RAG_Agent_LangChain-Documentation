# 🤖 RAG Agent - LangChain Documentation

A production-ready **Retrieval-Augmented Generation (RAG)** application for querying LangChain documentation with AI-powered answers. Deployed on Streamlit Cloud with Pinecone vector database.

## 🌟 Features

- **🔍 Semantic Search**: Search 45,547 document chunks across LangChain documentation
- **🤖 AI-Powered Answers**: Generate accurate, context-aware responses using OpenRouter LLM
- **💬 Chat History**: Full conversation tracking with expandable Q&A sections
- **🌐 Web Search Integration**: Optional Tavily Search for real-time information
- **⌨️ Keyboard Support**: Press Enter to submit queries instantly
- **📊 Real-time Feedback**: Processing indicators at each step
- **☁️ Cloud-Ready**: Deployed on Streamlit Cloud with Pinecone
- **🔐 Secure**: All API keys stored in environment variables

## 🏗️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector DB** | Pinecone (Cloud) | Scalable, cloud-hosted vector storage |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) | 384-dim semantic embeddings (free, local) |
| **LLM** | OpenRouter (Llama 3.2 3B) | Fast, free inference for answer generation |
| **Framework** | LangChain 1.0.3 | RAG orchestration & document processing |
| **Web UI** | Streamlit 1.40+ | Interactive interface with real-time updates |
| **Web Search** | Tavily Search API | Optional web context for recent information |
| **Python** | 3.12 | Latest features with zero dependency conflicts |

## 📊 Architecture

```
User Query
    ↓
[Streamlit UI]
    ↓
[HuggingFace Embeddings] → Query vector (384 dims)
    ↓
[Pinecone Cloud] → Similarity search (top 6 results)
    ↓
[Context Assembly] → Retrieved document chunks
    ↓
[OpenRouter LLM] → Generate answer
    ↓
[Display Results] → Answer + Sources + Chat History
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Pinecone API Key (free tier available)
- OpenRouter API Key (free tier available)
- Tavily API Key (optional, for web search)

### Installation

1. **Clone Repository**
```bash
git clone https://github.com/THENABILMAN/THENABILMA_RAG_Agent_LangChain-Documentation.git
cd "Lang rag ai"
```

2. **Create Environment**
```bash
conda create -n chainrag12 python=3.12
conda activate chainrag12
pip install -r requirements.txt
```

3. **Configure Secrets**
Create `.env` file:
```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=langchain-docs
OPENROUTER_API_KEY=your_openrouter_api_key
TAVILY_API_KEY=your_tavily_api_key (optional)
```

4. **Build Index** (First time only)
```bash
python build_pinecone_index.py
```
This uploads 45,547 document chunks to Pinecone (takes ~15 minutes).

5. **Run App**
```bash
streamlit run streamlit_app.py
```
Visit `http://localhost:8501`

## 📝 Usage

### Basic Query
1. Type your LangChain question in the search box
2. Press **Enter** or click **🔍 Search**
3. Get instant semantic results + AI-generated answer

### Advanced Features

**Search Modes:**
- **Local Documentation**: Search only LangChain docs
- **Web Search + Documentation**: Combine web results + docs

**Settings (Sidebar):**
- **Number of Results**: 1-15 documents (default: 6)
- **Search Mode**: Local or Web+Local
- **Tavily Status**: Check if web search is available

**Chat History:**
- View all previous Q&A
- Expand sections to see full question + answer
- See document sources for each query
- Clear entire history with one click

## 📦 Dependencies

Key packages with pinned versions (Python 3.12 compatible):

```
langchain==1.0.3                    # RAG orchestration
langchain-core==1.0.2               # Core components
langchain-community==0.4.1          # Community integrations
langchain-openai>=1.0.0             # OpenAI/OpenRouter LLM
langchain-huggingface==1.0.0        # HuggingFace embeddings
pinecone-client>=3.2.0              # Pinecone cloud DB
sentence-transformers>=2.2.0        # Embedding models
streamlit>=1.40.0                   # Web UI
python-dotenv>=1.0.0                # Environment variables
tavily-python>=0.3.0                # Web search (optional)
```

## 🗂️ Project Structure

```
Lang rag ai/
├── streamlit_app.py              # Main web application
├── build_pinecone_index.py       # Index building script
├── langchain_docs.txt             # Document source (~15MB)
├── requirements.txt               # Python dependencies
├── .env                           # API keys (git-ignored)
├── .gitignore                     # Git configuration
└── README.md                      # This file
```

## 🔧 Configuration

### Chunking Strategy
- **Chunk Size**: 600 characters (semantic boundaries)
- **Overlap**: 100 characters (context preservation)
- **Total Chunks**: 45,547
- **Retrieval**: 6 results default, max 15

### Embedding Model
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions**: 384
- **Device**: CPU (faster, no GPU needed)
- **Inference**: Local (no API calls)

### LLM Configuration
- **Model**: Meta Llama 3.2 3B Instruct
- **Provider**: OpenRouter
- **Temperature**: 0.7 (balanced creativity)
- **Cost**: Free tier available
- **Speed**: ~1-2 seconds per response

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| Documents Indexed | 1 (LangChain docs) |
| Document Chunks | 45,547 |
| Embedding Dimensions | 384 |
| Average Query Time | 1-3 seconds |
| Answer Generation | 1-2 seconds |
| Total Time Per Query | 3-5 seconds |
| Pinecone Storage | ~35MB |
| Local Disk (excluding chroma_db) | ~15MB |

## 🌐 Deployment on Streamlit Cloud

### Setup Secrets
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Deploy your GitHub repository
3. Add secrets in **Settings → Secrets**:
```
PINECONE_API_KEY = "your_key"
OPENROUTER_API_KEY = "your_key"
TAVILY_API_KEY = "your_key"
PINECONE_INDEX_NAME = "langchain-docs"
```

### Benefits
- ✅ Free hosting (Streamlit Community Cloud)
- ✅ Cloud vector DB (Pinecone) - no local database
- ✅ Automatic deployments from GitHub
- ✅ 24/7 availability
- ✅ Shareable public URL

## 🛠️ Development

### Add Custom Documents
1. Combine documents into a single `.txt` file
2. Update `build_pinecone_index.py`:
```python
DOCS_FILE = Path("your_docs.txt")
```
3. Run `python build_pinecone_index.py`
4. Restart app

### Modify Chunking
Edit `build_pinecone_index.py`:
```python
CHUNK_SIZE = 600      # Adjust chunk size
CHUNK_OVERLAP = 100   # Adjust overlap
```

### Change Retrieval Count
Edit `streamlit_app.py` sidebar:
```python
k = st.slider("Number of results:", 1, 15, 6)  # Change default (6)
```

## 🐛 Troubleshooting

### Issue: "Resource langchain-docs not found"
**Solution**: Create Pinecone index first:
- Go to [app.pinecone.io](https://app.pinecone.io)
- Create index: Name="langchain-docs", Dimension=384, Metric="cosine"

### Issue: "PINECONE_API_KEY not configured"
**Solution**: Add to `.env`:
```
PINECONE_API_KEY=pk-...
```

### Issue: Slow embedding generation
**Solution**: It's normal for first run. HuggingFace downloads model (~500MB) on first use.

### Issue: Web search not working
**Solution**: Add `TAVILY_API_KEY` to `.env` (optional feature)

## 📚 Key Concepts

### RAG (Retrieval-Augmented Generation)
1. **Retrieve**: Find relevant documents via semantic search
2. **Augment**: Add context to the query
3. **Generate**: Create answer using LLM with context

### Vector Database
- Stores embeddings of all documents
- Enables fast semantic similarity search
- Pinecone: managed cloud service (scalable, reliable)

### Semantic Search vs Keyword Search
- **Keyword**: Exact word matching (limited)
- **Semantic**: Meaning-based matching (powerful)
- Example: "How to use chains?" → Finds relevant docs even if exact words aren't present

## 📖 API Keys Setup

### Pinecone
1. Sign up: [pinecone.io](https://pinecone.io)
2. Create free index (dimension 384)
3. Copy API key

### OpenRouter
1. Sign up: [openrouter.ai](https://openrouter.ai)
2. Free tier: $5 credit
3. Copy API key

### Tavily (Optional)
1. Sign up: [tavily.com](https://tavily.com)
2. Free tier: 1,000 searches/month
3. Copy API key

## 👨‍💻 Creator

**Nabil** ([@THENABILMAN](https://github.com/THENABILMAN))

## 📄 License

MIT License - Free to use and modify

## 🤝 Contributing

Contributions welcome! Feel free to:
- Report bugs
- Suggest features
- Improve documentation
- Submit pull requests

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Open GitHub issue
3. Contact creator

## 🎯 Roadmap

- [ ] Add PDF document support
- [ ] Support multiple document sources
- [ ] Add query refinement suggestions
- [ ] Implement answer citations
- [ ] Add multilingual support
- [ ] Custom LLM model selection
- [ ] Rate limiting and analytics
- [ ] Dark mode UI

---

**Last Updated**: October 2025

**Status**: ✅ Production Ready | 🚀 Deployed | 📈 Optimized

Made with ❤️ for the LangChain community
