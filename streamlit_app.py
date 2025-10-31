

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# LangChain imports
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mxbai-embed-large")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")

# Config
PERSIST_DIR = "chroma_db"

st.set_page_config(page_title="RAG Agent - LangChain Docs", layout="wide")
st.title("🤖 RAG Agent - LangChain Documentation")

@st.cache_resource
def load_vectordb():
    """Load the persisted Chroma vectorstore."""
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL
    )
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectordb

# Load the vector database
try:
    vectordb = load_vectordb()
    st.success("✅ Vector database loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading vector database: {e}")
    st.info("Please run `python build_index.py` first to create the index.")
    st.stop()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Number of results to retrieve:", 1, 10, 4)
    search_mode = st.selectbox(
        "Search Mode:",
        ["Local Documentation", "Web Search + Documentation"]
    )
    st.divider()
    st.info(f"Using Ollama model: `{OLLAMA_MODEL}`")
    if TAVILY_API_KEY:
        st.success("✅ Tavily Search Available")
    else:
        st.warning("⚠️ Tavily API key not configured")

# Main search interface
st.subheader("🔍 Search LangChain Documentation")
query = st.text_input("Enter your question about LangChain:")

if query:
    try:
        results = []
        
        # Local Documentation Search
        local_results = vectordb.similarity_search(query, k=k)
        results.extend([(doc, "Local Docs") for doc in local_results])
        
        # Web Search if enabled
        if search_mode == "Web Search + Documentation" and TAVILY_API_KEY:
            try:
                from langchain_community.tools.tavily_search import TavilySearchResults
                
                tavily_search = TavilySearchResults(
                    max_results=3,
                    api_key=TAVILY_API_KEY
                )
                web_results = tavily_search.invoke({"query": query})
                
                if web_results:
                    st.info(f"🌐 Found {len(web_results)} web results from Tavily")
                    st.divider()
                    
                    for i, result in enumerate(web_results, 1):
                        with st.expander(f"🌐 Web Result {i}: {result.get('title', 'Result')}", expanded=False):
                            st.write(result.get('content', result))
                            if 'url' in result:
                                st.link_button("Visit Source", result['url'])
                    
                    st.divider()
            except Exception as e:
                st.warning(f"Web search unavailable: {str(e)}")
        
        # Display Local Results
        if local_results:
            st.success(f"Found {len(local_results)} relevant documents from Local Docs")
            st.divider()
            
            for i, doc in enumerate(local_results, 1):
                with st.expander(f"📄 Local Result {i}", expanded=(i==1)):
                    st.write(doc.page_content)
                    if doc.metadata:
                        st.caption(f"Source: {doc.metadata.get('source', 'Unknown')}")
        else:
            st.warning("No relevant documents found in local docs. Try a different query.")
    except Exception as e:
        st.error(f"❌ Error during search: {e}")

st.divider()
st.caption("Powered by LangChain, Chroma, and Ollama")
