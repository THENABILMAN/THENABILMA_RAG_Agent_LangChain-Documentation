

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

# LangChain imports
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mxbai-embed-large")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.2")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Config
PERSIST_DIR = "chroma_db"

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="RAG Agent - LangChain Docs", layout="wide")
st.title("🤖 RAG Agent - LangChain Documentation")

@st.cache_resource
def load_vectordb():
    """Load the persisted Chroma vectorstore with OpenRouter embeddings."""
    # Use OpenAI-compatible embeddings from OpenRouter
    embeddings = OpenAIEmbeddings(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="openai/text-embedding-3-small"
    )
    vectordb = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectordb

@st.cache_resource
def load_llm():
    """Load the LLM from OpenRouter."""
    if not OPENROUTER_API_KEY:
        st.error("❌ OPENROUTER_API_KEY not set!")
        st.info("Add your OpenRouter API key to .env file")
        st.stop()
    
    llm = ChatOpenAI(
        model="meta-llama/llama-3.2-3b-instruct:free",
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.7
    )
    return llm

# Load the vector database
try:
    vectordb = load_vectordb()
    st.success("✅ Vector database loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading vector database: {e}")
    st.info("Please run `python build_index.py` first to create the index.")
    st.stop()

# Load the LLM
try:
    llm = load_llm()
    st.success("✅ LLM loaded successfully! (OpenRouter - Llama 3.2 3B)")
except Exception as e:
    st.error(f"❌ Error loading LLM: {e}")
    st.info("Make sure OPENROUTER_API_KEY is set in .env")
    st.stop()

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Number of results to retrieve:", 1, 15, 6)
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

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("Enter your question about LangChain (or press Enter):", key="query_input")
with col2:
    search_button = st.button("🔍 Search", use_container_width=True)

# Trigger search on Enter key or button click
if query and (search_button):
    try:
        results = []
        
        # Local Documentation Search
        with st.spinner("🔎 Searching documentation..."):
            local_results = vectordb.similarity_search(query, k=k)
        results.extend([(doc, "Local Docs") for doc in local_results])
        st.success(f"✅ Found {len(local_results)} relevant documents")
        
        # Web Search if enabled
        if search_mode == "Web Search + Documentation" and TAVILY_API_KEY:
            try:
                with st.spinner("🌐 Searching the web..."):
                    from langchain_community.tools.tavily_search import TavilySearchResults
                    
                    tavily_search = TavilySearchResults(
                        max_results=3,
                        api_key=TAVILY_API_KEY
                    )
                    web_results = tavily_search.invoke({"query": query})
                
                if web_results:
                    st.info(f"✅ Found {len(web_results)} web results from Tavily")
                    st.divider()
                    
                    for i, result in enumerate(web_results, 1):
                        with st.expander(f"🌐 Web Result {i}: {result.get('title', 'Result')}", expanded=False):
                            st.write(result.get('content', result))
                            if 'url' in result:
                                st.link_button("Visit Source", result['url'])
                    
                    st.divider()
            except Exception as e:
                st.warning(f"Web search unavailable: {str(e)}")
        
        # Generate Answer using LLM
        if local_results:
            st.subheader("💡 Generated Answer")
            with st.spinner("⏳ Generating answer from OpenRouter LLM..."):
                # Prepare context from retrieved documents
                context = "\n\n".join([doc.page_content for doc in local_results])
                
                # Create a prompt for the LLM
                prompt = f"""Based on the following documentation context, answer the question concisely and accurately.

Context:
{context}

Question: {query}

Answer:"""
                
                try:
                    # Generate answer using LLM
                    answer = llm.invoke(prompt)
                    # Extract string content from AIMessage object
                    answer_text = answer.content if hasattr(answer, 'content') else str(answer)
                    st.markdown(answer_text)
                    st.success("✅ Answer generated successfully!")
                    
                    # Store in chat history
                    st.session_state.chat_history.append({
                        "question": query,
                        "answer": answer_text,
                        "sources": len(local_results)
                    })
                    
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")
            
            st.divider()
        
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
st.caption("Powered by LangChain, Chroma, and OpenRouter")

# Display Scrollable Chat History
if st.session_state.chat_history:
    st.divider()
    st.subheader("📋 Chat History")
    
    # Add clear history button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Total conversations: {len(st.session_state.chat_history)}")
    with col2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    
    st.divider()
    
    # Show full details in expandable sections
    st.subheader("📖 Previous Questions & Answers")
    for i, chat in enumerate(reversed(st.session_state.chat_history), 1):
        chat_num = len(st.session_state.chat_history) - i + 1
        with st.expander(f"Q{chat_num}: {chat['question'][:50]}...", expanded=False):
            st.write("**Full Question:**")
            st.write(chat['question'])
            st.write("**Full Answer:**")
            st.write(chat['answer'])
            st.caption(f"📚 Sources: {chat['sources']} documents")
