

import os
from dotenv import load_dotenv
import streamlit as st
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

load_dotenv()

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "langchain-docs")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

st.set_page_config(
    page_title="RAG Assistant - LangChain Docs",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "RAG Assistant powered by LangChain, Pinecone, and OpenRouter"
    }
)

# Custom CSS for modern interface
st.markdown("""
    <style>
    .main {
        max-width: 900px;
    }
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #3f0f64 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .header-container h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .header-container p {
        margin: 0.5rem 0 0 0;
        opacity: 1;
        font-size: 1.1rem;
        color: #e2e8f0;
    }
    .message-container {
        margin-bottom: 1.5rem;
        padding: 1.2rem;
        border-radius: 12px;
        animation: fadeIn 0.3s ease-in;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    }
    .user-message {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b0764 100%);
        color: #ffffff;
        margin-left: 2rem;
        border-left: 4px solid #60a5fa;
    }
    .user-message strong {
        display: block;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        opacity: 1;
        color: #dbeafe;
    }
    .assistant-message {
        background: linear-gradient(135deg, #2d1b4e 0%, #1e1b4b 100%);
        color: #ffffff;
        margin-right: 2rem;
        border-left: 4px solid #a855f7;
        line-height: 1.6;
    }
    .assistant-message strong {
        display: block;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        opacity: 1;
        color: #e9d5ff;
    }
    .source-container {
        background: #1e293b;
        border-left: 4px solid #6d28d9;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #e2e8f0;
    }
    .source-title {
        font-weight: 600;
        color: #a78bfa;
        margin-bottom: 0.5rem;
    }
    .stat-box {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem;
    }
    .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #ffffff;
    }
    .stat-label {
        font-size: 0.85rem;
        opacity: 1;
        color: #bfdbfe;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_vectordb():
    """Load Pinecone vector database with HuggingFace embeddings."""
    if not PINECONE_API_KEY:
        st.error("❌ PINECONE_API_KEY not configured!")
        st.stop()
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    return {"index": index, "embeddings": embeddings}

@st.cache_resource
def load_llm():
    """Load the LLM from Groq."""
    if not GROQ_API_KEY:
        st.error("❌ GROQ_API_KEY not set!")
        st.info("Add your Groq API key to .env file")
        st.stop()
    
    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
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

# Sidebar for settings and configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model settings
    st.subheader("🤖 Model Settings")
    k = st.slider("Documents to retrieve:", min_value=1, max_value=15, value=6)
    temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    
    st.divider()
    
    # Search mode
    st.subheader("🔍 Search Options")
    search_mode = st.selectbox(
        "Search Mode:",
        ["Local Documentation Only", "Local + Web Search"]
    )
    
    if TAVILY_API_KEY:
        st.success("✅ Web search enabled")
    else:
        st.warning("⚠️ Web search disabled")
    
    st.divider()
    
    # System status with detailed info
    st.subheader("📊 System Status")
    
    # Add animation CSS
    st.markdown("""
    <style>
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.15); opacity: 0.8; }
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Create status boxes
    col1, col2 = st.columns(2)
    
    with col1:
        if PINECONE_API_KEY:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); padding: 1.5rem; border-radius: 12px; text-align: center; color: white; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4); border: 2px solid rgba(255,255,255,0.3);">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">✅</div>
                <div style="font-size: 1rem; font-weight: 700; margin-bottom: 0.3rem; color: #ffffff;">Pinecone</div>
                <div style="font-size: 0.85rem; opacity: 1; background: rgba(255,255,255,0.15); padding: 0.4rem 0.8rem; border-radius: 6px; margin-top: 0.5rem; color: #d1fae5;">🟢 Connected</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); padding: 1.5rem; border-radius: 12px; text-align: center; color: white; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4); border: 2px solid rgba(255,255,255,0.3);">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">❌</div>
                <div style="font-size: 1rem; font-weight: 700; margin-bottom: 0.3rem; color: #ffffff;">Pinecone</div>
                <div style="font-size: 0.85rem; opacity: 1; background: rgba(255,255,255,0.15); padding: 0.4rem 0.8rem; border-radius: 6px; margin-top: 0.5rem; color: #fecaca;">🔴 Disconnected</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if GROQ_API_KEY:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #065f46 0%, #047857 100%); padding: 1.5rem; border-radius: 12px; text-align: center; color: white; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.4); border: 2px solid rgba(255,255,255,0.3);">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">✅</div>
                <div style="font-size: 1rem; font-weight: 700; margin-bottom: 0.3rem; color: #ffffff;">Groq LLM</div>
                <div style="font-size: 0.85rem; opacity: 1; background: rgba(255,255,255,0.15); padding: 0.4rem 0.8rem; border-radius: 6px; margin-top: 0.5rem; color: #d1fae5;">🟢 Ready</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%); padding: 1.5rem; border-radius: 12px; text-align: center; color: white; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4); border: 2px solid rgba(255,255,255,0.3);">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">❌</div>
                <div style="font-size: 1rem; font-weight: 700; margin-bottom: 0.3rem; color: #ffffff;">Groq LLM</div>
                <div style="font-size: 0.85rem; opacity: 1; background: rgba(255,255,255,0.15); padding: 0.4rem 0.8rem; border-radius: 6px; margin-top: 0.5rem; color: #fecaca;">🔴 Not Ready</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # Chat Statistics
    st.subheader("📈 Chat Statistics")
    stat_cols = st.columns(3)
    
    with stat_cols[0]:
        st.metric("Messages", len(st.session_state.messages))
    
    with stat_cols[1]:
        st.metric("Conversations", len(st.session_state.chat_history))
    
    with stat_cols[2]:
        st.metric("K-value", k)
    
    st.divider()
    
    # Clear history button
    st.subheader("🗑️ History Management")
    if st.button("🗑️ Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.success("Chat history cleared!", icon="✅")
        st.rerun()
    
    st.divider()
    
    # About section with more details
    with st.expander("ℹ️ About This App", expanded=False):
        st.markdown("""
        **RAG Assistant - LangChain Edition**
        
        Intelligent semantic search over LangChain documentation powered by AI.
        
        **✨ Features:**
        - 🔍 Semantic vector search
        - 🤖 AI-powered answer generation
        - 💬 Full conversation history
        - 🌐 Optional web search integration
        - ⚡ Real-time streaming responses
        - 📊 Statistics tracking
        
        **🛠️ Tech Stack:**
        - **Vector DB:** Pinecone (cloud)
        - **Embeddings:** HuggingFace (sentence-transformers)
        - **LLM:** Groq (Llama 3.3 70B)
        - **Framework:** Streamlit + LangChain
        
        **🚀 How It Works:**
        1. Your question is converted to embeddings
        2. Semantic search retrieves relevant documents
        3. Context is provided to the LLM
        4. AI generates a contextual answer
        5. Sources are displayed for verification
        
        **⏱️ Performance:**
        - Search: 1-3 seconds
        - Generation: 1-2 seconds  
        - Total: 2-5 seconds
        
        **👨‍💻 Creator:** [@THENABILMAN](https://github.com/THENABILMAN)
        
        **📦 Repository:** [GitHub](https://github.com/THENABILMAN/THENABILMA_RAG_Agent_LangChain-Documentation)
        """)

# Main Header
st.markdown("""
    <div class="header-container">
        <h1>📚 Langchain Doc Assistant</h1>
        <p>Ask anything about LangChain documentation</p>
    </div>
""", unsafe_allow_html=True)

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    if message["role"] == "user":
        st.markdown(f"""
            <div class="message-container user-message">
                <strong>👤 You</strong>
                {message['content']}
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div class="message-container assistant-message">
                <strong>🤖 Assistant</strong>
                {message['content']}
            </div>
        """, unsafe_allow_html=True)
        
        # Show sources if available
        if "sources" in message and message["sources"]:
            with st.expander(f"📚 View {len(message['sources'])} Sources", expanded=False):
                for j, source in enumerate(message["sources"], 1):
                    st.markdown(f"""
                        <div class="source-container">
                            <div class="source-title">📄 Source {j}</div>
                            {source[:500]}{'...' if len(source) > 500 else ''}
                        </div>
                    """, unsafe_allow_html=True)

# Input area
st.divider()
st.subheader("Ask a Question")

col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input(
        "Your question:",
        placeholder="e.g., How do I use LangChain with OpenAI?",
        label_visibility="collapsed"
    )
with col2:
    search_button = st.button("Send", use_container_width=True, type="primary")

# Process user input
if user_input and search_button:
    query = user_input  # Use user_input as query
    
    # Add user message to chat
    st.session_state.messages.append({
        "role": "user",
        "content": query
    })
    
    try:
        # Local Documentation Search with Pinecone
        with st.spinner("🔎 Searching documentation..."):
            db = load_vectordb()
            index = db["index"]
            embeddings = db["embeddings"]
            
            # Get query embedding
            query_embedding = embeddings.embed_query(query)
            
            # Query Pinecone
            results = index.query(
                vector=query_embedding,
                top_k=k,
                namespace="langchain-docs",
                include_metadata=True
            )
            
            local_results = results.get("matches", [])
        
        # Web Search if enabled
        web_results = []
        if search_mode == "Local + Web Search" and TAVILY_API_KEY:
            try:
                with st.spinner("🌐 Searching the web..."):
                    from langchain_community.tools.tavily_search import TavilySearchResults
                    
                    tavily_search = TavilySearchResults(
                        max_results=3,
                        api_key=TAVILY_API_KEY
                    )
                    web_results = tavily_search.invoke({"query": query})
            except Exception as e:
                st.warning(f"Web search unavailable: {str(e)}")
        
        # Generate Answer using LLM
        if local_results:
            # Prepare context from retrieved documents
            context = "\n\n".join([result["metadata"].get("text", "") for result in local_results])
            
            # Create a prompt for the LLM
            prompt = f"""Based on the following documentation context, answer the question concisely and accurately.

Context:
{context}

Question: {query}

Answer:"""
            
            try:
                # Generate answer using LLM
                llm_with_temp = ChatOpenAI(
                    model="llama-3.3-70b-versatile",
                    api_key=GROQ_API_KEY,
                    base_url="https://api.groq.com/openai/v1",
                    temperature=temperature
                )
                
                with st.spinner("⏳ Generating answer..."):
                    answer = llm_with_temp.invoke(prompt)
                    answer_text = answer.content if hasattr(answer, 'content') else str(answer)
                
                # Display answer word by word with streaming effect
                answer_placeholder = st.empty()
                displayed_text = ""
                words = answer_text.split()
                
                for word in words:
                    displayed_text += word + " "
                    answer_placeholder.markdown(f"""
                        <div class="assistant-message">
                            <strong>🤖 Assistant:</strong> {displayed_text}
                        </div>
                    """, unsafe_allow_html=True)
                    import time
                    time.sleep(0.02)  # Faster streaming (20ms between words)
                
                # Add assistant message with sources
                sources = [result["metadata"].get("text", "") for result in local_results[:3]]
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources
                })
                
                # Store in chat history
                st.session_state.chat_history.append({
                    "question": query,
                    "answer": answer_text,
                    "sources": len(local_results),
                    "timestamp": datetime.now().isoformat()
                })
                
                st.rerun()
                    
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
        else:
            st.warning("No relevant documents found. Try a different question.")
    
    except Exception as e:
        st.error(f"❌ Error during search: {e}")

# Footer
st.divider()

footer_cols = st.columns([1, 1, 1])

with footer_cols[0]:
    st.metric("📊 API Model", "Llama 3.3 70B", help="Groq LLM")

with footer_cols[1]:
    st.metric("🗄️ Vector Store", "Pinecone", help="Cloud-hosted vector database")

with footer_cols[2]:
    st.metric("⚙️ Framework", "Streamlit", help="Python app framework")

# Enhanced Footer Section
footer_html = """
<div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); border-top: 2px solid #3f0f64; margin-top: 2rem; padding: 2rem; border-radius: 12px; text-align: center;">
    <div style="margin-bottom: 1.5rem;">
        <p style="font-size: 1.1rem; margin: 0.5rem 0; font-weight: 600; color: #e0e7ff;">
            💡 <strong>Powered by Intelligence</strong>
        </p>
        <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin: 1rem 0; font-size: 0.95rem;">
            <span style="background: rgba(59, 130, 246, 0.2); padding: 0.5rem 1rem; border-radius: 6px; color: #bfdbfe; border: 1px solid rgba(59, 130, 246, 0.5);">⚡ Streamlit</span>
            <span style="background: rgba(59, 130, 246, 0.2); padding: 0.5rem 1rem; border-radius: 6px; color: #bfdbfe; border: 1px solid rgba(59, 130, 246, 0.5);">🔗 LangChain</span>
            <span style="background: rgba(168, 85, 247, 0.2); padding: 0.5rem 1rem; border-radius: 6px; color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.5);">📍 Pinecone</span>
            <span style="background: rgba(168, 85, 247, 0.2); padding: 0.5rem 1rem; border-radius: 6px; color: #d8b4fe; border: 1px solid rgba(168, 85, 247, 0.5);">🚀 OpenRouter</span>
            <span style="background: rgba(34, 197, 94, 0.2); padding: 0.5rem 1rem; border-radius: 6px; color: #86efac; border: 1px solid rgba(34, 197, 94, 0.5);">🤖 HuggingFace</span>
        </div>
    </div>
    <div style="border-top: 1px solid rgba(255, 255, 255, 0.1); padding-top: 1.5rem; margin-top: 1.5rem;">
        <p style="font-size: 1rem; color: #e2e8f0; margin: 0.5rem 0; font-weight: 600;">
            RAG Assistant - LangChain Documentation Search
        </p>
        <p style="font-size: 0.9rem; color: #cbd5e1; margin: 0.5rem 0;">
            Semantic search powered by AI | Real-time answers from documentation
        </p>
        <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 1rem; margin-bottom: 0.5rem;">
            © 2025 RAG Assistant | Built with ❤️ for developers
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; font-size: 0.9rem; flex-wrap: wrap;">
            <a href="https://github.com/THENABILMAN" target="_blank" style="color: #60a5fa; text-decoration: none;">👨‍💻 Creator: @THENABILMAN</a>
            <span style="color: #475569;">|</span>
            <a href="https://github.com/THENABILMAN/THENABILMA_RAG_Agent_LangChain-Documentation" target="_blank" style="color: #60a5fa; text-decoration: none;">📦 GitHub Repository</a>
        </div>
    </div>
    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
        <p style="font-size: 0.8rem; color: #64748b; margin: 0;">
            🔐 Privacy First | ⚡ Fast & Efficient | 🎯 Accurate Results | 🌍 Global Scale
        </p>
    </div>
</div>
"""

st.markdown(footer_html, unsafe_allow_html=True)