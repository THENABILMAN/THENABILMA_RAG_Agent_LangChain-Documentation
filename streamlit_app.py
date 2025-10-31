

import os
from dotenv import load_dotenv
import streamlit as st

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

load_dotenv()

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "langchain-docs")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="RAG Agent - LangChain Docs", layout="wide")
st.title("🤖 RAG Agent - LangChain Documentation")

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
    if TAVILY_API_KEY:
        st.success("✅ Tavily Search Available")
    else:
        st.warning("⚠️ Tavily API key not configured")
    
    # App Details Section
    st.divider()
    with st.expander("📱 About This App", expanded=False):
        st.markdown("""
        **RAG Agent - LangChain Documentation**
        
        Smart semantic search over LangChain docs with AI-powered answers.
        
        **Features:**
        - 🔍 45,547 document chunks indexed
        - 🤖 Llama 3.2 3B LLM answers
        - 💬 Full chat history tracking
        - 🌐 Optional web search
        - ⌨️ Enter key support
        - ☁️ Pinecone cloud database
        
        **Tech Stack:**
        - **Vector DB:** Pinecone (cloud-hosted)
        - **Embeddings:** HuggingFace (384 dims)
        - **LLM:** OpenRouter (Llama 3.2 3B)
        - **Framework:** LangChain + Streamlit
        
        **How It Works:**
        1. Query → 384-dim embedding
        2. Search Pinecone → Top 6 docs
        3. Add context to query
        4. LLM generates answer
        5. Display with sources
        
        **Performance:**
        - Query time: 1-3 sec
        - Answer time: 1-2 sec
        - Total: 3-5 sec per query
        
        **Creator:** [@THENABILMAN](https://github.com/THENABILMAN)
        
        **Repo:** [GitHub](https://github.com/THENABILMAN/THENABILMA_RAG_Agent_LangChain-Documentation)
        """)

# Main search interface
st.subheader("🔍 Search LangChain Documentation")

# Use form for Enter key support
with st.form("search_form", clear_on_submit=False):
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input("Enter your question about LangChain:", key="query_input")
    with col2:
        st.write("")  # Spacer for vertical alignment
        search_button = st.form_submit_button("🔍 Search", use_container_width=True)

# Trigger search on Enter key or button click
if query and search_button:
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
                context = "\n\n".join([result["metadata"].get("text", "") for result in local_results])
                
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
            
            for i, result in enumerate(local_results, 1):
                with st.expander(f"📄 Local Result {i}", expanded=(i==1)):
                    text = result["metadata"].get("text", "No content")
                    st.write(text)
                    st.caption(f"Score: {result.get('score', 'N/A'):.4f}")
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
