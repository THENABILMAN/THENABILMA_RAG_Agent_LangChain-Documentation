import os
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "mxbai-embed-large")

# Config
DOCS_FILE = Path("langchain_docs.txt")
PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 600      # Optimized for better semantic boundaries
CHUNK_OVERLAP = 100   # Maintains context continuity

def load_documents(docs_file: Path):
    """Load the langchain_docs.txt file."""
    if not docs_file.exists():
        raise FileNotFoundError(f"{docs_file} not found")
    
    loader = TextLoader(str(docs_file), encoding="utf-8")
    documents = loader.load()
    return documents

def main():
    docs_file = DOCS_FILE
    if not docs_file.exists():
        print(f"{docs_file} not found — please create the file.")
        return

    print("Loading documents...")
    docs = load_documents(docs_file)
    print(f"Loaded {len(docs)} document(s)")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    print("Splitting documents into chunks...")
    docs_chunks = text_splitter.split_documents(docs)
    print(f"Created {len(docs_chunks)} document chunks")

    print(f"Creating embeddings using Ollama ({OLLAMA_MODEL})...")
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL
    )
    print("Creating/persisting Chroma vectorstore...")
    vectordb = Chroma.from_documents(
        documents=docs_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    print(f"✅ Chroma index persisted at: {PERSIST_DIR}")

if __name__ == "__main__":
    main()
