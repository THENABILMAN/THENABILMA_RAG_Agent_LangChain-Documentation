import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
DOCS_FILE = Path("langchain_docs.txt")
PERSIST_DIR = "chroma_db"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

def main():
    if not DOCS_FILE.exists():
        print(f"{DOCS_FILE} not found")
        return

    print("Loading documents...")
    loader = TextLoader(str(DOCS_FILE), encoding="utf-8")
    docs = loader.load()
    print(f"Loaded {len(docs)} document(s)")

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    docs_chunks = text_splitter.split_documents(docs)
    print(f"Created {len(docs_chunks)} document chunks")

    print("Creating embeddings with OpenRouter...")
    embeddings = OpenAIEmbeddings(
        api_key=OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model="openai/text-embedding-3-small"
    )
    
    print("Building Chroma vectorstore...")
    vectordb = Chroma.from_documents(
        documents=docs_chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    print(f"✅ Chroma index persisted at: {PERSIST_DIR}")

if __name__ == "__main__":
    main()