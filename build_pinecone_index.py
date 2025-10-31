import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "langchain-docs")
DOCS_FILE = Path("langchain_docs.txt")
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

def main():
    if not DOCS_FILE.exists():
        print(f"{DOCS_FILE} not found")
        return

    if not PINECONE_API_KEY:
        print("PINECONE_API_KEY not found in .env")
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

    print("Creating embeddings with HuggingFace...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    print(f"Uploading to Pinecone index: {PINECONE_INDEX_NAME}...")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    # Upload in batches - embed and upload each batch immediately
    batch_size = 50  # Smaller batch for faster processing
    total_docs = len(docs_chunks)
    
    for batch_start in range(0, total_docs, batch_size):
        batch_end = min(batch_start + batch_size, total_docs)
        batch_docs = docs_chunks[batch_start:batch_end]
        
        # Embed this batch
        vectors_batch = []
        for i, doc in enumerate(batch_docs):
            doc_id = batch_start + i
            embedding = embeddings.embed_query(doc.page_content)
            vectors_batch.append({
                "id": f"doc_{doc_id}",
                "values": embedding,
                "metadata": {"text": doc.page_content[:1000]}
            })
        
        # Upload this batch
        index.upsert(vectors=vectors_batch, namespace="langchain-docs")
        print(f"  ✓ Uploaded {batch_end}/{total_docs} vectors")
    
    print(f"✅ Successfully uploaded {len(docs_chunks)} chunks to Pinecone!")
    print(f"Index: {PINECONE_INDEX_NAME}")
    print(f"Namespace: langchain-docs")

if __name__ == "__main__":
    main()
