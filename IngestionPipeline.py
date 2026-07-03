import os
import hashlib
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Cache the embedding model so it's only loaded once
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Cache the persistent Chroma client
@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path="chroma_db")

def overlapping_chunks(text, n=3, overlap=1):
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    step = n - overlap
    chunks = []
    for i in range(0, len(sentences), step):
        group = sentences[i:i + n]
        if group:
            chunks.append(". ".join(group) + ".")
    return chunks

def build_collection(text, file_name, embed_model):
    client = get_chroma_client()
    
    # Create a unique collection name based on the content hash to avoid re-indexing if uploaded again
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    collection_name = f"doc_{content_hash}"
    
    collection = client.get_or_create_collection(name=collection_name)
    chunk_count = collection.count()
    
    # If the collection doesn't have any chunks inside, parse, embed, and load them
    if chunk_count == 0:
        chunks = overlapping_chunks(text)
        embeddings = embed_model.encode(chunks).tolist()
        ids = [f"c_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids)
        chunk_count = len(chunks)
        
    return collection, chunk_count

# UI ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Document QA Ingestion", layout="centered")

st.title("Document Q&A Ingestion Pipeline")

embed_model = load_embed_model()

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

if uploaded_file:
    # Read and decode the file
    text = uploaded_file.read().decode("utf-8")
    
    try:
        with st.spinner("Indexing document..."):
            collection, chunk_count = build_collection(text, uploaded_file.name, embed_model)
        
        st.success(f"Indexed {chunk_count} chunks from {uploaded_file.name}")
    except Exception as e:
        st.error(f"Failed to index document: {e}")
        st.stop()
        
    # Collapsible document preview
    with st.expander("Document preview"):
        preview_text = text[:500] + "..." if len(text) > 500 else text
        st.text(preview_text)
else:
    st.info("Upload a document to get started.")