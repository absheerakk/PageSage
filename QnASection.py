import os
import hashlib
import streamlit as st
import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Cache the embedding model so it's only loaded once
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Cache the Generative Model
@st.cache_resource
def load_llm():
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=(
            "Answer using only the context provided. "
            "If the answer is not in the context, say 'I don't have that information.' "
            "Always be factual and grounded strictly in the context."
        ),
    )

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

def ask(question, collection, embed_model, llm):
    try:
        # Prevent querying more chunks than are available in short documents
        n_results = min(3, collection.count())
        if n_results == 0:
            return "No document content available.", []
            
        results = collection.query(
            query_embeddings=embed_model.encode([question]).tolist(),
            n_results=n_results,
        )
        
        used_chunks = results["documents"][0]
        context = "\n\n".join(f"- {c}" for c in used_chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {question}"
        
        response = llm.generate_content(prompt)
        answer = response.text.strip()
        
        return answer, used_chunks
        
    except Exception as e:
        st.error(f"API Error during generation: {e}")
        st.stop()

# ── UI ────────────────────────────────────────────────────────────

st.set_page_config(page_title="Document QA App", layout="centered")

st.title("Document Q&A Web App")

# Load models
embed_model = load_embed_model()
llm = load_llm()

uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])

collection = None
chunk_count = 0

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

# Question input (disabled until document is loaded and collection exists)
question = st.text_input(
    "Ask a question about the document:",
    disabled=(collection is None),
    placeholder="Upload a document first..." if collection is None else "Type your question...",
)

# Generate and display Answer & Sources
if collection and question.strip():
    with st.spinner("Thinking..."):
        answer, sources = ask(question, collection, embed_model, llm)
    
    st.markdown("### Answer")
    st.write(answer)
    
    st.markdown("### Sources")
    for i, source in enumerate(sources, 1):
        st.markdown(f"**[{i}]** {source}")