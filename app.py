import streamlit as st
import os
import faiss
import pickle
import numpy as np
import pdfplumber
import docx
import pandas as pd
from sentence_transformers import SentenceTransformer
from groq import Groq

# STREAMLIT CONFIG
st.set_page_config(page_title="AI Legal Assistant", layout="centered")
st.title("⚖️ AI Legal Assistant")
st.caption("Legal Document Q&A — Summarization & Obligation Extraction")

# LOAD FAISS + METADATA
@st.cache_resource
def load_faiss():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    FAISS_PATH = os.path.join(BASE_DIR, "model", "index.faiss")
    META_PATH  = os.path.join(BASE_DIR, "model", "metadata.pkl")
    index = faiss.read_index(FAISS_PATH)
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

index, metadata = load_faiss()

# LOAD EMBEDDING MODEL
@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

# GROQ CLIENT
client = Groq(api_key=st.secrets["GROQ_API_KEY"])

SYSTEM_PROMPT = """You are an expert AI Legal Assistant specializing in contract law, corporate agreements, and legal document analysis.

Your job is to:
- Summarize legal documents clearly and accurately
- Extract obligations, rights, and responsibilities of each party
- Identify key clauses: payment terms, termination, confidentiality, liability
- Answer specific legal questions based on document content
- Answer general legal questions from your knowledge

Always be precise, structured, and avoid hallucination. If information is not in the document, say so clearly."""

def ask_groq(context, question):
    if context:
        user_message = f"""Legal Document:
{context}

Question: {question}

Provide a clear, accurate, and well-structured answer based on the document above."""
    else:
        user_message = f"""Question: {question}

Answer this legal question accurately from your legal knowledge."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        temperature=0.1,
        max_tokens=1024
    )
    return response.choices[0].message.content

# RAG FUNCTIONS
def retrieve(query, top_k=4):
    emb = embed_model.encode([query])
    _, ids = index.search(np.array(emb).astype("float32"), top_k)
    return [metadata[i]["text"] for i in ids[0]]

def answer_from_index(query):
    chunks = retrieve(query)
    context = "\n\n".join(chunks)
    return ask_groq(context, query)

# FILE EXTRACTION
def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_docx(file):
    d = docx.Document(file)
    return "\n".join(p.text for p in d.paragraphs)

def extract_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

def answer_from_uploaded_doc(text, question):
    # Chunk the document
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    # Find most relevant chunks via FAISS
    embeddings = embed_model.encode(chunks)
    temp_index = faiss.IndexFlatL2(embeddings.shape[1])
    temp_index.add(np.array(embeddings).astype("float32"))

    q_emb = embed_model.encode([question])
    _, ids = temp_index.search(np.array(q_emb).astype("float32"), 4)

    # Use top 4 relevant chunks as context
    context = "\n\n".join([chunks[i] for i in ids[0]])
    return ask_groq(context, question)

# SIDEBAR
with st.sidebar:
    st.header("📄 Upload Document (Optional)")
    st.caption("Upload to ask questions about a specific document")
    uploaded_file = st.file_uploader("PDF / DOCX / XLSX", type=["pdf", "docx", "xlsx"])

    if uploaded_file:
        st.success(f"✅ {uploaded_file.name} uploaded")

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("💡 Try asking:")
    st.caption("• Summarize this agreement")
    st.caption("• What are the obligations of each party?")
    st.caption("• What are the termination clauses?")
    st.caption("• What is the payment terms?")
    st.caption("• What is bail in Indian law?")

# CHAT STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# CHAT INPUT
prompt = st.chat_input("Ask a legal question...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            try:
                if uploaded_file:
                    if uploaded_file.name.endswith(".pdf"):
                        text = extract_pdf(uploaded_file)
                    elif uploaded_file.name.endswith(".docx"):
                        text = extract_docx(uploaded_file)
                    else:
                        text = extract_excel(uploaded_file)
                    answer = answer_from_uploaded_doc(text, prompt)
                else:
                    answer = answer_from_index(prompt)
            except Exception as e:
                answer = f"❌ Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
