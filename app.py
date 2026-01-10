import streamlit as st
import os
import faiss
import pickle
import numpy as np
import re

import pdfplumber
import docx
import pandas as pd

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Legal Assistant",
    layout="wide"
)

st.title("AI-Based Legal Document Summarizer")
st.caption("Legal Document Analysis & Q&A")


# =========================================================
# LOAD FAISS + METADATA
# =========================================================

@st.cache_resource
def load_faiss():
    base = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(base, "faiss_index", "index.faiss")
    meta_path = os.path.join(base, "faiss_index", "metadata.pkl")

    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    return index, metadata

index, metadata = load_faiss()


# =========================================================
# LOAD MODELS
# =========================================================

@st.cache_resource
def load_models():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    summarizer = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        device=-1
    )
    return embed_model, summarizer

embed_model, summarizer = load_models()


# =========================================================
# UTILITIES
# =========================================================

def clean_text(text):
    text = re.sub(r'\b[a-d]\)\s*', '', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


def format_output(result):
    out = "### 🧾 Executive Summary\n"
    out += result["executive_summary"] + "\n\n"

    out += "### ⚠️ Key Risks\n"
    for r in result["key_risks"]:
        out += f"- {r}\n"

    out += "\n### 📌 Obligations\n"
    for o in result["obligations"]:
        out += f"- {o}\n"

    return out


# =========================================================
# RAG (INDEXED DOCS)
# =========================================================

def retrieve(query, top_k=3):
    emb = embed_model.encode([query])
    _, ids = index.search(np.array(emb).astype("float32"), top_k)
    return [metadata[i]["text"] for i in ids[0]]


def aggregate_from_chunks(chunks):
    summaries = []

    for c in chunks:
        prompt = f"""
You are a legal analyst.

Extract obligations and risks from the clause below.
Do not repeat text. Use bullet points.

Clause:
{c[:1000]}

Answer format:
Obligations:
- ...

Risks:
- ...
"""
        res = summarizer(prompt, max_new_tokens=180)[0]["generated_text"]
        summaries.append(clean_text(res))

    risks, obligations = [], []

    for s in summaries:
        low = s.lower()
        if "indemn" in low or "liable" in low or "risk" in low:
            risks.append(s)
        if "shall" in low or "must" in low:
            obligations.append(s)

    return {
        "executive_summary": f"{len(obligations)} obligations and {len(risks)} risks identified.",
        "key_risks": list(set(risks)),
        "obligations": list(set(obligations))
    }


# =========================================================
# FILE EXTRACTION
# =========================================================

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
    df = pd.read_excel(file, engine="openpyxl")
    return df.to_string()


# =========================================================
# SIDEBAR (UPLOAD + CONTROLS)
# =========================================================

with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader(
        "PDF / DOCX / XLSX",
        type=["pdf", "docx", "xlsx"]
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []


# =========================================================
# CHAT STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================================================
# CHAT INPUT (BOTTOM)
# =========================================================

query = st.chat_input("Ask a legal question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing document..."):
            try:
                if uploaded_file:
                    if uploaded_file.name.endswith(".pdf"):
                        text = extract_pdf(uploaded_file)
                    elif uploaded_file.name.endswith(".docx"):
                        text = extract_docx(uploaded_file)
                    else:
                        text = extract_excel(uploaded_file)

                    chunks = [text[i:i+600] for i in range(0, len(text), 600)]
                else:
                    chunks = retrieve(query)

                result = aggregate_from_chunks(chunks)
                answer = format_output(result)

            except Exception as e:
                answer = f"❌ Error: {e}"

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
