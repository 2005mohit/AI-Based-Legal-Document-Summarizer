import streamlit as st
import os, faiss, pickle, re
import numpy as np

import pdfplumber
import docx
import pandas as pd

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="AI Legal Assistant", layout="wide")

# =========================================================
# CSS (CHATGPT STYLE BOTTOM BAR)
# =========================================================
st.markdown("""
<style>
.chat-wrapper { max-width: 900px; margin: auto; }

.bottom-bar {
    position: fixed;
    bottom: 0;
    width: 100%;
    background: #0e1117;
    padding: 10px;
    border-top: 1px solid #333;
    z-index: 999;
}

.upload-box {
    width: 40px;
    height: 40px;
    border-radius: 8px;
    background: #262730;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
}

.upload-box:hover {
    background: #333;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD FAISS
# =========================================================
@st.cache_resource
def load_faiss():
    base = os.path.dirname(os.path.abspath(__file__))
    index = faiss.read_index(os.path.join(base, "faiss_index/index.faiss"))
    with open(os.path.join(base, "faiss_index/metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    return index, meta

index, metadata = load_faiss()

# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_models():
    return (
        SentenceTransformer("all-MiniLM-L6-v2"),
        pipeline("text2text-generation", model="google/flan-t5-small", device=-1)
    )

embed_model, summarizer = load_models()

# =========================================================
# UTIL
# =========================================================
def clean_text(t):
    t = re.sub(r'\n+', '\n', t)
    return t.strip()

def retrieve(q):
    emb = embed_model.encode([q])
    _, ids = index.search(np.array(emb).astype("float32"), 3)
    return [metadata[i]["text"] for i in ids[0]]

def aggregate(chunks):
    risks, obligations = [], []

    for c in chunks:
        res = summarizer(
            f"Extract obligations and risks:\n{c[:1000]}",
            max_new_tokens=150
        )[0]["generated_text"]

        s = clean_text(res).lower()
        if "liable" in s or "risk" in s:
            risks.append(res)
        if "shall" in s or "must" in s:
            obligations.append(res)

    return {
        "executive_summary": f"{len(obligations)} obligations and {len(risks)} risks identified.",
        "key_risks": list(set(risks)),
        "obligations": list(set(obligations))
    }

def format_out(r):
    out = f"### 🧾 Executive Summary\n{r['executive_summary']}\n\n"
    out += "### ⚠️ Key Risks\n" + "\n".join(f"- {i}" for i in r["key_risks"])
    out += "\n\n### 📌 Obligations\n" + "\n".join(f"- {i}" for i in r["obligations"])
    return out

# =========================================================
# FILE EXTRACT
# =========================================================
def extract(file):
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            return "".join(p.extract_text() or "" for p in pdf.pages)
    if file.name.endswith(".docx"):
        return "\n".join(p.text for p in docx.Document(file).paragraphs)
    return pd.read_excel(file).to_string()

# =========================================================
# SIDEBAR (ONLY HISTORY)
# =========================================================
with st.sidebar:
    st.title("🕘 History")
    if st.button("Clear Chat"):
        st.session_state.messages = []

# =========================================================
# CHAT STATE
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

st.markdown("<div class='chat-wrapper'>", unsafe_allow_html=True)
st.title("⚖️ AI Legal Assistant")
st.caption("Legal Document Analysis & Q&A")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# BOTTOM BAR (UPLOAD + INPUT)
# =========================================================
with st.container():
    st.markdown("<div class='bottom-bar'>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 10])

    with c1:
        uploaded_file = st.file_uploader("➕", label_visibility="collapsed",
                                         type=["pdf", "docx", "xlsx"])

    with c2:
        query = st.text_input("Ask a legal question...", label_visibility="collapsed")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================================================
# RESPONSE
# =========================================================
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            if uploaded_file:
                text = extract(uploaded_file)
                chunks = [text[i:i+600] for i in range(0, len(text), 600)]
            else:
                chunks = retrieve(query)

            result = aggregate(chunks)
            answer = format_out(result)

        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
