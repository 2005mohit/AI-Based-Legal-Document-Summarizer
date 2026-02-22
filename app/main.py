from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import faiss
import pickle
import numpy as np

import pdfplumber
import docx
import pandas as pd

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# ==============================
# FASTAPI INITIALIZATION
# ==============================

app = FastAPI(
    title="AI-Based Legal Document Summarizer & QnA",
    version="2.0",
    description="RAG-powered Legal Risk & Obligation Analyzer"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================
# GLOBAL VARIABLES (Loaded at Startup)
# ==============================

index = None
metadata = None
embed_model = None
summarizer = None


# ==============================
# PATH CONFIG
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_PATH = os.path.join(BASE_DIR, "model", "index.faiss")
META_PATH = os.path.join(BASE_DIR, "model", "metadata.pkl")


# ==============================
# STARTUP EVENT
# ==============================

@app.on_event("startup")
async def load_resources():
    global index, metadata, embed_model, summarizer

    print("Loading models and FAISS index...")

    # Load FAISS index
    if os.path.exists(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
    else:
        print("FAISS index not found.")
        index = None

    # Load metadata
    if os.path.exists(META_PATH):
        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)
    else:
        print("Metadata file not found.")
        metadata = None

    # Load embedding model
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load summarization model
    summarizer = pipeline("text-generation",
        model="google/flan-t5-small",
        device=-1
    )

    print("All models loaded successfully.")


# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "faiss_loaded": index is not None,
        "embedding_loaded": embed_model is not None,
        "summarizer_loaded": summarizer is not None
    }


# ==============================
# REQUEST SCHEMA
# ==============================

class Query(BaseModel):
    question: str


# ==============================
# RAG FUNCTIONS (STATIC INDEX)
# ==============================

def retrieve(query, top_k=3):
    if index is None or metadata is None:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")

    emb = embed_model.encode([query])
    _, ids = index.search(np.array(emb).astype("float32"), top_k)

    return [metadata[i]["text"] for i in ids[0] if i < len(metadata)]


def summarize_chunks(chunks):
    results = []

    for c in chunks:
        prompt = f"""
Extract legal risks and obligations from the clause below:

{c[:1000]}
"""
        output = summarizer(prompt, max_new_tokens=150)[0]["generated_text"]
        results.append(output)

    return results


def aggregate(query):
    chunks = retrieve(query)
    summaries = summarize_chunks(chunks)

    risks, obligations = [], []

    for s in summaries:
        t = s.lower()
        if "liability" in t or "indemn" in t or "penalty" in t:
            risks.append(s)
        if "shall" in t or "will" in t or "must" in t:
            obligations.append(s)

    return {
        "question": query,
        "executive_summary": f"{len(obligations)} obligations and {len(risks)} risks identified.",
        "key_risks": list(set(risks)),
        "obligations": list(set(obligations))
    }


# ==============================
# FILE TEXT EXTRACTION
# ==============================

def extract_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text


def extract_docx(file):
    document = docx.Document(file)
    return "\n".join(p.text for p in document.paragraphs)


def extract_excel(file):
    df = pd.read_excel(file)
    return df.to_string()


# ==============================
# TEMPORARY RAG (UPLOADED DOC)
# ==============================

def answer_from_uploaded_document(text, question, top_k=3):
    chunks = [text[i:i+500] for i in range(0, len(text), 500)]

    embeddings = embed_model.encode(chunks)
    temp_index = faiss.IndexFlatL2(embeddings.shape[1])
    temp_index.add(np.array(embeddings).astype("float32"))

    q_emb = embed_model.encode([question])
    _, ids = temp_index.search(np.array(q_emb).astype("float32"), top_k)

    relevant_chunks = [chunks[i] for i in ids[0]]

    answers = []

    for c in relevant_chunks:
        prompt = f"""
Answer the question strictly using the document context below:

Context:
{c}

Question:
{question}
"""
        output = summarizer(prompt, max_new_tokens=200)[0]["generated_text"]
        answers.append(output)

    return {
        "question": question,
        "answers": answers
    }


# ==============================
# API ROUTES
# ==============================

@app.get("/")
def root():
    return {
        "message": "Legal AI API is running",
        "docs": "/docs"
    }


@app.post("/summarize")
def summarize_api(q: Query):
    return aggregate(q.question)


@app.post("/ask-document")
async def ask_document(
    question: str,
    file: UploadFile = File(...)
):
    if file.filename.endswith(".pdf"):
        text = extract_pdf(file.file)

    elif file.filename.endswith(".docx"):
        text = extract_docx(file.file)

    elif file.filename.endswith(".xlsx"):
        text = extract_excel(file.file)

    else:
        raise HTTPException(status_code=400, detail="Unsupported file format")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No readable text found")

    return answer_from_uploaded_document(text, question)
