from fastapi import FastAPI
from pydantic import BaseModel

import faiss
import pickle
import numpy as np
import os

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# FASTAPI INIT

app = FastAPI(
    title="AI-Based Legal Document Summarizer",
    version="1.0"
)

# ABSOLUTE PATH RESOLUTION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAISS_PATH = os.path.join(BASE_DIR, "faiss_index", "index.faiss")
META_PATH  = os.path.join(BASE_DIR, "faiss_index", "metadata.pkl")

print("DEBUG FAISS PATH:", FAISS_PATH)
print("DEBUG META PATH:", META_PATH)

# LOAD INDEX & METADATA

index = faiss.read_index(FAISS_PATH)

with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# MODELS

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

summarizer = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    device=-1
)

# RAG LOGIC

def retrieve(query, top_k=3):
    emb = embed_model.encode([query])
    _, ids = index.search(np.array(emb).astype("float32"), top_k)
    return [metadata[i]["text"] for i in ids[0]]

def summarize_chunks(chunks):
    out = []
    for c in chunks:
        prompt = f"""
Extract obligations and risks from the clause below:

{c[:1000]}
"""
        res = summarizer(prompt, max_new_tokens=150)[0]["generated_text"]
        out.append(res)
    return out

def aggregate(query):
    chunks = retrieve(query)
    summaries = summarize_chunks(chunks)

    risks, obligations = [], []

    for s in summaries:
        t = s.lower()
        if "indemn" in t or "liability" in t:
            risks.append(s)
        if "shall" in t or "will" in t:
            obligations.append(s)

    return {
        "executive_summary": f"{len(obligations)} obligations and {len(risks)} risks found.",
        "key_risks": list(set(risks)),
        "obligations": list(set(obligations))
    }

# API

class Query(BaseModel):
    question: str

@app.get("/")
def health():
    return {"status": "running"}

@app.post("/summarize")
def summarize_api(q: Query):
    return aggregate(q.question)
