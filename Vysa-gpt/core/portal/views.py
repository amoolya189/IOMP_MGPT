import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

BASE_DIR = Path(__file__).resolve().parent
TEXT_FOLDER = BASE_DIR / "data" / "mahatxt"

# Load Sentence-BERT model
MODEL = SentenceTransformer('all-MiniLM-L6-v2')

def load_mahabharata_texts():
    """Load all .txt Parvas"""
    all_texts = []
    for filename in os.listdir(TEXT_FOLDER):
        if filename.lower().endswith(".txt"):
            filepath = TEXT_FOLDER / filename
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read()
                all_texts.append({"file": filename, "content": text})
    return all_texts

def chunk_text(text, chunk_size=500):
    """Split text into word chunks"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

def build_faiss_index(all_texts):
    """
    Create FAISS index from all text chunks
    Returns:
        index: FAISS index
        metadata: list of dicts with 'file' and 'chunk' for each vector
    """
    all_chunks = []
    vectors = []

    for entry in all_texts:
        chunks = chunk_text(entry["content"])
        for chunk in chunks:
            vector = MODEL.encode(chunk)
            vectors.append(vector)
            all_chunks.append({"file": entry["file"], "chunk": chunk})

    vectors = np.array(vectors).astype('float32')
    dim = vectors.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index, all_chunks

def semantic_search(query, index, all_chunks, top_k=1):
    """Return top_k most relevant chunks"""
    query_vector = MODEL.encode(query).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = [all_chunks[i] for i in indices[0]]
    return results
