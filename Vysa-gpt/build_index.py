# build_index.py
from portal.utils import (
    load_mahabharata_texts,
    chunk_text,
)
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "portal" / "data" / "index"
INDEX_DIR.mkdir(exist_ok=True)

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

texts = load_mahabharata_texts()

chunks = []
embeddings = []

for entry in texts:
    for chunk in chunk_text(entry["content"], 300):
        chunks.append({
            "file": entry["file"],
            "chunk": chunk
        })
        embeddings.append(MODEL.encode(chunk))

embeddings = np.array(embeddings).astype("float32")

# Build FAISS
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save
faiss.write_index(index, str(INDEX_DIR / "mahabharata.faiss"))
with open(INDEX_DIR / "chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ FAISS index built and saved")
