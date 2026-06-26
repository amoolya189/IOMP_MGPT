import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import matplotlib.pyplot as plt

# =========================
# TEST QUESTIONS
# =========================
TEST_QUESTIONS = [
    {"q": "Who is Karna's mother?", "expected": ["kunti"]},
    {"q": "Who raised Karna?", "expected": ["adhiratha"]},
    {"q": "Who killed Abhimanyu?", "expected": ["drona", "karna", "duryodhana"]},
    {"q": "How did Abhimanyu die?", "expected": ["chakravyuha"]},
    {"q": "What is the Kurukshetra war?", "expected": ["war", "battle"]},
    {"q": "Who are the Pandavas?", "expected": ["pandavas"]},
    {"q": "Who is the teacher of Arjuna?", "expected": ["drona", "dronacharya"]},
    {"q": "Who killed Dronacharya?", "expected": ["dhrishtadyumna"]},
    {"q": "Who is Bhishma?", "expected": ["bhishma"]},
    {"q": "Why did Bhishma take a vow?", "expected": ["vow"]},
    {"q": "Who is Krishna?", "expected": ["krishna"]},
    {"q": "What role did Krishna play in the war?", "expected": ["charioteer"]},
    {"q": "Who is the eldest Pandava?", "expected": ["yudhishthira"]},
    {"q": "Who killed Karna?", "expected": ["arjuna"]},
    {"q": "What is the Bhagavad Gita?", "expected": ["krishna", "dialogue"]},
]

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "portal" / "data" / "index"

# =========================
# LOAD MODEL
# =========================
print("🔄 Loading embedding model...")

# Use same model as index creation
model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# LOAD DATA
# =========================
print("🔄 Loading FAISS index...")
index = faiss.read_index(str(INDEX_DIR / "mahabharata.faiss"))

print("🔄 Loading chunks...")
with open(INDEX_DIR / "chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# =========================
# DIMENSION CHECK
# =========================
sample_embedding = model.encode(
    ["test"],
    normalize_embeddings=True
).astype("float32")

print("FAISS dimension:", index.d)
print("Embedding dimension:", sample_embedding.shape[1])

if sample_embedding.shape[1] != index.d:
    raise ValueError(
        f"❌ Dimension mismatch!\n"
        f"FAISS index dimension = {index.d}\n"
        f"Embedding dimension = {sample_embedding.shape[1]}\n"
        f"Use the same model used to build the FAISS index."
    )

# =========================
# MATCH LOGIC
# =========================
def is_match(chunk, expected_keywords):
    chunk = chunk.lower()

    # Match if ANY expected keyword appears
    return any(word.lower() in chunk for word in expected_keywords)

# =========================
# EVALUATION
# =========================
print("\n🚀 Running evaluation...\n")

correct = 0
mrr_total = 0
k_eval = 5

for i, item in enumerate(TEST_QUESTIONS, 1):

    q = item["q"]
    expected_keywords = item["expected"]

    q_embedding = model.encode(
        [q],
        normalize_embeddings=True
    ).astype("float32")

    D, I = index.search(q_embedding, k=k_eval)

    print(f"\n🔹 Question {i}: {q}")

    match_found = False
    rank = 0

    for r, idx in enumerate(I[0], start=1):

        chunk = chunks[idx]["chunk"]

        print(f"Rank {r}: {chunk[:120]}...")

        if is_match(chunk, expected_keywords) and not match_found:
            match_found = True
            rank = r

    if match_found:
        print("✅ Match Found")
        correct += 1
        mrr_total += 1 / rank
    else:
        print("❌ Not Found")

# =========================
# RESULTS
# =========================
accuracy = (correct / len(TEST_QUESTIONS)) * 100
mrr = mrr_total / len(TEST_QUESTIONS)

print("\n==============================")
print(f"🎯 Accuracy@{k_eval}: {accuracy:.2f}%")
print(f"📊 MRR@{k_eval}: {mrr:.4f}")
print("==============================")

# =========================
# TOP-K ANALYSIS
# =========================
k_values = [1, 3, 5, 7]
accuracies = []
mrr_scores = []

print("\n📈 Running Top-K analysis...\n")

for k in k_values:

    correct = 0
    mrr_total = 0

    for item in TEST_QUESTIONS:

        q = item["q"]
        expected_keywords = item["expected"]

        q_embedding = model.encode(
            [q],
            normalize_embeddings=True
        ).astype("float32")

        D, I = index.search(q_embedding, k=k)

        match_found = False
        rank = 0

        for r, idx in enumerate(I[0], start=1):

            chunk = chunks[idx]["chunk"]

            if is_match(chunk, expected_keywords):
                match_found = True
                rank = r
                break

        if match_found:
            correct += 1
            mrr_total += 1 / rank

    acc = (correct / len(TEST_QUESTIONS)) * 100
    mrr_k = mrr_total / len(TEST_QUESTIONS)

    accuracies.append(acc)
    mrr_scores.append(mrr_k)

    print(f"K={k} → Accuracy={acc:.2f}%, MRR={mrr_k:.4f}")

# =========================
# PLOTS
# =========================
plt.figure(figsize=(6, 4))
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("Top-K Value")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy vs Top-K")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(k_values, mrr_scores, marker="o")
plt.xlabel("Top-K Value")
plt.ylabel("MRR")
plt.title("MRR vs Top-K")
plt.grid(True)
plt.show()