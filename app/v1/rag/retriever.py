import faiss
import pickle
import os
from rag import embedder

FAISS_PATH = "rag/store/faiss_index"

def load_static_faiss():
    with open(f"{FAISS_PATH}/texts.pkl", "rb") as f:
        texts = pickle.load(f)
    index = faiss.read_index(f"{FAISS_PATH}/index.faiss")
    return index, texts

def retrieve(query, k=5,max_distance=1.2):
    index, texts = load_static_faiss()
    query_vec = embedder.embed([query])
    D, I = index.search(query_vec, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        print(f"Distance: {dist:.4f} | Text: {texts[idx][:80]}")
        if dist <= max_distance:
            results.append(texts[idx])
    print(f"✅ Matched chunks: {len(results)}")
    return results