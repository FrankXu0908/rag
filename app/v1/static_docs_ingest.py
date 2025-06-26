from rag import reader, embedder
import faiss, pickle, os

folder = "rag/store/docs"
texts = []

for file in os.listdir(folder):
    if file.endswith(".pdf"):
        full = os.path.join(folder, file)
        raw = reader.extract_text_from_pdf(full)
        texts.extend(reader.chunk_text(raw))

vectors = embedder.embed(texts)
index = faiss.IndexFlatL2(len(vectors[0]))
index.add(vectors)

os.makedirs("rag/store/faiss_index", exist_ok=True)
faiss.write_index(index, "rag/store/faiss_index/index.faiss")
with open("rag/store/faiss_index/texts.pkl", "wb") as f:
    pickle.dump(texts, f)
print(f"✅ Total chunks: {len(texts)}")