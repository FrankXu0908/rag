from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")  # 可选用其他中文语言模型作为encoder

def embed(texts):
    return model.encode(texts, convert_to_tensor=False)