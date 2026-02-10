# ============================
# RAG SYSTEM IN VISUAL STUDIO CODE
# ============================

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
import faiss
import numpy as np

# ----------------------------
# 1. Load knowledge base
# ----------------------------
with open("my_knowledge.txt", "r") as f:
    raw_text = f.read()

# ----------------------------
# 2. Chunk the text
# ----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
)

docs = text_splitter.split_text(raw_text)
print(f"Total chunks: {len(docs)}")

# ----------------------------
# 3. Embeddings + FAISS index
# ----------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embed_model.encode(docs, convert_to_numpy=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("FAISS index built with", index.ntotal, "vectors")

# ----------------------------
# 4. Simple retriever
# ----------------------------
def retrieve(query, k=3):
    query_emb = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, k)
    return [docs[i] for i in indices[0]]

# ----------------------------
# 5. Load LLM (local HF model)
# ----------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "gpt2"   # or TinyLlama

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    pad_token_id=tokenizer.eos_token_id
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200
)

# ----------------------------
# 6. Build RAG prompt
# ----------------------------
def build_prompt(query, context_chunks):
    context = "\n\n".join(context_chunks)
    return f"""
You are a helpful assistant answering questions about company policies.

Use ONLY the information in the context below.

Context:
{context}

Question: {query}

Answer:
"""

# ----------------------------
# 7. RAG answer function
# ----------------------------
def rag_answer(query, k=3):
    chunks = retrieve(query, k)
    prompt = build_prompt(query, chunks)
    output = llm(prompt)[0]["generated_text"]
    return output

# ----------------------------
# 8. Test queries
# ----------------------------
if __name__ == "__main__":
    print("\n=== RAG SYSTEM READY ===\n")

    questions = [
        "What days are employees required to be in the office?",
        "How many PTO days do full-time employees get?",
        "What is the official backend language?",
    ]

    for q in questions:
        print("\nQUESTION:", q)
        print(rag_answer(q))
        print("-" * 60)