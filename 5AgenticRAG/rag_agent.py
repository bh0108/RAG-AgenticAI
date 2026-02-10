import os
from typing import List, Dict, Any

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
# STEP1: Local Flan-T5 wrapper
class LocalFlanT5:
    def __init__(self, model_name: str = "google/flan-t5-small", device: str | None = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=4,
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Step 2: Ingest and index documents into Chroma

    PERSIST_DIR = "chroma_db"

def load_documents(data_dir: str = "data") -> List[Any]:
    docs = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".txt"):
            path = os.path.join(data_dir, fname)
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    return docs


def build_vectorstore(docs: List[Any]) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    vectordb.persist()
    return vectordb


def load_or_build_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    if os.path.exists(PERSIST_DIR):
        return Chroma(
            embedding_function=embeddings,
            persist_directory=PERSIST_DIR,
        )
    docs = load_documents()
    return build_vectorstore(docs)
# Step 3: A simple RAG “reasoning prompt”
RAG_SYSTEM_PROMPT = """You are a careful, honest assistant.
You answer questions using ONLY the provided context.
If the context does not contain the answer, say you don't know.

Context:
{context}

Question:
{question}

Answer in a clear, concise paragraph:
"""

# Step 4: Retrieval + generation function
def retrieve_context(vectordb: Chroma, query: str, k: int = 4) -> str:
    docs = vectordb.similarity_search(query, k=k)
    context_parts = []
    for i, d in enumerate(docs):
        context_parts.append(f"[Doc {i+1}]\n{d.page_content}")
    return "\n\n".join(context_parts)


def rag_answer(
    llm: LocalFlanT5,
    vectordb: Chroma,
    question: str,
    k: int = 4,
) -> Dict[str, Any]:
    context = retrieve_context(vectordb, question, k=k)
    prompt = RAG_SYSTEM_PROMPT.format(context=context, question=question)
    answer = llm.generate(prompt)
    return {
        "question": question,
        "context": context,
        "answer": answer,
        "prompt": prompt,
    }
# Step 5: Adding “agentic” behavior
class AgenticRAG:
    def __init__(self, llm: LocalFlanT5, vectordb: Chroma):
        self.llm = llm
        self.vectordb = vectordb
        self.history: List[Dict[str, str]] = []

    def _is_low_confidence(self, answer: str) -> bool:
    text = answer.lower()
    if "i don't know" in text or "cannot answer" in text:
        return True
    if len(answer.split()) < 8:
        return True
    return False

def _build_meta_prompt(self, question: str, context: str) -> str:
    history_str = ""
    if self.history:
        history_lines = []
        for turn in self.history[-5:]:
            history_lines.append(f"User: {turn['question']}")
            history_lines.append(f"Assistant: {turn['answer']}")
        history_str = "\n\nRecent conversation:\n" + "\n".join(history_lines)

    meta_prompt = f"""You are an assistant that reasons step by step.

{history_str}

Context:
{context}

User question:
{question}

Think briefly, then answer clearly:
"""
        return meta_prompt

    def ask(self, question: str) -> Dict[str, Any]:
        # First attempt with default k
        context = retrieve_context(self.vectordb, question, k=4)
        prompt = self._build_meta_prompt(question, context)
        answer = self.llm.generate(prompt)

        if self._is_low_confidence(answer):
            # Second attempt with expanded context
            expanded_context = retrieve_context(self.vectordb, question, k=8)
            prompt2 = self._build_meta_prompt(question, expanded_context)
            answer2 = self.llm.generate(prompt2)

            result = {
                "question": question,
                "context": expanded_context,
                "answer": answer2,
                "first_answer": answer,
                "used_retry": True,
            }
        else:
            result = {
                "question": question,
                "context": context,
                "answer": answer,
                "used_retry": False,
            }

        # Update history with final answer
        self.history.append({
            "question": question,
            "answer": result["answer"],
        })
        return result
    # Step 6: Wiring it up with a simple CLI loop
    def main():
        print("Loading vector store...")
        vectordb = load_or_build_vectorstore()
        print("Loading local Flan-T5 model...")
        llm = LocalFlanT5(model_name="google/flan-t5-small")
        agent = AgenticRAG(llm, vectordb)

    print("\nLocal Agentic RAG is ready. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        
        result = agent.ask(question)
        print("\nAssistant:", result["answer"])
        if result.get("used_retry"):
            print("(Note: I re-queried with expanded context.)")
        print("-" * 60)


if __name__ == "__main__":
    main()

