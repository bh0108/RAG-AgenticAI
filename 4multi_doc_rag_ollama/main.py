import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------------------------------------------------------------------
# 1. Config
# -------------------------------------------------------------------
load_dotenv()

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Ollama models (change if you prefer others)
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# -------------------------------------------------------------------
# 2. Load & chunk documents (multi-document)
# -------------------------------------------------------------------
def load_documents():
    """
    Load all supported documents from the data/ directory.
    Extend this as needed for more types.
    """
    docs = []

    # PDFs
    pdf_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs.extend(pdf_loader.load())

    # Text / Markdown
    text_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.[tT][xX][tT]",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs.extend(text_loader.load())

    md_loader = DirectoryLoader(
        str(DATA_DIR),
        glob="**/*.md",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs.extend(md_loader.load())

    if not docs:
        print("No documents found in data/. Add PDFs, .txt, or .md files.")
    else:
        print(f"Loaded {len(docs)} documents.")

    return docs


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


# -------------------------------------------------------------------
# 3. Build / load vector store (Chroma + Ollama embeddings)
# -------------------------------------------------------------------
def get_vectorstore(chunks):
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    vectorstore.persist()
    print("Vector store built and persisted.")
    return vectorstore


def load_existing_vectorstore():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    print("Loaded existing vector store.")
    return vectorstore


# -------------------------------------------------------------------
# 4. Build RAG chain (retriever + LLM)
# -------------------------------------------------------------------
def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )

    llm = Ollama(model=LLM_MODEL)
    prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": lambda x: x["query"]}
    | prompt
    | llm
    | StrOutputParser()
)
    return qa_chain


# -------------------------------------------------------------------
# 5. Simple CLI loop
# -------------------------------------------------------------------
def interactive_loop(qa_chain):
    print("\nMulti-Document RAG ready. Ask questions about your docs.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        query = input(">>> ")
        if query.lower() in {"exit", "quit"}:
            break

        result = qa_chain({"query": query})
        answer = result["result"]
        sources = result.get("source_documents", [])

        print("\n--- Answer ---")
        print(answer)
        print("\n--- Sources ---")
        for i, doc in enumerate(sources, start=1):
            print(f"[{i}] {doc.metadata.get('source', 'unknown')}")

        print("\n")


def main():
    # If Chroma DB already exists, reuse it; otherwise build from docs
    if CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir()):
        vectorstore = load_existing_vectorstore()
    else:
        docs = load_documents()
        if not docs:
            return
        chunks = chunk_documents(docs)
        vectorstore = get_vectorstore(chunks)

    qa_chain = build_rag_chain(vectorstore)
    interactive_loop(qa_chain)


if __name__ == "__main__":
    main()