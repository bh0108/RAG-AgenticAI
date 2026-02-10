import os
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from langchain.chains import RetrievalQA


DATA_DIR = "data"


def load_documents():
    docs = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".txt"):
            path = os.path.join(DATA_DIR, filename)
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
    return docs


def build_vectorstore(documents):
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = OpenAIEmbeddings()

    # In-memory Qdrant instance
    client = QdrantClient(":memory:")

    vectorstore = Qdrant.from_documents(
        chunks,
        embeddings,
        client=client,
        collection_name="rag_docs"
    )

    return vectorstore


def build_rag_chain():
    docs = load_documents()
    vectorstore = build_vectorstore(docs)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set. Add it to .env or environment variables.")

    print("Loading RAG system...")
    qa = build_rag_chain()
    print("Ready! Ask questions about your documents.\n")

    while True:
        query = input("Your question (or 'exit'): ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        result = qa({"query": query})
        print("\n--- Answer ---")
        print(result["result"])

        print("\n--- Sources ---")
        for i, doc in enumerate(result["source_documents"], start=1):
            print(f"[{i}] {doc.metadata.get('source')}")

        print("\n")


if __name__ == "__main__":
    main()