# main.py
from pathlib import Path

from config import CORPUS_PATH
from graph_rag import extract_triples, build_graph, answer_question_with_graph


def read_corpus(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def main():
    # 1. Read text
    text = read_corpus(CORPUS_PATH)
    print("=== Corpus ===")
    print(text)
    print()

    # 2. Extract triples
    print("Extracting triples...")
    triples = extract_triples(text)
    for t in triples:
        print("  ", t)
    print()

    # 3. Build graph
    print("Building graph...")
    G = build_graph(triples)
    print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    print()

    # 4. Ask a question
    question = "Where is Alice located?"
    print(f"Question: {question}")
    answer = answer_question_with_graph(G, question, max_hops=3)
    print("Answer:", answer)


if __name__ == "__main__":
    main()