# graph_rag.py
import json
from typing import List, Tuple

import networkx as nx
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

from config import OLLAMA_MODEL


def get_extractor_llm():
    return ChatOllama(model=OLLAMA_MODEL, temperature=0.0)


EXTRACTION_PROMPT = ChatPromptTemplate.from_template(
    """
You extract knowledge graph triples from text.

For the given text, return a JSON list of objects with keys:
"subject", "predicate", "object".

Text:
\"\"\"{text}\"\"\"


Return ONLY valid JSON, no explanation.
"""
)


def extract_triples(text: str) -> List[Tuple[str, str, str]]:
    llm = get_extractor_llm()
    chain = EXTRACTION_PROMPT | llm

    response = chain.invoke({"text": text})
    content = response.content

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # crude fallback: try to strip code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.strip("`")
            # remove possible "json" language tag
            content = "\n".join(content.splitlines()[1:])
        data = json.loads(content)

    triples: List[Tuple[str, str, str]] = []
    for item in data:
        s = item.get("subject", "").strip()
        p = item.get("predicate", "").strip()
        o = item.get("object", "").strip()
        if s and p and o:
            triples.append((s, p, o))
    return triples
# graph_rag.py (continued)
def build_graph(triples: List[Tuple[str, str, str]]) -> nx.MultiDiGraph:
    """
    Build a directed multigraph where edges are labeled with predicates.
    """
    G = nx.MultiDiGraph()
    for subj, pred, obj in triples:
        G.add_node(subj)
        G.add_node(obj)
        G.add_edge(subj, obj, predicate=pred)
    return G


def graph_to_text_context(G: nx.MultiDiGraph, start_node: str, max_hops: int = 2) -> str:
    """
    Walk the graph up to max_hops from start_node and serialize edges as text.
    This is our 'deep context' for RAG.
    """
    if start_node not in G:
        return f"No information found about {start_node}."

    visited = set([start_node])
    frontier = {start_node}
    hops = 0
    edges_collected = []

    while frontier and hops < max_hops:
        next_frontier = set()
        for node in frontier:
            for neighbor in G.successors(node):
                for key, edge_data in G.get_edge_data(node, neighbor).items():
                    pred = edge_data.get("predicate", "related_to")
                    edges_collected.append(f"{node} {pred} {neighbor}.")
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        hops += 1

    if not edges_collected:
        return f"No outgoing relationships found for {start_node}."

    return "\n".join(edges_collected)

# graph_rag.py (continued)
QA_PROMPT = ChatPromptTemplate.from_template(
    """
You are a reasoning assistant that answers questions using the provided graph context.

Graph context:
\"\"\"{context}\"\"\"

Question:
\"\"\"{question}\"\"\"

Use ONLY the information in the graph context. If the answer is not there, say you don't know.
Answer concisely.
"""
)

ENTITY_PROMPT = ChatPromptTemplate.from_template(
    """
You are given a question and must identify the single most relevant entity name
to start a graph search from.

Return ONLY the entity name as plain text, no explanation.

Question:
\"\"\"{question}\"\"\"
"""
)


def infer_start_entity(question: str) -> str:
    llm = get_extractor_llm()
    chain = ENTITY_PROMPT | llm
    response = chain.invoke({"question": question})
    return response.content.strip()


def answer_question_with_graph(G: nx.MultiDiGraph, question: str, max_hops: int = 2) -> str:
    start_entity = infer_start_entity(question)
    context = graph_to_text_context(G, start_entity, max_hops=max_hops)

    llm = get_extractor_llm()
    chain = QA_PROMPT | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content.strip()