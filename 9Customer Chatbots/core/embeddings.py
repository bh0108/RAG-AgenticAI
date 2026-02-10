import math
from typing import List


def simple_embed(text: str) -> List[float]:
    """
    Very simple character-count embedding.
    Replace with SentenceTransformer for production.
    """
    vec = [0] * 26
    for ch in text.lower():
        if 'a' <= ch <= 'z':
            vec[ord(ch) - ord('a')] += 1

    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_sim(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))