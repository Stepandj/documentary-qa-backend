"""Lexical (BM25) index for hybrid retrieval.

Dense embeddings capture meaning but can miss exact tokens — proper names, rare
keywords — when a chunk's overall topic dominates the vector. BM25 scores exact term
overlap, so fusing it with dense retrieval recovers those cases (e.g. "Who is Annie
Gray?"). See DESIGN.md for the fusion rationale.
"""
from __future__ import annotations

import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do", "does", "for",
    "from", "had", "has", "have", "how", "i", "in", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "them", "they", "this", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "with", "would", "you", "your",
}


def tokenize(text: str) -> list[str]:
    """Lowercase content tokens; drop 1-char tokens and common stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= 2 and t not in _STOPWORDS]


class BM25Index:
    def __init__(self, texts: list[str]):
        from rank_bm25 import BM25Okapi

        self._corpus_tokens = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._corpus_tokens)

    def scores(self, query: str) -> np.ndarray:
        """Return the (n,) array of BM25 scores for the query over all documents."""
        return np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float32)

    def overlap_count(self, query: str, doc_idx: int) -> int:
        """Return how many content tokens the query shares with the given document."""
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return 0
        return len(query_tokens & set(self._corpus_tokens[doc_idx]))
