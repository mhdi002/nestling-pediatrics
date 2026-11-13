"""BM25 keyword ranking for recalled memory rows.

Long-term memory ranks the rows it recalls by how well they match the current
message. This used to live in the retrieval (RAG) package, which is gone: the
care library is read by the RLM orchestrator now, and this ranking is all that
was still in use.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+|[؀-ۿ]+", re.I)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """Pure-Python BM25 over a small set of texts."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.df: Counter = Counter()
        self.N = 0

    def fit(self, texts: Iterable[str]):
        docs = [tokenize(t) for t in texts]
        # Per-doc term frequencies are computed once, not on every query.
        self.doc_tf = [Counter(toks) for toks in docs]
        self.doc_len = [len(toks) for toks in docs]
        self.N = len(docs)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.df = Counter()
        for tf in self.doc_tf:
            for term in tf:
                self.df[term] += 1

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        scores = np.zeros(self.N, dtype=np.float64)
        if not q or self.N == 0:
            return scores
        for i, tf in enumerate(self.doc_tf):
            dl = self.doc_len[i] or 1
            s = 0.0
            for term in q:
                if term not in tf:
                    continue
                n_q = self.df.get(term, 0)
                idf = math.log(1 + (self.N - n_q + 0.5) / (n_q + 0.5))
                freq = tf[term]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                s += idf * (freq * (self.k1 + 1)) / denom
            scores[i] = s
        return scores
