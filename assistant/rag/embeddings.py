#!/usr/bin/env python3
"""
Lexical retrieval only (BM25-style). No third-party embedding models.

Neural generation is handled by the local Qwen OpenAI-compatible sidecar in stores.py
(or extractive fallback when the LLM is down).
Tool calling is handled by Salesforce/xLAM-1b-fc-r (optional) or deterministic router.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+|[\u0600-\u06FF]+", re.I)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


class BM25Index:
    """Pure-Python BM25 for accurate offline retrieval without neural embedders."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.df: Counter = Counter()
        self.N = 0

    def fit(self, texts: Iterable[str]):
        self.docs_tokens = [tokenize(t) for t in texts]
        self.doc_len = [len(toks) for toks in self.docs_tokens]
        self.N = len(self.docs_tokens)
        self.avgdl = (sum(self.doc_len) / self.N) if self.N else 0.0
        self.df = Counter()
        for toks in self.docs_tokens:
            for term in set(toks):
                self.df[term] += 1

    def scores(self, query: str) -> np.ndarray:
        q = tokenize(query)
        scores = np.zeros(self.N, dtype=np.float64)
        if not q or self.N == 0:
            return scores
        for i, toks in enumerate(self.docs_tokens):
            tf = Counter(toks)
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


# Back-compat name used by older imports
class HashingEmbedder:
    """Deprecated alias — project no longer uses neural/hashing embedders for RAG."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self._bm25 = BM25Index()

    def embed(self, text: str) -> np.ndarray:
        # Not used for similarity; kept so old tests importing this don't crash hard.
        vec = np.zeros(self.dim, dtype=np.float64)
        for i, tok in enumerate(tokenize(text)[: self.dim]):
            vec[i % self.dim] += 1.0
        n = np.linalg.norm(vec)
        return vec / n if n else vec

    def embed_many(self, texts: Iterable[str]) -> np.ndarray:
        return np.vstack([self.embed(t) for t in texts])


def get_embedder(backend: str = "bm25", model_id: str | None = None):
    """
    Only BM25 is supported. model_id is ignored — neural embedders are disabled.
    """
    if backend in {"sentence-transformers", "hf", "pleias-embed"}:
        raise RuntimeError(
            "Neural embedding backends are disabled. "
            "RAG retrieval uses BM25; generation uses the local Qwen LLM sidecar."
        )
    return HashingEmbedder()
