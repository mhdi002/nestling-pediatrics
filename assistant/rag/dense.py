"""Optional dense embeddings for hybrid retrieval (BM25 + dense RRF).

When NESTLING_USE_DENSE=0 or the model is unavailable, callers fall back to BM25-only.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from collections import OrderedDict

import numpy as np

from assistant.settings import get_settings

log = logging.getLogger(__name__)

_model = None
_model_failed = False
_model_lock = threading.Lock()
# Bounded LRU — an unbounded dict grew with every distinct chunk/query seen.
_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_cache_lock = threading.Lock()


def dense_enabled() -> bool:
    return bool(get_settings().nestling_use_dense) and not _model_failed


def _get_model():
    global _model, _model_failed
    if _model_failed:
        return None
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if _model_failed:
            return None
        try:
            from sentence_transformers import SentenceTransformer

            name = get_settings().nestling_embedding_model
            _model = SentenceTransformer(name)
            return _model
        except Exception as exc:
            log.warning("Dense embeddings unavailable: %s", exc)
            _model_failed = True
            return None


def _cache_get(key: str) -> np.ndarray | None:
    with _cache_lock:
        vec = _cache.get(key)
        if vec is not None:
            _cache.move_to_end(key)
        return vec


def _cache_put(key: str, vec: np.ndarray, max_size: int) -> None:
    with _cache_lock:
        _cache[key] = vec
        _cache.move_to_end(key)
        while len(_cache) > max_size:
            _cache.popitem(last=False)


def embed_texts(texts: list[str]) -> np.ndarray | None:
    model = _get_model()
    if model is None:
        return None
    if not texts:
        return None
    max_size = max(1, get_settings().nestling_dense_cache_size)
    out = []
    for t in texts:
        key = hashlib.sha256((t or "").encode("utf-8")).hexdigest()
        vec = _cache_get(key)
        if vec is None:
            vec = model.encode([t or ""], normalize_embeddings=True)[0]
            _cache_put(key, vec, max_size)
        out.append(vec)
    return np.vstack(out)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int | None = None,
    top_k: int | None = None,
) -> list[str]:
    """RRF over lists of document ids (best-first)."""
    settings = get_settings()
    k = settings.nestling_dense_rrf_k if k is None else k
    top_k = settings.nestling_rag_top_k if top_k is None else top_k
    scores: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: -x[1])[:top_k]]


def hybrid_order(
    bm25_ids: list[str],
    query: str,
    id_to_text: dict[str, str],
    *,
    top_k: int | None = None,
) -> list[str]:
    """Fuse BM25 ranking with dense similarity when embeddings are available."""
    top_k = get_settings().nestling_rag_top_k if top_k is None else top_k
    if not dense_enabled() or not bm25_ids:
        return bm25_ids[:top_k]
    texts = [id_to_text.get(i, "") for i in bm25_ids]
    q = embed_texts([query])
    docs = embed_texts(texts)
    if q is None or docs is None:
        return bm25_ids[:top_k]
    sims = (docs @ q[0]).tolist()
    dense_order = [doc_id for doc_id, _ in sorted(zip(bm25_ids, sims), key=lambda x: -x[1])]
    return reciprocal_rank_fusion([bm25_ids, dense_order], top_k=top_k)
