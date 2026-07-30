"""Optional dense embeddings for hybrid retrieval (BM25 + dense RRF).

When NESTLING_USE_DENSE=0 or the model is unavailable, callers fall back to BM25-only.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

import numpy as np

from assistant.settings import get_settings

log = logging.getLogger(__name__)

_model = None
_model_failed = False
_cache: dict[str, np.ndarray] = {}


def dense_enabled() -> bool:
    return bool(get_settings().nestling_use_dense) and not _model_failed


def _get_model():
    global _model, _model_failed
    if _model_failed:
        return None
    if _model is not None:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        name = get_settings().nestling_embedding_model
        _model = SentenceTransformer(name)
        return _model
    except Exception as exc:
        log.warning("Dense embeddings unavailable: %s", exc)
        _model_failed = True
        return None


def embed_texts(texts: list[str]) -> np.ndarray | None:
    model = _get_model()
    if model is None:
        return None
    out = []
    for t in texts:
        key = hashlib.sha256((t or "").encode("utf-8")).hexdigest()
        if key in _cache:
            out.append(_cache[key])
            continue
        vec = model.encode([t or ""], normalize_embeddings=True)[0]
        _cache[key] = vec
        out.append(vec)
    return np.vstack(out)


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
    top_k: int = 5,
) -> list[str]:
    """RRF over lists of document ids (best-first)."""
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
    top_k: int = 5,
) -> list[str]:
    """Fuse BM25 ranking with dense similarity when embeddings are available."""
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
