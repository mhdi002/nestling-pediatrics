"""The lazy embedding-model load must not outlive its deadline."""
import time
import assistant.rag.dense as dense
from assistant.settings import reset_settings


def test_a_hanging_model_load_gives_up_and_falls_back(monkeypatch):
    monkeypatch.setenv("NESTLING_DENSE_LOAD_TIMEOUT", "1")
    monkeypatch.setenv("NESTLING_USE_DENSE", "1")
    reset_settings()
    monkeypatch.setattr(dense, "_model", None)
    monkeypatch.setattr(dense, "_model_failed", False)

    class _Slow:
        def __init__(self, name):
            time.sleep(30)

    import sys, types
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _Slow
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)

    t0 = time.time()
    assert dense._get_model() is None
    elapsed = time.time() - t0
    assert elapsed < 5, f"load blocked for {elapsed:.1f}s"
    assert dense.dense_enabled() is False, "must degrade to BM25 after failing"
