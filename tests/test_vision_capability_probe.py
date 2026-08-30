"""Vision readiness must reflect image capability, not URL reachability."""

from __future__ import annotations

import urllib.error

import assistant.llm.qwen_client as qc


def _client(monkeypatch, *, reachable=True, capable=True, transport_error=False):
    client = qc.QwenClient.__new__(qc.QwenClient)
    client._probe_cache = {}
    client._probe_lock = __import__("threading").Lock()
    client._ready_ttl = 5.0
    client.probe_timeout = 0.5
    client.vision_url = "http://llm:8000"
    client.base_url = "http://llm:8000"
    monkeypatch.setattr(qc.QwenClient, "_probe", lambda self, url: reachable)

    def _open(req, timeout=None):
        if transport_error:
            raise urllib.error.URLError("boom")
        if not capable:
            raise urllib.error.HTTPError(req.full_url, 400, "no images", {}, None)

        class _R:
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        return _R()

    monkeypatch.setattr(qc.urllib.request, "urlopen", _open)
    return client


def test_reachable_but_text_only_model_is_not_vision_ready(monkeypatch):
    """The served Qwen3.5-4B answers text and rejects every image."""
    client = _client(monkeypatch, reachable=True, capable=False)
    assert client.vision_ready is False


def test_a_model_that_accepts_an_image_is_vision_ready(monkeypatch):
    client = _client(monkeypatch, reachable=True, capable=True)
    assert client.vision_ready is True


def test_unreachable_endpoint_is_not_vision_ready(monkeypatch):
    client = _client(monkeypatch, reachable=False, capable=True)
    assert client.vision_ready is False


def test_a_transport_failure_is_not_cached_as_incapable(monkeypatch):
    """An undecided probe must not permanently mark a good model as blind."""
    client = _client(monkeypatch, reachable=True, transport_error=True)
    assert client.vision_ready is False
    assert f"{client.vision_url}#vision" not in client._probe_cache
