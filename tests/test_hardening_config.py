"""Regression tests: nothing that drives behaviour is hardcoded in business logic."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from assistant.refdata import care_topics, clinical_bounds
from assistant.settings import Settings, get_settings

ROOT = Path(__file__).resolve().parent.parent


def test_defaults_work_without_env_file():
    """A fresh checkout with no .env must still produce a usable configuration."""
    s = Settings(_env_file=None)
    assert s.nestling_llm_timeout > 0
    assert s.nestling_max_upload_bytes > 0
    assert s.nestling_rag_top_k > 0
    assert s.allowed_upload_types
    assert s.cors_origin_list == ["*"]
    # Secrets are env-only: never a baked-in default.
    assert s.nestling_api_key is None


def test_llm_urls_are_not_hardcoded():
    """The sidecar is opt-in; an unset URL must not point at a guessed host."""
    s = Settings(_env_file=None)
    assert s.nestling_llm_url == ""
    assert s.nestling_vision_llm_url == ""


def test_model_ids_only_come_from_settings():
    """No model identifier may appear in executable backend code."""
    model_id = re.compile(
        r"^[^#]*[\"'](?:Qwen/|Salesforce/|BAAI/|meta-llama/|mistralai/)[\w.\-]+[\"']"
    )
    offenders = []
    for path in sorted((ROOT / "assistant").rglob("*.py")) + sorted((ROOT / "app").rglob("*.py")):
        if path == ROOT / "assistant" / "settings.py":
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if model_id.match(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "Model IDs must come from assistant.settings:\n" + "\n".join(offenders)


def test_paths_derive_from_settings_root():
    from assistant import config as cfg

    root = Path(get_settings().nestling_root).resolve()
    for path in (cfg.DATA, cfg.EN_DIR, cfg.KNOWLEDGE_DIR, cfg.OVERLAY_DIR, cfg.UPLOAD_DIR):
        assert path.is_absolute()
        assert root in path.parents or path == root


def test_data_dir_override_relocates_state(tmp_path, monkeypatch):
    """NESTLING_DATA_DIR must move every derived path, not just some of them."""
    s = Settings(_env_file=None, nestling_data_dir=tmp_path)
    assert s.nestling_data_dir == tmp_path
    monkeypatch.setattr("assistant.settings.get_settings", lambda: s)
    import importlib

    import assistant.config as cfg

    reloaded = importlib.reload(cfg)
    try:
        assert reloaded.DATA == tmp_path
        assert reloaded.OVERLAY_DIR == tmp_path / "overlays"
        assert reloaded.CHILD_DB_PATH == tmp_path / "children" / "children.db"
    finally:
        monkeypatch.undo()
        importlib.reload(cfg)


def test_clinical_bounds_are_config_driven():
    b = clinical_bounds()
    for key in (
        "intergrowth_weeks_min",
        "intergrowth_weeks_max",
        "who_age_months_max",
        "preterm_ga_threshold_weeks",
        "full_term_weeks",
        "asq_age_months_max",
        "value_ranges",
    ):
        assert key in b, f"clinical_bounds.json is missing {key}"


def test_care_topic_routing_data_is_config_driven():
    cfg = care_topics()
    assert cfg.get("topics", {}).get("feeding", {}).get("keywords")
    assert cfg.get("feeding_age_bands")
    # The final band is the open-ended catch-all.
    assert cfg["feeding_age_bands"][-1]["max_age_months"] is None
    for key in ("safety_tail", "citation_tail", "no_match"):
        assert cfg["messages"][key].strip()


def test_safety_tail_is_topic_neutral():
    """The extractive tail is appended to every topic, so it must not be skin-specific."""
    from assistant.rag.stores import SAFETY_TAIL

    lowered = SAFETY_TAIL.lower()
    for skin_word in ("redness", "rash", "swelling"):
        assert skin_word not in lowered


@pytest.mark.parametrize(
    "field",
    [
        "nestling_llm_timeout",
        "nestling_llm_probe_timeout",
        "nestling_translate_timeout",
    ],
)
def test_every_outbound_call_has_a_configurable_timeout(field):
    assert getattr(Settings(_env_file=None), field) > 0
