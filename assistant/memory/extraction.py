"""Turn a remembered sentence into entities and relationships.

"She has an ulcer in her stomach, we saw a doctor at Mehr hospital" becomes

    child -has_condition-> ulcer
    ulcer -located_at-> stomach
    child -seen_at-> Mehr hospital

which is what lets the graph answer a question no keyword search can: which
hospital treated the ulcer.

Two extractors, and the order matters. The model reads paraphrase properly
and is tried first; a deterministic pass runs when the sidecar is down or the
model returns nothing usable. That fallback is not a nicety -- this project
serves a 4B model on one GPU, whose structured output is unreliable by its own
vendor's admission, so the graph has to keep growing without it.

The relationship vocabulary lives in config/memory.yaml. It is a starting
vocabulary, not a whitelist: an extraction proposing a relation outside it is
kept, because a parent will describe something nobody wrote down in advance.
"""

from __future__ import annotations

import json
import logging
import re

from assistant.refdata import memory_config

log = logging.getLogger(__name__)

CHILD = "child"

_SYSTEM = (
    "You turn a sentence from a parent into a small knowledge graph about "
    "their child. Reply with a JSON array of triples, each "
    '{"src": ..., "relation": ..., "dst": ..., "src_type": ..., "dst_type": ...}. '
    'Use "child" as src when the sentence is about the child themselves. '
    "Relations are lower_snake_case verbs. Keep labels short and in the "
    "parent's own words. Return [] if there is nothing to record."
)


def _clean(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _vocab() -> dict:
    cfg = (memory_config() or {}).get("graph") or {}
    return {
        "relations": [str(r) for r in (cfg.get("relations") or [])],
        "entity_types": [str(t) for t in (cfg.get("entity_types") or [])],
        "patterns": cfg.get("patterns") or [],
    }


def extract_with_llm(text: str) -> list[dict] | None:
    """Triples from the model. None when it could not be asked."""
    from assistant.llm.qwen_client import get_qwen, llm_enabled

    text = _clean(text)
    if not text or not llm_enabled():
        return None
    vocab = _vocab()
    hint = ""
    if vocab["relations"]:
        hint = "\n\nRelations already in use: " + ", ".join(vocab["relations"])
    try:
        raw = get_qwen().answer_with_context(text, hint.strip() or text, system=_SYSTEM)
    except Exception as exc:
        log.warning("Graph extraction call failed: %s", exc)
        return None
    return _parse(raw)


def _parse(raw: str) -> list[dict] | None:
    match = re.search(r"\[.*\]", (raw or "").strip(), re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, list):
        return None
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        src = _clean(str(item.get("src") or ""))
        dst = _clean(str(item.get("dst") or ""))
        rel = _clean(str(item.get("relation") or "")).lower().replace(" ", "_")
        if not (src and dst and rel):
            continue
        out.append(
            {
                "src": src,
                "relation": rel,
                "dst": dst,
                "src_type": _clean(str(item.get("src_type") or "entity")).lower(),
                "dst_type": _clean(str(item.get("dst_type") or "entity")).lower(),
            }
        )
    return out


def extract_deterministic(text: str) -> list[dict]:
    """Triples without a model, from patterns declared in config.

    Each pattern names the relation it produces and the type of the thing it
    captures, so adding "is taking <medicine>" is a config edit rather than a
    code change.
    """
    text = _clean(text)
    if not text:
        return []
    triples: list[dict] = []
    for spec in _vocab()["patterns"]:
        if not isinstance(spec, dict):
            continue
        pattern = spec.get("match")
        relation = spec.get("relation")
        if not pattern or not relation:
            continue
        # Case matters for proper nouns. Compiling everything IGNORECASE made
        # [A-Z] match lowercase, so "we saw a doctor at Mehr hospital" gave the
        # clinic the label "a doctor at Mehr hospital". Patterns that need to
        # see capitals declare ignore_case: false.
        flags = re.IGNORECASE if spec.get("ignore_case", True) else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error as exc:
            log.warning("Bad graph pattern %r: %s", pattern, exc)
            continue
        for m in rx.finditer(text):
            captured = _clean(m.group(m.lastindex or 1) if m.groups() else m.group(0))
            if not captured:
                continue
            triples.append(
                {
                    "src": str(spec.get("src") or CHILD),
                    "relation": str(relation),
                    "dst": captured,
                    "src_type": str(spec.get("src_type") or CHILD),
                    "dst_type": str(spec.get("dst_type") or "entity"),
                }
            )
    return triples


def extract(text: str, *, use_llm: bool = True) -> list[dict]:
    """Triples for one remembered sentence, model first then patterns."""
    if use_llm:
        triples = extract_with_llm(text)
        if triples:
            return triples
    return extract_deterministic(text)


def ingest(
    graph,
    text: str,
    *,
    subject: str,
    owner_user_id: str | None = None,
    fact_id: str | None = None,
    use_llm: bool = True,
) -> int:
    """Write the triples for one fact into the profile graph.

    The child node is created under the subject id rather than a name, so two
    children called Sara in one account stay separate.
    """
    triples = extract(text, use_llm=use_llm)
    if not triples:
        return 0
    written = 0
    for t in triples:
        src_label = t["src"]
        src_type = t["src_type"]
        if src_label.lower() in {CHILD, "the child", "baby", "the baby"}:
            src_label, src_type = subject, CHILD
        src_id = graph.upsert_node(
            subject=subject, label=src_label, type=src_type, owner_user_id=owner_user_id
        )
        dst_id = graph.upsert_node(
            subject=subject, label=t["dst"], type=t["dst_type"], owner_user_id=owner_user_id
        )
        if src_id and dst_id and src_id != dst_id:
            if graph.add_edge(
                subject=subject,
                src=src_id,
                relation=t["relation"],
                dst=dst_id,
                owner_user_id=owner_user_id,
                fact_id=fact_id,
            ):
                written += 1
    return written
