"""Consolidation: distil episodes into durable facts.

Sleep does this for people, and an agent needs the equivalent, because raw
conversation is a bad long-term store. It grows without bound, most of it is
small talk, and the one line that matters -- "she has an ulcer in her stomach"
-- is worth exactly as much attention as the twenty around it.

So every N turns the unconsolidated stretch is read and the durable claims are
written into semantic memory. Two properties matter more than cleverness:

  Nothing is lost. Episodes are never deleted by consolidation; a distilled
  fact sits alongside the turn it came from, and the watermark only ever moves
  forward, so no stretch is folded twice and none is skipped.

  It works without the model. Extraction prefers the LLM, which reads a
  paraphrase correctly; when the sidecar is down or the model returns nothing
  usable, a deterministic pass still captures first-person statements about
  the child. A memory system that only works when the GPU is up is not a
  memory system.
"""

from __future__ import annotations

import json
import logging
import re

from assistant.memory.types import Episode
from assistant.refdata import memory_config
from assistant.settings import get_settings

log = logging.getLogger(__name__)

_EXTRACT_SYSTEM = (
    "You extract durable facts about a child from a parent's conversation "
    "with a pediatric assistant. A durable fact is something still true next "
    "month: a diagnosis, an allergy, a symptom being followed, a clinic or "
    "clinician, a feeding or sleeping pattern, a milestone. Ignore greetings, "
    "questions, and anything the assistant said. Reply with a JSON array of "
    "short factual sentences in the parent's own terms, and nothing else. "
    "Return [] when the conversation holds no durable fact."
)

# A parent stating something about their child, as opposed to asking about it.
# Deliberately about grammatical shape, not about clinical vocabulary: a list
# of conditions could never cover what a parent might say.
_STATEMENT = re.compile(
    r"^\s*(?:my|our|she|he|they|her|his|the baby|the child)\b.*"
    r"\b(?:has|had|have|is|was|were|takes|took|started|stopped|got|gets|"
    r"eats|ate|sleeps|slept|weighs|weighed|saw|visited|allergic|diagnosed)\b",
    re.IGNORECASE,
)
_QUESTION = re.compile(r"\?\s*$|^\s*(?:what|when|how|why|where|who|should|can|is it|does)\b", re.I)


def _policy() -> dict:
    return (memory_config() or {}).get("consolidation") or {}


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def due(turns_since_last: int) -> bool:
    """Whether enough has been said to be worth consolidating."""
    every = get_settings().nestling_memory_consolidate_every
    return every > 0 and turns_since_last >= every


def extract_deterministic(episodes: list[Episode]) -> list[str]:
    """Durable claims without a model.

    Keeps first-person statements the parent made about the child and drops
    questions and assistant turns. Crude next to the model, and it is the
    reason consolidation still happens when the sidecar is down.
    """
    policy = _policy()
    min_chars = int(policy.get("min_fact_chars") or 12)
    facts: list[str] = []
    for ep in episodes:
        if (ep.role or "").lower() != "user":
            continue
        text = _clean(ep.content)
        if len(text) < min_chars or _QUESTION.search(text):
            continue
        if _STATEMENT.match(text):
            facts.append(text)
    return facts


def extract_with_llm(episodes: list[Episode]) -> list[str] | None:
    """Ask the model for durable facts. None when it could not be asked."""
    from assistant.llm.qwen_client import get_qwen, llm_enabled

    if not llm_enabled():
        return None
    transcript = "\n".join(
        e.as_line(get_settings().nestling_memory_line_chars) for e in episodes
    )
    if not transcript.strip():
        return []
    try:
        raw = get_qwen().answer_with_context(
            "Extract the durable facts about the child.",
            transcript,
            system=_EXTRACT_SYSTEM,
        )
    except Exception as exc:
        log.warning("Consolidation model call failed: %s", exc)
        return None
    return _parse_facts(raw)


def _parse_facts(raw: str) -> list[str] | None:
    """Read a JSON array out of a model reply that may carry prose around it."""
    text = (raw or "").strip()
    if not text:
        return None
    match = re.search(r"\[.*\]", text, re.DOTALL)
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
        if isinstance(item, str):
            out.append(_clean(item))
        elif isinstance(item, dict):
            # A small model sometimes answers [{"fact": "..."}] instead.
            value = item.get("fact") or item.get("text") or ""
            if isinstance(value, str) and value.strip():
                out.append(_clean(value))
    return out


class Consolidator:
    """Folds episodes into semantic memory, once each."""

    def __init__(self, semantic, episodic):
        self.semantic = semantic
        self.episodic = episodic

    def consolidate(
        self,
        *,
        session_id: str,
        subject: str,
        owner_user_id: str | None = None,
        episodes: list[Episode] | None = None,
        use_llm: bool = True,
    ) -> list[str]:
        """Distil a stretch of conversation. Returns the facts stored."""
        if not subject:
            # Nothing to attach a durable fact to yet.
            return []
        if episodes is None:
            episodes = self.episodic.recall(
                session_id=session_id, owner_user_id=owner_user_id, limit=200
            )
        if not episodes:
            return []

        policy = _policy()
        min_chars = int(policy.get("min_fact_chars") or 12)
        max_facts = int(policy.get("max_facts_per_pass") or 6)

        facts = extract_with_llm(episodes) if use_llm else None
        source = "consolidation_llm"
        if facts is None:
            facts = extract_deterministic(episodes)
            source = "consolidation_rule"

        # Never store the same claim twice: a fact already held for this child
        # is not news, and repeating it wastes the semantic budget.
        known = {
            _clean(r.text).lower()
            for r in self.semantic.recall(
                subject=subject, owner_user_id=owner_user_id, limit=200
            )
        }
        stored: list[str] = []
        confidence = get_settings().nestling_memory_inferred_confidence
        for fact in facts:
            fact = _clean(fact)
            if len(fact) < min_chars or fact.lower() in known:
                continue
            record = self.semantic.remember(
                fact,
                subject=subject,
                owner_user_id=owner_user_id,
                source=source,
                confidence=confidence,
                attributes={"session_id": session_id},
            )
            if record is not None:
                stored.append(fact)
                known.add(fact.lower())
            if len(stored) >= max_facts:
                break
        if stored:
            log.info(
                "Consolidated %d fact(s) for %s via %s", len(stored), subject, source
            )
        return stored
