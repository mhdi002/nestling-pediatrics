#!/usr/bin/env python3
"""
Who was a knowledge chunk written for?

Part of the corpus is adapted from manuals written for skilled health workers
(WHO's PCPNC guide, the IYCF counselling course). Those books carry procedures
a parent must never be walked through -- bag-mask ventilation of a newborn, IV
access, IM antibiotic doses, manual removal of a placenta. Retrieval had no
notion of audience, so a frightened parent typing "my baby is not breathing"
could be handed a resuscitation chart as if it were advice.

Nothing is deleted here. Every chunk keeps its place in the index; each one
just gains an `audience` label, and the parent-facing answer path retrieves
only what was written for parents. The label comes from two structural facts,
never from a list of alarming words:

  Provenance -- config/knowledge_audience.yaml records, per source document,
  the audience that document was written for. Undeclared sources fail closed.

  Grammatical person, in document order -- a clinician manual talks *about*
  the mother and the baby, while the pages it writes for the family talk *to*
  them. PCPNC gathers those pages in one block at the back of the book, which
  its own front matter describes. So each section is scored on grammatical
  person and the document is cut at the single point where the running total
  of that score bottoms out: the moment the book stops speaking about the
  family and starts speaking to them. The head stays clinician-only, the tail
  is released to parents. A manual with no such section (IYCF) yields an empty
  tail and stays clinician-only end to end.

The word lists this uses are closed grammatical classes -- pronouns, and the
nouns naming the people a maternity text is about. They determine whom a
sentence is about, never what it is about, so no clinical topic can be
smuggled in or out by editing them.
"""

from __future__ import annotations

import fnmatch
import logging
import re
from functools import lru_cache

from assistant.refdata import knowledge_audience
from assistant.settings import get_settings

log = logging.getLogger(__name__)

PARENT = "parent"
CLINICIAN = "clinician"

# Chunks are matched at word starts, not on raw substrings, so that "eating"
# counts as a person-free token rather than as the noun "a child" hidden
# inside another word.
_WORD_START = r"(?<![^\W\d_])"
_WORD_END = r"(?![^\W\d_])"


@lru_cache(maxsize=1)
def _patterns() -> tuple[re.Pattern[str], re.Pattern[str]]:
    """Compile the second-person and third-person-reference matchers."""
    person = knowledge_audience().get("person") or {}
    second = [re.escape(w) for w in person.get("second_person") or ()]
    roles = [re.escape(w) for w in person.get("role_nouns") or ()]
    dets = [re.escape(w) for w in person.get("determiners") or ()]
    if not second or not roles or not dets:
        raise ValueError("config/knowledge_audience.yaml: person markers are incomplete")
    second_re = re.compile(
        _WORD_START + "(?:" + "|".join(second) + ")" + _WORD_END, re.IGNORECASE
    )
    # "the mother", "her small baby", "every child" -- a determiner, up to two
    # intervening modifiers, then one of the role nouns.
    third_re = re.compile(
        _WORD_START
        + "(?:"
        + "|".join(dets)
        + r")\s+(?:\w+\s+){0,2}(?:"
        + "|".join(roles)
        + ")"
        + _WORD_END,
        re.IGNORECASE,
    )
    return second_re, third_re


def person_counts(text: str) -> tuple[int, int]:
    """(second-person mentions, third-person references to the people in the text)."""
    second_re, third_re = _patterns()
    body = text or ""
    return len(second_re.findall(body)), len(third_re.findall(body))


def address_score(text: str) -> float:
    """
    +1 when a passage speaks only to its reader, -1 when only about others.

    0.0 means "no evidence either way" -- a dose table or an equipment list
    names nobody -- and callers fall back to the document's declared audience.
    """
    second, third = person_counts(text)
    total = second + third
    if not total:
        return 0.0
    return (second - third) / total


@lru_cache(maxsize=1)
def _source_rules() -> tuple[tuple[str, str], ...]:
    raw = knowledge_audience().get("sources") or {}
    return tuple((str(pattern), str(value).strip().lower()) for pattern, value in raw.items())


@lru_cache(maxsize=1)
def _default_audience() -> str:
    value = str(knowledge_audience().get("default_audience") or CLINICIAN).strip().lower()
    return PARENT if value == PARENT else CLINICIAN


def declared_audience(source: str | None) -> str:
    """The audience the source document was written for; fail closed when unknown."""
    name = str(source or "")
    for pattern, value in _source_rules():
        if name == pattern or fnmatch.fnmatch(name, pattern):
            return PARENT if value == PARENT else CLINICIAN
    return _default_audience()


def is_source_declared(source: str | None) -> bool:
    """False when no rule in config/knowledge_audience.yaml covers this source."""
    name = str(source or "")
    return any(name == p or fnmatch.fnmatch(name, p) for p, _ in _source_rules())


def reader_facing_split(scores: list[float]) -> int:
    """
    Index at which a document turns from writing about the family to writing
    to them, or len(scores) when it never does.

    The running total of the address scores falls while the text is
    third-person and climbs once it is second-person, so the turning point is
    where that total is lowest. One pass, no tuning knob for *where* the cut
    lands -- only for whether the split is believed at all.
    """
    running = 0.0
    lowest = 0.0
    split = 0
    for i, value in enumerate(scores):
        running += value
        if running < lowest:
            lowest = running
            split = i + 1
    return split if split else len(scores)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _label_clinician_document(docs: list[dict]) -> None:
    """Label one clinician-written document, releasing its parent-facing tail."""
    settings = get_settings()
    for doc in docs:
        doc["audience"] = CLINICIAN
    if len(docs) < settings.nestling_audience_min_sections:
        return
    scores = [address_score(d.get("text") or "") for d in docs]
    split = reader_facing_split(scores)
    head, tail = scores[:split], scores[split:]
    if not tail:
        return
    tail_mean = _mean(tail)
    if tail_mean <= 0:
        return
    if tail_mean - _mean(head) < settings.nestling_audience_min_separation:
        return
    for doc in docs[split:]:
        doc["audience"] = PARENT


def label_chunks(chunks: list[dict]) -> dict[str, int]:
    """
    Stamp `audience` on every chunk in place. Returns a per-label tally.

    Document order matters, so chunks are grouped by source in the order they
    appear -- which is the order the rebuild script writes them, which is the
    order they appear in the source document.
    """
    by_source: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_source.setdefault(str(chunk.get("source") or ""), []).append(chunk)
    for source, docs in by_source.items():
        if declared_audience(source) == PARENT:
            for doc in docs:
                doc["audience"] = PARENT
            continue
        if not is_source_declared(source):
            log.warning(
                "Knowledge source %r is not declared in config/knowledge_audience.yaml; "
                "withholding its %d chunk(s) from parent answers",
                source,
                len(docs),
            )
        _label_clinician_document(docs)
    tally: dict[str, int] = {}
    for chunk in chunks:
        label = str(chunk.get("audience") or CLINICIAN)
        tally[label] = tally.get(label, 0) + 1
    return tally


def is_parent_facing(doc: dict) -> bool:
    """True when this chunk may be quoted back to a parent."""
    label = doc.get("audience")
    if label is None:
        # An index built before audience labelling existed. Fall back to the
        # document's declared audience rather than trusting an unlabelled row.
        return declared_audience(doc.get("source")) == PARENT
    return str(label).strip().lower() == PARENT


# WHY THERE IS NO SCORE-BASED EMERGENCY TEST HERE ANY MORE
#
# There used to be a `clinician_only_topic(pool, parent_hits)` in this module:
# a question was called an emergency when the corpus's strongest match for it
# was a clinician procedure and nothing written for parents came close. The
# idea is right; BM25 scores are the wrong instrument for it. Measured over
# emergency and ordinary phrasings on this corpus, the two classes overlap
# badly:
#
#     my baby is not breathing         clinician  8.19  parent  6.60
#     my baby is having convulsions    clinician  8.12  parent  6.89
#     what foods are good for her?     clinician 11.94  parent  8.74
#     is my baby gaining enough weight clinician 12.59  parent 10.36
#
# "What foods are good for her?" outranks four of six real emergencies on
# every ratio tried, and with escalation on it was answered as one. A pair of
# thresholds can be fitted to any sample of these, but that is a constant
# chosen to make the examples pass, not a signal, so the feature shipped
# switched off. Recognising an emergency needs a classifier that reads the
# sentence; it is now the `urgent` intent in assistant/agent/urgency.py, and
# the escalation is driven from there.
#
# None of that was ever the safety guarantee. Clinician procedures are
# withheld from parents by the audience filter above, which is
# provenance-based, always on, and independent of any of this.


def clear_cache() -> None:
    """Drop compiled patterns and provenance rules (tests reload the config)."""
    _patterns.cache_clear()
    _source_rules.cache_clear()
    _default_audience.cache_clear()
