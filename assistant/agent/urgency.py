#!/usr/bin/env python3
"""
Is this parent reporting an emergency?

WHY THIS IS NOT DONE WITH RETRIEVAL SCORES

The first version of the escalation compared BM25 scores: if the corpus's
strongest match for a question was a clinician procedure and nothing written
for parents came close, the question was called an emergency. Measured on this
corpus that measure does not separate the classes at all -- "what foods are
good for her?" (clinician 11.94 / parent 8.74) outranks four of six genuine
emergencies, and with the feature on a question about solid food was answered
as an emergency. The measurements are recorded in
`clinician_only_topic` in assistant/rag/audience.py, which is now dead weight
kept only for that record.

WHAT REPLACED IT

An intent, with two paths:

  1. The LLM intent router (assistant/agent/router.py) is the primary
     classifier. It reads the sentence; that is what a classifier is for.

  2. This module is the deterministic fallback, and it runs on every turn
     whether or not the sidecar answered. The app runs on one GPU that can be
     down, and an emergency check that only works while the model is up is not
     a safety check.

THE SIGNAL THE FALLBACK USES

Two things have to be true at once, and neither of them is a keyword list:

  A RED-FLAG SEMANTIC CLASS IS NAMED. The classes live in
  config/urgency_signs.yaml, and every one of them is keyed on words taken
  from the danger-sign items WHO's own material in this corpus enumerates --
  the bullet lists in curated_who_danger_signs_en.md and in PCPNC's newborn
  danger-sign sheets. `danger_sign_items()` below extracts those items from
  the corpus, and tests/test_urgency_intent.py fails if a class claims a word
  the corpus does not use. So the classes are the corpus's, not mine. What the
  config adds on top is surface forms -- "went blue" for "Diffuse cyanosis",
  "کبود شده" for the same thing -- because the corpus is a clinical text in
  English and the parent is not writing one.

  IT IS REPORTED, NOT ASKED ABOUT. This is the part that killed the previous
  attempt and it is structural, not lexical. "What are the danger signs I
  should watch for" and "what should I do if my baby has a seizure" name the
  class as plainly as "my baby is having a seizure" does. The difference is
  grammatical: in the first two the class sits inside an interrogative or
  conditional frame, in the third it is predicated of the child. So each
  clause is cut at its first question or conditional word (and a clause that
  opens with an auxiliary is a yes/no question end to end), and a sign only
  counts when it is named BEFORE that cut, in a message that refers to this
  child.

  A vital function -- breathing, movement, responsiveness, bleeding -- also
  has to be reported as FAILING before it counts, using a closed class of
  negation / inability / cessation words plus the qualifiers the corpus's own
  items attach to a failing sign ("Difficulty breathing", "Severe chest
  indrawing"). "Is he breathing?" and "he is breathing fast when he cries" are
  not emergencies; "he is not breathing" is.

There is no score and no threshold here, deliberately. A threshold is what the
last attempt had, and any threshold on a sample of these examples is a
constant chosen to make the examples pass.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from assistant.refdata import urgency_signs

log = logging.getLogger(__name__)

INTENT = "urgent"

# Matched at word starts, as elsewhere in the project: a Persian or English
# suffix must not stop a form matching ("کبود" in "کبودشده"), but a form must
# not be found inside an unrelated word either.
_WORD_START = r"(?<![^\W\d_])"
_WORD_END = r"(?![^\W\d_])"

# Clause boundaries: sentence punctuation in both scripts, plus the
# coordinators that join two independent statements.
_CLAUSE_SPLIT = re.compile(
    r"[.!?;:\n،؛؟]+|\s+(?:and|but|or|so|because|while|then|though|although)\s+",
    re.IGNORECASE,
)

_LATIN = re.compile(r"[A-Za-z]")


def _normalise(text: str) -> str:
    """Fold the spelling variants that would otherwise hide a match.

    Persian is written with and without the zero-width non-joiner
    ("نمی‌کشه" / "نمیکشه") and with Arabic ي/ك in place of Persian ی/ک; typed
    English uses three different apostrophes. None of that is a difference in
    meaning, and all of it breaks literal matching.
    """
    out = unicodedata.normalize("NFKC", text or "")
    out = out.replace("‌", "").replace("‏", "").replace("‎", "")
    out = out.replace("ي", "ی").replace("ك", "ک")
    out = out.replace("’", "'").replace("ʼ", "'").replace("`", "'")
    return re.sub(r"[ \t]+", " ", out.lower())


def _alternation(forms: tuple[str, ...], *, anchor_end: bool) -> str | None:
    """A regex alternation over `forms`, whitespace-tolerant."""
    parts = []
    for form in sorted({f for f in forms if f}, key=len, reverse=True):
        words = _normalise(form).split()
        if not words:
            continue
        parts.append(r"\s+".join(re.escape(w) for w in words))
    if not parts:
        return None
    body = _WORD_START + "(?:" + "|".join(parts) + ")"
    return body + _WORD_END if anchor_end else body


def _matcher(forms: list[str]) -> re.Pattern[str] | None:
    """
    One pattern for a set of surface forms.

    Latin-script forms are anchored at both ends -- without a right anchor
    "limp" matches "limping" and "no" matches "nose", which would report a
    limping toddler and a runny nose as emergencies. Persian forms are
    anchored on the left only, because Persian glues its inflection on the
    right ("کبود" in "کبودشده") and a right anchor would lose every inflected
    form.
    """
    latin = [f for f in forms if _LATIN.search(f)]
    other = [f for f in forms if not _LATIN.search(f)]
    bodies = [
        b
        for b in (
            _alternation(tuple(latin), anchor_end=True),
            _alternation(tuple(other), anchor_end=False),
        )
        if b
    ]
    if not bodies:
        return None
    return re.compile("|".join(bodies), re.IGNORECASE)


@lru_cache(maxsize=1)
def _signs() -> tuple[tuple[str, frozenset[str], re.Pattern[str]], ...]:
    """(class name, impairment kinds it needs, matcher) for every red flag."""
    out = []
    for name, spec in (urgency_signs().get("signs") or {}).items():
        forms = list(spec.get("forms_en") or []) + list(spec.get("forms_fa") or [])
        pattern = _matcher(forms)
        if pattern is None:
            continue
        kinds = frozenset(str(k) for k in (spec.get("impairment_kinds") or ()))
        unknown = kinds - set(_impairment())
        if unknown:
            raise ValueError(
                f"config/urgency_signs.yaml: sign {name!r} asks for unknown "
                f"impairment kind(s) {sorted(unknown)}"
            )
        out.append((str(name), kinds, pattern))
    if not out:
        raise ValueError("config/urgency_signs.yaml declares no signs")
    return tuple(out)


@lru_cache(maxsize=1)
def _impairment() -> dict[str, re.Pattern[str]]:
    """One matcher per kind of failure (negation / cessation / severity)."""
    out: dict[str, re.Pattern[str]] = {}
    for kind, langs in (urgency_signs().get("impairment_markers") or {}).items():
        forms = [f for words in (langs or {}).values() for f in (words or [])]
        pattern = _matcher(forms)
        if pattern is not None:
            out[str(kind)] = pattern
    if not out:
        raise ValueError("config/urgency_signs.yaml declares no impairment markers")
    return out


@lru_cache(maxsize=1)
def _question() -> re.Pattern[str] | None:
    cfg = urgency_signs()
    forms = list(cfg.get("question_words_en") or []) + list(cfg.get("question_words_fa") or [])
    latin = [f for f in forms if _LATIN.search(f)]
    other = [f for f in forms if not _LATIN.search(f)]
    bodies = [
        b
        for b in (
            _alternation(tuple(latin), anchor_end=True),
            # Persian question words are anchored on both ends too: "چی" would
            # otherwise match inside "چیزی" and veto a real report.
            _alternation(tuple(other), anchor_end=True),
        )
        if b
    ]
    return re.compile("|".join(bodies), re.IGNORECASE) if bodies else None


@lru_cache(maxsize=1)
def _initial_aux() -> re.Pattern[str] | None:
    forms = tuple(urgency_signs().get("clause_initial_auxiliaries_en") or ())
    body = _alternation(forms, anchor_end=True)
    return re.compile(r"^\s*(?:" + body + ")", re.IGNORECASE) if body else None


@lru_cache(maxsize=1)
def _child_nouns() -> re.Pattern[str] | None:
    cfg = urgency_signs()
    return _matcher(
        list(cfg.get("child_nouns_en") or []) + list(cfg.get("child_nouns_fa") or [])
    )


@lru_cache(maxsize=1)
def _determiners() -> frozenset[str]:
    return frozenset(_normalise(w) for w in (urgency_signs().get("possessives_en") or []))


@lru_cache(maxsize=1)
def _pronouns() -> re.Pattern[str] | None:
    cfg = urgency_signs()
    return _matcher(
        list(cfg.get("subject_pronouns_en") or []) + list(cfg.get("subject_pronouns_fa") or [])
    )


@lru_cache(maxsize=1)
def _existentials() -> re.Pattern[str] | None:
    return _matcher(list(urgency_signs().get("existentials_en") or []))


# How far back from a child noun a determiner still binds to it: "my baby",
# "my 3 month old". Three tokens covers a determiner, a number and a unit,
# which is the longest such phrase English builds without a second noun.
_DETERMINER_WINDOW = 3


def refers_to_child(text: str) -> bool:
    """True when the message is about a specific person, not about babies in general.

    "when do babies start talking" is a question about the category; "my baby
    is not breathing" is a report about one child. The difference is the
    determiner, so a bare category noun does not count.
    """
    body = _normalise(text)
    if not body.strip():
        return False
    pronouns = _pronouns()
    if pronouns and pronouns.search(body):
        return True
    existentials = _existentials()
    if existentials and existentials.search(body):
        return True
    nouns = _child_nouns()
    if nouns:
        tokens = re.findall(r"[^\W\d_]+", body)
        for match in nouns.finditer(body):
            head = body[: match.start()]
            recent = re.findall(r"[^\W\d_]+", head)[-_DETERMINER_WINDOW:]
            if _determiners() & set(recent):
                return True
            # Persian marks the possessive as a suffix ("بچم" = my child), so a
            # Persian child noun needs no separate determiner.
            if not _LATIN.search(match.group(0)):
                return True
        if not tokens:
            return False
    # A message with no subject at all is still a report -- a parent typing
    # "not breathing!!" is not asking a general question.
    return not re.search(r"(?<![^\W\d_])(?:i|we|you|my|our)(?![^\W\d_])", body)


def _ask_cut(clause: str) -> int:
    """Index at which the clause stops reporting and starts asking.

    Everything from the first question or conditional word to the end of the
    clause is being asked about. A clause that opens with an auxiliary is a
    yes/no question from its first character.
    """
    aux = _initial_aux()
    if aux and aux.match(clause):
        return 0
    question = _question()
    if not question:
        return len(clause)
    match = question.search(clause)
    return match.start() if match else len(clause)


def urgent_signs(text: str) -> list[str]:
    """Names of the red-flag classes this message REPORTS (not asks about)."""
    body = _normalise(text)
    if not body.strip():
        return []
    if not refers_to_child(body):
        return []
    found: list[str] = []
    for clause in _CLAUSE_SPLIT.split(body):
        clause = clause.strip()
        if not clause:
            continue
        cut = _ask_cut(clause)
        if cut <= 0:
            continue
        reported = {
            kind
            for kind, pattern in _impairment().items()
            if any(m.start() < cut for m in pattern.finditer(clause))
        }
        for name, kinds, pattern in _signs():
            if kinds and not (kinds & reported):
                continue
            for match in pattern.finditer(clause):
                if match.start() >= cut:
                    continue
                if name not in found:
                    found.append(name)
                break
    return found


def is_urgent(text: str, *, en_text: str | None = None) -> bool:
    """
    True when this turn reports an emergency, without asking any model.

    Both the parent's own words and the translated English are checked: a
    Persian turn reaches the router already translated, but that translation
    is an HTTP call to an outside service and it is exactly the kind of thing
    that is unavailable at the same time as everything else.
    """
    if urgent_signs(text):
        return True
    return bool(en_text and en_text != text and urgent_signs(en_text))


# --------------------------------------------------------------------------
# Corpus derivation. Not used to answer a turn -- used to prove that the
# classes in config/urgency_signs.yaml are the corpus's classes and not ones
# somebody invented. tests/test_urgency_intent.py is the caller.
# --------------------------------------------------------------------------

# A bullet in this corpus survives extraction either as a markdown "- " at the
# start of a line, or as the source PDF's bullet glyph, which lands as C0/C1
# control characters. Both mark an enumerated item; neither is a word.
_BULLET = re.compile(r"(?:^|\n)\s*-\s+|[\x00-\x08\x0b-\x1f\x7f-\x9f]+")
# A shouted heading ("POSSIBLE SERIOUS ILLNESS") ends the list of signs and
# starts the block of instructions to the health worker.
_SHOUTED = re.compile(r"\b[A-Z]{3,}(?:\s+[A-Z]{3,})*\b")


def _enumerated_items(text: str) -> list[str]:
    body = re.sub(r"(?m)^##.*$", "", text or "")
    parts = _BULLET.split(body)
    if len(parts) == 1:
        return []
    out: list[str] = []
    for segment in parts[1:]:
        segment = " ".join(segment.split())
        if not segment:
            continue
        shouted = _SHOUTED.search(segment)
        if shouted:
            head = segment[: shouted.start()].strip()
            if head:
                out.append(head)
            break
        out.append(segment)
    return out


def _chunks_path() -> Path:
    from assistant.config import KNOWLEDGE_DIR

    return Path(KNOWLEDGE_DIR) / "chunks.json"


def _load_chunks() -> list[dict]:
    path = _chunks_path()
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return []
    return data if isinstance(data, list) else []


def danger_sign_items(*, child_only: bool = True) -> list[str]:
    """The danger signs the corpus enumerates, one string per item.

    `child_only` selects the sections config/urgency_signs.yaml declares as
    the child danger-sign material; False widens it to every danger-sign
    section in the corpus, the maternal counselling sheets included.
    """
    cfg = urgency_signs()
    items: list[str] = []
    if child_only:
        selectors = [
            (str(s.get("source") or ""), str(s.get("title") or "").strip())
            for s in (cfg.get("danger_sign_sections") or [])
        ]
        for chunk in _load_chunks():
            for source, title in selectors:
                if chunk.get("source") != source:
                    continue
                if title and str(chunk.get("title") or "").strip() != title:
                    continue
                items.extend(_enumerated_items(chunk.get("text") or ""))
                break
        return items
    needle = str(cfg.get("danger_sign_title_contains") or "danger sign").lower()
    for chunk in _load_chunks():
        if needle in str(chunk.get("title") or "").lower():
            items.extend(_enumerated_items(chunk.get("text") or ""))
    return items


def danger_sign_terms(*, child_only: bool = True) -> frozenset[str]:
    """Every word the corpus uses inside a danger-sign item."""
    words: set[str] = set()
    for item in danger_sign_items(child_only=child_only):
        words.update(re.findall(r"[a-z]+", item.lower()))
    return frozenset(words)


# How many leading letters two words must share before one is treated as an
# inflection of the other. Four is what it takes to relate the config's
# "difficulties"/"severely"/"heavily" to the corpus's "difficulty"/"severe"/
# "heavy" without relating anything else in either vocabulary.
_ATTESTED_PREFIX = 4


def corpus_attests(word: str, *, child_only: bool = True) -> bool:
    """True when the danger-sign items use this word, or an inflection of it."""
    body = (word or "").lower().strip()
    if not body:
        return False
    corpus = danger_sign_terms(child_only=child_only)
    if body in corpus:
        return True
    head = body[:_ATTESTED_PREFIX]
    if len(head) < _ATTESTED_PREFIX:
        return False
    return any(term.startswith(head) for term in corpus)


def clear_cache() -> None:
    """Drop compiled matchers (tests reload the config)."""
    for cached in (
        _signs,
        _impairment,
        _question,
        _initial_aux,
        _child_nouns,
        _determiners,
        _pronouns,
        _existentials,
    ):
        cached.cache_clear()
