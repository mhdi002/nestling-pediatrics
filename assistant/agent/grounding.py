"""Separate what the parent told us from what the corpus says.

The orchestrator prepends memory blocks to the medical query, so a turn arrives
as "<memory>\n[CURRENT_USER]\n<question>". Retrieval already ignores the memory
half, but generation did not: the whole string went in as the *question* while
the retrieved WHO chunks got the privileged *context* slot. A grounded model
told to "answer only from the care notes" then answers every turn from the
corpus, which is how "where was her ulcer?" came back as vaccination-site
guidance -- the ulcer was in the prompt, just not where the model was told to
look.

So we split the turn and hand the model both sources as labelled context, with
the parent's own words as the question. Deciding which source answers a given
turn is a judgement call the model is well suited to make, and one that no
threshold on term overlap can make for it: on a corpus this size the retrieved
notes contain most of a question's vocabulary whatever the question was about.
"""

from __future__ import annotations

import re

_CURRENT_USER = "[CURRENT_USER]"
_META_LINE = re.compile(r"^\s*Known [^\n]*$", re.IGNORECASE | re.MULTILINE)

PARENT_NOTES_HEADING = "WHAT THIS PARENT HAS TOLD YOU ABOUT THEIR CHILD"
CARE_NOTES_HEADING = "GENERAL CARE NOTES (WHO guidance)"

GROUNDED_SYSTEM = (
    "You are Nestling, a warm pediatric parent assistant. Two labelled sources "
    "follow. If the parent is asking about their own child -- something they "
    f"told you earlier, a name, an age, a symptom -- answer from '{PARENT_NOTES_HEADING}'. "
    f"Use '{CARE_NOTES_HEADING}' for general guidance and for what to do next. "
    "That section also carries the transcript, including your own earlier "
    "replies: treat what you said there as something you said, not as an "
    "established fact about the child. "
    # Both sources are working material, not something to talk about. Told to
    # say when it had not been told a thing, the model narrated its own notes
    # on every turn -- "the care notes you shared focus on newborns", "I don't
    # have your child's name or age yet" -- and asked again for an age the
    # parent had given one message earlier. A parent wants the answer, not a
    # report on the assistant's filing.
    "Use both sources silently. Never mention care notes, your records, or "
    "what you have or have not been told, and never restate or re-answer an "
    "earlier turn -- you already know it, so just carry on. Do not ask for "
    "anything the parent has already given. "
    # A question about something from the conversation kept being answered
    # from whatever chunk matched its words. Asked "what is the photo", the
    # corpus returned a page on preparing photographs for a clinician -- a
    # checklist -- and the model recited the checklist back, asking when the
    # spots started and whether they blanch, having just described them.
    "When the parent refers to something from earlier in this conversation -- "
    "a photo they sent, a symptom they described -- answer from the parent's "
    "notes above and never ask them for details those notes already contain. "
    "Only when the parent asks about a specific thing they told you earlier "
    "and it is genuinely not there, say briefly that you do not have it. "
    "Never guess, and never substitute a different topic for the one asked. "
    "Answer ONLY the current question's topic. Paraphrase in your own words. "
    "If the parent's notes state a chronological age in months, use ONLY that "
    "age; never infer one from a care-note section title. "
    "Do NOT include chain-of-thought, analysis steps, or 'Thinking Process'. "
    "Never invent drug doses. Always remind parents this is not a diagnosis "
    "and to see a clinician when worried."
)


def memory_context(query: str) -> str:
    """The memory blocks the orchestrator prepended, without the current turn."""
    text = query or ""
    if _CURRENT_USER not in text:
        return ""
    return text.split(_CURRENT_USER, 1)[0].strip()


def parent_question(query: str) -> str:
    """The parent's own words, free of memory blocks and injected metadata."""
    from assistant.websearch import current_user_query

    return current_user_query(query)


def compose_context(memory: str, care_notes: str) -> str:
    """Label both sources so the model can tell recall from guidance.

    Each half is held to its share of the prompt cap. Without that the two
    were concatenated at whatever length they happened to be -- eight thousand
    characters in a long conversation -- and answer_with_context chopped the
    tail at the cap. Since the care notes come second, they were the part
    thrown away: the model was handed a prompt with no guidance in it at all,
    and no error to say so. Sharing the cap explicitly means both halves
    survive, in the proportions configured for memory.
    """
    from assistant.memory.assembly import budget

    caps = budget()
    # Procedural rules ride in the system prompt, not here, so the cap is
    # split between what is remembered and what was retrieved.
    remembered_share = max(1, caps.semantic + caps.episodic)
    notes_share = max(1, caps.working)
    total = remembered_share + notes_share

    mem = _flatten_markers(_META_LINE.sub(" ", memory or "")).strip()
    notes = (care_notes or "").strip()

    # An absent half hands its room to the other rather than wasting it.
    if not mem:
        remembered_share, notes_share = 0, total
    elif not notes:
        remembered_share, notes_share = total, 0

    blocks = []
    if mem:
        blocks.append(f"[{PARENT_NOTES_HEADING}]\n{_fit(mem, remembered_share)}")
    if notes:
        blocks.append(f"[{CARE_NOTES_HEADING}]\n{_fit(notes, notes_share)}")
    return "\n\n".join(blocks)


_INNER_MARKER = re.compile(r"^\[([A-Z][A-Z_]*)\]\s*$", re.MULTILINE)


def _flatten_markers(text: str) -> str:
    """Turn the orchestrator's internal markers into plain sub-labels.

    The memory half arrives carrying [CHILD_MEMORY], [RECENT_CHAT] and friends,
    and is then wrapped in a section heading of its own. That left the prompt
    with two indistinguishable levels of [BRACKETS], so [RECENT_CHAT] read as a
    peer of [GENERAL CARE NOTES] rather than as part of what the parent said --
    undoing the very labelling that stopped a question about a child's ulcer
    being answered with vaccination guidance.

    The markers stay as they are in the query, because retrieval, slot
    extraction and the web-search gate all split on them. Only the copy the
    model reads is flattened, so there is exactly one level of section.

    Converted by rule rather than by a table, so a marker added later is
    handled without anyone remembering to come back here.
    """
    return _INNER_MARKER.sub(
        lambda m: f"{m.group(1).replace('_', ' ').capitalize()}:", text or ""
    )


def _fit(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"
