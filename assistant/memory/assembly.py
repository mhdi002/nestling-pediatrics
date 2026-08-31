"""Assemble the four memories into one prompt, within a budget.

This is where a memory system is usually lost. Each layer can be individually
correct and the prompt still useless, because everything was concatenated and
the model was left to guess which part answered the question. That is not
hypothetical here: a question about a child's ulcer was answered with
vaccination-site guidance, because the guidance was in the prompt and looked
just as authoritative as the memory.

Two rules fix it.

Label the sources. The model is told which text is this child's history and
which is general guidance, so it can tell a fact about this baby from advice
about babies.

Give each kind a share of the budget and enforce it. The shares live in
settings as fractions of the prompt cap and are normalised here, so no kind
can crowd out another however much of it exists -- a chatty session cannot
push out the one line recording an allergy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from assistant.settings import get_settings

# Headings the model sees. Named for what the text IS to the parent, not for
# the subsystem that produced it.
PROCEDURAL_HEADING = "HOW TO ANSWER"
SEMANTIC_HEADING = "WHAT THIS PARENT HAS TOLD YOU ABOUT THEIR CHILD"
EPISODIC_HEADING = "EARLIER IN YOUR CONVERSATION"
WORKING_HEADING = "GENERAL CARE NOTES (WHO guidance)"


@dataclass
class Budget:
    """How many characters each memory kind may spend on this prompt."""

    procedural: int
    semantic: int
    episodic: int
    working: int

    @property
    def total(self) -> int:
        return self.procedural + self.semantic + self.episodic + self.working

    def as_dict(self) -> dict[str, int]:
        return {
            "procedural": self.procedural,
            "semantic": self.semantic,
            "episodic": self.episodic,
            "working": self.working,
        }


def budget(total_chars: int | None = None) -> Budget:
    """Split the prompt cap between the four kinds.

    Shares are normalised rather than required to sum to one, so an operator
    can raise a single share without recomputing the others.
    """
    s = get_settings()
    total = int(total_chars if total_chars is not None else s.llm_prompt_context_chars)
    shares = {
        "procedural": max(0.0, s.nestling_memory_share_procedural),
        "semantic": max(0.0, s.nestling_memory_share_semantic),
        "episodic": max(0.0, s.nestling_memory_share_episodic),
        "working": max(0.0, s.nestling_memory_share_working),
    }
    denom = sum(shares.values())
    if denom <= 0:
        # Degenerate configuration: spend it all on the child's own facts,
        # which is the part a parent is least willing to lose.
        return Budget(0, total, 0, 0)
    return Budget(**{k: int(total * v / denom) for k, v in shares.items()})


@dataclass
class AssembledContext:
    """The prompt context, plus what it cost, for tests and debugging."""

    text: str
    used: dict[str, int] = field(default_factory=dict)
    budget: dict[str, int] = field(default_factory=dict)
    sections: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


def _section(heading: str, body: str) -> str:
    return f"[{heading}]\n{body.strip()}"


def assemble(
    *,
    procedural: str = "",
    semantic: str = "",
    episodic: str = "",
    working: str = "",
    total_chars: int | None = None,
) -> AssembledContext:
    """Build the labelled, budgeted context block.

    Order is deliberate: how to answer, then who this child is, then what was
    said, then general guidance. The child's own facts come before the care
    notes so that a question about this child is answered from this child.
    """
    caps = budget(total_chars)
    parts: list[tuple[str, str, str, int]] = [
        ("procedural", PROCEDURAL_HEADING, procedural, caps.procedural),
        ("semantic", SEMANTIC_HEADING, semantic, caps.semantic),
        ("episodic", EPISODIC_HEADING, episodic, caps.episodic),
        ("working", WORKING_HEADING, working, caps.working),
    ]

    # An empty kind hands its budget to the kinds that follow rather than
    # wasting it: with no photo and no prior turns, the child's facts and the
    # care notes should be free to use the whole prompt.
    spare = sum(cap for _, _, body, cap in parts if not (body or "").strip())
    blocks: list[str] = []
    used: dict[str, int] = {}
    for name, heading, body, cap in parts:
        body = (body or "").strip()
        if not body:
            used[name] = 0
            continue
        allowance = cap + spare
        if len(body) > allowance:
            body = body[: max(0, allowance - 1)].rstrip() + "…"
            # It consumed its own share AND the spare, so there is none left
            # to hand on. Not zeroing it here spent the same characters twice
            # and pushed the assembled prompt past the cap.
            spare = 0
        else:
            spare = allowance - len(body)
        block = _section(heading, body)
        blocks.append(block)
        used[name] = len(body)

    return AssembledContext(
        text="\n\n".join(blocks),
        used=used,
        budget=caps.as_dict(),
        sections=[b.split("]", 1)[0].lstrip("[") for b in blocks],
    )
