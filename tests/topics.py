"""A wide range of things parents actually ask about.

tests/scenarios.py generates one shape of conversation: a condition, a clinic,
an allergen, a medication. That shape found real bugs, but it is still one
shape, and code can be right for it and wrong for everything else. A parent
asks about sleep, teething, tantrums, screens, travel, nursery, siblings and
whether any of it is normal.

These are TOPICS, not scripts: each is a template with slots filled from the
same generated scenario, so no test depends on a particular sentence. Nothing
here is matched by the implementation -- it is only ever used to ask questions
and to check what comes back.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

# Statements a parent makes about their child. {name}/{p}/{poss} are filled
# from a scenario; {p} is the pronoun and {poss} the possessive.
STATEMENTS = [
    "{p} wakes up screaming around midnight most nights",
    "{p} has started refusing the bottle",
    "{p} only naps for twenty minutes at a time",
    "{p} is cutting {poss} first teeth and chews everything",
    "{p} throws food on the floor at every meal",
    "{p} has a birthmark on {poss} shoulder",
    "we moved house last month and {p} has been unsettled since",
    "{p} started nursery two weeks ago",
    "{poss} older brother had the same thing at this age",
    "{p} hates having {poss} nappy changed",
    "{p} watches cartoons while we eat dinner",
    "we are flying next week and {p} has never been on a plane",
    "{p} pulls {poss} own hair when tired",
    "{p} has not said any words yet",
    "{p} prefers to be held facing outwards",
    "{p} was in hospital overnight when {p} was born",
    "we swaddle {p} to get {p} to settle",
    "{p} sweats a lot when sleeping",
]

# Questions covering the ground a pediatric assistant should handle.
QUESTIONS = [
    "is that normal at this age?",
    "should I be worried?",
    "how long does this usually last?",
    "is there anything I can do to help?",
    "when should I see a doctor about it?",
    "how much sleep does {p} need?",
    "what foods should I be offering now?",
    "how do I know if {p} is getting enough milk?",
    "when do babies usually start walking?",
    "how much screen time is too much?",
    "how do I handle a tantrum in public?",
    "is it safe to travel by plane at this age?",
    "how do I clean {poss} teeth?",
    "what vaccines are due around now?",
    "how do I know if {p} has a temperature?",
    "should {p} be talking more by now?",
    "how do I get {p} to sleep through the night?",
    "what should I pack for nursery?",
]

# Bare follow-ups: they carry no topic of their own and only make sense
# against the previous turn. These are where context is most easily lost.
FOLLOW_UPS = [
    "how often?",
    "and at night?",
    "is that normal?",
    "for how long?",
    "what about during the day?",
    "and if it gets worse?",
    "why does that happen?",
    "should I change anything?",
    "what else should I watch for?",
]

# Persian, because the app serves Persian-speaking parents and an English-only
# test suite would never notice it breaking.
PERSIAN_STATEMENTS = [
    "دخترم شب‌ها بی‌قرار است",
    "پسرم غذا نمی‌خورد",
    "او تازه راه رفتن را شروع کرده",
]
PERSIAN_QUESTIONS = [
    "طبیعی است؟",
    "چه کار کنم؟",
    "چند وقت طول می‌کشد؟",
]


@dataclass(frozen=True)
class Turn:
    text: str
    kind: str  # statement | question | follow_up


def conversation(seed: int, scenario, *, length: int = 6, persian: bool = False) -> list[Turn]:
    """A varied conversation for one child.

    Mixes statements, questions and bare follow-ups so a test exercises the
    shapes a real chat has, rather than one question type repeated.
    """
    rng = random.Random(seed * 7919)
    p = scenario.pronoun
    poss = scenario.possessive
    statements = PERSIAN_STATEMENTS if persian else STATEMENTS
    questions = PERSIAN_QUESTIONS if persian else QUESTIONS

    turns: list[Turn] = []
    # Open with something the parent tells us, so there is memory to test.
    turns.append(Turn(rng.choice(statements).format(p=p, poss=poss, name=scenario.name),
                      "statement"))
    for _ in range(max(0, length - 1)):
        roll = rng.random()
        if roll < 0.35:
            text = rng.choice(statements)
            kind = "statement"
        elif roll < 0.75:
            text = rng.choice(questions)
            kind = "question"
        else:
            text = rng.choice(FOLLOW_UPS) if not persian else rng.choice(PERSIAN_QUESTIONS)
            kind = "follow_up"
        turns.append(Turn(text.format(p=p, poss=poss, name=scenario.name), kind))
    return turns
