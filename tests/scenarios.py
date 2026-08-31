"""Generated conversation scenarios, so tests are not fitted to examples.

Every memory test so far used the same two stories -- an ulcer at Mehr
hospital, peanuts and eczema. Code tuned to those would pass them and fail a
real parent, and the tests would never say so.

So scenarios are composed here from vocabularies the implementation has never
seen. The condition names deliberately avoid the ones written into the
fallback patterns in config/memory.yaml: if recall only works for words
somebody listed in advance, these tests are meant to expose that rather than
hide it.

Seeded, so a failure names the seed that produced it and can be replayed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# Names from several scripts: a memory that only works in ASCII is not one
# this project can ship, given it serves Persian-speaking parents.
NAMES = [
    "Arman", "Bita", "Kian", "Roya", "Nika", "Sina", "Yalda", "Omid",
    "Parisa", "Reza", "Sahar", "Tara", "Vida", "Zari", "Mehrdad", "Golnar",
    "مریم", "علی", "زهرا", "امیر",
]

# Deliberately NOT the words in the config fallback patterns. If anything only
# works for ulcer/rash/eczema, these break it.
CONDITIONS = [
    "bronchiolitis", "colic", "reflux", "thrush", "impetigo", "croup",
    "constipation", "conjunctivitis", "roseola", "hand foot and mouth",
    "jaundice", "cradle cap", "otitis media", "gastroenteritis",
]

BODY_PARTS = [
    "left elbow", "right knee", "scalp", "tummy", "back", "neck", "chin",
    "left ear", "right foot", "chest", "forehead", "thigh",
]

CLINICS = [
    "Razi clinic", "Milad hospital", "Pars health centre", "Shafa clinic",
    "Nur Hospital", "Erfan clinic", "Sina hospital", "Atieh clinic",
]

ALLERGENS = [
    "shellfish", "sesame", "cow's milk", "soy", "wheat", "kiwi", "cashews",
    "penicillin", "dust mites", "egg white",
]

MEDICATIONS = [
    "paracetamol suspension", "vitamin D drops", "iron syrup", "salbutamol",
    "hydrocortisone cream", "oral rehydration salts", "zinc syrup",
]

# Filler turns: on-topic but carrying no durable fact. These are what bury a
# remembered detail when recall is by recency.
CHATTER = [
    "how much should she sleep at this age?",
    "is it normal for him to be fussy in the evening?",
    "when do babies start crawling?",
    "should I be worried about screen time?",
    "how often should I bathe him?",
    "what temperature should the room be?",
    "he keeps kicking his blanket off",
    "she does not like tummy time",
    "how many teeth should he have by now?",
    "is it okay if she skips a nap?",
]


@dataclass
class Scenario:
    """One child, several durable facts, and questions that recall them."""

    seed: int
    name: str
    condition: str
    body_part: str
    clinic: str
    allergen: str
    medication: str
    pronoun: str
    facts: list[str] = field(default_factory=list)
    chatter: list[str] = field(default_factory=list)

    # -- what the parent said ---------------------------------------------
    @property
    def condition_fact(self) -> str:
        return f"{self.pronoun} has {self.condition} on {self.possessive} {self.body_part}"

    @property
    def clinic_fact(self) -> str:
        return f"we went to {self.clinic} about it"

    @property
    def allergy_fact(self) -> str:
        return f"{self.pronoun} is allergic to {self.allergen}"

    @property
    def medication_fact(self) -> str:
        return f"{self.pronoun} is taking {self.medication}"

    @property
    def possessive(self) -> str:
        return "her" if self.pronoun == "she" else "his"

    # -- what the parent asks later ---------------------------------------
    @property
    def where_question(self) -> str:
        return f"where is {self.possessive} {self.condition}?"

    @property
    def clinic_question(self) -> str:
        return "which clinic did we go to?"

    @property
    def allergy_question(self) -> str:
        return f"what is {self.pronoun} allergic to?"

    @property
    def medication_question(self) -> str:
        return f"what is {self.pronoun} taking?"

    def __str__(self) -> str:  # pragma: no cover - shown on failure
        return (
            f"Scenario(seed={self.seed}, name={self.name!r}, "
            f"condition={self.condition!r}, part={self.body_part!r}, "
            f"clinic={self.clinic!r}, allergen={self.allergen!r})"
        )


def make(seed: int) -> Scenario:
    """A reproducible scenario. The seed is printed on failure."""
    rng = random.Random(seed)
    pronoun = rng.choice(["she", "he"])
    scenario = Scenario(
        seed=seed,
        name=rng.choice(NAMES),
        condition=rng.choice(CONDITIONS),
        body_part=rng.choice(BODY_PARTS),
        clinic=rng.choice(CLINICS),
        allergen=rng.choice(ALLERGENS),
        medication=rng.choice(MEDICATIONS),
        pronoun=pronoun,
    )
    scenario.facts = [
        scenario.condition_fact,
        scenario.clinic_fact,
        scenario.allergy_fact,
        scenario.medication_fact,
    ]
    scenario.chatter = rng.sample(CHATTER, k=rng.randint(4, len(CHATTER)))
    return scenario


def many(count: int, *, start: int = 0) -> list[Scenario]:
    return [make(start + i) for i in range(count)]
