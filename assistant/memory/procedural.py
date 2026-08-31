"""Procedural memory: how the assistant should act.

Distinct from the other kinds in that it is about the assistant, not the
child. It is also the only memory a parent writes by correcting us -- "stop
repeating yourself", "you asked me that already" -- so it has two sources:
house rules declared in config/memory.yaml, and learned rules stored per
account when a parent tells us how they want to be spoken to.

Rules are text, not code, so behaviour can be corrected without a release.
They are selected by the kind of turn in hand and ordered by priority, which
is what lets a tight context budget drop the least important rule rather than
an arbitrary one.
"""

from __future__ import annotations

from dataclasses import dataclass

from assistant.memory.types import PROCEDURAL, MemoryRecord
from assistant.refdata import memory_config


@dataclass(frozen=True)
class Rule:
    id: str
    text: str
    priority: int = 50
    applies_to: tuple[str, ...] = ()
    source: str = "config"

    def relevant_to(self, intents: set[str] | None) -> bool:
        """A rule with no `applies_to` is a house rule and always applies."""
        if not self.applies_to:
            return True
        if not intents:
            return False
        return bool(set(self.applies_to) & set(intents))


def _clean(text: str) -> str:
    return " ".join((text or "").split())


def house_rules() -> list[Rule]:
    """The declared rules, highest priority first."""
    raw = (memory_config() or {}).get("procedural") or []
    rules = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = _clean(item.get("text") or "")
        rid = str(item.get("id") or "").strip()
        if not text or not rid:
            continue
        applies = item.get("applies_to") or []
        rules.append(
            Rule(
                id=rid,
                text=text,
                priority=int(item.get("priority") or 50),
                applies_to=tuple(str(a) for a in applies),
            )
        )
    return sorted(rules, key=lambda r: (-r.priority, r.id))


class ProceduralMemory:
    """House rules, plus what this account has taught us."""

    def __init__(self, backend=None):
        self.backend = backend

    def learn(
        self,
        text: str,
        *,
        owner_user_id: str | None,
        rule_id: str = "",
        priority: int = 60,
    ) -> MemoryRecord | None:
        """Record a correction the parent made about how we should behave.

        Stored against the account rather than a child: it is about the
        conversation, not about one baby.
        """
        text = _clean(text)
        if not text or self.backend is None:
            return None
        record = MemoryRecord(
            text=text,
            kind=PROCEDURAL,
            subject=owner_user_id or "",
            owner_user_id=owner_user_id,
            source="parent_correction",
            attributes={"rule_id": rule_id or "", "priority": int(priority)},
        )
        return self.backend.remember_fact(record)

    def learned_rules(self, *, owner_user_id: str | None) -> list[Rule]:
        if self.backend is None or not owner_user_id:
            return []
        records = self.backend.recall_facts(
            owner_user_id, owner_user_id=owner_user_id, limit=50
        )
        rules = []
        for r in records:
            if r.kind != PROCEDURAL:
                continue
            rules.append(
                Rule(
                    id=str(r.attributes.get("rule_id") or r.id),
                    text=r.text,
                    priority=int(r.attributes.get("priority") or 60),
                    source="parent_correction",
                )
            )
        return rules

    def rules_for(
        self, *, intents: set[str] | None = None, owner_user_id: str | None = None
    ) -> list[Rule]:
        """Every rule that applies to this turn, most important first.

        A learned rule replaces a house rule of the same id: a parent telling
        us how they want to be spoken to outranks the default.
        """
        by_id: dict[str, Rule] = {}
        for rule in house_rules():
            if rule.relevant_to(intents):
                by_id[rule.id] = rule
        for rule in self.learned_rules(owner_user_id=owner_user_id):
            by_id[rule.id] = rule
        return sorted(by_id.values(), key=lambda r: (-r.priority, r.id))

    def render(
        self,
        *,
        intents: set[str] | None = None,
        owner_user_id: str | None = None,
        budget_chars: int | None = None,
    ) -> str:
        """The rules as prompt lines, dropping the least important on overflow."""
        lines: list[str] = []
        used = 0
        for rule in self.rules_for(intents=intents, owner_user_id=owner_user_id):
            line = f"- {rule.text}"
            if budget_chars is not None and used + len(line) + 1 > budget_chars:
                # Stop rather than skip: skipping kept whichever lower rules
                # happened to be short enough and dropped the most important
                # one, which is backwards for a priority list. But a budget
                # too small for even the first rule must still say something
                # about how to answer, so truncate that one instead of
                # returning nothing at all.
                if not lines and budget_chars > 1:
                    lines.append(line[: budget_chars - 1].rstrip() + "…")
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)
