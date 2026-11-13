#!/usr/bin/env python3
"""JSON stdin/stdout wrapper so the TypeScript RLM host can call clinical tools.

Two requests:

  {"describe": true}
      -> the tool specs (name, description, JSON-schema parameters). The
         TypeScript host builds its tool surface from this, so the model is
         shown each tool's real signature and a tool added in Python appears
         in the agent without a second list to keep in sync.

  {"name": ..., "args": {...}, "child_id": "..."}
      -> runs the tool. Arguments the caller left out are filled from the
         active child's own record where the record knows them (sex,
         gestational age, chronological age, id), so "what centile is 7.4 kg?"
         works for the child in front of us without the model re-typing --
         or mis-typing -- facts the profile already holds. Anything the
         caller did pass always wins.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant.tools.clinical import TOOL_SPECS, dispatch_tool  # noqa: E402

DAYS_PER_MONTH = 365.25 / 12


def _spec(name: str) -> dict:
    for spec in TOOL_SPECS:
        if spec.get("name") == name:
            return spec
    return {}


def _age_months(date_of_birth: str | None) -> float | None:
    if not date_of_birth:
        return None
    try:
        dob = datetime.strptime(str(date_of_birth).strip()[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    today = datetime.now(timezone.utc).date()
    if dob > today:
        return None
    return round((today - dob).days / DAYS_PER_MONTH, 2)


def fill_from_child(name: str, args: dict, child: dict | None) -> dict:
    """Default the tool's declared parameters from the child's record.

    Driven by the tool's own schema: only a parameter the spec declares is
    ever filled, so this needs no per-tool branch and a new tool picks it up
    by declaring the parameter.
    """
    if not child:
        return args
    props = ((_spec(name).get("parameters") or {}).get("properties") or {})
    known = {
        "child_id": child.get("child_id"),
        "sex": child.get("sex"),
        "gestational_age_weeks": child.get("gestational_age_weeks"),
        "age_months": _age_months(child.get("date_of_birth")),
    }
    out = dict(args)
    # An explicit age in weeks is a statement about which age to use; do not
    # put a second, different age beside it.
    gave_age = out.get("weeks") is not None or out.get("age_months") is not None
    for key, value in known.items():
        if key not in props or value is None or out.get(key) is not None:
            continue
        if key == "age_months" and gave_age:
            continue
        out[key] = value
    return out


def main() -> int:
    req = json.load(sys.stdin)
    if req.get("describe"):
        json.dump(TOOL_SPECS, sys.stdout, default=str)
        sys.stdout.write("\n")
        return 0
    name = str(req.get("name") or "")
    args = req.get("args") if isinstance(req.get("args"), dict) else {}
    db = None
    db_path = str(req.get("child_db") or os.environ.get("NESTLING_CHILD_DB") or "")
    if db_path:
        from assistant.memory.child_db import ChildMemoryDB

        db = ChildMemoryDB(path=Path(db_path))
    child_id = str(req.get("child_id") or "")
    child = None
    if db is not None and child_id:
        try:
            child = db.get_child(child_id)
        except Exception:
            child = None
    out = dispatch_tool(name, fill_from_child(name, args, child), db=db)
    json.dump(out, sys.stdout, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
