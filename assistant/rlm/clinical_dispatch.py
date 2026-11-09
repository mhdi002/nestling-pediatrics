#!/usr/bin/env python3
"""JSON stdin/stdout wrapper so the TypeScript RLM host can call clinical tools."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from assistant.tools.clinical import dispatch_tool  # noqa: E402


def main() -> int:
    req = json.load(sys.stdin)
    name = str(req.get("name") or "")
    args = req.get("args") if isinstance(req.get("args"), dict) else {}
    db = None
    db_path = os.environ.get("NESTLING_CHILD_DB") or ""
    if db_path:
        from assistant.memory.child_db import ChildMemoryDB

        db = ChildMemoryDB(path=Path(db_path))
    out = dispatch_tool(name, args, db=db)
    json.dump(out, sys.stdout, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
