#!/usr/bin/env python3
"""CLI for the parent assistant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.agent.orchestrator import ParentAssistant
from assistant.memory.child_db import ChildMemoryDB


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Pediatric parent assistant")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("create-child")
    c.add_argument("--name", required=True)
    c.add_argument("--sex", required=True, choices=["male", "female"])
    c.add_argument("--ga", type=float, default=None, help="gestational age weeks")

    g = sub.add_parser("growth")
    g.add_argument("--child", required=True)
    g.add_argument("--sex", required=True)
    g.add_argument("--measure", required=True, choices=["weight", "length", "head_circumference"])
    g.add_argument("--weeks", type=float, required=True)
    g.add_argument("--value", type=float, required=True)

    a = sub.add_parser("asq")
    a.add_argument("--child", required=True)
    a.add_argument("--age", type=int, required=True)
    a.add_argument("--answers-json", required=True, help="JSON file of domain_answers")

    m = sub.add_parser("mchat")
    m.add_argument("--child", required=True)
    m.add_argument("--answers-json", required=True)

    q = sub.add_parser("ask")
    q.add_argument("--query", required=True)
    q.add_argument("--child", default=None)
    q.add_argument("--medical", action="store_true")
    q.add_argument("--memory", action="store_true")

    args = p.parse_args()
    db = ChildMemoryDB()
    asst = ParentAssistant(db=db)

    if args.cmd == "create-child":
        cid = db.create_child(args.name, args.sex, gestational_age_weeks=args.ga)
        asst.refresh_child_index(cid)
        print(json.dumps({"child_id": cid}, indent=2))
    elif args.cmd == "growth":
        out = asst.record_growth_and_overlay(args.child, args.sex, args.measure, args.weeks, args.value)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "asq":
        answers = json.loads(Path(args.answers_json).read_text(encoding="utf-8"))
        out = asst.run_asq_session(args.child, args.age, answers)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "mchat":
        answers = json.loads(Path(args.answers_json).read_text(encoding="utf-8"))
        answers = {int(k): v for k, v in answers.items()}
        out = asst.run_mchat_session(args.child, answers)
        print(json.dumps(out, indent=2, default=str))
    elif args.cmd == "ask":
        if args.medical:
            print(json.dumps(asst.ask_medical(args.query), indent=2, default=str))
        elif args.memory:
            if not args.child:
                raise SystemExit("--child required for memory")
            print(json.dumps(asst.ask_child(args.child, args.query), indent=2, default=str))
        else:
            print(json.dumps(asst.handle(args.query, child_id=args.child), indent=2, default=str))


if __name__ == "__main__":
    main()
