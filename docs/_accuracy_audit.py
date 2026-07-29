#!/usr/bin/env python3
"""Strict accuracy audit — prints results and writes docs/AUDIT_ACCURACY.json."""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import intergrowth_preterm_equations as ig
from assistant.config import EN_DIR
from assistant.tools.clinical import (
    MCHAT_REVERSE,
    score_asq_domain,
    score_asq_questionnaire,
    score_mchat,
)
from assistant.agent.orchestrator import ParentAssistant, classify_intent
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB

OUT_PATH = ROOT / "docs" / "AUDIT_ACCURACY.json"


def _check(name: str, ok: bool, detail: dict | None = None, error: str | None = None) -> dict:
    row = {"name": name, "ok": bool(ok)}
    if detail is not None:
        row["detail"] = detail
    if error:
        row["error"] = error
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {error}" if error else ""))
    if detail and not ok:
        print(f"       detail: {json.dumps(detail, ensure_ascii=False)[:400]}")
    return row


def audit_intergrowth() -> list[dict]:
    checks: list[dict] = []
    sexes = ("male", "female")
    measures = ("weight", "length", "head_circumference")
    weeks_list = (27, 40, 64)
    table = {}
    mono_ok = True
    mono_failures = []
    for sex in sexes:
        table[sex] = {}
        for meas in measures:
            table[sex][meas] = {}
            for w in weeks_list:
                p3 = ig.percentile(sex, meas, w, 3)
                p50 = ig.percentile(sex, meas, w, 50)
                p97 = ig.percentile(sex, meas, w, 97)
                table[sex][meas][str(w)] = {"p3": p3, "p50": p50, "p97": p97}
                if not (p3 < p50 < p97) or not all(math.isfinite(x) for x in (p3, p50, p97)):
                    mono_ok = False
                    mono_failures.append({"sex": sex, "measure": meas, "weeks": w, "p3": p3, "p50": p50, "p97": p97})

    checks.append(
        _check(
            "intergrowth_monotonic_p3_p50_p97",
            mono_ok,
            detail={"sample": {s: {m: table[s][m]["40"] for m in measures} for s in sexes}, "failures": mono_failures},
            error=None if mono_ok else f"{len(mono_failures)} non-monotonic cells",
        )
    )
    checks.append(
        _check(
            "intergrowth_full_grid",
            True,
            detail={"weeks": list(weeks_list), "sexes": list(sexes), "measures": list(measures), "values": table},
        )
    )

    c_m = ig.centile_from_measurement("male", "weight", 27, 0.99)
    checks.append(
        _check(
            "published_male_wt_27w_0.99kg",
            abs(c_m - 96.89) < 0.05,
            detail={"centile": c_m, "expected": 96.89, "tol": 0.05},
            error=None if abs(c_m - 96.89) < 0.05 else f"got {c_m}",
        )
    )
    c_f = ig.centile_from_measurement("female", "weight", 27, 0.91)
    checks.append(
        _check(
            "published_female_wt_27w_0.91kg",
            abs(c_f - 97.12) < 0.05,
            detail={"centile": c_f, "expected": 97.12, "tol": 0.05},
            error=None if abs(c_f - 97.12) < 0.05 else f"got {c_f}",
        )
    )
    z_f = ig.z_score("female", "length", 64, 64.68)
    checks.append(
        _check(
            "published_female_length_64w_64.68_z0",
            abs(z_f) < 0.05,
            detail={"z": z_f, "expected": 0.0, "tol": 0.05},
            error=None if abs(z_f) < 0.05 else f"got {z_f}",
        )
    )
    return checks


def audit_asq() -> list[dict]:
    checks: list[dict] = []
    yes = score_asq_domain(["yes"] * 6)
    checks.append(
        _check(
            "asq_6x_yes_equals_60",
            yes.get("ok") and yes.get("total") == 60 and yes.get("below_cutoff") is False,
            detail=yes,
            error=None if yes.get("total") == 60 else f"total={yes.get('total')}",
        )
    )
    ny = score_asq_domain(["not_yet"] * 6)
    checks.append(
        _check(
            "asq_6x_not_yet_0_below_cutoff",
            ny.get("ok") and ny.get("total") == 0 and ny.get("below_cutoff") is True,
            detail=ny,
            error=None if ny.get("total") == 0 and ny.get("below_cutoff") else str(ny),
        )
    )
    mixed = score_asq_questionnaire(
        {"communication": ["not_yet"] * 6, "gross_motor": ["yes"] * 6}
    )
    ok_mixed = (
        mixed.get("ok")
        and mixed.get("needs_referral") is True
        and "communication" in mixed.get("referral_domains", [])
        and "gross_motor" not in mixed.get("referral_domains", [])
    )
    checks.append(
        _check(
            "asq_mixed_referral",
            ok_mixed,
            detail=mixed,
            error=None if ok_mixed else "mixed referral mismatch",
        )
    )
    return checks


def audit_mchat() -> list[dict]:
    checks: list[dict] = []
    checks.append(
        _check(
            "mchat_reverse_items",
            MCHAT_REVERSE == {2, 5, 12},
            detail={"MCHAT_REVERSE": sorted(MCHAT_REVERSE)},
            error=None if MCHAT_REVERSE == {2, 5, 12} else f"got {MCHAT_REVERSE}",
        )
    )

    # low: all pass (yes on non-reverse, no on reverse)
    low_ans = {i: "yes" for i in range(1, 21)}
    for i in (2, 5, 12):
        low_ans[i] = "no"
    low = score_mchat(low_ans)
    checks.append(
        _check(
            "mchat_low_risk",
            low.get("ok") and low.get("total_failed") == 0 and low.get("risk") == "low",
            detail=low,
            error=None if low.get("risk") == "low" else str(low),
        )
    )

    # medium: only reverse fails (3 fails)
    med_ans = {i: "yes" for i in range(1, 21)}
    med = score_mchat(med_ans)
    checks.append(
        _check(
            "mchat_medium_risk",
            med.get("ok") and med.get("total_failed") == 3 and med.get("risk") == "medium",
            detail=med,
            error=None if med.get("risk") == "medium" else str(med),
        )
    )

    # high: all no → 17 non-reverse fails
    high_ans = {i: "no" for i in range(1, 21)}
    high = score_mchat(high_ans)
    checks.append(
        _check(
            "mchat_high_risk",
            high.get("ok") and high.get("total_failed") == 17 and high.get("risk") == "high",
            detail=high,
            error=None if high.get("risk") == "high" else str(high),
        )
    )
    return checks


def audit_intent_routing() -> list[dict]:
    checks: list[dict] = []
    asst = ParentAssistant(use_xlam=False, use_pleias=False)
    try:
        sid = asst.start_session()
        help_out = asst.chat(sid, "hi, how can you help me?")
        help_ok = (
            "help" in help_out.get("intents", [])
            and "medical" not in help_out.get("intents", [])
            and "medical_rag" not in help_out
            and help_out.get("tools", {}).get("tool_calls") == []
        )
        checks.append(
            _check(
                "intent_help_no_medical",
                help_ok,
                detail={"intents": help_out.get("intents"), "has_medical_rag": "medical_rag" in help_out},
                error=None if help_ok else "help routed medical or tools",
            )
        )

        iron = asst.chat(sid, "tell me about iron")
        cites = (iron.get("medical_rag") or {}).get("citations") or []
        cite_blob = json.dumps(cites, ensure_ascii=False).lower()
        iron_ok = (
            "medical" in iron.get("intents", [])
            and "medical_rag" in iron
            and "mchat_q" not in cite_blob
            and iron.get("tools", {}).get("tool_calls") == []
        )
        checks.append(
            _check(
                "intent_iron_medical_clean",
                iron_ok,
                detail={
                    "intents": iron.get("intents"),
                    "citation_ids": [c.get("id") for c in cites if isinstance(c, dict)],
                    "mchat_q_in_citations": "mchat_q" in cite_blob,
                },
                error=None if iron_ok else "iron medical unclean or missing RAG",
            )
        )
    finally:
        asst.close()

    # history no growth overlay (multi-turn covered below too; also classifier)
    hist_intents = classify_intent("what was my child's last growth result?")
    hist_ok = "history" in hist_intents and "growth" not in hist_intents
    checks.append(
        _check(
            "intent_history_no_growth",
            hist_ok,
            detail={"intents": sorted(hist_intents)},
            error=None if hist_ok else "history also classified as growth",
        )
    )
    return checks


def audit_translation() -> list[dict]:
    asq_dir = EN_DIR / "asq"
    marker = "[FA→EN pending"
    marker_ascii = "[FA->EN pending"
    pending = 0
    files = list(asq_dir.glob("*.json")) if asq_dir.exists() else []
    for path in files:
        text = path.read_text(encoding="utf-8")
        pending += text.count(marker) + text.count(marker_ascii)
    ok = pending == 0 and len(files) > 0
    return [
        _check(
            "translation_fa_en_pending_zero",
            ok,
            detail={"pending_count": pending, "en_asq_files": len(files), "path": str(asq_dir)},
            error=None if ok else f"pending={pending} files={len(files)}",
        )
    ]


def audit_parent_assistant_multiturn() -> list[dict]:
    checks: list[dict] = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        child_db = ChildMemoryDB(Path(td) / "c.db")
        chat_db = ChatMemory(Path(td) / "chat.db")
        asst = ParentAssistant(db=child_db, chat_memory=chat_db, use_xlam=False, use_pleias=False)
        try:
            cid = child_db.create_child("AuditBoy", "male", gestational_age_weeks=32)
            sid = asst.start_session(cid)

            t1 = asst.chat(sid, "boy", child_id=cid)
            t1_ok = t1.get("slots", {}).get("sex") == "male" and t1.get("tools", {}).get("tool_calls") == []
            checks.append(
                _check(
                    "multiturn_boy_slot",
                    t1_ok,
                    detail={"slots": t1.get("slots"), "intents": t1.get("intents")},
                    error=None if t1_ok else "boy slot failed",
                )
            )

            t2 = asst.chat(sid, "weight 40 weeks 3.2 kg overlay", child_id=cid)
            names2 = [c["name"] for c in t2.get("tools", {}).get("tool_calls", [])]
            centile = None
            for tc in t2.get("tools", {}).get("tool_calls", []):
                res = tc.get("result") or {}
                if res.get("centile") is not None:
                    centile = res["centile"]
            cent_ok = centile is not None and abs(centile - 30.8) <= 0.2
            overlay_ok = "overlay_growth_on_chart" in names2 and "growth" in t2.get("intents", [])
            checks.append(
                _check(
                    "multiturn_overlay_centile_30_8",
                    overlay_ok and cent_ok,
                    detail={"tools": names2, "centile": centile, "expected": 30.8, "tol": 0.2},
                    error=None if (overlay_ok and cent_ok) else f"tools={names2} centile={centile}",
                )
            )

            t3 = asst.chat(sid, "what was my child's last growth result?", child_id=cid)
            names3 = [c["name"] for c in t3.get("tools", {}).get("tool_calls", [])]
            hist_ok = (
                "history" in t3.get("intents", [])
                and "growth" not in t3.get("intents", [])
                and "overlay_growth_on_chart" not in names3
                and "growth_percentile" not in names3
            )
            checks.append(
                _check(
                    "multiturn_history_excludes_overlay",
                    hist_ok,
                    detail={"intents": t3.get("intents"), "tools": names3},
                    error=None if hist_ok else f"intents={t3.get('intents')} tools={names3}",
                )
            )
        finally:
            asst.close()
    return checks


def main() -> int:
    print("=== STRICT ACCURACY AUDIT ===")
    sections = {
        "intergrowth": audit_intergrowth(),
        "asq": audit_asq(),
        "mchat": audit_mchat(),
        "intent_routing": audit_intent_routing(),
        "translation": audit_translation(),
        "parent_assistant": audit_parent_assistant_multiturn(),
    }
    all_checks = [c for checks in sections.values() for c in checks]
    passed = sum(1 for c in all_checks if c["ok"])
    failed = [c["name"] for c in all_checks if not c["ok"]]
    report = {
        "ok": len(failed) == 0,
        "passed": passed,
        "failed": len(failed),
        "failed_names": failed,
        "total_checks": len(all_checks),
        "sections": sections,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("---")
    print(f"passed={passed}/{len(all_checks)} ok={report['ok']}")
    print(f"wrote {OUT_PATH}")
    if failed:
        print("FAILED:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
