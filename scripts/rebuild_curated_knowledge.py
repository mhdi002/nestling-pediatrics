#!/usr/bin/env python3
"""Merge all data/en/curated_*.md sections into chunks.json and rebuild medical RAG index."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.rag.stores import MedicalRAG

EN_DIR = ROOT / "data" / "en"
CHUNKS_PATH = ROOT / "data" / "knowledge" / "chunks.json"


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return s[:60] or "section"


def parse_sections(md_text: str) -> list[tuple[str, str]]:
    parts = re.split(r"\n(?=## )", md_text.strip())
    sections: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part.startswith("## "):
            continue
        lines = part.split("\n", 1)
        title = lines[0].lstrip("# ").strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        if len(body.split()) < 40:
            continue
        sections.append((title, body))
    return sections


def chunk_id_for(source_stem: str, title: str) -> str:
    # Stable ids for known high-value topics
    low = title.lower()
    overrides = {
        "newborn feeding": "info_feeding_newborn",
        "feeding 0–3 months": "info_feeding_0_3m",
        "feeding 0-3 months": "info_feeding_0_3m",
        "feeding 4–5 months": "info_feeding_4_5m",
        "starting solids at 6 months": "info_feeding_solids_6m",
        "feeding 7–9 months": "info_feeding_7_9m",
        "feeding 10–12 months": "info_feeding_10_12m",
        "feeding 12–24 months": "info_feeding_12_24m",
        "safe sleep for newborns": "info_sleep_safe_abc",
        "sleep 0–3 months": "info_sleep_0_3m",
        "sleep 4–7 months": "info_sleep_4_7m",
        "sleep 8–12 months": "info_sleep_8_12m",
        "sleep 12–24 months": "info_sleep_12_24m",
        "why iron matters": "info_iron_why",
        "iron for breastfed": "info_iron_breastfed",
        "iron for formula-fed": "info_iron_formula",
        "vitamin d": "info_vitamins_infant",
        "communication 0–6 months": "info_speech_0_6m",
        "communication 9–18 months": "info_speech_9_18m",
        "walking 9–18 months": "info_motor_walking_9_18m",
        "motor and play 6–12 months": "info_motor_play_6_12m",
    }
    for needle, cid in overrides.items():
        if needle in low:
            return cid
    return f"guidance_{source_stem}_{slugify(title)}"


def main() -> None:
    if not CHUNKS_PATH.exists():
        raise SystemExit(f"Missing {CHUNKS_PATH} — run assistant.translate first")

    chunks: list[dict] = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in chunks}
    added = 0
    updated = 0

    for md_path in sorted(EN_DIR.glob("curated_*.md")):
        stem = md_path.stem
        sections = parse_sections(md_path.read_text(encoding="utf-8"))
        print(f"{md_path.name}: {len(sections)} sections")
        for title, body in sections:
            cid = chunk_id_for(stem, title)
            chunk = {
                "id": cid,
                "source": md_path.name,
                "collection": "medical",
                "title": title,
                "text": f"## {title}\n{body}",
            }
            if cid in by_id:
                updated += 1
            else:
                added += 1
            by_id[cid] = chunk

    # Drop chunks that came from a curated_*.md which no longer exists.
    # Without this the merge is append-only: renaming or deleting a source
    # file leaves orphans behind that duplicate the replacement content and
    # compete with it in retrieval.
    present = {p.name for p in EN_DIR.glob("curated_*.md")}
    orphans = [
        cid
        for cid, c in by_id.items()
        if str(c.get("source", "")).startswith("curated_")
        and c.get("source") not in present
    ]
    for cid in orphans:
        by_id.pop(cid, None)
    if orphans:
        print(f"pruned {len(orphans)} orphaned chunk(s) from removed curated files")

    merged = list(by_id.values())
    CHUNKS_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    n = MedicalRAG().build_from_chunks()
    print(f"\nChunks: {len(chunks)} -> {len(merged)} (+{added} new, {updated} updated)")
    print(f"MedicalRAG index rebuilt: {n} chunks")


if __name__ == "__main__":
    main()
