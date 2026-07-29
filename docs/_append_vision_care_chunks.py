"""Append curated_pediatric_vision_care_en.md sections to knowledge chunks and rebuild RAG index."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from assistant.rag.stores import MedicalRAG

MD_PATH = ROOT / "data" / "en" / "curated_pediatric_vision_care_en.md"
CHUNKS_PATH = ROOT / "data" / "knowledge" / "chunks.json"
SOURCE = "curated_pediatric_vision_care_en.md"

# Map ## heading substring (case-insensitive match) -> chunk id
SECTION_IDS: list[tuple[str, str]] = [
    ("hand, foot, and mouth", "info_hfmd_parent_guide"),
    ("common infant rashes", "info_rash_common_infant"),
    ("skin redness, wounds, and blisters", "info_wound_first_aid_parent"),
    ("fever with rash", "info_fever_rash_red_flags"),
    ("photographs of your child's skin", "info_vision_photo_tips_clinician"),
    ("growth charts", "info_growth_parent_centiles"),
    ("speech and language milestones", "info_speech_milestones_0_24m"),
]


def parse_sections(md_text: str) -> list[tuple[str, str]]:
    """Return list of (title, body) for each ## section."""
    parts = re.split(r"\n(?=## )", md_text.strip())
    sections: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part.startswith("## "):
            continue
        lines = part.split("\n", 1)
        title = lines[0].lstrip("# ").strip()
        body = lines[1].strip() if len(lines) > 1 else ""
        sections.append((title, body))
    return sections


def match_id(title: str) -> str | None:
    low = title.lower()
    for needle, chunk_id in SECTION_IDS:
        if needle in low:
            return chunk_id
    return None


def main() -> None:
    md_text = MD_PATH.read_text(encoding="utf-8")
    sections = parse_sections(md_text)
    if len(sections) != len(SECTION_IDS):
        raise SystemExit(
            f"Expected {len(SECTION_IDS)} sections, parsed {len(sections)}: "
            + ", ".join(t for t, _ in sections)
        )

    chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    existing_ids = {c["id"] for c in chunks}
    added: list[str] = []

    for title, body in sections:
        chunk_id = match_id(title)
        if not chunk_id:
            raise SystemExit(f"No chunk id mapping for section: {title}")
        text = f"## {title}\n{body}"
        word_count = len(body.split())
        if word_count < 150 or word_count > 400:
            print(f"WARNING: {chunk_id} has {word_count} words (target 150-400)")

        new_chunk = {
            "id": chunk_id,
            "source": SOURCE,
            "collection": "medical",
            "title": title,
            "text": text,
        }

        if chunk_id in existing_ids:
            chunks = [new_chunk if c["id"] == chunk_id else c for c in chunks]
            print(f"Updated existing chunk: {chunk_id}")
        else:
            chunks.append(new_chunk)
            added.append(chunk_id)
            print(f"Added chunk: {chunk_id} ({word_count} words)")

    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    n = MedicalRAG().build_from_chunks()
    print(f"\nMedicalRAG index rebuilt: {n} total chunks in index")
    print(f"Chunks added this run: {len(added)}")
    print("Sample chunk ids:", ", ".join(added[:5] if added else [c[1] for c in SECTION_IDS[:5]]))


if __name__ == "__main__":
    main()
