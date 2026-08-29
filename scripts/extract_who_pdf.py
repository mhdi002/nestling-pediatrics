#!/usr/bin/env python3
"""
Extract a WHO publication into the curated-knowledge markdown format.

The curated pipeline (scripts/rebuild_curated_knowledge.py) splits
data/en/curated_*.md on `## ` headings, so this script's job is to turn a PDF
into exactly that shape: one `## ` per numbered section, with the body text
cleaned of page furniture.

Headings are detected by font size rather than by regex on the text, because
running heads and figure captions repeat the same numbering patterns as real
headings and would otherwise be picked up as sections.

Only use this on publications whose licence permits adaptation (for example
CC BY-NC-SA), and always record the source in the generated file's header.

Usage:
  scripts/extract_who_pdf.py INPUT.pdf OUTPUT.md \
      --title "..." --source "..." \
      [--heading-size 15] [--body-size 11] [--skip-sections 1,2]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - dependency is in requirements-core
    print("PyMuPDF (fitz) is required: pip install pymupdf", file=sys.stderr)
    raise SystemExit(1)

# Default heading shape: "7.1 Nipple pain", "3.10 Continue breastfeeding".
# Publications differ (some use letter-coded sections such as "K3 Cord care"),
# so the pattern is overridable with --heading-pattern.
DEFAULT_HEADING_PATTERN = r"^(\d+(?:\.\d+)+)\s+(.{3,120})$"
# Inline citation markers such as "(1,2)" or "(3–5)" carry no meaning once the
# reference list is dropped.
CITATION_RE = re.compile(r"\s*\((?:\d+(?:[–,-]\s*\d+)*)\)")


def _spans(page):
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if text:
                    yield round(span["size"], 1), text


def parse(
    doc,
    heading_size: float,
    body_size: float,
    heading_re: re.Pattern,
    tolerance: float = 0.6,
):
    """Yield (section_number, title, body_text) in document order."""
    sections: list[list] = []
    for page in doc:
        for size, text in _spans(page):
            if abs(size - heading_size) <= tolerance:
                m = heading_re.match(text)
                if m:
                    # Two groups = "id title"; one group = title only, in
                    # publications whose headings carry no section number.
                    if m.re.groups >= 2:
                        number, title = m.group(1), m.group(2).strip()
                    else:
                        number, title = str(len(sections) + 1), m.group(1).strip()
                    sections.append([number, title, []])
                    continue
            if abs(size - body_size) <= tolerance and sections:
                sections[-1][2].append(text)
    for number, title, parts in sections:
        yield number, title, " ".join(parts)


def clean(text: str) -> str:
    text = CITATION_RE.sub("", text)
    # Bullet glyphs arrive as separate spans; normalise to markdown list items.
    text = text.replace("•", "\n- ").replace("– ", "- ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s*-\s*", "\n- ", text)
    # Re-wrap into paragraphs at sentence boundaries for readable chunks.
    text = re.sub(r"(?<=[.!?]) (?=[A-Z])", " ", text)
    return text.strip()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("pdf", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--title", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--heading-size", type=float, default=15.0)
    ap.add_argument(
        "--heading-pattern",
        default=DEFAULT_HEADING_PATTERN,
        help="regex with two groups: section id and title",
    )
    ap.add_argument("--body-size", type=float, default=11.0)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument(
        "--skip-sections",
        default="",
        help="comma-separated top-level section numbers to omit (e.g. 1,2)",
    )
    args = ap.parse_args(argv)

    skip = {s.strip() for s in args.skip_sections.split(",") if s.strip()}
    doc = fitz.open(args.pdf)

    written = 0
    lines = [f"# {args.title}", "", args.source, ""]
    seen: set[str] = set()
    heading_re = re.compile(args.heading_pattern)
    for number, title, body in parse(doc, args.heading_size, args.body_size, heading_re):
        if re.split(r"[.\d]", number)[0] in skip or number.split(".")[0] in skip:
            continue
        body = clean(body)
        # Short stubs are usually figure captions or continuation fragments.
        if len(body) < args.min_chars:
            continue
        key = f"{number} {title}".lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"## {title}")
        lines.append(body)
        lines.append("")
        written += 1

    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"{args.pdf.name}: wrote {written} sections -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
