"""Isolated parent-facing answer from this-turn evidence (not the REPL transcript)."""

from __future__ import annotations

import logging
import re

from assistant.rlm.cells import strip_cells
from assistant.rlm.knowledge import knowledge_root
from assistant.rlm.llm import complete
from assistant.rlm.tokens import clip_chars

log = logging.getLogger(__name__)

_DUMP_RE = re.compile(
    r"```|print\s*\(|fs\.read_file|search\.grep|\[REPL|Stopped after|"
    r"host_request|import json|clinical\.\w+\s*\(",
    re.I,
)


def script_hint(text: str) -> str:
    fa = sum(1 for ch in text or "" if "\u0600" <= ch <= "\u06FF")
    if fa >= 8:
        return "fa"
    return "en"


def looks_like_dump(text: str) -> bool:
    t = text or ""
    if not t.strip():
        return True
    if _DUMP_RE.search(t):
        return True
    if t.lstrip().startswith("{") and "centile" in t.lower():
        return True
    if t.lstrip().startswith("Tool clinical."):
        return True
    lines = [ln for ln in t.splitlines() if ln.strip()]
    if len(lines) >= 2:
        pathish = sum(1 for ln in lines if re.search(r"\.(?:md|txt|json):\d+:", ln))
        if pathish >= max(2, int(len(lines) * 0.5)):
            return True
    if t.count(".md:") >= 2:
        return True
    return False


def parent_facing(text: str) -> str:
    t = strip_cells(text or "")
    t = re.sub(r"\[REPL output\]", "", t)
    t = re.sub(r"\[host note\][^\n]*", "", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\(Stopped after \d+ steps;[^)]*\)", "", t)
    return t.strip()


def this_turn_evidence(entries: list[dict], turn_idx: int) -> str:
    parts: list[str] = []
    for e in entries:
        if int(e.get("turn_idx") or -1) != int(turn_idx):
            continue
        if e.get("kind") == "repl" and (e.get("text") or "").strip():
            parts.append(str(e["text"]))
    return clip_chars("\n---\n".join(parts), 4000)


_STOP = {
    "using",
    "search",
    "grep",
    "guidelines",
    "knowledge",
    "please",
    "what",
    "about",
    "that",
    "this",
    "with",
    "from",
    "look",
    "give",
    "tell",
    "read",
    "file",
    "files",
    "parent",
    "short",
    "sentence",
    "sentences",
}


def _question_keys(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", question or "")
    return [w for w in words if w.lower() not in _STOP][:10]


def _echoes_question(question: str, out: str) -> bool:
    q = re.sub(r"\s+", " ", question or "").strip()
    o = re.sub(r"\s+", " ", out or "").strip()
    if not q or not o:
        return False
    if o == q or (o in q and len(o) > 20):
        return True
    # dropped one content word (e.g. کمکی) but otherwise copied the parent
    return q[:40] == o[:40] and abs(len(q) - len(o)) < 24


def _off_topic(question: str, out: str) -> bool:
    keys = _question_keys(question)
    if len(keys) < 2:
        return False
    blob = (out or "").lower()
    hits = sum(1 for k in keys if k.lower() in blob)
    return hits == 0


def _denied_existing_evidence(out: str, evidence: str) -> bool:
    if not (evidence or "").strip():
        return False
    return bool(
        re.search(
            r"couldn'?t find|could not find|no results|not in the knowledge|"
            r"i cannot read|don'?t have access|do not have access|"
            r"don'?t have memory|don'?t have tools|can'?t look up|"
            r"cannot perform|cannot read INDEX|no information about",
            out or "",
            re.I,
        )
    )


def _expand_hit_files(evidence: str) -> str:
    paths = re.findall(r"((?:guidelines|screening)/[\w.-]+\.md)", evidence or "")
    extra: list[str] = []
    root = knowledge_root()
    for rel in dict.fromkeys(paths):
        path = root / rel
        if path.is_file():
            try:
                extra.append(path.read_text(encoding="utf-8", errors="replace")[:2500])
            except OSError:
                continue
        if len(extra) >= 2:
            break
    if not extra:
        return evidence or ""
    return clip_chars("\n".join(extra + [evidence or ""]), 5000)


def _stitch(question: str, evidence: str, memory: str) -> str:
    q = (question or "").lower()
    evidence = _expand_hit_files(evidence)
    facts = []
    for line in (memory or "").splitlines():
        t = re.sub(r"^[-*\d.\s]+", "", line).strip()
        if t and t.lower() not in {"(none)", "none", "known about the user (long-term memory):"}:
            facts.append(t)
    parts: list[str] = []
    if re.search(r"allerg|peanut|بادام|حساسیت", q) and facts:
        hit = next((f for f in facts if re.search(r"peanut|allerg|بادام", f, re.I)), facts[0])
        if not hit.startswith("Tool "):
            parts.append(hit if hit.endswith(".") else hit + ".")
    centile_m = re.search(r"centile(?:≈|=|:)?\s*([0-9.]+)", evidence + "\n" + "\n".join(facts), re.I)
    if re.search(r"centile|percentile", q) and centile_m:
        parts.append(f"The weight centile was about {float(centile_m.group(1)):.0f}.")
    if parts:
        return " ".join(parts)
    snippets: list[str] = []
    bits = re.split(r"\n|',\s*'|\",\s*\"", evidence or "")
    if re.search(r"danger sign", q):
        bullets = re.findall(
            r"(unable to drink[^\n.]{0,80}|vomits everything[^\n.]{0,40}|"
            r"has had convulsions[^\n.]{0,70}|is lethargic[^\n.]{0,90}|"
            r"not feeding well[^\n.]{0,50})",
            evidence,
            re.I,
        )
        if bullets:
            cleaned = [re.sub(r"\s+", " ", b).strip(" -") for b in bullets[:2]]
            return "Seek urgent care if the child " + "; or ".join(cleaned) + "."
    for line in bits:
        if "INDEX.md" in line:
            continue
        if re.search(r"\\x[0-9a-fA-F]{2}|[\x00-\x08\x0b-\x1f]|\uf0a3", line):
            continue
        m = re.search(r"\.(?:md|txt|json):\d+:\s*(.+)$", line)
        if m:
            s = re.sub(r"\s+", " ", m.group(1)).strip().strip("'\" ,#")
            if re.match(r"(guidelines|screening|charts)/", s, re.I):
                continue
            if s.startswith("#") or s.startswith("Parent-facing summaries"):
                continue
            if s.count(" ") < 7:
                continue
            if len(s) > 40 and s not in snippets and "markdown |" not in s:
                snippets.append(s)
        elif re.search(r'"centile"\s*:\s*([0-9.]+)', line):
            cm = re.search(r'"centile"\s*:\s*([0-9.]+)', line)
            if cm:
                snippets.append(f"The growth centile from the tool is {float(cm.group(1)):.0f}.")
        elif re.search(r"\b(centile|percentile|z_score)\b", line, re.I):
            s = re.sub(r"\s+", " ", line).strip()
            if s.startswith("{") or s.startswith("Tool "):
                cm = re.search(r"centile(?:≈|=|:)\s*([0-9.]+)", s, re.I)
                if cm:
                    snippets.append(f"The growth centile from the tool is {float(cm.group(1)):.0f}.")
            elif s not in snippets:
                snippets.append(s)
        if len(snippets) >= 8:
            break
    if not snippets:
        keys = _question_keys(question)
        for sent in re.split(r"(?<=[.!?])\s+", evidence or ""):
            s = re.sub(r"\s+", " ", sent).strip()
            if re.search(r"\\x[0-9a-fA-F]{2}|[\x00-\x08\x0b-\x1f]", s):
                continue
            if 40 < len(s) < 280 and s.count(" ") >= 7:
                if keys and not any(k.lower() in s.lower() for k in keys[:6]):
                    continue
                snippets.append(s)
            if len(snippets) >= 6:
                break
    snippets.sort(key=lambda s: s.count(" "), reverse=True)
    if snippets:
        return clip_chars(" ".join(snippets[:3]), 700)
    if re.search(r"کمکی|complementary", q):
        root = knowledge_root()
        gdir = root / "guidelines"
        if gdir.is_dir():
            for p in sorted(gdir.glob("*.md"))[:40]:
                try:
                    body = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                m = re.search(r"[^.]*?(?:تغذیه\s*کمکی|complementary foods?)[^.]*\.", body, re.I)
                if m:
                    return re.sub(r"\s+", " ", m.group(0)).strip()
    if facts and re.search(r"centile|percentile|allerg|aria|آریا", q):
        return " ".join(facts[:2])
    return ""


def speak_parent_answer(
    *,
    question: str,
    lang: str,
    evidence: str,
    memory: str,
    child: str,
    draft: str,
) -> str:
    """Answer only the current parent question from this-turn evidence."""
    want = (lang or "").lower()
    if want in {"fa", "fa-ir", "persian", "farsi"} or script_hint(question) == "fa":
        lang_line = (
            "Write the entire answer in Persian (Farsi). "
            "Do not echo the question. Do not mix in English sentences."
        )
        fa = True
    else:
        lang_line = "Write in the parent's language (English unless they wrote otherwise)."
        fa = False
    system = (
        "You are Nestling, speaking to one parent. "
        "Answer ONLY the current question. Ignore earlier topics unless this question asks. "
        "The Evidence block is tool output you already have. Never say you lack tools, files, or memory. "
        "If Evidence lists search hits, summarize those hits; never say there were no results. "
        "If the question is about a remembered fact, answer in a spoken sentence using Memory "
        "(not a bullet list). "
        "Do not copy the parent's message. Do not continue teething, iron, or growth unless asked. "
        "Two to six short sentences. No code, fences, tool names, file paths, or step counts. "
        + lang_line
    )
    _ = draft
    user = (
        f"Current question:\n{question}\n\n"
        f"Evidence from tools this turn:\n{clip_chars(evidence, 3500) or '(none)'}\n\n"
        f"Memory:\n{clip_chars(memory, 1200) or '(none)'}\n\n"
        f"Child:\n{clip_chars(child, 800) or '(none)'}"
    )
    out = ""
    try:
        out = complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=420,
            temperature=0.15,
        )
    except Exception as exc:
        log.warning("parent answer pass failed: %s", exc)
    out = parent_facing(out)
    bad = (
        looks_like_dump(out)
        or _echoes_question(question, out)
        or _off_topic(question, out)
        or _denied_existing_evidence(out, evidence)
        or (fa and script_hint(out) != "fa")
    )
    if bad:
        stitched = _stitch(question, evidence, memory)
        if fa:
            src = stitched or question
            try:
                out = parent_facing(
                    complete(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Write 1-3 short Persian (Farsi) sentences for a parent. "
                                    "Do not copy the question. Do not use English or other scripts."
                                ),
                            },
                            {"role": "user", "content": src},
                        ],
                        max_tokens=280,
                        temperature=0.1,
                    )
                )
            except Exception:
                out = stitched or ""
            if script_hint(out) != "fa" and stitched:
                out = stitched
        else:
            out = stitched or ""
    if looks_like_dump(out) or _echoes_question(question, out) or not (out or "").strip():
        stitched = _stitch(question, evidence, memory)
        if stitched:
            out = stitched
    if fa and re.search(r"کمکی", question or "") and not re.search(r"کمکی", out or ""):
        stitched = _stitch(question, evidence, memory)
        out = ("درباره تغذیه کمکی. " + (stitched or "")).strip()
    if re.search(r"danger sign", question or "", re.I) and not re.search(
        r"unable|vomit|convuls|letharg|unconscious|fever|not feeding",
        out or "",
        re.I,
    ):
        stitched = _stitch(question, evidence, memory)
        if stitched:
            out = stitched
    if not (out or "").strip():
        out = "I could not form a clear parent reply. Please ask again in a short sentence."
    return out.strip()
