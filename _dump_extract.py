#!/usr/bin/env python3
"""Extract structured text from pediatrics PDFs and info.txt."""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import fitz

# Avoid UnicodeEncodeError on Windows consoles (cp1252) when printing Persian paths.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "extracted"

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

DOMAIN_PATTERNS = [
    (re.compile(r"حيطه\s*برقراری\s*ارتباط"), "برقراری ارتباط", "communication"),
    (re.compile(r"حيطه\s*حرکات\s*درشت"), "حرکات درشت", "gross_motor"),
    (re.compile(r"حيطه\s*حرکات\s*ظریف"), "حرکات ظریف", "fine_motor"),
    (re.compile(r"حيطه\s*حل\s*مسئله"), "حل مسئله", "problem_solving"),
    (re.compile(r"حيطه\s*شخصی\s*[-–]?\s*اجتماعی"), "شخصی-اجتماعی", "personal_social"),
    (re.compile(r"موارد\s*کلی"), "موارد کلی", "overall"),
]

QUESTION_RE = re.compile(
    r"^([0-9]{1,2})\s*[-–]\s*(.+)$",
    re.DOTALL,
)

HEADER_FIELD_KEYS = [
    ("نام کودک", "child_name"),
    ("نام خانوادگی کودک", "family_name"),
    ("تاریخ تولد", "date_of_birth"),
    ("سن اصلاح شده", "corrected_age"),
    ("تاریخ تکمیل", "completion_date"),
    ("نام تکمیل کننده", "completer_name"),
    ("نسبت تکمیل کننده با کودک", "completer_relation"),
    ("تلفن تماس تکمیل کننده", "completer_phone"),
    ("جنس", "sex"),
    ("استان", "province"),
    ("شهرستان", "county"),
    ("روستا", "village"),
    ("نام مرکز", "center_name"),
    ("نام پرسشگر", "interviewer"),
    ("شماره تلفن مرکز", "center_phone"),
]


def nfkc(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(ARABIC_DIGITS)
    # Unify Persian Yeh (U+06CC) with Arabic Yeh (U+064A) so domain regexes match all ASQ PDFs.
    text = text.replace("\u06cc", "\u064a").replace("\u06a9", "\u0643")
    text = text.replace("\u200c", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def page_blocks(page: fitz.Page) -> list[tuple[float, float, str]]:
    raw = page.get_text("blocks")
    items = []
    for b in raw:
        if len(b) < 5:
            continue
        x0, y0, _x1, _y1, text, *_rest = b
        t = nfkc(text)
        if t:
            items.append((y0, x0, t))
    items.sort(key=lambda it: (round(it[0], 1), round(it[1], 1)))
    return items


def detect_domain(text: str):
    for pattern, title_fa, key in DOMAIN_PATTERNS:
        if pattern.search(text):
            return title_fa, key
    return None


def clean_question_text(text: str) -> str:
    text = re.sub(r"[.…\-–—_]{2,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" :،,")
    return text


def parse_asq(path: Path) -> dict:
    doc = fitz.open(path)
    age_m = int(re.match(r"(\d+)m\.pdf$", path.name, re.I).group(1))

    all_blocks: list[str] = []
    for page in doc:
        for _y, _x, text in page_blocks(page):
            all_blocks.append(text)
    doc.close()

    header_fields = {k: "" for _, k in HEADER_FIELD_KEYS}
    flat = "\n".join(all_blocks)
    for label_fa, key in HEADER_FIELD_KEYS:
        m = re.search(rf"{re.escape(label_fa)}\s*[:：]?\s*([^\n]*)", flat)
        if m:
            header_fields[key] = m.group(1).strip(" .-/")

    instructions = []
    domains: dict[str, dict] = {}
    current_key = None
    pending_q = None

    def flush_pending():
        nonlocal pending_q
        if pending_q and current_key and pending_q["text"]:
            domains[current_key]["questions"].append(pending_q)
        pending_q = None

    for text in all_blocks:
        dom = detect_domain(text)
        if dom:
            flush_pending()
            title_fa, key = dom
            current_key = key
            domains.setdefault(
                key,
                {
                    "id": key,
                    "title_fa": title_fa,
                    "instruction": "",
                    "answer_options": (
                        ["بله", "خیر"] if key == "overall" else ["بله", "گاهی", "هنوز نه"]
                    ),
                    "questions": [],
                },
            )
            # instruction often follows header on same block
            rest = re.split(r"حيطه\s*[^\n]+|موارد\s*کلی", text, maxsplit=1)
            if len(rest) > 1 and rest[-1].strip():
                domains[key]["instruction"] = clean_question_text(rest[-1])
            continue

        if current_key is None:
            if re.match(r"^[0-9]{1,2}\s*[-–]", text) or "پرسشنامه" in text or "ASQ" in text.upper():
                if "ASQ" in text.upper() or "پرسشنامه" in text or text.startswith(
                    tuple("123456789")
                ):
                    if not re.match(r"^[0-9]{1,2}\s*[-–]\s*آیا", text):
                        instructions.append(text)
            continue

        # skip totals / option headers alone
        if re.fullmatch(r"(بله|گاهی|هنوز\s*نه|هنوزنه|خیر|خير|جمع\s*کل.*)", text):
            continue
        if text.startswith("جمع کل"):
            flush_pending()
            continue

        m = QUESTION_RE.match(text)
        if m:
            flush_pending()
            qnum = int(m.group(1))
            qtext = clean_question_text(m.group(2))
            opts = domains[current_key]["answer_options"]
            pending_q = {
                "id": qnum,
                "text": qtext,
                "options": list(opts),
                "answer": None,
            }
            continue

        # continuation of previous question
        if pending_q is not None:
            cont = clean_question_text(text)
            if cont and not cont.startswith("پیش از پاسخ"):
                pending_q["text"] = clean_question_text(pending_q["text"] + " " + cont)

    flush_pending()

    # ordered domains
    order = [
        "communication",
        "gross_motor",
        "fine_motor",
        "problem_solving",
        "personal_social",
        "overall",
    ]
    domain_list = [domains[k] for k in order if k in domains]

    return {
        "source": path.name,
        "type": "ASQ",
        "age_months": age_m,
        "title_fa": "پرسشنامه سنین و مراحل A.S.Q",
        "header_fields": header_fields,
        "instructions": instructions[:20],
        "domains": domain_list,
    }


def asq_to_markdown(data: dict) -> str:
    lines = [
        f"# {data['title_fa']} — {data['age_months']} ماهگی",
        "",
        f"منبع: `{data['source']}`",
        "",
        "## اطلاعات کودک / تکمیل‌کننده",
        "",
    ]
    labels = {v: k for k, v in HEADER_FIELD_KEYS}
    for key, val in data["header_fields"].items():
        lines.append(f"- **{labels.get(key, key)}**: {val or '…………'}")

    if data.get("instructions"):
        lines += ["", "## راهنما", ""]
        for i, instr in enumerate(data["instructions"], 1):
            lines.append(f"{i}. {instr}")

    for dom in data["domains"]:
        lines += ["", f"## {dom['title_fa']}", ""]
        if dom.get("instruction"):
            lines.append(f"_{dom['instruction']}_")
            lines.append("")
        opts = " / ".join(f"[ ] {o}" for o in dom["answer_options"])
        lines.append(f"گزینه‌ها: {opts}")
        lines.append("")
        for q in dom["questions"]:
            lines.append(f"### سؤال {q['id']}")
            lines.append("")
            lines.append(q["text"])
            lines.append("")
            lines.append("پاسخ: " + " · ".join(f"[ ] {o}" for o in q["options"]))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_mchat(path: Path) -> dict:
    doc = fitz.open(path)
    blocks: list[str] = []
    for page in doc:
        for _y, _x, text in page_blocks(page):
            blocks.append(text)
    doc.close()

    flat_lines = []
    for b in blocks:
        flat_lines.extend(b.splitlines())
    flat_lines = [nfkc(x) for x in flat_lines if nfkc(x)]

    title = "چک‌لیست اصلاح‌شده ارزیابی اوتیسم در کودکان (M-CHAT-R)"
    instructions = []
    questions = []
    buf = []
    current_num = None

    def flush():
        nonlocal current_num, buf
        if current_num is None:
            return
        text = clean_question_text(" ".join(buf))
        text = re.sub(r"\s*(آری|آری\s+خیر|خیر)\s*$", "", text).strip()
        questions.append(
            {
                "id": current_num,
                "text": text,
                "options": ["آری", "خیر"],
                "answer": None,
            }
        )
        current_num = None
        buf = []

    joined = "\n".join(flat_lines)
    # Prefer digit-hyphen items; M-CHAT often splits number and dash across lines
    chunk = re.split(r"(?m)(?=^[0-9]{1,2}\s*[-–])", joined)
    preamble = chunk[0] if chunk else ""
    for line in preamble.splitlines():
        if line.strip():
            instructions.append(line.strip())

    item_re = re.compile(r"^([0-9]{1,2})\s*[-–]\s*(.*)$", re.S)
    for part in chunk[1:]:
        m = item_re.match(part.strip())
        if not m:
            continue
        qnum = int(m.group(1))
        body = clean_question_text(m.group(2))
        body = re.sub(r"\s*(آری\s*)?خیر\s*$", "", body).strip()
        body = re.sub(r"\s*آری\s*$", "", body).strip()
        questions.append(
            {
                "id": qnum,
                "text": body,
                "options": ["آری", "خیر"],
                "answer": None,
            }
        )

    # Deduplicate by id keeping first substantial text
    by_id = {}
    for q in questions:
        prev = by_id.get(q["id"])
        if prev is None or len(q["text"]) > len(prev["text"]):
            by_id[q["id"]] = q
    questions = [by_id[k] for k in sorted(by_id)]

    return {
        "source": path.name,
        "type": "M-CHAT-R",
        "title_fa": title,
        "instructions": instructions[:30],
        "answer_options": ["آری", "خیر"],
        "questions": questions,
    }


def mchat_to_markdown(data: dict) -> str:
    lines = [
        f"# {data['title_fa']}",
        "",
        f"منبع: `{data['source']}`",
        "",
        "## راهنما",
        "",
    ]
    for instr in data["instructions"]:
        lines.append(instr)
        lines.append("")
    lines += ["## سؤالات", ""]
    for q in data["questions"]:
        lines.append(f"### {q['id']}")
        lines.append("")
        lines.append(q["text"])
        lines.append("")
        lines.append("پاسخ: [ ] آری · [ ] خیر")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def extract_chart_labels(path: Path) -> dict:
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        text = nfkc(page.get_text("text"))
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return {
        "source": path.name,
        "type": "INTERGROWTH-21st_chart",
        "note": "Curves are graphical; mathematical model is in intergrowth_preterm_equations.py",
        "pages": pages,
    }


def extract_booklet_ocr(path: Path) -> dict:
    """Best-effort text extraction; scanned pages may yield little text."""
    doc = fitz.open(path)
    pages = []
    ocr_note = "PyMuPDF text layer only (no Tesseract OCR installed by default)."
    try:
        import pytesseract  # noqa: F401
        from PIL import Image
        import io

        has_ocr = True
    except Exception:
        has_ocr = False

    for i, page in enumerate(doc):
        text = nfkc(page.get_text("text"))
        ocr_text = ""
        if has_ocr and len(text) < 200:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text = nfkc(pytesseract.image_to_string(img, lang="fas+eng"))
            except Exception as exc:
                ocr_text = f"[OCR failed: {exc}]"
        pages.append(
            {
                "page": i + 1,
                "text": text,
                "ocr_text": ocr_text,
                "char_count": len(text),
            }
        )
    doc.close()
    return {
        "source": path.name,
        "type": "iranian_growth_booklet_scan",
        "ocr_available": has_ocr,
        "note": ocr_note if not has_ocr else "OCR attempted for sparse pages.",
        "pages": pages,
    }


def write_info_md():
    src = ROOT / "info.txt"
    text = src.read_text(encoding="utf-8")
    out = OUT / "info.md"
    out.write_text("# رشد کودک — یادداشت‌ها\n\n" + text, encoding="utf-8")
    return out


def main():
    (OUT / "asq").mkdir(parents=True, exist_ok=True)
    (OUT / "screens").mkdir(parents=True, exist_ok=True)
    (OUT / "charts").mkdir(parents=True, exist_ok=True)
    (OUT / "booklets").mkdir(parents=True, exist_ok=True)

    summary = {"asq": [], "screens": [], "charts": [], "booklets": [], "info": None}

    for pdf in sorted(ROOT.glob("*m.pdf"), key=lambda p: int(re.match(r"(\d+)", p.name).group(1))):
        print(f"ASQ: {pdf.name}")
        data = parse_asq(pdf)
        stem = pdf.stem
        (OUT / "asq" / f"{stem}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT / "asq" / f"{stem}.md").write_text(asq_to_markdown(data), encoding="utf-8")
        nq = sum(len(d["questions"]) for d in data["domains"])
        summary["asq"].append({"file": pdf.name, "domains": len(data["domains"]), "questions": nq})

    mchat = ROOT / "غربالگری-اوتیسم.pdf"
    if mchat.exists():
        print(f"M-CHAT: {mchat.name}")
        data = parse_mchat(mchat)
        (OUT / "screens" / "mchat-r.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (OUT / "screens" / "mchat-r.md").write_text(mchat_to_markdown(data), encoding="utf-8")
        summary["screens"].append({"file": mchat.name, "questions": len(data["questions"])})

    for pdf in sorted(ROOT.glob("نمودار*.pdf")):
        print(f"Chart: {pdf.name}")
        data = extract_chart_labels(pdf)
        safe = re.sub(r"[^\w\-]+", "_", pdf.stem)[:80]
        (OUT / "charts" / f"{safe}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        md = f"# {pdf.name}\n\n" + "\n\n".join(
            f"## صفحه {p['page']}\n\n```\n{p['text']}\n```" for p in data["pages"]
        )
        (OUT / "charts" / f"{safe}.md").write_text(md, encoding="utf-8")
        summary["charts"].append({"file": pdf.name})

    for pdf in [ROOT / "chart-boy-95.pdf", ROOT / "chart-girl-95.pdf"]:
        if not pdf.exists():
            continue
        print(f"Booklet: {pdf.name}")
        data = extract_booklet_ocr(pdf)
        (OUT / "booklets" / f"{pdf.stem}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        md_parts = [f"# {pdf.name}", "", data["note"], ""]
        for p in data["pages"]:
            md_parts += [f"## صفحه {p['page']}", "", "### Text layer", "", "```", p["text"] or "(empty)", "```", ""]
            if p.get("ocr_text"):
                md_parts += ["### OCR", "", "```", p["ocr_text"], "```", ""]
        (OUT / "booklets" / f"{pdf.stem}.md").write_text("\n".join(md_parts), encoding="utf-8")
        summary["booklets"].append({"file": pdf.name, "pages": len(data["pages"])})

    info_out = write_info_md()
    summary["info"] = str(info_out.name)
    (OUT / "extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
