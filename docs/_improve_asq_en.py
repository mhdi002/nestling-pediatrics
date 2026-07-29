"""Re-translate ASQ English from Persian source; clean OCR junk."""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.runtime_translate import translate_fa_to_en

ASQ_DIR = ROOT / "data" / "en" / "asq"
JUNK = re.compile(r"[\uf000-\uf0ff]|☐|☑|□|■|\u00a0")


def clean(s: str) -> str:
    s = JUNK.sub(" ", s or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def improve_file(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    changed = 0

    def fix_q(q: dict) -> None:
        nonlocal changed
        fa = clean(q.get("text_fa") or q.get("text") or "")
        en = clean(q.get("text_en") or "")
        bad = (
            not en
            or "Total" in en
            or en.count("?") > 3
            or "childish game" in en.lower()
            or "Give him" in en
            or len(en) < 12
            or "\uf0a3" in (q.get("text_en") or "")
        )
        if fa and (bad or not q.get("text_en")):
            neu = clean(translate_fa_to_en(fa))
            # Soft post-edits for common awkward MT
            neu = neu.replace("childish game", "simple play action")
            neu = re.sub(r"\s+", " ", neu).strip()
            if neu and neu != en:
                q["text_en"] = neu
                changed += 1
        else:
            q["text_en"] = en
        if fa:
            q["text_fa"] = fa
        opts = q.get("options_en")
        if not opts or any(not o for o in opts):
            q["options_en"] = ["Yes", "Sometimes", "Not yet"]
            changed += 1

    domains = data.get("domains")
    if isinstance(domains, list):
        for d in domains:
            for q in d.get("questions") or []:
                fix_q(q)
                time.sleep(0.05)
    elif isinstance(domains, dict):
        for _id, d in domains.items():
            qs = d if isinstance(d, list) else (d.get("questions") or [])
            for q in qs:
                fix_q(q)
                time.sleep(0.05)

    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def main():
    total = 0
    for path in sorted(ASQ_DIR.glob("*.json")):
        n = improve_file(path)
        print(f"{path.name}: {n} updated")
        total += n
    print("TOTAL", total)


if __name__ == "__main__":
    main()
