# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import extract_texts as et
for pat, title, key in et.DOMAIN_PATTERNS:
    print(f"=== {key} ===")
    print("pattern:", pat.pattern)
    print("cps:", " ".join(f"{c} U+{ord(c):04X}" for c in pat.pattern if not c.isascii() or c in "\\"))
    print("title cps:", " ".join(f"{c} U+{ord(c):04X}" for c in title))
