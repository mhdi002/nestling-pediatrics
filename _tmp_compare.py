# -*- coding: utf-8 -*-
import sys, unicodedata, re
from pathlib import Path
import fitz

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load DOMAIN_PATTERNS from extract_texts by exec/import
import extract_texts as et

doc4 = fitz.open("4m.pdf")
doc10 = fitz.open("10m.pdf")

def all_blocks(doc):
    blocks = []
    for page in doc:
        for y,x,t in et.page_blocks(page):
            blocks.append((y,x,t))
    return blocks

b4 = all_blocks(doc4)
b10 = all_blocks(doc10)
doc4.close(); doc10.close()

print("=== DOMAIN_PATTERNS (repr) ===")
for pat, title, key in et.DOMAIN_PATTERNS:
    print(f"key={key} title={title!r} pattern={pat.pattern!r}")

print("\n=== Blocks matching any domain pattern in 4m ===")
for y,x,t in b4:
    d = et.detect_domain(t)
    if d:
        print(f"MATCH {d} y={y:.1f} text={t[:150]!r}")

print("\n=== Blocks matching any domain pattern in 10m ===")
for y,x,t in b10:
    d = et.detect_domain(t)
    if d:
        print(f"MATCH {d} y={y:.1f} text={t[:150]!r}")

# Search for حیطه / حيطه and related substrings
needles = [
    "حیطه",
    "حيطه",
    "حوزه",
    "ارتباطی",
    "حرکتی",
    "حل مسئله",
    "شخصی",
    "اطلاعات کلی",
    "کلی",
]
# also Presentation Forms
print("\n=== Needle search in 10m (codepoints) ===")
flat10 = "\n".join(t for _,_,t in b10)
flat4 = "\n".join(t for _,_,t in b4)

for needle in needles:
    c10 = flat10.count(needle)
    c4 = flat4.count(needle)
    cps = " ".join(f"U+{ord(c):04X}" for c in needle)
    print(f"{needle!r} ({cps}): 4m={c4} 10m={c10}")

# Find blocks containing Arabic Presentation Forms or similar headers
print("\n=== 10m blocks containing 'ح' and looking like headers ===")
for y,x,t in b10:
    if "ح" in t and len(t) < 80:
        # show codepoints of first 40 chars
        cps = " ".join(f"U+{ord(c):04X}" for c in t[:40])
        print(f"y={y:.1f} len={len(t)} text={t[:80]!r}")
        print(f"  cps: {cps}")

print("\n=== Compare unique short header-like lines ===")
def headerish(blocks):
    out=[]
    for y,x,t in blocks:
        first = t.split("\n")[0].strip()
        if len(first) < 60 and ("حیطه" in first or "حيطه" in first or "اطلاعات" in first or "حوزه" in first):
            out.append(first)
        # also check for presentation forms of heh/yeh
        if any(ord(c) >= 0xFB50 for c in first) and len(first)<80:
            out.append("PF:"+first)
    return out

print("4m headers:", headerish(b4))
print("10m headers:", headerish(b10))

# Dump first 200 chars of each block that looks like domain start for both
print("\n=== Lines with digit-start questions counts ===")
print("4m q-like:", sum(1 for _,_,t in b4 if re.match(r"^[0-9]{1,2}\s*[-–—]", t)))
print("10m q-like:", sum(1 for _,_,t in b10 if re.match(r"^[0-9]{1,2}\s*[-–—]", t)))
