# -*- coding: utf-8 -*-
"""Show exact domain header lines from debug dump and sample question blocks."""
from pathlib import Path
t = Path("_debug_10m_blocks.txt").read_text(encoding="utf-8")
for line in t.splitlines():
    if "حیطه" in line or "حيطه" in line or "موارد کلی" in line or "برقراری" in line or "حرکات" in line:
        print(line)
print("--- q samples ---")
for line in t.splitlines():
    if "y=" in line and ("1-" in line or "1\n" in line or "| 1" in line or "1- " in line or "| 4" in line):
        if any(x in line for x in ["1-", "2-", "3-", "4-", "5-", "6-", "| 1", "| 4", "| 5", "| 6"]):
            print(line[:160])
