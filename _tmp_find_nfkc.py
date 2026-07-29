# -*- coding: utf-8 -*-
from pathlib import Path

path = Path("extract_texts.py")
text = path.read_text(encoding="utf-8")

old_nfkc = '''def nfkc(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.translate(ARABIC_DIGITS)
    text = text.replace("\\u200c", " ").replace("\\xa0", " ")
    text = re.sub(r"[ \\t]+", " ", text)
    return text.strip()'''

# The file uses actual escape sequences as characters in source - let me find the function properly
import re
m = re.search(r"def nfkc\(text: str\) -> str:\n(?:    .*\n)+?    return text\.strip\(\)\n", text)
if not m:
    # try CRLF
    m = re.search(r"def nfkc\(text: str\) -> str:\r?\n(?:    .*\r?\n)+?    return text\.strip\(\)\r?\n", text)
print("found nfkc:", bool(m))
if m:
    print(repr(m.group(0)))
