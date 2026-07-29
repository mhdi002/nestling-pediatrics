import os, unicodedata, fitz
b = r"C:\Users\mhf\Desktop\pedriatrics"

def norm(s):
    return unicodedata.normalize("NFKC", s)

doc = fitz.open(os.path.join(b, "4m.pdf"))
lines = []
for pi in range(doc.page_count):
    blocks = sorted(doc[pi].get_text("blocks"), key=lambda x: (x[1], x[0]))
    for bb in blocks:
        t = norm(bb[4].strip())
        if "حيط" in t or "حوز" in t or t.startswith("ح"):
            if len(t) < 80:
                lines.append("p%d y=%.0f x=%.0f %s" % (pi, bb[1], bb[0], t.replace("\n"," / ").encode("unicode_escape").decode()))
full = norm("\n".join(doc[i].get_text("text") for i in range(doc.page_count)))
for pat in ["حيطه", "حل مس", "شخص", "ریز", "ظریف", "خیر", "خير"]:
    lines.append("count %s=%d" % (pat.encode("unicode_escape").decode(), full.count(pat)))
doc.close()
open(os.path.join(b, "_probe_headers.txt"), "w", encoding="utf-8").write("\n".join(lines))
