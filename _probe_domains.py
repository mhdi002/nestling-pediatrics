import os, re, unicodedata, fitz
b = r"C:\Users\mhf\Desktop\pedriatrics"

def norm(s):
    return unicodedata.normalize("NFKC", s)

doc = fitz.open(os.path.join(b, "4m.pdf"))
full = norm("\n".join(doc[i].get_text("text") for i in range(doc.page_count)))
lines = []
keys = ["حرک", "ریز", "درشت", "برقرار", "مسئله", "مساله", "ناحیه", "شخص", "خود", "بله", "خیر", "نه"]
for k in keys:
    lines.append("%s count=%d" % (k.encode("unicode_escape").decode(), full.count(k)))
page = doc[1]
blocks = sorted(page.get_text("blocks"), key=lambda x: (x[1], x[0]))
lines.append("header-like blocks page1:")
for bb in blocks:
    t = norm(bb[4].strip())
    if any(k in t for k in ["برقرار", "درشت", "ریز", "مس", "ناح", "شخص"]):
        lines.append("y=%.0f x=%.0f | %s" % (bb[1], bb[0], t.encode("unicode_escape").decode()[:120]))
# unique short lines that look like section titles (len<40, contains one domain word)
cands = set()
for bb in blocks:
    t = norm(bb[4].strip().replace("\n", " "))
    if 5 < len(t) < 45 and any(w in t for w in ["ارتباط", "درشت", "ریز", "مس", "شخص", "خود"]):
        cands.add(t)
lines.append("title candidates page1:")
for t in sorted(cands):
    lines.append(t.encode("unicode_escape").decode())
doc.close()
open(os.path.join(b, "_probe_domains.txt"), "w", encoding="utf-8").write("\n".join(lines))
