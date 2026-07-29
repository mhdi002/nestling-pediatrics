import os, unicodedata, fitz, re
b = r"C:\Users\mhf\Desktop\pedriatrics"

def norm(s):
    return unicodedata.normalize("NFKC", s)

doc = fitz.open(os.path.join(b, "4m.pdf"))
lines = []
for pi in range(doc.page_count):
    for bb in doc[pi].get_text("blocks"):
        t = norm(bb[4])
        if "حيطه" in t:
            lines.append("p%d y=%.0f x=%.0f | %s" % (pi, bb[1], bb[0], t.replace("\n"," ").strip()[:100].encode("unicode_escape").decode()))
full = norm("\n".join(doc[i].get_text("text") for i in range(doc.page_count)))
# questions per domain using حيطه headers
hdrs = []
for m in re.finditer(r"حيطه[^\n]{0,40}", full):
    hdrs.append(m.group(0).strip())
lines.append("headers found: %s" % hdrs)
# split by headers
for i, h in enumerate(hdrs):
    start = full.find(h)
    end = full.find(hdrs[i+1]) if i+1 < len(hdrs) else len(full)
    chunk = full[start:end]
    nums = re.findall(r"(?m)^[\s\u200c]*([\u0660-\u0669\u06f0-\u06f90-9]{1,2})\s*[-–]", chunk)
    lines.append("section %d %r q=%d nums=%s" % (i, h.encode("unicode_escape").decode(), len(nums), nums))
doc.close()
open(os.path.join(b, "_domain_sections.txt"), "w", encoding="utf-8").write("\n".join(lines))
