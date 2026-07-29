import os, re, unicodedata, fitz
b = r"C:\Users\mhf\Desktop\pedriatrics"

def norm(s):
    return unicodedata.normalize("NFKC", s)

doc = fitz.open(os.path.join(b, "4m.pdf"))
pages = [norm(doc[i].get_text("text")) for i in range(doc.page_count)]
full = "\n".join(pages)
domains = [
    "برقراری ارتباط",
    "حرکات درشت",
    "حرکات ریز",
    "حل مسئله",
    "ناحیه خود",
]
opts = ["بله", "گاهی", "خیر"]
lines = []
lines.append("pages=%d chars=%d" % (doc.page_count, len(full)))
for kw in domains:
    lines.append("domain %r count=%d" % (kw, full.count(kw)))
for o in opts:
    lines.append("opt %r count=%d" % (o, full.count(o)))
# question patterns: Persian/Arabic/Western digit + dash
pat = re.compile(r"(?m)^[\s\u200c]*([\u0660-\u0669\u06f0-\u06f90-9]{1,2})\s*[-–\.]\s*")
all_nums = pat.findall(full)
lines.append("dash_numbered_items total=%d sample=%s" % (len(all_nums), all_nums[:20]))
for pi, t in enumerate(pages):
    nums = pat.findall(t)
    if nums:
        lines.append("page %d nums=%s" % (pi, nums))
# domain sections
for kw in domains:
    idx = full.find(kw)
    if idx < 0:
        lines.append("domain %r NOT FOUND" % kw)
        continue
    nxt = len(full)
    for other in domains:
        if other == kw:
            continue
        j = full.find(other, idx + len(kw))
        if j >= 0:
            nxt = min(nxt, j)
    chunk = full[idx:nxt]
    nums = pat.findall(chunk)
    lines.append("domain %r ~questions=%d nums=%s" % (kw, len(nums), nums))
# sample lines with options
for m in re.finditer(r".{0,40}(بله|گاهی|خیر).{0,40}", full):
    lines.append("opt_ctx: ...%s..." % m.group(0).replace("\n", " "))
    if len([x for x in lines if x.startswith("opt_ctx")]) >= 8:
        break
doc.close()
open(os.path.join(b, "_parser_metrics_nfkc.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines[:25]))
