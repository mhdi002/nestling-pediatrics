import os, re, fitz
b = r"C:\Users\mhf\Desktop\pedriatrics"
doc = fitz.open(os.path.join(b, "4m.pdf"))
pages = [doc[i].get_text("text") for i in range(doc.page_count)]
full = "\n".join(pages)
domains = [
    "\u0628\u0631\u0642\u0631\u0627\u0631\u06cc \u0627\u0631\u062a\u0628\u0627\u0637",
    "\u062d\u0631\u0643\u0627\u062a \u062f\u0631\u0634\u062a",
    "\u062d\u0631\u0643\u0627\u062a \u0631\u06cc\u0632",
    "\u062d\u0644 \u0645\u0633\u0626\u0644\u0647",
    "\u0646\u0627\u062d\u06cc\u0647 \u062e\u0648\u062f",
]
lines = []
lines.append("ASQ 4m.pdf meta: pages=%d total_chars=%d" % (doc.page_count, len(full)))
lines.append("")
for pi, t in enumerate(pages):
    nums = re.findall(r"(?m)^\s*(\d{1,2})\.\s", t)
    lines.append("page %d: numbered_lines=%d nums=%s" % (pi, len(nums), nums))
lines.append("")
for kw in domains:
    lines.append("domain_count[%s]=%d" % (kw.encode("unicode_escape").decode(), full.count(kw)))
lines.append("")
for pi in range(min(2, doc.page_count)):
    bl = doc[pi].get_text("blocks")
    lines.append("page %d blocks=%d" % (pi, len(bl)))
    for i, bb in enumerate(bl[:15]):
        t = bb[4].replace("\n", " ").strip()[:100]
        lines.append("  b%d y=%.0f: %s" % (i, bb[1], t.encode("unicode_escape").decode()))
lines.append("")
# domain question heuristic
pos = 0
for kw in domains:
    idx = full.find(kw, pos)
    if idx < 0:
        continue
    nxt = len(full)
    for other in domains:
        if other == kw:
            continue
        j = full.find(other, idx + len(kw))
        if j >= 0:
            nxt = min(nxt, j)
    chunk = full[idx:nxt]
    nums = re.findall(r"(?m)^\s*(\d{1,2})\.\s", chunk)
    lines.append("domain_q[%s]=%s" % (kw.encode("unicode_escape").decode(), nums))
doc.close()
open(os.path.join(b, "_parser_metrics.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("wrote metrics")
