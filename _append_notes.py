import os, re, unicodedata, fitz
b = r"C:\Users\mhf\Desktop\pedriatrics"

def norm(s):
    return unicodedata.normalize("NFKC", s)

doc = fitz.open(os.path.join(b, "4m.pdf"))
# per-domain q count using known header phrases in order
headers = [
    "\u062d\u064a\u0637\u0647 \u0628\u0631\u0642\u0631\u0627\u0631\u06cc \u0627\u0631\u062a\u0628\u0627\u0637",
    "\u062d\u064a\u0637\u0647 \u062d\u0631\u06a9\u0627\u062a \u062f\u0631\u0634\u062a",
    "\u062d\u064a\u0637\u0647 \u062d\u0631\u06a9\u0627\u062a \u0638\u0631\u06cc\u0641",
    "\u062d\u064a\u0637\u0647 \u062d\u0644 \u0645\u0633\u0626\u0644\u0647",
    "\u062d\u064a\u0637\u0647 \u0634\u062e\u0635\u06cc-\u0627\u062c\u062a\u0645\u0627\u0639\u06cc",
]
full = norm("\n".join(doc[i].get_text("text") for i in range(doc.page_count)))
pat = re.compile(r"(?m)^[\s\u200c]*([\u0660-\u0669\u06f0-\u06f90-9]{1,2})\s*[-–]\s*")
# page0 before first header
idx0 = full.find(headers[0])
pre = full[:idx0] if idx0 >= 0 else ""
pre_nums = pat.findall(pre)
lines = ["PARSER NOTES (auto)", "page0_preamble_numbered_items=%s" % pre_nums]
for i, h in enumerate(headers):
    start = full.find(h)
    if start < 0:
        lines.append("%d NOT FOUND" % i)
        continue
    end = len(full)
    for h2 in headers[i + 1 :]:
        j = full.find(h2, start + len(h))
        if j >= 0:
            end = min(end, j)
    nums = pat.findall(full[start:end])
    lines.append("domain_%d count=%d nums=%s" % (i, len(nums), nums))
doc.close()
append = "\n\n" + "=" * 72 + "\n" + "\n".join(lines) + "\n"
path = os.path.join(b, "_asq_sample_4m.txt")
open(path, "a", encoding="utf-8").write(append)
open(os.path.join(b, "_asq_domain_counts.txt"), "w", encoding="utf-8").write("\n".join(lines))
