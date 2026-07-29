import os
b = r"C:\Users\mhf\Desktop\pedriatrics"
asq = open(os.path.join(b, "_asq_sample_4m.txt"), encoding="utf-8").read()
markers = ["FULL DOCUMENT", "HEURISTIC", "Numbered questions"]
out = []
for m in markers:
    i = asq.find(m)
    if i >= 0:
        out.append(asq[i : i + 3000])
        out.append("\n" + "=" * 40 + "\n")
open(os.path.join(b, "_asq_structure_only.txt"), "w", encoding="utf-8").write("\n".join(out))
# first page plain text block
p0 = asq.split("PAGE 0", 1)[1].split("PAGE 1", 1)[0][:4500]
open(os.path.join(b, "_asq_p0_excerpt.txt"), "w", encoding="utf-8").write(p0)
open(os.path.join(b, "_mchat_excerpt.txt"), "w", encoding="utf-8").write(
    open(os.path.join(b, "_mchat_sample.txt"), encoding="utf-8").read()[:4000]
)
print(len(asq))
