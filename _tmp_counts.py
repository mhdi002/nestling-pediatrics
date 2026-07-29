import json
from pathlib import Path
d=json.loads(Path("extracted/extraction_summary.json").read_text(encoding="utf-8"))
print("=== ALL ASQ ages ===")
for a in d["asq"]:
    print(f"{a['file']}: domains={a['domains']} questions={a['questions']}")
