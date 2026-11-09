import { readdir, readFile, stat } from "node:fs/promises";
import path from "node:path";

async function* walk(dir: string): AsyncGenerator<string> {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const e of entries) {
    const full = path.join(dir, e.name);
    if (e.isDirectory()) yield* walk(full);
    else if (e.isFile() && e.name.endsWith(".md")) yield full;
  }
}

export async function grepKnowledge(pattern: string, max = 25): Promise<string[]> {
  const root =
    process.env.NESTLING_KNOWLEDGE_DIR ||
    path.resolve(process.env.NESTLING_ROOT || path.resolve(import.meta.dirname, "../../../.."), "knowledge");
  let re: RegExp;
  try {
    re = new RegExp(pattern, "i");
  } catch {
    re = new RegExp(pattern.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "i");
  }
  const tokens = pattern
    .split("|")
    .map((t) => t.toLowerCase())
    .filter((t) => t.length >= 4);
  const scored: Array<{ sc: number; line: string }> = [];
  for await (const file of walk(root)) {
    const st = await stat(file);
    if (st.size > 400_000) continue;
    const rel = path.relative(root, file);
    const relLow = rel.toLowerCase();
    const text = await readFile(file, "utf8");
    const lines = text.split("\n");
    let fileHits = 0;
    let pathScore = 0;
    for (const tok of tokens) {
      if (relLow.includes(tok)) pathScore += 12;
    }
    for (let i = 0; i < lines.length; i++) {
      if (!re.test(lines[i]!)) continue;
      const line = `${rel}:${i + 1}: ${lines[i]!.trim().slice(0, 300)}`;
      scored.push({ sc: pathScore + (re.test(rel) ? 8 : 0), line });
      fileHits++;
      if (fileHits >= 8) break;
    }
  }
  scored.sort((a, b) => b.sc - a.sc);
  return scored.slice(0, max).map((row) => row.line);
}
