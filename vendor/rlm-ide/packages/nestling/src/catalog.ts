import { readdirSync, readFileSync, statSync } from "node:fs";
import path from "node:path";

/** The care library as the agent sees it up front: every readable file with what it covers.
 *
 * The files are the agent's skills. It cannot open the right one if all it is told is that files
 * exist, and a small model does not go exploring on its own: asked about iron with only a table of
 * paths and byte sizes to go on, it called an unrelated tool and then said it had no information.
 * So the catalog is built from the files themselves, their title and section headings, and sized
 * to a share of the context window. Nothing here names a topic; adding, renaming or removing a
 * file changes the catalog on the next turn. */

const TEXT = new Set([".md", ".txt"]);
/** A markdown heading, parsed by hand: a regex with a lazy group and a trailing `\s*#*\s*$` backtracks
 * quadratically on the long single-line paragraphs these books contain and never returned. */
function heading(line: string): { level: number; text: string } | null {
  if (line.charCodeAt(0) !== 35 || line.length > 300) return null;
  let level = 0;
  while (level < line.length && line[level] === "#") level++;
  if (level > 3 || line[level] !== " ") return null;
  const text = line.slice(level).trim();
  return text ? { level, text } : null;
}

interface Doc {
  rel: string;
  dir: string;
  title: string;
  sections: string[];
}

function walk(root: string, dir = ""): string[] {
  const out: string[] = [];
  let names: string[];
  try {
    names = readdirSync(path.join(root, dir)).sort();
  } catch {
    return out;
  }
  for (const name of names) {
    if (name.startsWith(".")) continue;
    const rel = dir ? `${dir}/${name}` : name;
    let st;
    try {
      st = statSync(path.join(root, rel));
    } catch {
      continue;
    }
    if (st.isDirectory()) out.push(...walk(root, rel));
    else out.push(rel);
  }
  return out;
}

function clip(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, Math.max(0, n - 1)).trimEnd()}…`;
}

function readDoc(root: string, rel: string, headBytes: number): Doc {
  let text = "";
  try {
    text = readFileSync(path.join(root, rel), "utf8").slice(0, headBytes);
  } catch {
    /* unreadable: listed by name only */
  }
  let title = "";
  const sections: string[] = [];
  const seen = new Set<string>();
  for (const line of text.split(/\r?\n/)) {
    const m = heading(line);
    if (!m) continue;
    const h = m.text;
    if (!title && m.level === 1) {
      title = h;
      continue;
    }
    const key = h.toLowerCase();
    if (m.level === 2 && !seen.has(key)) {
      seen.add(key);
      sections.push(h);
    }
  }
  return { rel, dir: rel.includes("/") ? rel.slice(0, rel.lastIndexOf("/")) : "", title: title || path.basename(rel), sections };
}

/** Every file under `root`, relative, forward-slashed. */
export function listFiles(root: string): string[] {
  return walk(root);
}

function grams(s: string): Set<string> {
  const t = s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  const out = new Set<string>();
  for (const w of t.split(" ")) for (let i = 0; i + 3 <= w.length; i++) out.add(w.slice(i, i + 3));
  return out;
}

/** The files whose paths look most like `wanted`, best first. For "no such file" errors: a model
 * that mistypes a path is told which real paths are near it instead of being left to guess again. */
export function nearestFiles(root: string, wanted: string, limit = 5): string[] {
  const w = grams(wanted);
  if (!w.size) return [];
  return walk(root)
    .map((f) => {
      const g = grams(f);
      let hit = 0;
      for (const x of w) if (g.has(x)) hit++;
      return { f, score: hit / Math.sqrt(g.size || 1) };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, limit)
    .map((x) => x.f);
}

let cache: { key: string; text: string } | null = null;

/** `budgetChars` is the most the catalog may occupy; the caller derives it from the context window. */
export function buildCatalog(root: string, budgetChars: number): string {
  const files = walk(root).filter((f) => f.toUpperCase() !== "INDEX.MD");
  const key = `${root}|${budgetChars}|${files
    .map((f) => {
      try {
        const st = statSync(path.join(root, f));
        return `${f}:${st.size}:${Math.floor(st.mtimeMs)}`;
      } catch {
        return f;
      }
    })
    .join("|")}`;
  if (cache?.key === key) return cache.text;

  const docs = files.filter((f) => TEXT.has(path.extname(f).toLowerCase())).map((f) => readDoc(root, f, 400_000));
  const other = new Map<string, number>();
  for (const f of files) {
    const ext = path.extname(f).toLowerCase();
    if (TEXT.has(ext)) continue;
    const dir = f.includes("/") ? f.slice(0, f.lastIndexOf("/")) : "";
    other.set(`${dir}|${ext || "(none)"}`, (other.get(`${dir}|${ext || "(none)"}`) ?? 0) + 1);
  }
  const dirs = [...new Set(docs.map((d) => d.dir))].sort();
  // Most of the budget names the files; the rest says what is inside them. When the titles do
  // not all fit, the folders whose files are most alike (the same form at different ages, the same
  // chart for different groups) are named once as a folder, most uniform first, so the files that
  // differ from each other keep a line each.
  const half = Math.floor(budgetChars * 0.6);
  const titled = (d: Doc) => `- ${d.rel} — ${clip(d.title, 70)}`;
  const alike = (dir: string) => {
    const mine = docs.filter((d) => d.dir === dir);
    const shapes = new Set(mine.map((d) => d.sections.join("|").replace(/\d+/g, "#")));
    return mine.length < 2 ? 0 : 1 - shapes.size / mine.length;
  };
  const folded = new Set<string>();
  const foldedLine = (dir: string) => {
    const mine = docs.filter((d) => d.dir === dir);
    return `- ${dir || "."}/ — ${mine.length} files like "${clip(mine[0]!.title, 60)}": ${mine.map((d) => path.basename(d.rel)).join(", ")}`;
  };
  const cost = () =>
    dirs.reduce((n, dir) => n + (folded.has(dir) ? Math.min(foldedLine(dir).length, Math.floor(half / Math.max(2, dirs.length))) : docs.filter((d) => d.dir === dir).reduce((m, d) => m + titled(d).length + 1, 0)), 0);
  for (const dir of [...dirs].sort((a, b) => alike(b) - alike(a))) {
    // A folder where most files share one shape gains nothing from a line per file: the lines
    // would repeat each other, and the room is better spent describing the files that differ.
    if (alike(dir) <= 0.5 && cost() <= half) break;
    folded.add(dir);
  }
  const foldedShare = Math.floor(half / Math.max(2, dirs.length));
  const lines: { doc?: Doc; text: string }[] = [];
  // Files described one by one come first: they are what a question is most often matched to.
  for (const dir of dirs) if (!folded.has(dir)) lines.push(...docs.filter((d) => d.dir === dir).map((d) => ({ doc: d, text: titled(d) })));
  for (const dir of dirs) if (folded.has(dir)) lines.push({ text: clip(foldedLine(dir), foldedShare) });
  let used = lines.reduce((n, l) => n + l.text.length + 1, 0);
  // Spend what is left on section headings, one per file per round, so every file gets some
  // description before any file gets a long one.
  const added = new Map<Doc, string[]>();
  for (let round = 0; ; round++) {
    let any = false;
    for (const l of lines) {
      if (!l.doc || round >= l.doc.sections.length) continue;
      const h = clip(l.doc.sections[round]!, 70);
      if (used + h.length + 2 > budgetChars) continue;
      added.set(l.doc, [...(added.get(l.doc) ?? []), h]);
      used += h.length + 2;
      any = true;
    }
    if (!any) break;
  }
  const body = lines.map((l) => (l.doc && added.get(l.doc)?.length ? `${l.text}\n  covers: ${added.get(l.doc)!.join("; ")}` : l.text));
  const extras = [...other.entries()].sort().map(([k, n]) => {
    const [dir, ext] = k.split("|");
    return `${dir || "."}/ ${n} ${ext} file${n === 1 ? "" : "s"}`;
  });
  const text = [
    "## Care library",
    "Files you can read with fs.read_file(path=...) or search with search.grep. Paths are relative to your working directory.",
    ...body,
    extras.length ? `Other files (data, charts, scans): ${extras.join("; ")}.` : "",
  ]
    .filter(Boolean)
    .join("\n");
  cache = { key, text };
  return text;
}
