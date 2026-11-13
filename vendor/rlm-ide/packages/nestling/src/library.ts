import { readFileSync, statSync } from "node:fs";
import path from "node:path";
import type { ToolImpl } from "@rlm/core";
import { nearestFiles } from "./catalog.ts";
import { childScope } from "./clinical.ts";

/** `library.open`: read a care-library file the way a person opens a handbook.
 *
 * A short file comes back whole. A long one (the WHO chapters run to hundreds of kilobytes) comes
 * back as its outline plus the sections that match what is being looked for, so the reader sees
 * where the answer sits without the file flooding the context window. Sections are found by the
 * file's own headings and ranked by the words asked for; nothing about any subject is built in. */

interface Section {
  heading: string;
  level: number;
  line: number;
  body: string;
}

function headingOf(line: string): { level: number; text: string } | null {
  if (line.charCodeAt(0) !== 35 || line.length > 300) return null;
  let level = 0;
  while (level < line.length && line[level] === "#") level++;
  if (level > 4 || line[level] !== " ") return null;
  const text = line.slice(level).trim();
  return text ? { level, text } : null;
}

export function splitSections(text: string): Section[] {
  const out: Section[] = [];
  let cur: Section = { heading: "", level: 0, line: 1, body: "" };
  const lines = text.split(/\r?\n/);
  lines.forEach((line, i) => {
    const h = headingOf(line);
    if (h) {
      if (cur.heading || cur.body.trim()) out.push(cur);
      cur = { heading: h.text, level: h.level, line: i + 1, body: "" };
    } else cur.body += `${line}\n`;
  });
  if (cur.heading || cur.body.trim()) out.push(cur);
  return out;
}

const UNIT_MONTHS: Record<string, number> = { week: 12 / 52.18, month: 1, year: 12 };

/** The age range a heading is written for, in months, when it names one ("Feeding 7–9 months",
 * "Growth 1 to 3 years"). The range runs to the end of its last unit, so "4–7 months" still holds
 * a child of 7.5 months. Headings without a range apply at any age and return null. */
export function ageRangeOf(heading: string): { lo: number; hi: number } | null {
  const m = /(\d+(?:\.\d+)?)\s*(?:[-–—]|to)\s*(\d+(?:\.\d+)?)\s*(week|month|year)/i.exec(heading);
  if (!m) return null;
  const k = UNIT_MONTHS[m[3]!.toLowerCase()]!;
  const lo = Number(m[1]) * k;
  const hi = (Number(m[2]) + 1) * k;
  return hi > lo ? { lo, hi } : null;
}

/** Which of a file's age-ranged sections are written for a child of this age. A small model asked
 * to place 7.1 months picked "8–12 months"; whether a number lies in a range is arithmetic, so it
 * is worked out here and stated, for whatever file and whatever age. */
export function ageNote(sections: Section[], ageMonths: number | undefined): { note: string; fits: (s: Section) => boolean | null } {
  const fits = (s: Section): boolean | null => {
    const r = ageRangeOf(s.heading);
    if (!r || ageMonths === undefined) return null;
    return ageMonths >= r.lo && ageMonths < r.hi;
  };
  if (ageMonths === undefined) return { note: "", fits };
  const ranged = sections.filter((s) => fits(s) !== null);
  if (!ranged.length) return { note: "", fits };
  const hit = ranged.filter((s) => fits(s));
  const age = `${Math.round(ageMonths * 10) / 10} months`;
  const note = hit.length
    ? `The child is ${age} old. The sections written for that age are: ${hit.map((s) => `"${s.heading}"`).join("; ")}. Sections headed with another age range are about other ages.`
    : `The child is ${age} old. No section of this file is headed with an age range that includes that age.`;
  return { note, fits };
}

const BM25_K = 1.2;
const BM25_B = 0.75;

function terms(find: string): string[] {
  return [...new Set(find.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((w) => w.length >= 3))];
}

function count(hay: string, needle: string): number {
  let n = 0;
  for (let i = hay.indexOf(needle); i >= 0; i = hay.indexOf(needle, i + needle.length)) n++;
  return n;
}

function clip(s: string, n: number): string {
  return s.length <= n ? s : `${s.slice(0, Math.max(0, n - 1)).trimEnd()}…`;
}

/** The text handed back for `find` in a file too long to return whole. */
export function openLong(rel: string, text: string, find: string, budget: number, ageMonths?: number): string {
  const sections = splitSections(text);
  const age = ageNote(sections, ageMonths);
  const want = terms(find);
  const lower = sections.map((s) => ({ head: s.heading.toLowerCase(), body: s.body.toLowerCase() }));
  // A word that turns up in most sections of this file says little about any one of them, so each
  // word counts by how few sections hold it. That is what lets the whole question be passed as
  // `find`: "the", "baby" and the file's own subject fade out, the distinctive words decide.
  const weight = new Map<string, number>();
  for (const t of want) {
    const df = lower.filter((l) => l.head.includes(t) || l.body.includes(t)).length;
    weight.set(t, df ? Math.log(1 + sections.length / df) : 0);
  }
  const avgLen = lower.reduce((n, l) => n + l.body.length, 0) / Math.max(1, lower.length) || 1;
  const scored = sections
    .map((s, i) => {
      const { head, body } = lower[i]!;
      // A word in the heading says the section is about it; in the body, only that it is mentioned.
      // Counts are set against the section's length, so a long section does not win on bulk.
      const norm = BM25_K * (1 - BM25_B + (BM25_B * body.length) / avgLen);
      const words = want.reduce((n, t) => {
        const tf = count(body, t);
        return n + (weight.get(t) ?? 0) * ((tf * (BM25_K + 1)) / (tf + norm) + (head.includes(t) ? 1.5 : 0));
      }, 0);
      // A section written for another age answers a different child; one written for this age
      // is where to look first.
      const fit = age.fits(s);
      const score = fit === null ? words : fit ? words * 2 + 1 : words * 0.25;
      return { s, i, score };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score || a.i - b.i);

  const outlineBudget = Math.floor(budget * 0.2);
  const heads = sections.filter((s) => s.heading && s.level <= 2).map((s) => `${"  ".repeat(Math.max(0, s.level - 1))}${s.heading}`);
  let outline = "";
  for (const h of heads) {
    if (outline.length + h.length + 1 > outlineBudget) {
      outline += `… (${heads.length} headings in all)\n`;
      break;
    }
    outline += `${h}\n`;
  }
  const parts = [`${rel} is long (${text.length} characters), so this is its outline and the matching sections, not the whole file.`, `Outline:\n${outline.trimEnd()}`];
  if (age.note) parts.push(age.note);
  let left = budget - parts.join("\n\n").length;
  const picked = scored.length ? scored : sections.slice(0, 1).map((s, i) => ({ s, i, score: 0 }));
  const shown: typeof picked = [];
  for (const p of picked) {
    if (left < 200) break;
    const body = clip(`## ${p.s.heading || "(start)"}\n${p.s.body.trim()}`, Math.min(left, Math.max(600, Math.floor(budget * 0.35))));
    shown.push({ ...p, s: { ...p.s, body } });
    left -= body.length + 2;
  }
  // Back in reading order, so the passages read as the book has them.
  shown.sort((a, b) => a.i - b.i);
  parts.push(...shown.map((p) => p.s.body));
  if (!want.length) parts.push('Pass find="words to look for" to read the sections about something specific.');
  else if (!scored.length) parts.push(`No section mentions: ${want.join(", ")}. Try other words, or search.grep across the library.`);
  return parts.join("\n\n");
}

interface Parsed {
  mtimeMs: number;
  sections: Array<{ head: string; body: string }>;
}
const parsed = new Map<string, Parsed>();

function sectionsOf(abs: string): Parsed["sections"] {
  try {
    const st = statSync(abs);
    const hit = parsed.get(abs);
    if (hit && hit.mtimeMs === st.mtimeMs) return hit.sections;
    const sections = splitSections(readFileSync(abs, "utf8")).map((s) => ({ head: s.heading.toLowerCase(), body: s.body.toLowerCase() }));
    parsed.set(abs, { mtimeMs: st.mtimeMs, sections });
    return sections;
  } catch {
    return [];
  }
}

/** The files whose text has most to say about `query`, best first.
 *
 * A file is chosen for a question by the model from the catalog, which shows a title and a few
 * headings. That is enough for "sleep" and not for a question whose answer sits inside a chapter
 * the headings do not name: asked how to express milk by hand, the model picked two short leaflets
 * and the WHO chapter that covers it was never opened. So the library is also asked directly.
 * Every section of every file is scored by the question's words, each weighted by how few sections
 * in the whole library contain it, and a file stands on its best section. Nothing here knows a
 * subject: it is the text on disk that decides. */
export function rankFiles(root: string, files: string[], query: string): Array<{ file: string; score: number }> {
  const want = terms(query);
  if (!want.length) return [];
  const all = files.map((file) => ({ file, sections: sectionsOf(path.resolve(root, file)) }));
  const total = all.reduce((n, f) => n + f.sections.length, 0);
  if (!total) return [];
  const weight = new Map<string, number>();
  for (const t of want) {
    let df = 0;
    for (const f of all) for (const s of f.sections) if (s.head.includes(t) || s.body.includes(t)) df++;
    weight.set(t, df ? Math.log(1 + total / df) : 0);
  }
  // A long section mentions more of everything, so each word's count is set against the
  // section's length (the usual BM25 saturation), and a chapter does not win on bulk.
  const avg = all.reduce((n, f) => n + f.sections.reduce((m, s) => m + s.body.length, 0), 0) / total || 1;
  const K = BM25_K;
  const B = BM25_B;
  return all
    .map(({ file, sections }) => {
      let best = 0;
      for (const s of sections) {
        const norm = K * (1 - B + (B * s.body.length) / avg);
        let score = 0;
        for (const t of want) {
          const w = weight.get(t)!;
          if (!w) continue;
          const tf = count(s.body, t);
          score += w * ((tf * (K + 1)) / (tf + norm) + (s.head.includes(t) ? 1.5 : 0));
        }
        if (score > best) best = score;
      }
      return { file, score: best };
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score);
}

/** Whether a file is written for a child of this age, judged by the age ranges in its own
 * headings: false when it names age ranges and none of them holds this age (a file for 2 to 6 year
 * olds, asked about a baby of 7 months), true when one does, null when it names none. */
export function fileFitsAge(abs: string, ageMonths: number | undefined): boolean | null {
  if (ageMonths === undefined) return null;
  const heads = sectionsOf(abs).map((s) => s.head);
  let ranged = false;
  for (const h of heads) {
    const r = ageRangeOf(h);
    if (!r) continue;
    ranged = true;
    if (ageMonths >= r.lo && ageMonths < r.hi) return true;
  }
  return ranged ? false : null;
}

export function createLibraryTools(root: string, budgetChars: () => number): ToolImpl[] {
  return [
    {
      descriptor: {
        id: "library.open",
        group: "library",
        description:
          "Open a care-library file. A short file is returned whole. A long file returns its outline and the sections that match `find`. Use this to read a file from the Care library list.",
        parameters: {
          type: "object",
          properties: {
            path: { type: "string", description: "File path from the Care library list" },
            find: { type: "string", description: "Words describing what you are looking for; needed for long files" },
            max_chars: { type: "integer", description: "Return at most this many characters (optional; for reading several files in one step)" },
          },
          required: ["path"],
        },
        returns: "str",
        sideEffects: false,
        timeoutSec: 30,
      },
      async run(args, ctx) {
        const rel = String(args.path ?? "").replace(/\\/g, "/").replace(/^\.?\//, "");
        const abs = path.resolve(root, rel);
        if (abs !== root && !abs.startsWith(root + path.sep)) throw new Error("path is outside the care library");
        let text: string;
        try {
          if (!statSync(abs).isFile()) throw new Error("ENOENT");
          text = readFileSync(abs, "utf8");
        } catch {
          const near = nearestFiles(root, rel);
          throw new Error(`no such file: ${rel}.${near.length ? ` Closest existing files: ${near.join(", ")}` : ""}`);
        }
        if (text.slice(0, 2000).includes("\0")) throw new Error(`${rel} is not a text file. Open the .md file with the same name instead.`);
        const cap = Number(args.max_chars);
        const budget = Number.isFinite(cap) && cap > 0 ? Math.max(1000, Math.min(budgetChars(), cap)) : budgetChars();
        const ageMonths = childScope(ctx.chatId)?.ageMonths;
        if (text.length <= budget) {
          const { note } = ageNote(splitSections(text), ageMonths);
          return note ? `${note}\n\n${text}` : text;
        }
        return openLong(rel, text, String(args.find ?? ""), budget, ageMonths);
      },
    },
  ];
}
