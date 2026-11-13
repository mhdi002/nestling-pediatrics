import type { ChatModelPort } from "@rlm/core";

/** Turns the engine's working answer into what the parent reads, and refuses to pass on what the
 * turn cannot back up.
 *
 * Every check here is about the shape of the text or about whether it is supported by what the
 * turn actually saw. None of them knows a topic: there is no list of subjects, symptoms or
 * expected words, so the same rules hold for a question nobody has asked yet. */

const PERSIAN = /[؀-ۿ]/g;
const LATIN = /[A-Za-z]/g;

/** Which script a text is mostly written in; `null` when it has too few letters to say. */
export function dominantScript(text: string): "fa" | "en" | null {
  const fa = (text.match(PERSIAN) ?? []).length;
  const en = (text.match(LATIN) ?? []).length;
  if (fa + en < 4) return null;
  return fa >= en ? "fa" : "en";
}

/** Engine working material rather than an answer: code, raw tool output, file:line search hits. */
export function looksLikeWorkOutput(text: string): boolean {
  const t = (text || "").trim();
  if (!t) return true;
  if (/```/.test(t) || /\[REPL output\]/i.test(t)) return true;
  if (/^[\[{]/.test(t) && /[\]}]$/.test(t)) return true;
  const lines = t.split("\n").filter((l) => l.trim());
  const hits = lines.filter((l) => /^\S+\.\w{1,5}:\d+:/.test(l.trim())).length;
  if (hits >= 2 && hits >= lines.length / 2) return true;
  // A call to one of the kernel's namespaces left in the prose, e.g. `print(fs.read_file(...))`.
  return /\b[a-z_]+\.[a-z_]+\s*\(\s*[a-z_]+\s*=/.test(t);
}

/** Removes fenced blocks and engine bookkeeping lines, leaving the prose. */
export function stripWorkMarkup(text: string): string {
  return (text || "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/```[\s\S]*$/g, "")
    .replace(/^\s*\[(?:REPL output|IDE|hint|stderr|result|error)\].*$/gim, "")
    .replace(/\(Stopped after \d+ steps[^)]*\)/g, "")
    // The parent reads this as plain text in a chat bubble: emphasis marks and heading hashes
    // would show as stray symbols, so the words are kept and the marks dropped.
    .replace(/(\*\*|__)(?=\S)([\s\S]*?\S)\1/g, "$2")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/^(\s*)[*+]\s+/gm, "$1• ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

const DIGIT_MAP: Record<string, string> = {};
"۰۱۲۳۴۵۶۷۸۹".split("").forEach((d, i) => (DIGIT_MAP[d] = String(i)));
"٠١٢٣٤٥٦٧٨٩".split("").forEach((d, i) => (DIGIT_MAP[d] = String(i)));

/** Numbers written as words. A figure spelled out ("three meals and two snacks") is as much a
 * claim as one in digits, and slipped past a check that only looked for digits. "One" and "یک"
 * are left out: they are articles and pronouns far more often than counts. */
const NUMBER_WORDS: Record<string, number> = {};
"two three four five six seven eight nine ten eleven twelve".split(" ").forEach((w, i) => (NUMBER_WORDS[w] = i + 2));
// Persian "نه" (nine) is left out as well: it is also the word for "no".
"دو سه چهار پنج شش هفت هشت".split(" ").forEach((w, i) => (NUMBER_WORDS[w] = i + 2));
"ده یازده دوازده".split(" ").forEach((w, i) => (NUMBER_WORDS[w] = i + 10));
const NUMBER_WORD_RE = new RegExp(`(?<![\\p{L}\\p{N}])(${Object.keys(NUMBER_WORDS).join("|")})(?![\\p{L}\\p{N}])`, "giu");

function asciiDigits(text: string): string {
  return (text || "")
    .replace(/[۰-۹٠-٩]/g, (d) => DIGIT_MAP[d] ?? d)
    .replace(/[٫]/g, ".")
    .replace(/(?<=\d)[,٬](?=\d{3}\b)/g, "")
    .replace(NUMBER_WORD_RE, (w) => String(NUMBER_WORDS[w.toLowerCase()] ?? w));
}

/** Every number written in digits, normalised so 6.0 and ۶ and 6 compare equal. */
export function numbersIn(text: string): Set<string> {
  const out = new Set<string>();
  for (const m of asciiDigits(text).matchAll(/\d+(?:\.\d+)?/g)) out.add(String(Number(m[0])));
  return out;
}

interface Quantity {
  n: number;
  /** The word the number counts ("kg", "month", "hour"), singular and lower-cased; "" when none follows. */
  unit: string;
}

const RANGE_JOIN = /^\s*(?:[-–—]|to|or|and|تا|یا|و)\s*(?=\d)/i;

/** Every number in a text together with what it measures. "4–6 months" yields 4 month and 6 month;
 * "7-month-old" yields 7 month; a bare JSON value yields the number with no unit. */
export function quantitiesIn(text: string): Quantity[] {
  const t = asciiDigits(text);
  const out: Quantity[] = [];
  const re = /\d+(?:\.\d+)?/g;
  let m: RegExpExecArray | null;
  let pending: number[] = [];
  while ((m = re.exec(t))) {
    // A list marker ("1." or "2)" opening a line) numbers the list, not the world.
    const lineStart = t.lastIndexOf("\n", m.index - 1) + 1;
    if (!t.slice(lineStart, m.index).trim() && /^[.)]\s/.test(t.slice(re.lastIndex, re.lastIndex + 2))) continue;
    const n = Number(m[0]);
    const rest = t.slice(re.lastIndex, re.lastIndex + 40);
    if (RANGE_JOIN.test(rest)) {
      pending.push(n);
      continue;
    }
    const u = /^\s?-?\s?([\p{L}%°]+)/u.exec(rest);
    const unit = u ? u[1]!.toLowerCase().replace(/(?<=\p{L}{2})s$/u, "") : "";
    for (const p of [...pending, n]) out.push({ n: p, unit });
    pending = [];
  }
  for (const p of pending) out.push({ n: p, unit: "" });
  return out;
}

/** Whether two unit words name the same thing. Inflected forms agree ("month", "months",
 * "monthly"; "ماه", "ماهگی"). Units written in different scripts cannot be compared word for word
 * (a Persian answer is written from English guidance), so there the number has to carry it. */
function sameUnit(a: string, b: string): boolean {
  if (!a || !b || a === b) return true;
  const latin = (s: string) => /^[a-z%°]+$/.test(s);
  if (latin(a) !== latin(b)) return true;
  const [short, long] = a.length <= b.length ? [a, b] : [b, a];
  return short.length >= 2 && long.startsWith(short);
}

/** Numbers in `answer` that `sources` do not back. A figure the turn never saw is one the model
 * made up: this is what turned "about 11 mg of iron" into "1.5-2.5 grams", and what put a "typical
 * range of 6.0 to 7.0 kg" beside a weight when no tool had returned one.
 *
 * A number is backed when the sources hold the same number measuring the same thing, or hold it
 * bare (a tool's JSON value, which the answer is free to name). Matching the number alone is not
 * enough: "7" is in every child record as an age, and would then license "7 kg". A value within
 * rounding distance of a source value is accepted, so 45.2 may be said as 45. */
export function unsupportedNumbers(answer: string, sources: string): string[] {
  const src = quantitiesIn(sources);
  const same = (a: number, b: number) => a === b || Math.round(b) === a || Math.round(b * 10) / 10 === a;
  const bad = new Set<string>();
  for (const q of quantitiesIn(answer)) {
    const ok = src.some((s) => same(q.n, s.n) && sameUnit(q.unit, s.unit));
    if (!ok) bad.add(q.unit ? `${q.n} ${q.unit}` : String(q.n));
  }
  return [...bad];
}

function sentences(text: string): string[] {
  return text.split(/(?<=[.!?؟。])\s+|\n+/).filter((s) => s.trim());
}

/** Drops the sentences that carry an unsupported number; the rest of the answer stands. */
function withoutUnsupported(answer: string, sources: string): string {
  return sentences(answer)
    .filter((s) => unsupportedNumbers(s, sources).length === 0)
    .join(" ")
    .trim();
}

export interface Exchange {
  question: string;
  answer: string;
}

export interface SpeakArgs {
  model: ChatModelPort;
  question: string;
  /** "fa" or "en": the language the parent wrote in. */
  lang: string;
  /** What this turn's cells and tools returned. */
  evidence: string;
  memory: string;
  child: string;
  /** Earlier turns of this chat, so a number established before may be repeated now. */
  history: string;
  /** Earlier exchanges of this chat, newest last: what was asked and what the parent was told. */
  previous: Exchange[];
  /** The last few exchanges as text. */
  recent: string;
  /** The engine's own final text. */
  draft: string;
  /** Most characters of evidence to hand the model; derived from its context window by the caller. */
  evidenceChars: number;
  /** Whether the turn read or computed anything. When it did not, the engine's answer rests on
   * nothing but the model's own recall, and a check that counts numbers cannot tell: three
   * paragraphs of feeding and sleep advice with no figure in them passed it. So the reply is then
   * written here, from the child record, the saved facts and the conversation only. */
  lookedUp: boolean;
  /** The parts of a request that was carried out as a plan; empty for a single question. */
  parts: string[];
}

export interface Spoken {
  text: string;
  /** How the text was arrived at, for the turn log and tests. */
  via: "draft" | "rewrite" | "retry" | "trimmed" | "none";
  unsupported: string[];
  /** Why the engine's own answer was not used as it stood; empty when it was. */
  draftRejected: string;
}

function words(text: string): Set<string> {
  return new Set(text.toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((w) => w.length > 2));
}

function overlap(a: Set<string>, b: Set<string>): number {
  let both = 0;
  for (const w of a) if (b.has(w)) both++;
  const all = a.size + b.size - both;
  return all ? both / all : 0;
}

/** Whether `text` says again what an earlier reply in this chat already said, to a different
 * question. A small model that has just written an answer tends to write it again for the next
 * question, whatever was asked. The same answer to the same question is not that: a parent who
 * asks again (the reply did not arrive, or they want to hear it once more) is owed it again. */
export function repeatsEarlier(text: string, question: string, previous: Exchange[]): boolean {
  const a = words(text);
  if (a.size < 12) return false;
  const q = words(question);
  return previous.some((p) => overlap(a, words(p.answer)) >= 0.7 && overlap(q, words(p.question)) < 0.5);
}

interface Verdict {
  ok: boolean;
  unsupported: string[];
  repeated: boolean;
  why: string;
}

function acceptable(text: string, want: "fa" | "en", sources: string, question: string, previous: Exchange[]): Verdict {
  if (!text.trim()) return { ok: false, unsupported: [], repeated: false, why: "empty" };
  if (looksLikeWorkOutput(text)) return { ok: false, unsupported: [], repeated: false, why: "work output" };
  const script = dominantScript(text);
  if (script && script !== want) return { ok: false, unsupported: [], repeated: false, why: `written in ${script}, wanted ${want}` };
  const repeated = repeatsEarlier(text, question, previous);
  const unsupported = unsupportedNumbers(text, sources);
  const why = [unsupported.length ? `unsupported numbers: ${unsupported.join(", ")}` : "", repeated ? "repeats an earlier answer" : ""].filter(Boolean).join("; ");
  return { ok: unsupported.length === 0 && !repeated, unsupported, repeated, why };
}

async function generate(model: ChatModelPort, system: string, user: string, maxTokens: number): Promise<string> {
  let out = "";
  for await (const c of model.stream({
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    maxTokens,
    temperature: 0.2,
    enableThinking: false,
  })) {
    if (c.type === "text") out += c.text;
  }
  return stripWorkMarkup(out);
}

export async function speakParent(args: SpeakArgs): Promise<Spoken> {
  const want: "fa" | "en" = args.lang === "fa" || dominantScript(args.question) === "fa" ? "fa" : "en";
  const evidence = (args.evidence || "").slice(0, args.evidenceChars);
  const sources = [evidence, args.memory, args.child, args.history, args.question].join("\n");

  // 1. The engine's own answer, when it is already something a parent can read and every number in
  //    it is backed. No second generation, so most turns cost one answer, not two.
  const draft = stripWorkMarkup(args.draft);
  const d: Verdict = args.lookedUp ? acceptable(draft, want, sources, args.question, args.previous) : { ok: false, unsupported: [], repeated: false, why: "nothing was looked up" };
  if (d.ok) return { text: draft, via: "draft", unsupported: [], draftRejected: "" };

  // 1b. An answer whose only fault is a figure or two it cannot back keeps everything else: the
  //     sentences carrying those figures are dropped. Writing it again from scratch risks losing
  //     a good answer over one stray number.
  if (args.lookedUp && d.unsupported.length && !d.repeated) {
    const kept = withoutUnsupported(draft, sources);
    if (kept.length >= draft.length * 0.6 && !looksLikeWorkOutput(kept)) return { text: kept, via: "trimmed", unsupported: d.unsupported, draftRejected: d.why };
  }

  // 2. Otherwise write the parent's answer from what the turn found.
  const langLine = want === "fa" ? "Write the whole answer in Persian (Farsi)." : "Write the answer in English.";
  const system = [
    "You are Nestling, speaking to one parent. Reply to the parent's message and nothing else.",
    "The name in the Child record is the child's; the child cannot read. Speak to the parent and refer to the child in the third person, by name.",
    "Use only what is under Evidence, Saved facts, Child record and Conversation so far. Every number you write must appear there.",
    "If the message asks something they do not answer, say plainly that you could not find it in the care library and suggest asking the child's clinician. Do not guess.",
    "If the message asks nothing and only tells you something or greets you, reply briefly and warmly, saying you have noted what they told you; give no advice they did not ask for.",
    args.parts.length
      ? `The request has ${args.parts.length} parts: ${args.parts.join("; ")}. Write a short paragraph for each part, in that order, starting each with the part's name. If the evidence has nothing for a part, say so for that part. No file names, code or tool names.`
      : "Write 2 to 6 plain sentences. No headings, file names, code or tool names.",
    langLine,
  ].join(" ");
  const user = [
    `Parent's message:\n${args.question}`,
    `Evidence:\n${evidence || "(nothing was looked up)"}`,
    `Saved facts:\n${args.memory || "(none)"}`,
    `Child record:\n${args.child || "(none)"}`,
    `Conversation so far:\n${args.recent || "(this is the first message)"}`,
  ].join("\n\n");
  const maxTokens = !args.lookedUp ? 220 : args.parts.length ? 420 + 160 * args.parts.length : 420;
  const first = await generate(args.model, system, user, maxTokens);
  const f = acceptable(first, want, sources, args.question, args.previous);
  if (f.ok) return { text: first, via: "rewrite", unsupported: [], draftRejected: d.why };

  // 3. One more attempt, told exactly what was wrong with the last one.
  const problems: string[] = [];
  if (f.unsupported.length) problems.push(`These numbers are not in the evidence and must not be used: ${f.unsupported.join(", ")}.`);
  if (f.repeated) problems.push("The last attempt repeated an earlier answer. Answer this question, in new words.");
  if (looksLikeWorkOutput(first)) problems.push("The last attempt contained code or raw tool output. Write sentences only.");
  const script = dominantScript(first);
  if (script && script !== want) problems.push(langLine);
  const second = await generate(args.model, `${system} ${problems.join(" ")}`, user, maxTokens);
  const s = acceptable(second, want, sources, args.question, args.previous);
  if (s.ok) return { text: second, via: "retry", unsupported: [], draftRejected: d.why };

  // 4. Keep what can be backed, drop what cannot.
  for (const cand of args.lookedUp ? [second, first, draft] : [second, first]) {
    if (!cand.trim() || looksLikeWorkOutput(cand)) continue;
    const kept = withoutUnsupported(cand, sources);
    const sc = dominantScript(kept);
    if (kept && (!sc || sc === want) && !repeatsEarlier(kept, args.question, args.previous)) return { text: kept, via: "trimmed", unsupported: unsupportedNumbers(cand, sources), draftRejected: d.why };
  }
  // Saying again what was said before is a fault, but a smaller one than saying nothing: when
  // that is the only thing wrong with every candidate, the parent still gets the answer.
  for (const cand of args.lookedUp ? [second, first, draft] : [second, first]) {
    if (!cand.trim() || looksLikeWorkOutput(cand)) continue;
    const kept = withoutUnsupported(cand, sources);
    const sc = dominantScript(kept);
    if (kept && (!sc || sc === want)) return { text: kept, via: "trimmed", unsupported: unsupportedNumbers(cand, sources), draftRejected: d.why };
  }
  return { text: "", via: "none", unsupported: f.unsupported, draftRejected: d.why };
}
