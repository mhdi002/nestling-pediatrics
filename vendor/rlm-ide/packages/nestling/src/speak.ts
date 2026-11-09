import type { ChatModelPort } from "@rlm/core";

function scriptHint(text: string): "fa" | "en" {
  let fa = 0;
  for (const ch of text) {
    const c = ch.codePointAt(0) ?? 0;
    if (c >= 0x0600 && c <= 0x06ff) fa++;
  }
  return fa >= 8 ? "fa" : "en";
}

function dumpLike(text: string): boolean {
  const t = text || "";
  if (!t.trim()) return true;
  if (/```|print\s*\(|fs\.read_file|search\.grep|\[REPL|Stopped after/i.test(t)) return true;
  if (t.trim().startsWith("#") && t.includes("guidelines/")) return true;
  if (t.trim().startsWith("{") && /centile/i.test(t)) return true;
  return (t.match(/\.md:\d+:/g) ?? []).length >= 2;
}

function sanitizeParent(text: string): string {
  return (text || "")
    .replace(/```[\s\S]*?```/g, "")
    .replace(/\bsearch\.grep\b/gi, "the notes")
    .replace(/\bfs\.read_file\b/gi, "the files")
    .replace(/\bclinical\.growth_percentile\b/gi, "the growth calculation")
    .replace(/\[REPL output\]/gi, "")
    .trim();
}

function missesEvidence(question: string, answer: string, evidence: string): boolean {
  const ev = (evidence || "").toLowerCase();
  const ans = (answer || "").toLowerCase();
  const q = (question || "").toLowerCase();
  if (q.includes("iron") && ev.includes("iron") && !/iron|anemia|fortif|breast/.test(ans)) return true;
  if (q.includes("danger") && ev.includes("danger") && !/vomit|feed|convuls|letharg|fever|drink|breath/.test(ans)) return true;
  if (q.includes("centile") && /centile|percentile/.test(ev) && !/\d/.test(ans)) return true;
  return false;
}

function deniedLike(text: string): boolean {
  return /don'?t have access|can'?t look up|could not find|no results|cannot search|can't help with this|do not have access|do not have the ability/i.test(
    text || "",
  );
}

export async function speakParent(args: {
  model: ChatModelPort;
  question: string;
  lang: string;
  evidence: string;
  memory: string;
  child: string;
  draft: string;
}): Promise<string> {
  const fa = args.lang === "fa" || scriptHint(args.question) === "fa";
  const langLine = fa
    ? "Write the entire answer in Persian (Farsi). Do not echo the question. Do not mix English sentences."
    : "Write fluent English for a parent.";
  const system =
    "You are Nestling, speaking to one parent. Answer ONLY the current question. " +
    "Use Evidence, Memory, and Child. Write 2–6 fluent sentences. No headings, file paths, code, tool names, or grep dumps. " +
    "If Evidence has search hits or JSON, summarize the concrete facts (ages, foods, doses, warning signs, numbers). " +
    "Never apologize, never say you lack files or tools, and never change the topic. " +
    langLine;
  const user =
    `Current question:\n${args.question}\n\n` +
    `Evidence:\n${(args.evidence || "(none)").slice(0, 3500)}\n\n` +
    `Memory:\n${(args.memory || "(none)").slice(0, 1200)}\n\n` +
    `Child:\n${(args.child || "(none)").slice(0, 800)}`;
  let out = "";
  for await (const c of args.model.stream({
    messages: [
      { role: "system", content: system },
      { role: "user", content: user },
    ],
    maxTokens: 420,
    temperature: 0.2,
    enableThinking: false,
  })) {
    if (c.type === "text") out += c.text;
  }
  out = sanitizeParent(out);
  const evidenceUseful = Boolean(args.evidence.trim()) && !/^\s*(\[\s*\]|null|none|\(none\))\s*$/i.test(args.evidence.trim());
  const denied = evidenceUseful && deniedLike(out);
  const missed = evidenceUseful && missesEvidence(args.question, out, args.evidence);
  if (!dumpLike(out) && !denied && !missed && (!fa || scriptHint(out) === "fa")) return out;
  if (evidenceUseful && (dumpLike(out) || denied || missed || !out)) {
    let retry = "";
    for await (const c of args.model.stream({
      messages: [
        {
          role: "system",
          content:
            "You already have the tool output. Summarize it for a parent in 2–6 fluent sentences. " +
            "Never say you lack files or tools. No headings or paths. " +
            "If the question asks for warning signs, name two concrete symptoms from the text " +
            "(for example not feeding, vomiting, convulsions, lethargy, fever, fast breathing). " +
            "If the question is about iron, mention complementary foods or supplementation age. " +
            (fa ? "Persian only." : "English."),
        },
        { role: "user", content: `Question:\n${args.question}\n\nTool output:\n${args.evidence.slice(0, 2800)}` },
      ],
      maxTokens: 360,
      temperature: 0.15,
      enableThinking: false,
    })) {
      if (c.type === "text") retry += c.text;
    }
    retry = sanitizeParent(retry);
    if (retry && !dumpLike(retry) && !deniedLike(retry)) return retry;
  }
  const draft = sanitizeParent(args.draft || "");
  if (draft && !dumpLike(draft) && (!fa || scriptHint(draft) === "fa") && !deniedLike(draft)) return draft;
  const cleaned = sanitizeParent(out);
  if (evidenceUseful && cleaned && !dumpLike(cleaned) && !deniedLike(cleaned)) return cleaned;
  return cleaned || draft || "I could not form a clear parent reply. Please ask again in a short sentence.";
}
