import { validateArgs, type ChatModelPort, type JsonSchema, type PromptsPort, type ToolDescriptor } from "@rlm/core";
import { numbersIn } from "./speak.ts";

/** The first step of a turn, chosen by a narrow, constrained question to the model.
 *
 * A small model asked to "work out what to do" in free text answers from memory or wanders. The
 * same model asked one closed question (which of these files cover this message, and is one of
 * these tools needed?) with its reply constrained to a schema built from the files and tools that
 * actually exist, answers it well. So the turn starts with that question, and the answer is turned
 * into the `repl` cells the model would have had to write. Everything in the schema comes from the
 * library on disk and the tool descriptors: a new file or tool is offered on the next turn, and
 * nothing here knows any topic. */

export interface RouteInput {
  model: ChatModelPort;
  prompts: PromptsPort;
  message: string;
  /** The care-library catalog shown to the model. */
  catalog: string;
  /** Files it may choose from, relative paths. */
  files: string[];
  /** Tools it may ask for (those that compute or fetch something for this child). */
  tools: ToolDescriptor[];
  child: string;
  memory: string;
  /** The last few exchanges, so a follow-up is read in context. */
  recent: string;
  /** The message is known to need the library: the choice left is which files. */
  mustRead?: boolean;
}

export interface Route {
  files: string[];
  find: string;
  tool: string;
  toolArgs: Record<string, unknown>;
  /** The model's own statement of where the answer comes from, for the turn log. */
  answerFrom: string;
  /** The separate things a request asks for, each with the files that cover it; empty for a
   * request that is one question. */
  parts: RoutePart[];
}

export interface RoutePart {
  topic: string;
  files: string[];
}

const SIMPLE = new Set(["string", "number", "integer", "boolean"]);

/** Most separate parts one request is split into. */
const MAX_PARTS = 5;

/** What a message wants, for `answerableFromKnown`. */
const ASKS = ["nothing", "about_this_family_or_what_was_said", "care_or_health_information"];

/** Where the answer to a message comes from. Only the first reads the library. */
const ANSWER_FROM = ["care_library", "child_record_or_saved_facts", "earlier_in_this_conversation", "no_lookup_needed"];

/** One flat object holding every simple parameter of the offered tools, so the model can fill in
 * the arguments of whichever tool it names. */
function argsSchema(tools: ToolDescriptor[]): JsonSchema {
  const properties: Record<string, JsonSchema> = {};
  for (const t of tools) {
    for (const [name, p] of Object.entries(t.parameters.properties ?? {})) {
      if (!p.type || !SIMPLE.has(p.type) || properties[name]) continue;
      properties[name] = { type: p.type, ...(p.enum ? { enum: p.enum } : {}) };
    }
  }
  return { type: "object", properties };
}

function toolLines(tools: ToolDescriptor[]): string {
  return tools
    .map((t) => {
      const params = Object.entries(t.parameters.properties ?? {})
        .filter(([, p]) => p.type && SIMPLE.has(p.type))
        .map(([n, p]) => `${n}${(t.parameters.required ?? []).includes(n) ? "*" : ""}${p.enum ? ` (${p.enum.join("|")})` : ""}`)
        .join(", ");
      return `- ${t.id}: ${t.description.split(/(?<=\.)\s/)[0]}${params ? ` Arguments: ${params}` : ""}`;
    })
    .join("\n");
}

function render(template: string, vars: Record<string, string>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, k: string) => vars[k] ?? "");
}

export async function route(input: RouteInput): Promise<Route | null> {
  if (!input.files.length && !input.tools.length) return null;
  const schema = {
    type: "object",
    properties: {
      // What the answer rests on is decided first, as a closed choice, and the files follow from
      // it. Asked for files straight away, the model always found some to name, and a question
      // about what it had been told was answered out of a handbook.
      answer_from: { type: "string", enum: input.mustRead ? ANSWER_FROM.slice(0, 1) : ANSWER_FROM },
      // `find` comes before the files so the model names the subject in its own words first.
      find: { type: "string", maxLength: 80 },
      files: { type: "array", items: { type: "string", enum: input.files }, maxItems: 2, ...(input.mustRead ? { minItems: 1 } : {}) },
      tool: { type: "string", enum: ["none", ...input.tools.map((t) => t.id)] },
      tool_args: argsSchema(input.tools),
      parts: {
        type: "array",
        maxItems: MAX_PARTS,
        items: {
          type: "object",
          properties: { topic: { type: "string", maxLength: 80 }, files: { type: "array", items: { type: "string", enum: input.files }, maxItems: 2 } },
          required: ["topic", "files"],
        },
      },
    },
    required: ["answer_from", "find", "files", "tool", "tool_args", "parts"],
  };
  const system = render(input.prompts.get("route.md"), {
    catalog: input.catalog,
    tools: toolLines(input.tools) || "(none)",
    child: input.child || "(none)",
    memory: input.memory || "(none)",
    recent: input.recent || "(this is the first message)",
  });
  let raw = "";
  for await (const c of input.model.stream({
    messages: [
      { role: "system", content: system },
      { role: "user", content: input.message },
    ],
    // Room for the parts list too: a reply cut short is not JSON, and the whole route is lost.
    maxTokens: 220 + MAX_PARTS * 90,
    temperature: 0,
    enableThinking: false,
    jsonSchema: { name: "route", schema },
  })) {
    if (c.type === "text") raw += c.text;
  }
  let parsed: { answer_from?: unknown; files?: unknown; find?: unknown; tool?: unknown; tool_args?: unknown; parts?: unknown };
  try {
    parsed = JSON.parse(raw.slice(raw.indexOf("{"), raw.lastIndexOf("}") + 1));
  } catch {
    return null;
  }
  const known = new Set(input.files);
  const reads = parsed.answer_from === undefined || parsed.answer_from === ANSWER_FROM[0];
  const files = reads ? [...new Set((Array.isArray(parsed.files) ? parsed.files : []).map(String))].filter((f) => known.has(f)).slice(0, 2) : [];
  let tool = typeof parsed.tool === "string" && parsed.tool !== "none" ? parsed.tool : "";
  let toolArgs: Record<string, unknown> = {};
  const desc = input.tools.find((t) => t.id === tool);
  if (desc) {
    // Only the arguments this tool declares; and if what is left is not a valid call, there is no
    // call. A tool named with nothing to compute (no measurement given) is dropped here.
    const allowed = desc.parameters.properties ?? {};
    const given = parsed.tool_args && typeof parsed.tool_args === "object" ? (parsed.tool_args as Record<string, unknown>) : {};
    // A number passed to a tool must be one the parent wrote. Asked to assess "7.4 kg", the model
    // also filled in an age, a percentile and a week count of its own; computing with those gives
    // a confident answer about a different child. Anything not stated is left for the child
    // record to supply.
    const stated = numbersIn(input.message);
    toolArgs = Object.fromEntries(
      Object.entries(given).filter(([k, v]) => k in allowed && v !== null && v !== "" && (typeof v !== "number" || stated.has(String(Number(v))))),
    );
    try {
      toolArgs = validateArgs(desc.parameters, toolArgs);
    } catch {
      tool = "";
      toolArgs = {};
    }
  } else tool = "";
  const parts: RoutePart[] = reads
    ? (Array.isArray(parsed.parts) ? parsed.parts : [])
        .map((p) => {
          const o = (p ?? {}) as { topic?: unknown; files?: unknown };
          return { topic: String(o.topic ?? "").trim(), files: [...new Set((Array.isArray(o.files) ? o.files : []).map(String))].filter((f) => known.has(f)).slice(0, 2) };
        })
        .filter((p) => p.topic)
        .slice(0, MAX_PARTS)
    : [];
  // One part is just the question again; a plan is for two or more.
  return { files, find: typeof parsed.find === "string" ? parsed.find.trim() : "", tool, toolArgs, answerFrom: String(parsed.answer_from ?? ""), parts: parts.length >= 2 ? parts : [] };
}

/** Whether a message can be replied to from what is already known, with nothing read.
 *
 * Skipping the library is the one routing mistake that costs the parent something: the reply is
 * then written from the record and the conversation alone, and a real question gets "I could not
 * find that" or, worse, an answer from recall. So that choice is not taken on the router's word.
 * It is put to the model again as a single yes-or-no about the material itself, and anything but
 * a clear yes sends the turn to the library. */
export async function answerableFromKnown(input: Pick<RouteInput, "model" | "prompts" | "message" | "child" | "memory" | "recent">): Promise<boolean> {
  const system = render(input.prompts.get("known_check.md"), {
    child: input.child || "(none)",
    memory: input.memory || "(none)",
    recent: input.recent || "(this is the first message)",
  });
  let raw = "";
  for await (const c of input.model.stream({
    messages: [
      { role: "system", content: system },
      { role: "user", content: input.message },
    ],
    maxTokens: 60,
    temperature: 0,
    enableThinking: false,
    jsonSchema: {
      name: "known",
      schema: {
        type: "object",
        properties: { asks: { type: "string", enum: ASKS }, quote: { type: "string", maxLength: 200 } },
        required: ["asks", "quote"],
      },
    },
  })) {
    if (c.type === "text") raw += c.text;
  }
  try {
    const p = JSON.parse(raw.slice(raw.indexOf("{"), raw.lastIndexOf("}") + 1)) as { asks?: string; quote?: string };
    if (p.asks === ASKS[0]) return true;
    if (p.asks !== ASKS[1]) return false;
    // "It is already known" has to be shown, not asserted: the model quotes the line that holds
    // the answer, and the quote has to be there. Asked whether a screening questionnaire had
    // already been given in the conversation, it said yes; it could not have quoted one.
    const flat = (s: string) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
    const quote = flat(p.quote ?? "");
    return quote.length >= 12 && flat([input.child, input.memory, input.recent].join("\n")).includes(quote);
  } catch {
    return false;
  }
}

/** The route as the assistant message the model would have written: one `repl` cell per action.
 * A long file is searched for the model's own words for the subject together with the question as
 * asked: the model's words alone have been a file name or a copied heading, and the library
 * weighs each word by how distinctive it is in that file, so the extra words cost nothing. */
export function routeToCells(r: Route, question = "", readChars = 0): string {
  const cells: string[] = [];
  const find = [r.find, question].filter(Boolean).join(" ").replace(/\s+/g, " ").trim().slice(0, 400);
  if (r.parts.length) {
    // A request with several parts is carried out as a plan, and the plan is kept by the host:
    // it is written on the task board, each part is read, and the part is ticked when its reading
    // is done. Left to write and tick its own plan, a small model ticked steps it had not done.
    // What is read for each part is capped so that all of them fit the context window together.
    const opens = r.parts.reduce((n, p) => n + Math.max(1, p.files.length), 0);
    const each = readChars > 0 ? Math.max(1200, Math.floor(readChars / opens)) : 0;
    cells.push(`print(task.plan(goal=${JSON.stringify(question.slice(0, 280))}, steps=${JSON.stringify(r.parts.map((p) => p.topic))}))`);
    r.parts.forEach((p, i) => {
      const lines = p.files.map((f) => `print(library.open(path=${JSON.stringify(f)}, find=${JSON.stringify(p.topic)}${each ? `, max_chars=${each}` : ""}))`);
      // A part no file covers stays open on the board; the answer has to say so.
      lines.push(p.files.length ? `_ = task.step(index=${i}, status="done")` : `_ = task.step(index=${i}, status="blocked", note="no file in the library covers this")`);
      cells.push([`print(${JSON.stringify(`--- Part ${i + 1}: ${p.topic} ---`)})`, ...lines].join("\n"));
    });
  } else
  for (const f of r.files) {
    cells.push(`print(library.open(path=${JSON.stringify(f)}${find ? `, find=${JSON.stringify(find)}` : ""}))`);
  }
  if (r.tool) {
    cells.push(`import json\nprint(json.dumps(${r.tool}(**json.loads(${JSON.stringify(JSON.stringify(r.toolArgs))})), ensure_ascii=False, default=str))`);
  }
  return cells.map((c) => `\`\`\`repl\n${c}\n\`\`\``).join("\n");
}
