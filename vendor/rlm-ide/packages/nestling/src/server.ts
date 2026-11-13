import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { composeNestling } from "./compose.ts";
import { setChildScope } from "./clinical.ts";
import { answerableFromKnown, route, routeToCells } from "./router.ts";
import { speakParent } from "./speak.ts";
import { fileFitsAge, rankFiles } from "./library.ts";
import path from "node:path";

const port = Number(process.env.RLM_TS_PORT || 8120);

/** The context window is the served model's, so ask the server instead of trusting a setting that
 * has to be kept equal to it by hand. An explicit RLM_CONTEXT_LENGTH still wins. */
async function detectContextLength(): Promise<void> {
  if (process.env.RLM_CONTEXT_LENGTH) return;
  const base = (process.env.NESTLING_LLM_URL || "http://127.0.0.1:8100").replace(/\/+$/, "");
  const want = process.env.NESTLING_LLM_MODEL || "";
  try {
    const r = await fetch(`${base}/v1/models`, { signal: AbortSignal.timeout(5000) });
    const data = ((await r.json()) as { data?: Array<{ id?: string; max_model_len?: number }> }).data ?? [];
    const m = data.find((x) => x.id === want) ?? data[0];
    if (m?.max_model_len) {
      process.env.RLM_CONTEXT_LENGTH = String(m.max_model_len);
      console.log(`[nestling-rlm] context window ${m.max_model_len} tokens (from ${base})`);
    }
  } catch {
    /* the model server is not up yet: the configured default applies */
  }
}
await detectContextLength();
const host = composeNestling();
// Wake-ups are what carry a planned job past its step budget. Without the scheduler running they
// are written on the board and never fire, and the request that scheduled one waits for a turn
// that does not come. Starting it also re-arms the ones left pending by a restart.
await host.orch.startScheduler();

type Ev = Record<string, unknown>;

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on("data", (c) => chunks.push(Buffer.isBuffer(c) ? c : Buffer.from(c)));
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", reject);
  });
}

function json(res: ServerResponse, code: number, body: unknown): void {
  res.writeHead(code, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

/** How much better the library's own best match must score than the files the model picked before it is read as well. */
const LEXICAL_LEAD = Number(process.env.RLM_LEXICAL_LEAD || 1.5);
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const isEnd = (e: Ev) => e.type === "turn_done" || e.type === "turn_error" || e.type === "turn_cancelled";

/** Waits for one engine turn to end and returns everything it emitted. */
async function awaitTurn(turnId: string, deadline: number): Promise<{ events: Ev[]; end: Ev | null }> {
  for (;;) {
    const events = host.orch.runner.bufferedEvents(turnId) as Ev[];
    const end = events.find(isEnd) ?? null;
    if (end || Date.now() > deadline) return { events, end };
    await sleep(50);
  }
}

/** A job that ran out of steps with work still open on its task board is continued by the engine
 * in a new turn of the same chat. The parent asked once and is waiting on this one HTTP request,
 * so follow those turns to the end and answer with the finished job instead of a note saying it
 * will carry on later. Returns the continuation turns' events, in order. */
async function followContinuations(chatId: string, afterTurnId: string, deadline: number): Promise<Array<{ events: Ev[]; end: Ev | null }>> {
  const out: Array<{ events: Ev[]; end: Ev | null }> = [];
  let last = afterTurnId;
  while (Date.now() < deadline) {
    const board = await host.boards.get(chatId);
    const turns = await host.chats.getTurns(chatId);
    const next = turns[turns.findIndex((t) => t.id === last) + 1];
    if (next) {
      out.push(await awaitTurn(next.id, deadline));
      last = next.id;
      continue;
    }
    if (!board.wakeup && !host.orch.runner.isActive(chatId)) break;
    await sleep(200);
  }
  return out;
}

function textOf(v: unknown): string {
  return typeof v === "string" ? v : JSON.stringify(v ?? "");
}

async function handleTurn(body: Record<string, unknown>): Promise<Record<string, unknown>> {
  const sessionId = String(body.session_id || "");
  const userMessage = String(body.user_message || "");
  const replyLang = String(body.reply_lang || "en");
  const childContext = String(body.child_context || "");
  const memoryBlock = String(body.memory_block || "");
  host.settings.userId = String(body.owner_user_id || "anonymous");
  host.refreshPersona();

  let chatId = host.chats.resolve(sessionId);
  if (!chatId) {
    const chat = await host.orch.createChat(sessionId || "nestling");
    host.chats.bindAlias(sessionId, chat.id);
    chatId = chat.id;
  }
  // Whose child this chat is about travels with the chat, not in process-wide state.
  const ageMonths = Number(body.child_age_months);
  setChildScope(chatId, {
    childId: String(body.child_id || ""),
    childDb: String(body.child_db || ""),
    ...(body.child_age_months != null && Number.isFinite(ageMonths) ? { ageMonths } : {}),
  });

  const pinText = [
    childContext ? `Child record (this is the child, who cannot read this; the person writing to you is the child's parent):\n${childContext}` : "",
    memoryBlock ? `Saved facts the parent told you earlier (use them when relevant; do not recite them as a list):\n${memoryBlock}` : "",
    body.en_gloss ? `English gloss of the parent's message, for searching only:\n${body.en_gloss}` : "",
    `Reply language: ${replyLang}.`,
  ]
    .filter(Boolean)
    .join("\n\n");
  if (pinText) await host.pinned.upsert({ chatId, kind: "constraint", text: pinText });

  // Exchanges the parent actually got an answer to; a turn that failed is not something said.
  const priorTurns = (await host.chats.getTurns(chatId)).filter((t) => t.assistantText.trim());
  // A new message from the parent is a new request. A plan left over from an earlier one is not
  // carried into it: a continuation still pending is called off, and a board nobody is working
  // on is cleared, so an old to-do list is not read as part of every later question.
  // Whatever is still running in this chat (a continuation that woke up, or an earlier message
  // sent twice) gives way to the message the parent has just sent, instead of that message
  // being refused with "a turn is already running".
  const running = host.orch.runner.activeTurn(chatId);
  if (running) {
    host.orch.stop(running);
    const until = Date.now() + 15000;
    while (host.orch.runner.activeTurn(chatId) && Date.now() < until) await sleep(50);
  }
  const stale = await host.boards.get(chatId);
  if (stale.wakeup) await host.orch.cancelWakeup(chatId);
  if (stale.steps.length && !host.orch.runner.isActive(chatId)) await host.boards.drop(chatId);
  const prevSteps = host.settings.rlm.maxSteps;
  if (body.max_steps) host.settings.rlm.maxSteps = Math.max(1, Number(body.max_steps));
  let turnId = "";
  let planParts: string[] = [];
  try {
    const groups = ["library", "fs", "search", "clinical", "task"];
    const recent = priorTurns
      .slice(-3)
      .map((t) => `Parent: ${t.userText.slice(0, 300)}\nNestling: ${t.assistantText.slice(0, 400)}`)
      .join("\n");
    const sent = await host.orch.send({
      chatId,
      text: userMessage,
      editorContext: {},
      toolGroups: groups,
      // The turn opens with a constrained choice of what to read or compute; see router.ts.
      firstStep: async () => {
        const ask = {
          model: host.agent,
          prompts: host.prompts,
          message: body.en_gloss ? `${userMessage}\n(English: ${String(body.en_gloss)})` : userMessage,
          catalog: host.settings.persona,
          files: host.readable(),
          tools: host.tools.list().filter((t) => t.group === "clinical"),
          child: childContext,
          memory: memoryBlock,
          recent,
        };
        let r = await route(ask);
        if (!r) console.log("[nestling-rlm] route: no decision (the reply was not valid JSON); the model starts the turn itself");
        // "Nothing to read" is checked before it is acted on; see answerableFromKnown.
        if (r && !r.files.length && !r.tool && !(await answerableFromKnown(ask))) {
          console.log(`[nestling-rlm] route: ${r.answerFrom || "?"} was not confirmed; reading the library`);
          r = (await route({ ...ask, mustRead: true })) ?? r;
        }
        // The model picks files by their titles; the library is also asked which file's text
        // matches the question best, and that file is read too when the model did not pick it.
        // A file written for another age band answers a different child. Which band a file is for
        // is read from its own headings, so the check holds for whatever the library contains.
        const fits = (f: string) => fileFitsAge(path.resolve(host.knowledgeRoot, f), Number.isFinite(ageMonths) && body.child_age_months != null ? ageMonths : undefined) !== false;
        const forThisAge = ask.files.filter(fits);
        const withBestMatch = (picked: string[], query: string): string[] => {
          const files = picked.filter(fits);
          if (files.length < picked.length) console.log(`[nestling-rlm] route: set aside for another age: ${picked.filter((f) => !fits(f)).join(", ")}`);
          const ranked = rankFiles(host.knowledgeRoot, forThisAge, query);
          const top = ranked[0];
          if (!top || files.includes(top.file)) return files;
          // Every file the model named was for another age: the best match of the right age stands in.
          if (!files.length) return picked.length ? [top.file] : [];
          const pickedScore = Math.max(0, ...files.map((f) => ranked.find((x) => x.file === f)?.score ?? 0));
          // Only when it is clearly the better match: a general question matches many files a little.
          return top.score >= pickedScore * LEXICAL_LEAD ? [files[0]!, top.file] : files;
        };
        // A part is read from one file: of the files the model named for it and the library's own
        // best match, the one whose text matches the part best.
        const bestFor = (picked: string[], topic: string): string[] => {
          const top = rankFiles(host.knowledgeRoot, forThisAge, topic)[0]?.file;
          const candidates = [...new Set([...picked.filter(fits), ...(top ? [top] : [])])];
          const ranked = rankFiles(host.knowledgeRoot, candidates, topic);
          return ranked.length ? [ranked[0]!.file] : candidates.slice(0, 1);
        };
        if (r?.parts.length) r = { ...r, parts: r.parts.map((p) => ({ ...p, files: bestFor(p.files, p.topic) })) };
        else if (r?.files.length) r = { ...r, files: withBestMatch(r.files, String(body.en_gloss || userMessage)) };
        if (r?.parts.length) console.log(`[nestling-rlm] plan: ${r.parts.map((p) => `${p.topic} -> ${p.files.join(", ") || "no file"}`).join(" | ")}`);
        if (r) console.log(`[nestling-rlm] route: ${r.answerFrom || "?"}; files ${r.files.join(", ") || "none"}; tool ${r.tool || "none"}`);
        planParts = r?.parts.map((p) => p.topic) ?? [];
        // A plan reads one file per part; together they get twice what a single read gets.
        return r ? routeToCells(r, String(body.en_gloss || userMessage), host.libraryReadChars * 2) : null;
      },
    });
    turnId = sent.turnId;
  } finally {
    host.settings.rlm.maxSteps = prevSteps;
  }

  const deadline = Date.now() + (host.settings.rlm.modelStallSec + 90) * 1000;
  const first = await awaitTurn(turnId, deadline);
  if (!first.end) throw new Error("the turn did not finish in time");
  const chain = [first, ...(first.end.type === "turn_done" ? await followContinuations(chatId, turnId, deadline) : [])];
  const events = chain.flatMap((c) => c.events);
  const done = [...chain].reverse().find((c) => c.end)?.end ?? first.end;
  const failed = done.type !== "turn_done";
  if (failed) console.error("[nestling-rlm] turn ended with", done.type, textOf(done.message ?? done.error ?? "").slice(0, 300));

  const toolCalls = events
    .filter((e) => e.type === "tool_call")
    .map((e) => {
      const call = (e.call ?? {}) as { toolId?: string; args?: Record<string, unknown>; callId?: string };
      const resultEv = events.find((r) => r.type === "tool_result" && r.callId === call.callId);
      const name = String(call.toolId ?? "");
      return {
        name: name.startsWith("clinical.") ? name.slice("clinical.".length) : name,
        arguments: call.args ?? {},
        ok: resultEv ? Boolean(resultEv.ok) : false,
        result: resultEv && resultEv.ok ? resultEv.result : undefined,
      };
    });

  // What the turn actually saw: the output of its cells. Tool results are already inside it,
  // because a tool only reaches the model through a cell that prints it.
  const evidenceChars = Math.floor(host.contextLength * 3.2 * 0.45);
  const cellOut = new Map<string, string>();
  for (const e of events) if (e.type === "cell_output") cellOut.set(String(e.cellId), (cellOut.get(String(e.cellId)) ?? "") + textOf(e.text));
  const perCell = Math.max(800, Math.floor(evidenceChars / Math.max(1, cellOut.size)));
  const evidence = [...cellOut.values()].map((t) => t.slice(0, perCell)).join("\n---\n");
  const history = priorTurns.map((t) => `${t.userText}\n${t.assistantText}`).join("\n");

  let reply = "";
  let via = "none";
  let unsupported: string[] = [];
  if (!failed) {
    try {
      const spoken = await speakParent({
        model: host.agent,
        question: userMessage,
        lang: replyLang,
        evidence,
        memory: memoryBlock,
        child: childContext,
        history,
        previous: priorTurns.slice(-4).map((t) => ({ question: t.userText, answer: t.assistantText })),
        recent: priorTurns
          .slice(-3)
          .map((t) => `Parent: ${t.userText.slice(0, 300)}\nNestling: ${t.assistantText.slice(0, 500)}`)
          .join("\n"),
        draft: String(done.finalText ?? ""),
        evidenceChars,
        lookedUp: [...cellOut.values()].some((t) => t.trim()),
        parts: planParts,
      });
      reply = spoken.text;
      via = spoken.via;
      unsupported = spoken.unsupported;
      if (spoken.draftRejected) console.log(`[nestling-rlm] reply via ${via}; the engine's answer was not used: ${spoken.draftRejected}`);
    } catch (e) {
      console.error("[nestling-rlm] speak failed", e);
    }
  }
  if (unsupported.length) console.error("[nestling-rlm] dropped unsupported numbers", unsupported.join(","));
  // The conversation keeps what the parent was told, not the draft it was made from. A draft that
  // failed the checks would otherwise stay in context and be quoted back as an earlier answer.
  if (!failed && reply !== String(done.finalText ?? "")) {
    const lastTurn = (await host.chats.getTurns(chatId)).at(-1);
    if (lastTurn) await host.orch.amendAnswer(chatId, lastTurn.id, reply);
  }

  const usageEv = [...events].reverse().find((e) => e.type === "context_usage");
  const compactions = events.filter((e) => e.type === "compaction").map((e) => String((e as { checkpoint?: { compactedBy?: string } }).checkpoint?.compactedBy || "rlm"));
  const board = await host.boards.get(chatId);

  return {
    // An empty reply is reported as such; the caller words the apology in the parent's language.
    // It is never papered over with an answer written from nothing.
    reply,
    reply_via: via,
    failed,
    error: failed ? textOf(done.message ?? done.error ?? done.type).slice(0, 400) : null,
    work_log: "",
    steps: events.filter((e) => e.type === "generation").length || 1,
    turns: chain.length,
    usage: done.usage ?? {},
    tools: { tool_calls: toolCalls },
    task_board: board.steps.length ? { goal: board.goal, steps: board.steps.map((s) => ({ text: s.text, status: s.status })) } : null,
    rlm: {
      engine: "typescript",
      steps: events.filter((e) => e.type === "generation").length || 1,
      turns: chain.length,
      compacted: compactions.length > 0,
      compaction_reason: compactions.includes("in_turn") ? "in_turn" : compactions[0] ?? null,
      stopped_at_limit: false,
      stm_vectors: 0,
      events: events.filter((e) => ["compaction", "status", "context_usage", "code_cell", "cell_done"].includes(String(e.type))),
    },
    medical_rag: null,
    child_rag: null,
    used: usageEv?.usedTokens ?? null,
    trigger: usageEv?.triggerTokens ?? null,
  };
}

const server = createServer(async (req, res) => {
  try {
    if (req.method === "GET" && req.url === "/health") {
      json(res, 200, { ok: true, engine: "typescript" });
      return;
    }
    if (req.method === "POST" && req.url === "/turn") {
      const body = JSON.parse((await readBody(req)) || "{}") as Record<string, unknown>;
      json(res, 200, await handleTurn(body));
      return;
    }
    json(res, 404, { error: "not found" });
  } catch (e) {
    console.error("[nestling-rlm] turn failed", e);
    json(res, 500, { error: e instanceof Error ? e.message : String(e) });
  }
});

server.listen(port, "127.0.0.1", () => {
  console.log(`[nestling-rlm] TypeScript orchestrator on http://127.0.0.1:${port}`);
});
